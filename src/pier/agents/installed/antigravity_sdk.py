import json
import shlex
from pathlib import Path
from typing import Any, override

from pier.agents.installed.base import BaseInstalledAgent, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import Trajectory
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.utils.trajectory_utils import format_trajectory_json

_SDK_VERSION = "0.1.9"
_UV_VERSION = "0.7.13"
_UV_INSTALLER_SHA256 = (
    "1bd6dcfae3377079ed9ebeb47cc814b0a9b6f42a81089e97de49cd26f7b9d2d2"
)
_REASONING_EFFORTS = frozenset(("minimal", "low", "medium", "high"))
_REQUIREMENTS_LOCK = "antigravity_sdk_requirements.lock"


class AntigravitySDK(BaseInstalledAgent):
    """Google Antigravity SDK coding agent."""

    SUPPORTS_ATIF = True
    _OUTPUT_FILENAME = "antigravity-sdk.txt"
    _TRAJECTORY_FILENAME = "trajectory.json"
    _RUNNER_PATH = "/installed-agent/antigravity_sdk_runner.py"
    _PYTHON_PATH = "/installed-agent/venv/bin/python"

    DEFAULT_SKILL_PATHS = [
        "~/.claude/skills",
        "~/.codex/skills",
        "~/.agents/skills",
        "~/.gemini/skills",
        "~/.config/opencode/skills",
    ]

    def __init__(
        self,
        *args: Any,
        reasoning_effort: str | None = "medium",
        load_skills: bool = True,
        skill_paths: list[str] | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        if version is not None and version != _SDK_VERSION:
            raise ValueError(
                f"Antigravity SDK runner requires google-antigravity=={_SDK_VERSION}; "
                f"got version={version!r}"
            )
        normalized_effort = (
            reasoning_effort if reasoning_effort is not None else "medium"
        ).lower()
        if normalized_effort not in _REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort {reasoning_effort!r}. Valid values: "
                f"{', '.join(sorted(_REASONING_EFFORTS))}"
            )
        self._reasoning_effort = normalized_effort
        self._load_skills = load_skills
        self._skill_paths = (
            list(skill_paths)
            if skill_paths is not None
            else list(self.DEFAULT_SKILL_PATHS)
        )
        super().__init__(*args, version=_SDK_VERSION, **kwargs)

    @staticmethod
    @override
    def name() -> str:
        return AgentName.ANTIGRAVITY_SDK.value

    @override
    def get_version_command(self) -> str | None:
        return (
            f'{self._PYTHON_PATH} -c "from importlib.metadata import version; '
            "print(version('google-antigravity'))\""
        )

    @override
    def install_spec(self) -> AgentInstallSpec:
        requirements_lock = Path(__file__).with_name(_REQUIREMENTS_LOCK).read_text()
        install = f"""
set -euo pipefail
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache bash ca-certificates curl
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y ca-certificates curl
elif command -v yum >/dev/null 2>&1; then
  yum install -y ca-certificates curl
else
  command -v curl >/dev/null 2>&1
fi
command -v sha256sum >/dev/null 2>&1
UV_INSTALLER=/tmp/uv-{_UV_VERSION}-installer.sh
curl -LsSf https://astral.sh/uv/{_UV_VERSION}/install.sh -o "$UV_INSTALLER"
echo "{_UV_INSTALLER_SHA256}  $UV_INSTALLER" | sha256sum -c -
env UV_INSTALL_DIR=/usr/local/bin sh "$UV_INSTALLER"
mkdir -p /installed-agent
export UV_PYTHON_INSTALL_DIR=/installed-agent/python
uv venv --python 3.12 /installed-agent/venv
cat > /installed-agent/{_REQUIREMENTS_LOCK} <<'PIER_ANTIGRAVITY_REQUIREMENTS'
{requirements_lock.rstrip()}
PIER_ANTIGRAVITY_REQUIREMENTS
uv pip install --python {self._PYTHON_PATH} \
  --require-hashes \
  --no-deps \
  -r /installed-agent/{_REQUIREMENTS_LOCK}
chmod -R a+rX /installed-agent
chmod +x /installed-agent/venv/lib/python3.12/site-packages/google/antigravity/bin/localharness
""".strip()
        return AgentInstallSpec(
            agent_name=self.name(),
            version=_SDK_VERSION,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=install,
                ),
                # Restore the task's configured agent user after the privileged
                # Dockerfile layer, and verify that user can execute the runtime.
                InstallStep(user="agent", run=self.get_version_command() or "true"),
            ],
            verification_command=self.get_version_command(),
            metadata={
                "python": "3.12",
                "uv": _UV_VERSION,
                "google-antigravity": _SDK_VERSION,
                "requirements-lock": _REQUIREMENTS_LOCK,
            },
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        runner = Path(__file__).with_name("antigravity_sdk_runner.py")
        await environment.upload_file(runner, self._RUNNER_PATH)
        await self.exec_as_root(environment, f"chmod a+r {self._RUNNER_PATH}")

    @override
    def network_allowlist(self) -> NetworkAllowlist:
        urls = [server.url for server in self.mcp_servers if server.url]
        return allowlist_from_urls(urls, default_domains=[".googleapis.com"])

    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> float | None:
        if not self.model_name:
            return None
        try:
            import litellm
        except ImportError:
            self.logger.warning("litellm not available; cost_usd left as None")
            return None

        pricing: dict[str, Any] | None = None
        for key in (self.model_name, self.model_name.split("/", 1)[-1]):
            if entry := litellm.model_cost.get(key):
                pricing = entry
                break
        if pricing is None:
            self.logger.warning(
                "No LiteLLM pricing for '%s'; cost_usd left as None", self.model_name
            )
            return None

        input_rate = pricing.get("input_cost_per_token") or 0.0
        output_rate = pricing.get("output_cost_per_token") or 0.0
        cache_rate = pricing.get("cache_read_input_token_cost") or input_rate
        cached = cached_tokens or 0
        uncached = max(0, (prompt_tokens or 0) - cached)
        return (
            uncached * input_rate
            + cached * cache_rate
            + (completion_tokens or 0) * output_rate
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory_path = self.logs_dir / self._TRAJECTORY_FILENAME
        if not trajectory_path.exists():
            self.logger.debug("No Antigravity trajectory found at %s", trajectory_path)
            return
        try:
            trajectory = Trajectory.model_validate_json(trajectory_path.read_text())
        except (OSError, ValueError):
            self.logger.exception("Failed to parse Antigravity trajectory")
            return
        if trajectory.final_metrics is None:
            return

        metrics = trajectory.final_metrics
        if metrics.total_cost_usd is None:
            metrics.total_cost_usd = self._compute_cost_from_pricing(
                metrics.total_prompt_tokens,
                metrics.total_completion_tokens,
                metrics.total_cached_tokens,
            )
            if metrics.total_cost_usd is not None:
                trajectory_path.write_text(
                    format_trajectory_json(trajectory.to_json_dict())
                )
        populate_context_from_final_metrics(context, metrics)
        context.n_agent_steps = sum(step.source == "agent" for step in trajectory.steps)

    def _mcp_servers_json(self) -> str | None:
        if not self.mcp_servers:
            return None
        servers: list[dict[str, Any]] = []
        for server in self.mcp_servers:
            if server.transport == "sse":
                raise ValueError(
                    f"Antigravity SDK {_SDK_VERSION} does not support MCP SSE "
                    f"transport (server {server.name!r}); use streamable-http"
                )
            entry: dict[str, Any] = {
                "name": server.name,
                "transport": server.transport,
            }
            if server.transport == "stdio":
                entry.update({"command": server.command, "args": server.args})
            else:
                entry["url"] = server.url
            servers.append(entry)
        return json.dumps(servers)

    def _skills_json(self) -> str:
        if not self._load_skills:
            return "[]"
        paths = list(self._skill_paths)
        if self.skills_dir and self.skills_dir not in paths:
            paths.insert(0, self.skills_dir)
        return json.dumps(paths)

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        provider, separator, model = (self.model_name or "").partition("/")
        if provider != "google" or not separator or not model:
            raise ValueError(
                "Antigravity SDK model_name must be in the format google/<model>"
            )
        api_key = self._get_env("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        env = self.build_process_env(
            {
                "GEMINI_API_KEY": api_key,
                "MODEL_NAME": self.model_name,
                "REASONING_EFFORT": self._reasoning_effort,
                "SKILLS_PATHS_JSON": self._skills_json(),
                "SESSION_ID": environment.session_id,
            }
        )
        if mcp_json := self._mcp_servers_json():
            env["MCP_SERVERS_JSON"] = mcp_json

        await self.exec_as_agent(
            environment,
            command=(
                f"{self._PYTHON_PATH} {self._RUNNER_PATH} "
                f"--instruction {shlex.quote(instruction)} "
                f"--logs-dir /logs/agent "
                f"--trajectory-path /logs/agent/{self._TRAJECTORY_FILENAME} "
                f"2>&1 </dev/null | stdbuf -oL tee "
                f"/logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env=env,
        )

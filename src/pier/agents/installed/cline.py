"""Pier harness for the Cline CLI."""

from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pier.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json


class Cline(BaseInstalledAgent):
    """Run Cline headlessly and convert its native session to ATIF v1.7."""

    SUPPORTS_ATIF = True
    _OUTPUT_FILENAME = "cline.txt"
    _NVM_VERSION = "0.40.2"

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--thinking",
            type="enum",
            choices=["none", "low", "medium", "high", "xhigh"],
        ),
        CliFlag(
            "max_consecutive_mistakes",
            cli="--retries",
            type="int",
        ),
    ]

    _PROVIDER_API_KEY_ENVS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "cline": "CLINE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    def __init__(
        self,
        *args: Any,
        tarball_url: str | None = None,
        tarball_sha256: str | None = None,
        reasoning_effort: str | None = None,
        max_consecutive_mistakes: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        if tarball_url is None:
            tarball_url = kwargs.pop("tarball-url", None)
        else:
            kwargs.pop("tarball-url", None)
        if tarball_sha256 is None:
            tarball_sha256 = kwargs.pop("tarball-sha256", None)
        else:
            kwargs.pop("tarball-sha256", None)
        if reasoning_effort is None:
            reasoning_effort = kwargs.pop("reasoning-effort", None)
        else:
            kwargs.pop("reasoning-effort", None)
        if max_consecutive_mistakes is None:
            max_consecutive_mistakes = kwargs.pop("max-consecutive-mistakes", None)
        else:
            kwargs.pop("max-consecutive-mistakes", None)

        if tarball_sha256 is not None:
            tarball_sha256 = tarball_sha256.strip().lower()
            if tarball_url is None:
                raise ValueError("tarball_sha256 requires tarball_url")
            if re.fullmatch(r"[0-9a-f]{64}", tarball_sha256) is None:
                raise ValueError(
                    "tarball_sha256 must be a 64-character SHA-256 hex digest"
                )

        super().__init__(
            *args,
            reasoning_effort=reasoning_effort,
            max_consecutive_mistakes=max_consecutive_mistakes,
            **kwargs,
        )

        max_mistakes = self._resolved_flags.get("max_consecutive_mistakes")
        if max_mistakes is not None and max_mistakes < 1:
            raise ValueError("max_consecutive_mistakes must be at least 1")

        self._tarball_url = tarball_url
        self._tarball_sha256 = tarball_sha256

    @staticmethod
    def name() -> str:
        return AgentName.CLINE.value

    @staticmethod
    def _source_nvm() -> str:
        return (
            'export NVM_DIR="$HOME/.nvm"; '
            'if [ -s "$NVM_DIR/nvm.sh" ]; then . "$NVM_DIR/nvm.sh"; fi'
        )

    def get_version_command(self) -> str | None:
        return f"{self._source_nvm()}; cline --version || cline version"

    def _install_package_command(self) -> str:
        if self._tarball_url is not None:
            url = shlex.quote(self._tarball_url)
            tarball_path = "/tmp/pier-cline-cli.tgz"
            commands = [f"curl -fsSL --retry 3 {url} -o {tarball_path}"]
            if self._tarball_sha256 is not None:
                checksum_line = shlex.quote(f"{self._tarball_sha256}  {tarball_path}")
                commands.append(f"printf '%s\\n' {checksum_line} | sha256sum -c -")
            commands.append(
                f"npm install -g --ignore-scripts -- {shlex.quote(tarball_path)}"
            )
            return " && ".join(commands)

        version_spec = f"@{self._version}" if self._version else "@latest"
        return f"npm install -g cline{version_spec}"

    def install_spec(self) -> AgentInstallSpec:
        root_run = (
            "set -e; "
            "if [ -f /etc/alpine-release ]; then"
            "  apk add --no-cache bash curl ca-certificates nodejs npm;"
            " elif command -v apt-get >/dev/null 2>&1; then"
            "  apt-get update && apt-get install -y --no-install-recommends curl ca-certificates;"
            " elif command -v yum >/dev/null 2>&1; then"
            "  yum install -y curl ca-certificates;"
            " else"
            "  command -v curl >/dev/null 2>&1 || { echo 'curl is required' >&2; exit 1; };"
            " fi"
        )
        agent_run = (
            "set -euo pipefail; "
            "if [ ! -f /etc/alpine-release ]; then "
            f"curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v{self._NVM_VERSION}/install.sh | bash; "
            f"{self._source_nvm()}; "
            "command -v nvm >/dev/null 2>&1 || { echo 'nvm failed to load' >&2; exit 1; }; "
            "nvm install 22; nvm alias default 22; "
            "fi; "
            "npm --version; "
            f"{self._install_package_command()}; "
            "cline --version || cline version"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="agent", run=agent_run),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        domains: set[str] = set()
        provider, _ = self._provider_and_model()
        defaults = {
            "anthropic": ["api.anthropic.com"],
            "cline": ["app.cline.bot", "api.cline.bot"],
            "gemini": ["generativelanguage.googleapis.com"],
            "google": ["generativelanguage.googleapis.com"],
            "openai": ["api.openai.com"],
            "openrouter": ["openrouter.ai"],
        }
        domains.update(defaults.get(provider, []))
        for key in (
            "CLINE_API_URL",
            "ANTHROPIC_BASE_URL",
            "OPENAI_BASE_URL",
        ):
            value = self._get_env(key)
            if value:
                hostname = urlparse(
                    value if "://" in value else f"https://{value}"
                ).hostname
                if hostname:
                    domains.add(hostname)
        return NetworkAllowlist(domains=sorted(domains or {"api.cline.bot"}))

    def _provider_and_model(self) -> tuple[str, str]:
        value = self.model_name or ""
        if ":" in value:
            provider, model = value.split(":", 1)
            return provider, model
        if "/" in value:
            provider, model = value.split("/", 1)
            return provider, model
        return "cline", value

    def _runtime_env(self, provider: str) -> dict[str, str]:
        env = self.build_process_env()
        for key in (
            "API_KEY",
            "CLINE_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "CLINE_API_URL",
            "ANTHROPIC_BASE_URL",
            "OPENAI_BASE_URL",
        ):
            if value := self._get_env(key):
                env[key] = value

        provider_key = self._PROVIDER_API_KEY_ENVS.get(provider)
        if provider_key and provider_key not in env and env.get("API_KEY"):
            env[provider_key] = env["API_KEY"]
        if provider_key and not env.get(provider_key):
            raise ValueError(
                f"{provider_key} or API_KEY environment variable is required"
            )

        api_key = env.get(provider_key) if provider_key else env.get("API_KEY")
        if api_key:
            env["PIER_CLINE_API_KEY"] = api_key

        env.update(
            {
                "CLINE_WRITE_PROMPT_ARTIFACTS": "1",
                "CLINE_PROMPT_ARTIFACT_DIR": "/logs/agent",
            }
        )
        return env

    def _build_run_command(self, instruction: str) -> tuple[str, dict[str, str]]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction is empty before invoking Cline")

        provider, model = self._provider_and_model()
        env = self._runtime_env(provider)
        api_key_argument = "__PIER_CLINE_API_KEY_REFERENCE__"
        args = [
            "cline",
            "--json",
            "--auto-approve",
            "true",
            "--provider",
            provider,
            "--key",
            api_key_argument,
        ]
        if model:
            args.extend(["--model", model])
        cli_flags = self.build_cli_flags()
        if cli_flags:
            args.extend(shlex.split(cli_flags))
        args.extend(["--", instruction])
        rendered_args = " ".join(
            '"$PIER_CLINE_API_KEY"'
            if argument == api_key_argument
            else shlex.quote(argument)
            for argument in args
        )

        global_state = shlex.quote('{"welcomeViewCompleted":true,"isNewUser":false}')
        command = (
            f"{self._source_nvm()}; "
            "mkdir -p /logs/agent ~/.cline/data; "
            f"printf '%s\\n' {global_state} > ~/.cline/data/globalState.json; "
            "set -o pipefail; "
            f"{rendered_args} "
            "</dev/null 2>&1 | tee /logs/agent/cline.txt"
        )
        return command, env

    def _find_session_messages_file(self) -> Path | None:
        sessions_dir = self.logs_dir / "sessions"
        if not sessions_dir.is_dir():
            return None
        candidates = list(sessions_dir.rglob("*.messages.json"))
        if not candidates:
            return None
        try:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        except OSError:
            return None

    @staticmethod
    def _timestamp(value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
            except (OSError, ValueError, OverflowError):
                return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _split_blocks(
        content: Any,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], str]:
        if isinstance(content, str):
            return ([content] if content else [], [], [], "")
        if not isinstance(content, list):
            return [], [], [], ""

        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif block_type == "thinking":
                reasoning = block.get("text", block.get("thinking"))
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
            elif block_type == "tool_use":
                tool_uses.append(block)
            elif block_type == "tool_result":
                tool_results.append(block)
            elif block_type == "image":
                media_type = block.get("mediaType", block.get("media_type", "image"))
                text_parts.append(f"[image: {media_type}]")
        return text_parts, tool_uses, tool_results, "\n".join(reasoning_parts)

    @staticmethod
    def _tool_result_content(content: Any) -> str | None:
        if content is None or isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _attach_tool_results(
        steps: list[Step], tool_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        unmatched: list[dict[str, Any]] = []
        for result in tool_results:
            tool_use_id = result.get("tool_use_id")
            target: Step | None = None
            if isinstance(tool_use_id, str):
                for step in reversed(steps):
                    if step.source != "agent" or not step.tool_calls:
                        continue
                    if any(
                        call.tool_call_id == tool_use_id for call in step.tool_calls
                    ):
                        target = step
                        break
            if target is None:
                unmatched.append(result)
                continue
            observation_result = ObservationResult(
                source_call_id=tool_use_id,
                content=Cline._tool_result_content(result.get("content")),
            )
            if target.observation is None:
                target.observation = Observation(results=[observation_result])
            else:
                target.observation.results.append(observation_result)
        return unmatched

    def _convert_session_to_trajectory(
        self, document: dict[str, Any], *, session_id: str
    ) -> Trajectory:
        messages = document.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Cline session contains no messages")

        steps: list[Step] = []
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0
        saw_metrics = False
        saw_cost = False
        default_model = self.model_name

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            text_parts, tool_uses, tool_results, reasoning = self._split_blocks(
                message.get("content")
            )
            timestamp = self._timestamp(message.get("ts"))

            if role == "user":
                unmatched = self._attach_tool_results(steps, tool_results)
                if unmatched:
                    text_parts.append(json.dumps(unmatched, ensure_ascii=False))
                text = "\n".join(part for part in text_parts if part).strip()
                if text:
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=timestamp,
                            source="user",
                            message=text,
                        )
                    )
                continue

            if role != "assistant":
                continue

            model_info = message.get("modelInfo")
            if isinstance(model_info, dict) and isinstance(model_info.get("id"), str):
                default_model = model_info["id"]

            raw_metrics = message.get("metrics")
            metrics: Metrics | None = None
            if isinstance(raw_metrics, dict):
                prompt = raw_metrics.get("inputTokens")
                completion = raw_metrics.get("outputTokens")
                cached = raw_metrics.get("cacheReadTokens")
                cache_write = raw_metrics.get("cacheWriteTokens")
                cost = raw_metrics.get("cost")
                numeric_values = (prompt, completion, cached, cache_write, cost)
                if any(value is not None for value in numeric_values):
                    saw_metrics = True
                    prompt_value = prompt if isinstance(prompt, int) else None
                    completion_value = (
                        completion if isinstance(completion, int) else None
                    )
                    cached_value = cached if isinstance(cached, int) else None
                    cost_value = (
                        float(cost)
                        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                        else None
                    )
                    if prompt_value is not None:
                        total_prompt += prompt_value
                    if completion_value is not None:
                        total_completion += completion_value
                    if cached_value is not None:
                        total_cached += cached_value
                    if cost_value is not None:
                        total_cost += cost_value
                        saw_cost = True
                    extra = (
                        {"cache_write_tokens": cache_write}
                        if isinstance(cache_write, int)
                        else None
                    )
                    metrics = Metrics(
                        prompt_tokens=prompt_value,
                        completion_tokens=completion_value,
                        cached_tokens=cached_value,
                        cost_usd=cost_value,
                        extra=extra,
                    )

            tool_calls: list[ToolCall] | None = None
            if tool_uses:
                tool_calls = []
                for index, tool_use in enumerate(tool_uses):
                    raw_id = tool_use.get("id")
                    tool_call_id = (
                        raw_id
                        if isinstance(raw_id, str) and raw_id
                        else f"tool-{len(steps) + 1}-{index + 1}"
                    )
                    arguments = tool_use.get("input")
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_call_id=tool_call_id,
                            function_name=str(tool_use.get("name") or "unknown"),
                            arguments=arguments,
                        )
                    )

            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="agent",
                    model_name=default_model,
                    message="\n".join(part for part in text_parts if part).strip(),
                    reasoning_content=reasoning.strip() or None,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    llm_call_count=1,
                )
            )

        if not steps:
            raise ValueError("Cline session contains no convertible messages")

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_prompt if saw_metrics else None,
            total_completion_tokens=total_completion if saw_metrics else None,
            total_cached_tokens=total_cached if saw_metrics else None,
            total_cost_usd=total_cost if saw_cost else None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=None,
            ),
        )
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=str(document.get("sessionId") or session_id),
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=default_model,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        session_path = self._find_session_messages_file()
        if session_path is None:
            return
        try:
            document = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(document, dict):
            return

        trajectory = self._convert_session_to_trajectory(
            document, session_id=session_path.parent.name
        )
        (self.logs_dir / "trajectory.json").write_text(
            format_trajectory_json(trajectory.to_json_dict()), encoding="utf-8"
        )
        if trajectory.final_metrics is not None:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        command, env = self._build_run_command(instruction)
        try:
            await self.exec_as_agent(environment, command=command, env=env)
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        "if [ -d ~/.cline/data/sessions ]; then "
                        "mkdir -p /logs/agent/sessions && "
                        "cp -r ~/.cline/data/sessions/* /logs/agent/sessions/; "
                        "fi; "
                        "if [ -f ~/.cline/data/taskHistory.json ]; then "
                        "cp ~/.cline/data/taskHistory.json /logs/agent/taskHistory.json; "
                        "fi"
                    ),
                )
            except Exception:
                self.logger.debug("Failed to preserve Cline session artifacts")

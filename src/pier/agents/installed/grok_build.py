import os
import shlex
from pathlib import Path
from typing import Any

from pier.agents.installed.base import CliFlag, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.agents.installed.claude_code import ClaudeCode
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths

_DEFAULT_AUTH_FILE = "~/.grok/auth.json"


class GrokBuild(ClaudeCode):
    """xAI's Grok Build CLI (https://github.com/xai-org/grok-build).

    Runs ``grok -p`` headless with ``--output-format streaming-messages-json``,
    which emits the same NDJSON wire format as Claude Code's stream-json mode
    (system/init, assistant, user, result events with Anthropic-style message
    bodies). We therefore subclass :class:`ClaudeCode` purely to reuse its
    event-to-ATIF trajectory conversion; install, auth, and invocation are
    Grok-specific.

    Auth resolution order:
      1. ``XAI_API_KEY`` env (API auth against api.x.ai)
      2. a Grok CLI session ``auth.json`` (kwarg ``auth_file``, default
         ``~/.grok/auth.json``) uploaded into an ephemeral in-container
         ``GROK_HOME`` under /tmp, removed when the run finishes so the
         credential never lands in the persisted logs mount.
    """

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "grok-build.txt"
    _STDERR_FILENAME = "grok-build-stderr.txt"
    # Outside /logs so the session credential is never persisted to the host.
    _GROK_HOME = "/tmp/grok-build-home"

    CLI_FLAGS = [
        CliFlag(
            "max_turns",
            cli="--max-turns",
            type="int",
            env_fallback="GROK_BUILD_MAX_TURNS",
        ),
        CliFlag(
            "reasoning_effort",
            cli="--reasoning-effort",
            type="enum",
            choices=["low", "medium", "high", "xhigh"],
            env_fallback="GROK_BUILD_EFFORT_LEVEL",
        ),
        CliFlag(
            "disable_web_search",
            cli="--disable-web-search",
            type="bool",
            default=True,
        ),
        CliFlag(
            "no_plan",
            cli="--no-plan",
            type="bool",
            default=True,
        ),
        CliFlag(
            # Default on: pier's converter has no inline-subagent support, so
            # subagent events would interleave into the mainline trajectory.
            # Opt back in with no_subagents=False if step attribution does not
            # matter for your analysis.
            "no_subagents",
            cli="--no-subagents",
            type="bool",
            default=True,
        ),
    ]
    ENV_VARS = []

    def __init__(
        self,
        logs_dir: Path,
        auth_file: str | None = None,
        *args,
        **kwargs,
    ):
        # Inherited ClaudeCode kwargs with no grok CLI equivalent would be
        # silently dropped by BaseInstalledAgent's kwarg extraction; reject
        # them eagerly instead.
        claude_only_kwargs = (
            {flag.kwarg for flag in ClaudeCode.CLI_FLAGS}
            | {var.kwarg for var in ClaudeCode.ENV_VARS}
        ) - {flag.kwarg for flag in self.CLI_FLAGS}
        rejected = sorted(claude_only_kwargs & set(kwargs))
        if rejected:
            raise ValueError(
                "kwargs not supported by the grok-build agent (Claude Code "
                f"only): {', '.join(rejected)}"
            )

        self._auth_file = auth_file or _DEFAULT_AUTH_FILE
        self._instruction_used: str | None = None
        super().__init__(logs_dir, *args, **kwargs)

        if self.memory_dir:
            # memory_dir can only come from the agent config; fail fast.
            raise ValueError("The grok-build agent does not support memory_dir yet.")
        if self.mcp_servers or self.skills_dir:
            # These may be injected from task.config.environment, which other
            # agents run with; degrade to a warning instead of failing trials.
            self.logger.warning(
                "The grok-build agent does not support MCP servers or skills "
                "yet; ignoring them for this run."
            )
        # Fail before the environment is built, not after the CLI install.
        self._session_auth_path()

    @staticmethod
    def name() -> str:
        return AgentName.GROK_BUILD.value

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.grok/bin:$PATH"; grok --version'

    def install_spec(self) -> AgentInstallSpec:
        root_run = (
            "if command -v apt-get &> /dev/null; then"
            "  apt-get update && apt-get install -y curl ca-certificates;"
            " elif command -v apk &> /dev/null; then"
            # coreutils provides stdbuf, which busybox lacks
            "  apk add --no-cache curl bash ca-certificates coreutils;"
            " elif command -v yum &> /dev/null; then"
            "  yum install -y curl ca-certificates;"
            " else"
            '  echo "Warning: no known package manager, assuming curl exists" >&2;'
            " fi"
        )
        version_arg = f" -s {shlex.quote(self._version)}" if self._version else ""
        agent_run = (
            "set -euo pipefail; "
            f"curl -fsSL https://x.ai/cli/install.sh | bash{version_arg} && "
            "echo 'export PATH=\"$HOME/.grok/bin:$PATH\"' >> ~/.bashrc && "
            'export PATH="$HOME/.grok/bin:$PATH" && '
            "grok --version"
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
        default_domains = [
            # installer script + release binaries
            "x.ai",
            "storage.googleapis.com",
            # session-auth inference proxy + OIDC token refresh
            "cli-chat-proxy.grok.com",
            "auth.x.ai",
        ]
        if self._get_env("XAI_API_KEY"):
            default_domains.append("api.x.ai")
        override_urls = [
            self._get_env(env_key)
            for env_key in ("GROK_CLI_CHAT_PROXY_BASE_URL", "GROK_XAI_API_BASE_URL")
        ]
        return allowlist_from_urls(
            [url for url in override_urls if url], default_domains=default_domains
        )

    def _session_auth_path(self) -> Path | None:
        """Host path of the session auth.json, or None for API-key auth.

        Raises ValueError when neither auth source is available.
        """
        auth_path = Path(self._auth_file).expanduser()
        if self._get_env("XAI_API_KEY"):
            if (
                "XAI_API_KEY" not in self._extra_env
                and os.environ.get("XAI_API_KEY")
                and auth_path.is_file()
            ):
                # Available model ids can differ per credential/auth mode, so
                # an ambient key silently overriding a session file is worth a
                # visible note.
                self.logger.warning(
                    "Using XAI_API_KEY from the host environment; ignoring "
                    f"the Grok session file at {auth_path}"
                )
            return None
        if not auth_path.is_file():
            raise ValueError(
                f"Grok auth file not found at {auth_path} and XAI_API_KEY is not "
                "set. Run `grok login` on the host or provide auth_file/XAI_API_KEY."
            )
        return auth_path

    def _resolved_model(self) -> str | None:
        """Model id for --model, stripping a pier-style provider prefix
        (e.g. "xai/grok-4.6"). None defers to the CLI's own default."""
        if not self.model_name:
            return None
        return self.model_name.split("/", 1)[-1]

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        self._instruction_used = instruction
        escaped_instruction = shlex.quote(instruction)
        grok_home = self._GROK_HOME

        # Resolve the auth mode BEFORE scrubbing empty values below — an
        # explicit XAI_API_KEY="" override must keep forcing session auth.
        auth_path = self._session_auth_path()

        env: dict[str, str | None] = {
            "GROK_HOME": grok_home,
            "GROK_TELEMETRY_ENABLED": "0",
            "GROK_FEEDBACK_ENABLED": "0",
            "XAI_API_KEY": self._get_env("XAI_API_KEY"),
            "GROK_CLI_CHAT_PROXY_BASE_URL": self._get_env(
                "GROK_CLI_CHAT_PROXY_BASE_URL"
            ),
            "GROK_XAI_API_BASE_URL": self._get_env("GROK_XAI_API_BASE_URL"),
        }
        process_env = self.build_process_env(env)
        # Drop empty credentials so the CLI does not select API-key auth on an
        # empty XAI_API_KEY while pier prepared session auth. _exec re-merges
        # self._extra_env over the env we pass, so the empty key must be kept
        # out of there too — scoped to this run and restored afterwards, since
        # _extra_env is shared agent state.
        process_env = {key: value for key, value in process_env.items() if value}
        original_extra_env = self._extra_env
        if original_extra_env.get("XAI_API_KEY") == "":
            self._extra_env = {
                key: value
                for key, value in original_extra_env.items()
                if key != "XAI_API_KEY"
            }

        await self.exec_as_agent(
            environment, command=f"mkdir -p {grok_home}", env=process_env
        )

        if auth_path:
            # upload_file keeps the credential out of both the command line
            # and the exec env (docker materializes exec env as host argv).
            remote_auth_path = f"{grok_home}/auth.json"
            await environment.upload_file(auth_path, remote_auth_path)
            # upload_file copies as root; fix ownership for the agent user.
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=(
                        f"chown {environment.default_user} {remote_auth_path}"
                        f" && chmod 600 {remote_auth_path}"
                    ),
                )

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        stderr_path = (EnvironmentPaths.agent_dir / self._STDERR_FILENAME).as_posix()

        model = self._resolved_model()
        model_flag = f"--model {shlex.quote(model)} " if model else ""

        # No auto-update checks mid-trial; the installed version is
        # authoritative. -p= keeps hyphen-leading instructions from parsing as
        # CLI options; stderr stays out of the NDJSON stream the trajectory is
        # parsed from.
        grok_command = (
            f"printf '[cli]\\nauto_update = false\\n' > {grok_home}/config.toml"
            " && "
            'export PATH="$HOME/.grok/bin:$PATH" && '
            f"grok {model_flag}"
            f"--output-format streaming-messages-json "
            f"--permission-mode bypassPermissions "
            f"{extra_flags}"
            f"-p={escaped_instruction} </dev/null "
            f"2>{stderr_path} | stdbuf -oL tee {output_path}"
        )

        try:
            await self.exec_as_agent(environment, command=grok_command, env=process_env)
        finally:
            self._extra_env = original_extra_env
            # GROK_HOME holds the session credential; never leave it behind.
            # Best-effort with a short timeout — a wedged container must not
            # hang the trial or mask the run's own error (the environment is
            # torn down after the trial either way).
            try:
                await self.exec_as_agent(
                    environment,
                    command=f"rm -rf {grok_home} || true",
                    timeout_sec=60,
                )
            except Exception:
                self.logger.debug("grok-build credential cleanup failed")

    def _load_stream_events(self) -> tuple[list[dict[str, Any]], str | None]:
        """Parse the captured NDJSON stream into Claude-session-shaped events.

        Collects assistant/user message events plus the session id; the final
        result event's cost is read by the inherited
        ``_parse_total_cost_from_stream_json`` (via ``_OUTPUT_FILENAME``).
        """
        events: list[dict[str, Any]] = []
        session_id: str | None = None
        for event in self._iter_stream_events():
            if session_id is None and isinstance(event.get("session_id"), str):
                session_id = event["session_id"]
            if event.get("type") in ("assistant", "user") and isinstance(
                event.get("message"), dict
            ):
                # NOTE: subagent events (parent_tool_use_id set; only emitted
                # when no_subagents=False) are kept in stream order — do NOT
                # map them onto isSidechain: the inherited converter hoists
                # sidechain events ahead of the mainline, which reorders the
                # trajectory when events carry no timestamps.
                # The inherited converter sorts on string timestamps; drop
                # any non-string form the CLI may emit.
                if not isinstance(event.get("timestamp"), str):
                    event.pop("timestamp", None)
                events.append(event)
        return events, session_id

    def populate_context_post_run(self, context: AgentContext) -> None:
        events, session_id = self._load_stream_events()
        if not events:
            self.logger.debug("No grok-build stream events found")
            return

        # Prepend the instruction as the opening user step unless the CLI
        # already echoed the prompt as a user event.
        first_user_message = next(
            (
                event["message"]
                for event in events
                if event.get("type") == "user" and not event.get("isSidechain")
            ),
            None,
        )
        prompt_echoed = bool(
            first_user_message
            and first_user_message.get("content") == self._instruction_used
        )
        if self._instruction_used and not prompt_echoed:
            events.insert(
                0,
                {
                    "type": "user",
                    "message": {"role": "user", "content": self._instruction_used},
                },
            )

        try:
            # The stream is wire-compatible with Claude Code session events, so
            # the inherited converter consumes it directly.
            trajectory = self._convert_raw_events_to_trajectory(
                events, session_id or "grok-build"
            )
        except Exception:
            self.logger.exception("Failed to convert grok-build stream")
            return
        if not trajectory:
            self.logger.debug("Failed to convert grok-build stream to trajectory")
            return

        self._write_trajectory(trajectory, context)

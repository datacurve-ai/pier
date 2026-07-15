from __future__ import annotations

import json
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from pier.models.trial.paths import EnvironmentPaths
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.utils.trajectory_utils import format_trajectory_json


@dataclass
class _SessionMetrics:
    input_tokens: int | None
    cache_read_tokens: int | None
    output_tokens: int | None
    peak_context_tokens: int | None
    summarization_count: int
    n_turns: int
    aiu: float | None


class CopilotCli(BaseInstalledAgent):
    """Run GitHub Copilot CLI inside a Pier-managed task environment."""

    SUPPORTS_ATIF = True

    _JSONL_FILENAME = "copilot-cli.jsonl"
    _OUTPUT_FILENAME = "copilot-cli.txt"
    _COPILOT_HOME = EnvironmentPaths.agent_dir / "copilot-home"
    _LOG_ROOT = EnvironmentPaths.agent_dir / "copilot-logs"
    _RE_VERSION = re.compile(r"(\d+\.\d+(?:\.\d+)?(?:-\d+)?)")

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--effort",
            type="enum",
            choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            env_fallback="COPILOT_CLI_EFFORT",
        ),
        CliFlag(
            "mode",
            cli="--mode",
            type="enum",
            choices=["interactive", "plan", "autopilot"],
            env_fallback="COPILOT_CLI_MODE",
        ),
        CliFlag(
            "context_tier",
            cli="--context",
            type="enum",
            choices=["default", "long_context"],
            env_fallback="COPILOT_CLI_CONTEXT_TIER",
        ),
        CliFlag("agent", cli="--agent", type="str", env_fallback="COPILOT_CLI_AGENT"),
        CliFlag("allow_all_tools", cli="--allow-all-tools", type="bool", default=True),
        CliFlag("no_ask_user", cli="--no-ask-user", type="bool", default=True),
        CliFlag("no_auto_update", cli="--no-auto-update", type="bool", default=True),
    ]

    def __init__(
        self,
        *args: Any,
        command_model_name: str | None = None,
        extra_args: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._command_model_name = command_model_name
        self._extra_args = extra_args
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return AgentName.COPILOT_CLI.value

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; copilot --version'

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        match = self._RE_VERSION.search(text)
        return match.group(1) if match else text

    def network_allowlist(self) -> NetworkAllowlist:
        # Squid's dstdomain ACL rejects overlapping bare and dotted domains.
        return NetworkAllowlist(
            domains=[
                ".github.com",
                ".githubcopilot.com",
                ".githubusercontent.com",
                "gh.io",
            ]
        )

    def install_spec(self) -> AgentInstallSpec:
        version_env = f" VERSION={shlex.quote(self._version)}" if self._version else ""
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=(
                        "if command -v apk >/dev/null 2>&1; then "
                        "echo 'Copilot CLI official installer does not support "
                        "musl-based images' >&2; exit 1; "
                        "elif command -v apt-get >/dev/null 2>&1; then "
                        "apt-get update && "
                        "apt-get install -y bash ca-certificates curl git; "
                        "elif command -v yum >/dev/null 2>&1; then "
                        "yum install -y bash ca-certificates curl git; "
                        "else echo 'No supported package manager found' >&2; exit 1; fi"
                    ),
                ),
                InstallStep(
                    user="agent",
                    run=(
                        "set -euo pipefail; "
                        f"curl -fsSL https://gh.io/copilot-install |{version_env} bash; "
                        'export PATH="$HOME/.local/bin:$PATH"; '
                        "copilot --version"
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def _extra_args_string(self) -> str:
        if self._extra_args is None:
            return ""
        if isinstance(self._extra_args, str):
            return self._extra_args.strip()
        return shlex.join([str(arg) for arg in self._extra_args])

    def _build_mcp_config_flag(self) -> str:
        if not self.mcp_servers:
            return ""

        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "type": "stdio",
                    "command": server.command,
                    "args": server.args,
                }
            else:
                transport = "http" if server.transport == "streamable-http" else "sse"
                servers[server.name] = {"type": transport, "url": server.url}

        config = json.dumps({"mcpServers": servers}, separators=(",", ":"))
        return f"--additional-mcp-config {shlex.quote(config)}"

    def _build_register_skills_command(self) -> str:
        if not self.skills_dir:
            return ""
        skills_dir = shlex.quote(self.skills_dir)
        return (
            f"if [ -d {skills_dir} ]; then "
            'mkdir -p "$COPILOT_HOME/skills" && '
            f'cp -r {skills_dir}/* "$COPILOT_HOME/skills/" 2>/dev/null || true; '
            "fi"
        )

    def _copilot_auth_env(self) -> dict[str, str]:
        token = (
            self._get_env("COPILOT_GITHUB_TOKEN")
            or self._get_env("GH_TOKEN")
            or self._get_env("GITHUB_TOKEN")
        )
        if not token:
            raise ValueError(
                "GitHub Copilot CLI authentication is required. Set "
                "COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN in the process "
                "environment or pass --env-file."
            )
        return {"COPILOT_GITHUB_TOKEN": token}

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        session_id = str(uuid.uuid4())
        model = self._command_model_name or (
            self.model_name.split("/", 1)[-1] if self.model_name else None
        )

        flags = [self.build_cli_flags()]
        if model:
            flags.append(f"--model {shlex.quote(model)}")
        flags.extend(
            [
                f"--session-id {shlex.quote(session_id)}",
                f"--log-dir {shlex.quote(self._LOG_ROOT.as_posix())}",
            ]
        )
        if mcp_flag := self._build_mcp_config_flag():
            flags.append(mcp_flag)
        if extra_args := self._extra_args_string():
            flags.append(extra_args)
        flag_text = " ".join(flag for flag in flags if flag)

        env = self.build_process_env(self._copilot_auth_env())
        env["COPILOT_HOME"] = self._COPILOT_HOME.as_posix()
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        jsonl_path = (EnvironmentPaths.agent_dir / self._JSONL_FILENAME).as_posix()
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()

        setup_commands = [
            (
                f"mkdir -p {shlex.quote(agent_dir)} "
                f"{shlex.quote(self._COPILOT_HOME.as_posix())} "
                f"{shlex.quote(self._LOG_ROOT.as_posix())}"
            ),
            'export PATH="$HOME/.local/bin:$PATH"',
        ]
        if skills_command := self._build_register_skills_command():
            setup_commands.append(skills_command)

        command = self._build_run_command(
            setup=" && ".join(setup_commands),
            instruction=instruction,
            flag_text=flag_text,
            jsonl_path=jsonl_path,
            output_path=output_path,
        )
        await self.exec_as_agent(environment, command=command, env=env)

    def _build_run_command(
        self,
        *,
        setup: str,
        instruction: str,
        flag_text: str,
        jsonl_path: str,
        output_path: str,
    ) -> str:
        run_script = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "set -o pipefail; "
            f"copilot -p {shlex.quote(instruction)} --output-format json --no-color "
            f"{flag_text} 2>&1 | tee {shlex.quote(jsonl_path)} | "
            f"tee {shlex.quote(output_path)}; "
            "exit ${PIPESTATUS[0]}"
        )
        return f"{setup} && bash -lc {shlex.quote(run_script)}"

    def populate_context_post_run(self, context: AgentContext) -> None:
        events_path = find_copilot_session_events(self.logs_dir)
        jsonl_path = self.logs_dir / self._JSONL_FILENAME
        events = _read_jsonl(events_path or jsonl_path)
        if not events:
            return

        trajectory = self._convert_events_to_trajectory(events)
        if trajectory is None:
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(
            format_trajectory_json(trajectory.to_json_dict()),
            encoding="utf-8",
        )

        if trajectory.final_metrics is not None:
            populate_context_from_final_metrics(context, trajectory.final_metrics)
        context.n_agent_steps = sum(step.source == "agent" for step in trajectory.steps)

        session_metrics = _parse_session_metrics(events)
        metadata = dict(context.metadata or {})
        if events_path is not None:
            metadata["copilot_session_events"] = str(
                events_path.relative_to(self.logs_dir)
            )
        if session_metrics.aiu is not None:
            metadata["copilot_aiu"] = session_metrics.aiu
        context.metadata = metadata or None

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        steps: list[Step] = []
        call_owners: dict[str, Step] = {}
        assistant_steps_by_call: dict[str, Step] = {}
        session_id: str | None = None

        def append_step(step: Step) -> None:
            step.step_id = len(steps) + 1
            steps.append(step)

        for event in events:
            event_type = event.get("type")
            data = event.get("data") or {}
            timestamp = event.get("timestamp")

            if event_type == "session.start":
                session_id = _string_or_none(
                    data.get("sessionId") or data.get("session_id")
                )
                continue

            if event_type == "user.message":
                if message := _flatten_content(data.get("content")):
                    append_step(
                        Step(
                            step_id=1,
                            timestamp=timestamp,
                            source="user",
                            message=message,
                        )
                    )
                continue

            if event_type == "assistant.message":
                tool_calls = _tool_calls(data.get("toolRequests"))
                prompt_tokens = _optional_int(data.get("inputTokens"))
                completion_tokens = _optional_int(data.get("outputTokens"))
                api_call_id = _string_or_none(
                    data.get("apiCallId")
                    or data.get("api_call_id")
                    or data.get("modelCallId")
                )
                if api_call_id and (
                    existing_step := assistant_steps_by_call.get(api_call_id)
                ):
                    _merge_assistant_event(
                        existing_step,
                        message=_flatten_content(data.get("content")),
                        tool_calls=tool_calls,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    for tool_call in tool_calls:
                        if tool_call.tool_call_id:
                            call_owners[tool_call.tool_call_id] = existing_step
                    continue

                metrics = (
                    Metrics(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    if prompt_tokens is not None or completion_tokens is not None
                    else None
                )
                step = Step(
                    step_id=1,
                    timestamp=timestamp,
                    source="agent",
                    model_name=data.get("model") or self.model_name,
                    message=(
                        _flatten_content(data.get("content"))
                        or ("Tool call" if tool_calls else "")
                    ),
                    tool_calls=tool_calls or None,
                    metrics=metrics,
                    llm_call_count=1,
                )
                append_step(step)
                if api_call_id:
                    assistant_steps_by_call[api_call_id] = step
                for tool_call in tool_calls:
                    if tool_call.tool_call_id:
                        call_owners[tool_call.tool_call_id] = step
                continue

            if event_type == "tool.execution_complete":
                content = _stringify_tool_result(data.get("result"))
                if data.get("error") is not None:
                    error = _stringify_tool_result(data.get("error"))
                    content = error if not content else f"{content}\n{error}"
                _attach_observation(
                    call_owners,
                    steps,
                    data.get("toolCallId"),
                    content,
                    timestamp,
                )
                continue

            if event_type == "message":
                source = "agent" if event.get("role") == "assistant" else "user"
                append_step(
                    Step(
                        step_id=1,
                        timestamp=timestamp,
                        source=source,
                        message=_flatten_content(event.get("content")),
                        model_name=(
                            event.get("model") or self.model_name
                            if source == "agent"
                            else None
                        ),
                        llm_call_count=1 if source == "agent" else None,
                    )
                )
                continue

            if event_type == "tool_use":
                call_id = str(event.get("id") or "")
                step = Step(
                    step_id=1,
                    timestamp=timestamp,
                    source="agent",
                    model_name=event.get("model") or self.model_name,
                    message=f"Executed {event.get('name') or 'tool'}",
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=str(event.get("name") or ""),
                            arguments=_normalize_arguments(event.get("input")),
                        )
                    ],
                    llm_call_count=1,
                )
                append_step(step)
                if call_id:
                    call_owners[call_id] = step
                continue

            if event_type == "tool_result":
                _attach_observation(
                    call_owners,
                    steps,
                    event.get("tool_use_id"),
                    _flatten_content(event.get("content")),
                    timestamp,
                )

        if not steps:
            return None

        session_metrics = _parse_session_metrics(events)
        extra: dict[str, Any] = {}
        if session_metrics.aiu is not None:
            extra["copilot_aiu"] = session_metrics.aiu
        if session_metrics.peak_context_tokens is not None:
            extra["peak_context_tokens"] = session_metrics.peak_context_tokens
        extra["summarization_count"] = session_metrics.summarization_count

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id or "copilot-cli",
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=session_metrics.input_tokens,
                total_completion_tokens=session_metrics.output_tokens,
                total_cached_tokens=session_metrics.cache_read_tokens,
                total_steps=len(steps),
                extra=extra,
            ),
        )


def find_copilot_session_events(agent_logs_dir: Path) -> Path | None:
    candidates = sorted(
        (agent_logs_dir / "copilot-home" / "session-state").glob("**/events.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _parse_session_metrics(events: list[dict[str, Any]]) -> _SessionMetrics:
    shutdowns: list[dict[str, Any]] = []
    n_turns = 0
    summarization_count = 0
    peak_context_tokens = 0
    fallback_input = 0
    fallback_output = 0
    has_fallback_input = False
    has_fallback_output = False
    fallback_by_call: dict[str, tuple[int | None, int | None]] = {}

    for event in events:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "session.shutdown":
            shutdowns.append(data)
            peak_context_tokens = max(
                peak_context_tokens, _optional_int(data.get("currentTokens")) or 0
            )
        elif event_type == "assistant.turn_start":
            n_turns += 1
        elif event_type == "session.compaction_complete":
            summarization_count += 1
            peak_context_tokens = max(
                peak_context_tokens,
                _optional_int(data.get("preCompactionTokens")) or 0,
            )
        elif event_type == "session.truncation":
            peak_context_tokens = max(
                peak_context_tokens,
                _optional_int(data.get("preTruncationTokensInMessages")) or 0,
            )
        elif event_type == "assistant.message":
            input_tokens = _optional_int(data.get("inputTokens"))
            output_tokens = _optional_int(data.get("outputTokens"))
            api_call_id = _string_or_none(
                data.get("apiCallId")
                or data.get("api_call_id")
                or data.get("modelCallId")
            )
            if api_call_id:
                previous_input, previous_output = fallback_by_call.get(
                    api_call_id, (None, None)
                )
                fallback_by_call[api_call_id] = (
                    _max_optional(previous_input, input_tokens),
                    _max_optional(previous_output, output_tokens),
                )
            else:
                if input_tokens is not None:
                    fallback_input += input_tokens
                    has_fallback_input = True
                if output_tokens is not None:
                    fallback_output += output_tokens
                    has_fallback_output = True

    for input_tokens, output_tokens in fallback_by_call.values():
        if input_tokens is not None:
            fallback_input += input_tokens
            has_fallback_input = True
        if output_tokens is not None:
            fallback_output += output_tokens
            has_fallback_output = True

    if not shutdowns:
        return _SessionMetrics(
            input_tokens=fallback_input if has_fallback_input else None,
            cache_read_tokens=None,
            output_tokens=fallback_output if has_fallback_output else None,
            peak_context_tokens=peak_context_tokens or None,
            summarization_count=summarization_count,
            n_turns=n_turns,
            aiu=None,
        )

    noncached = sum(_token_count(shutdown, "input") for shutdown in shutdowns)
    cache_read = sum(_token_count(shutdown, "cache_read") for shutdown in shutdowns)
    cache_write = sum(_token_count(shutdown, "cache_write") for shutdown in shutdowns)
    output = sum(_token_count(shutdown, "output") for shutdown in shutdowns)
    nano_aiu = sum(
        float(value)
        for shutdown in shutdowns
        if isinstance(value := shutdown.get("totalNanoAiu"), (int, float))
    )

    return _SessionMetrics(
        input_tokens=noncached + cache_read + cache_write,
        cache_read_tokens=cache_read,
        output_tokens=output,
        peak_context_tokens=peak_context_tokens or None,
        summarization_count=summarization_count,
        n_turns=n_turns,
        aiu=round(nano_aiu / 1_000_000_000, 6) if nano_aiu else None,
    )


def _token_count(shutdown: dict[str, Any], token_type: str) -> int:
    token_details = shutdown.get("tokenDetails")
    if not isinstance(token_details, dict):
        return 0
    details = token_details.get(token_type)
    if not isinstance(details, dict):
        return 0
    return _optional_int(details.get("tokenCount")) or 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _tool_calls(value: Any) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[ToolCall] = []
    for request in value:
        if not isinstance(request, dict):
            continue
        calls.append(
            ToolCall(
                tool_call_id=str(request.get("toolCallId") or request.get("id") or ""),
                function_name=str(request.get("name") or ""),
                arguments=_normalize_arguments(request.get("arguments")),
            )
        )
    return calls


def _normalize_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"value": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if arguments is None:
        return {}
    return {"value": arguments}


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text for part in content if (text := _flatten_content(part))
        )
    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            if key in content:
                return _flatten_content(content[key])
    return str(content)


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("content", "output", "stdout", "text", "message"):
            value = result.get(key)
            if isinstance(value, str):
                remainder = {k: v for k, v in result.items() if k != key}
                if remainder:
                    return f"{value}\n{json.dumps(remainder, ensure_ascii=False)}"
                return value
        return json.dumps(result, ensure_ascii=False)
    return _flatten_content(result)


def _attach_observation(
    call_owners: dict[str, Step],
    steps: list[Step],
    call_id: Any,
    content: str,
    timestamp: str | None,
) -> None:
    call_id_text = str(call_id or "")
    owner = call_owners.get(call_id_text) if call_id_text else None
    if owner is None:
        steps.append(
            Step(
                step_id=len(steps) + 1,
                timestamp=timestamp,
                source="agent",
                message=content or "Tool result",
                llm_call_count=0,
                extra={"source_call_id": call_id_text} if call_id_text else None,
            )
        )
        return

    result = ObservationResult(source_call_id=call_id_text, content=content or None)
    if owner.observation is None:
        owner.observation = Observation(results=[result])
    else:
        owner.observation.results.append(result)


def _merge_assistant_event(
    step: Step,
    *,
    message: str,
    tool_calls: list[ToolCall],
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    if message:
        existing_message = step.message if isinstance(step.message, str) else ""
        if not existing_message:
            step.message = message
        elif message.startswith(existing_message):
            step.message = message
        elif not existing_message.startswith(message):
            step.message = f"{existing_message}\n{message}"

    if tool_calls:
        existing_ids = {call.tool_call_id for call in step.tool_calls or []}
        step.tool_calls = [
            *(step.tool_calls or []),
            *(call for call in tool_calls if call.tool_call_id not in existing_ids),
        ]

    if prompt_tokens is not None or completion_tokens is not None:
        metrics = step.metrics or Metrics()
        if prompt_tokens is not None:
            metrics.prompt_tokens = max(metrics.prompt_tokens or 0, prompt_tokens)
        if completion_tokens is not None:
            metrics.completion_tokens = max(
                metrics.completion_tokens or 0, completion_tokens
            )
        step.metrics = metrics


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _max_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None

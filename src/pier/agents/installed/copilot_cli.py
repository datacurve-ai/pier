from __future__ import annotations

import json
import os
import re
import shlex
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
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
    SubagentTrajectoryRef,
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
    aiu: float | None
    premium_requests: int | None


@dataclass
class _CallUsage:
    """Token usage reported for a single upstream API call."""

    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class _Scope:
    """Steps and lifecycle facts collected for one agent's event stream."""

    steps: list[Step] = field(default_factory=list)
    call_owners: dict[str, Step] = field(default_factory=dict)
    session_id: str | None = None
    abort_reasons: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    selected_agents: list[str] = field(default_factory=list)


@dataclass
class _Subagent:
    """Delegation metadata reported by Copilot CLI's `subagent.*` events."""

    agent_id: str
    trajectory_id: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    display_name: str | None = None
    model: str | None = None
    status: str | None = None
    error: str | None = None
    description: str | None = None
    tools: list[Any] | None = None
    total_tool_calls: int | None = None
    total_tokens: int | None = None
    duration_ms: int | None = None

    def metrics_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.total_tokens is not None:
            extra["copilot_total_tokens"] = self.total_tokens
        if self.total_tool_calls is not None:
            extra["copilot_total_tool_calls"] = self.total_tool_calls
        if self.duration_ms is not None:
            extra["copilot_duration_ms"] = self.duration_ms
        return extra

    def reference_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"agent_id": self.agent_id}
        for key, value in (
            ("agent_name", self.name),
            ("agent_display_name", self.display_name),
            ("agent_description", self.description),
            ("model", self.model),
            ("status", self.status),
            ("error", self.error),
            ("tools", self.tools),
        ):
            if value is not None:
                extra[key] = value
        return extra


class CopilotCli(BaseInstalledAgent):
    """Run GitHub Copilot CLI inside a Pier-managed task environment."""

    SUPPORTS_ATIF = True

    _JSONL_FILENAME = "copilot-cli.jsonl"
    _OUTPUT_FILENAME = "copilot-cli.txt"
    _COMMAND_LOG_PATH = EnvironmentPaths.agent_dir / "command-0" / "stdout.txt"
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
        CliFlag(
            "agent",
            cli="--agent",
            type="str",
            env_fallback="COPILOT_CLI_AGENT",
            quote=True,
        ),
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
        self._session_id: str | None = None
        super().__init__(*args, **kwargs)
        # COPILOT_HOME is always controlled by the agent itself; _exec()
        # re-applies _extra_env on top of the constructed env dict, so a
        # caller-supplied value would redirect session state away from the
        # mounted log directory.
        self._extra_env.pop("COPILOT_HOME", None)
        # An empty COPILOT_GITHUB_TOKEN in _extra_env would mask a valid
        # token resolved by _copilot_auth_env() after _exec() re-applies
        # _extra_env on top of the constructed env dict.
        if not self._extra_env.get("COPILOT_GITHUB_TOKEN"):
            self._extra_env.pop("COPILOT_GITHUB_TOKEN", None)

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
                        "if ldd --version 2>&1 | grep -qi musl || "
                        "[ -f /etc/alpine-release ]; then "
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
            try:
                extra_args = shlex.split(self._extra_args)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid extra_args string: {self._extra_args!r}"
                ) from exc
            return shlex.join(extra_args)
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
            self._resolve_token("COPILOT_GITHUB_TOKEN")
            or self._resolve_token("GH_TOKEN")
            or self._resolve_token("GITHUB_TOKEN")
        )
        if not token:
            raise ValueError(
                "GitHub Copilot CLI authentication is required. Set "
                "COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN in the process "
                "environment or pass --env-file."
            )
        return {"COPILOT_GITHUB_TOKEN": token}

    def _resolve_token(self, key: str) -> str | None:
        """Read a token, treating an empty ``agent.env`` value as unset.

        ``_get_env()`` lets ``extra_env`` shadow the process environment, so an
        empty override would otherwise hide a usable token instead of falling
        back to it.
        """
        return self._extra_env.get(key) or os.environ.get(key)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        session_id = str(uuid.uuid4())
        self._session_id = session_id
        model = self._command_model_name or (
            self.model_name.split("/")[-1] if self.model_name else None
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
                f"{shlex.quote(self._COMMAND_LOG_PATH.parent.as_posix())} "
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
            f"tee {shlex.quote(output_path)} | "
            f"tee {shlex.quote(self._COMMAND_LOG_PATH.as_posix())}; "
            "exit ${PIPESTATUS[0]}"
        )
        return f"{setup} && bash -lc {shlex.quote(run_script)}"

    def populate_context_post_run(self, context: AgentContext) -> None:
        events_path = find_copilot_session_events(
            self.logs_dir, session_id=self._session_id
        )
        jsonl_path = self.logs_dir / self._JSONL_FILENAME
        persisted_events = _read_jsonl(events_path) if events_path is not None else []
        captured_events = _read_jsonl(jsonl_path)
        events = persisted_events or captured_events
        if not events:
            return

        metrics_events = _combine_event_streams(persisted_events, captured_events)
        trajectory = self._convert_events_to_trajectory(
            events, metrics_events=metrics_events
        )
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

        session_metrics = _parse_session_metrics(metrics_events)
        metadata = dict(context.metadata or {})
        if events_path is not None:
            metadata["copilot_session_events"] = str(
                events_path.relative_to(self.logs_dir)
            )
        if session_metrics.aiu is not None:
            metadata["copilot_aiu"] = session_metrics.aiu
        if session_metrics.premium_requests is not None:
            metadata["copilot_premium_requests"] = session_metrics.premium_requests
        context.metadata = metadata or None

    def _convert_events_to_trajectory(
        self,
        events: list[dict[str, Any]],
        *,
        metrics_events: list[dict[str, Any]] | None = None,
    ) -> Trajectory | None:
        root_events: list[dict[str, Any]] = []
        scoped_events: dict[str, list[dict[str, Any]]] = {}
        subagents: dict[str, _Subagent] = {}
        for event in events:
            agent_id = _subagent_id(event)
            if agent_id is None:
                root_events.append(event)
                continue
            scoped_events.setdefault(agent_id, []).append(event)
            _record_subagent(subagents, agent_id, event)

        usage_by_call = _usage_by_api_call(metrics_events or events)
        # One usage report describes one inference, so the scopes share a
        # ledger of the reports already spent.
        claimed_usage: set[tuple[str | None, int, str]] = set()
        root = self._build_scope(
            root_events, usage_by_call=usage_by_call, claimed_usage=claimed_usage
        )
        if not root.steps:
            return None

        scopes = {None: root}
        subagent_trajectories = self._build_subagent_trajectories(
            scoped_events, subagents, usage_by_call, claimed_usage, scopes
        )
        # A subagent may itself delegate, so every scope has to be offered the
        # references — not just the root — or a nested trajectory is embedded
        # without anything pointing at it.
        for owner_id, scope in scopes.items():
            _attach_subagent_references(scope, subagents, owner_id)

        # Permission events are always emitted on the root stream even when the
        # call they gate belongs to a subagent, so they are applied once every
        # scope exists rather than while the root is being built.
        permissions = _collect_permissions(events)
        for scope in scopes.values():
            for call_id, permission in permissions.items():
                _enrich_tool_call(scope, call_id, permission)

        session_metrics = _parse_session_metrics(metrics_events or events)
        extra: dict[str, Any] = {}
        if session_metrics.aiu is not None:
            extra["copilot_aiu"] = session_metrics.aiu
        if session_metrics.premium_requests is not None:
            extra["copilot_premium_requests"] = session_metrics.premium_requests
        if session_metrics.peak_context_tokens is not None:
            extra["peak_context_tokens"] = session_metrics.peak_context_tokens
        extra["summarization_count"] = session_metrics.summarization_count
        if root.abort_reasons:
            extra["abort_reasons"] = root.abort_reasons
        if root.errors:
            extra["session_errors"] = root.errors
        if root.selected_agents:
            extra["copilot_selected_agents"] = root.selected_agents

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=root.session_id or "copilot-cli",
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=root.steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=session_metrics.input_tokens,
                total_completion_tokens=session_metrics.output_tokens,
                total_cached_tokens=session_metrics.cache_read_tokens,
                total_steps=len(root.steps),
                extra=extra,
            ),
            subagent_trajectories=subagent_trajectories or None,
        )

    def _build_subagent_trajectories(
        self,
        scoped_events: dict[str, list[dict[str, Any]]],
        subagents: dict[str, _Subagent],
        usage_by_call: dict[tuple[str | None, int, str], _CallUsage],
        claimed_usage: set[tuple[str | None, int, str]],
        scopes: dict[str | None, _Scope],
    ) -> list[Trajectory]:
        """Convert every `agentId`-tagged event group into its own trajectory.

        Copilot CLI reports delegated work inline on the parent session stream,
        tagging each event with the subagent's `agentId`. ATIF v1.7 models the
        same relationship with embedded `subagent_trajectories` resolved via
        `trajectory_id`, so each group becomes an independently valid
        trajectory embedded in whichever scope actually delegated it — a
        subagent that delegates further nests its own children.
        """
        trajectories: dict[str, Trajectory] = {}
        for agent_id, events in scoped_events.items():
            info = subagents.setdefault(agent_id, _Subagent(agent_id=agent_id))
            scope = self._build_scope(
                events,
                usage_by_call=usage_by_call,
                claimed_usage=claimed_usage,
                agent_id=agent_id,
            )
            if not scope.steps:
                continue
            scopes[agent_id] = scope

            completion_tokens, has_completion = _sum_optional(
                step.metrics.completion_tokens
                for step in scope.steps
                if step.metrics is not None
            )
            extra = info.metrics_extra()
            if scope.abort_reasons:
                extra["abort_reasons"] = scope.abort_reasons
            if scope.errors:
                extra["session_errors"] = scope.errors
            info.trajectory_id = f"subagent-{agent_id}"
            trajectories[agent_id] = Trajectory(
                schema_version="ATIF-v1.7",
                trajectory_id=info.trajectory_id,
                agent=Agent(
                    name=f"{self.name()}:{info.name}" if info.name else self.name(),
                    version=self.version() or "unknown",
                    model_name=info.model or self.model_name,
                ),
                steps=scope.steps,
                final_metrics=FinalMetrics(
                    total_completion_tokens=(
                        completion_tokens if has_completion else None
                    ),
                    total_steps=len(scope.steps),
                    extra=extra or None,
                ),
            )

        children: dict[str | None, list[Trajectory]] = {}
        for agent_id, trajectory in trajectories.items():
            info = subagents[agent_id]
            call_id = info.tool_call_id or agent_id
            parent = next(
                (
                    owner_id
                    for owner_id, scope in scopes.items()
                    if owner_id != agent_id and call_id in scope.call_owners
                ),
                None,
            )
            children.setdefault(parent, []).append(trajectory)
        for agent_id, trajectory in trajectories.items():
            if nested := children.get(agent_id):
                trajectory.subagent_trajectories = nested
        return children.get(None, [])

    def _build_scope(
        self,
        events: list[dict[str, Any]],
        *,
        usage_by_call: dict[tuple[str | None, int, str], _CallUsage],
        claimed_usage: set[tuple[str | None, int, str]],
        agent_id: str | None = None,
    ) -> _Scope:
        scope = _Scope()
        assistant_steps_by_id: dict[str, Step] = {}
        resolved_usage: dict[str, _CallUsage | None] = {}
        requested_call_ids = _requested_tool_call_ids(events)
        current_model = self.model_name
        current_effort = self._resolved_flags.get("reasoning_effort")
        pending_reasoning = ""
        pending_reasoning_timestamp: str | None = None
        last_assistant_step: Step | None = None
        turn_ordinal = 0

        def append_step(step: Step) -> Step:
            step.step_id = len(scope.steps) + 1
            scope.steps.append(step)
            return step

        def end_turn() -> None:
            """Close the current API call so the next one starts a new step.

            `apiCallId` is only unique *within* a turn: OpenAI-compatible
            servers hand out sequential `chatcmpl-N` ids that repeat once a
            tool round-trip has completed. Forgetting the ids at every turn
            boundary keeps contiguous streaming chunks of one call merged
            while never folding two real inferences into a single step.
            """
            nonlocal last_assistant_step
            last_assistant_step = None
            assistant_steps_by_id.clear()
            resolved_usage.clear()

        def _resolve_usage(api_call_id: str) -> _CallUsage | None:
            """Resolve this call's usage report, spending it only once.

            Streaming delivers one call as several `assistant.message` events
            that merge into a single step, so the report is memoised for the
            rest of the turn instead of being claimed again per chunk.
            """
            if api_call_id in resolved_usage:
                return resolved_usage[api_call_id]
            usage = _scoped_usage(
                usage_by_call, agent_id, turn_ordinal, api_call_id, claimed_usage
            )
            resolved_usage[api_call_id] = usage
            return usage

        for event in events:
            event_type = event.get("type")
            data = event.get("data")
            data = data if isinstance(data, dict) else {}
            timestamp = event.get("timestamp")

            if event_type in _TURN_BOUNDARY_EVENTS:
                turn_ordinal += 1
                end_turn()

            if event_type in ("session.start", "session.resume"):
                if session_id := _string_or_none(
                    data.get("sessionId") or data.get("session_id")
                ):
                    scope.session_id = session_id
                if model := _string_or_none(data.get("selectedModel")):
                    current_model = model
                if effort := _string_or_none(data.get("reasoningEffort")):
                    current_effort = effort
                continue

            if event_type == "session.model_change":
                if model := _string_or_none(data.get("newModel")):
                    current_model = model
                if effort := _string_or_none(data.get("reasoningEffort")):
                    current_effort = effort
                continue

            if event_type == "assistant.turn_start":
                continue

            if event_type == "user.message":
                message = _flatten_content(data.get("content")) or _flatten_content(
                    data.get("transformedContent")
                )
                if message:
                    append_step(
                        Step(
                            step_id=1,
                            timestamp=timestamp,
                            source="user",
                            message=message,
                            extra=_user_message_extra(data),
                        )
                    )
                continue

            if event_type == "system.message":
                if message := _flatten_content(data.get("content")):
                    append_step(
                        Step(
                            step_id=1,
                            timestamp=timestamp,
                            source="system",
                            message=message,
                        )
                    )
                continue

            if event_type == "session.compaction_complete":
                # A completed compaction replaces conversation history with a
                # model-written summary; that summary is real context the run
                # continued from, so it becomes a system step in place.
                if data.get("success") is not False:
                    if summary := _flatten_content(data.get("summaryContent")):
                        append_step(
                            Step(
                                step_id=1,
                                timestamp=timestamp,
                                source="system",
                                message=summary,
                                extra=_compaction_extra(data),
                            )
                        )
                continue

            if event_type == "subagent.selected":
                if name := _string_or_none(data.get("agentName")):
                    if name not in scope.selected_agents:
                        scope.selected_agents.append(name)
                continue

            if event_type == "assistant.reasoning":
                # Copilot CLI emits reasoning summaries *after* the assistant
                # message of the API call they belong to and before that call's
                # tool results, so reasoning merges into the message step that
                # is still open; anything emitted outside an open turn is
                # buffered for the next assistant message instead.
                reasoning = _flatten_content(data.get("content"))
                if reasoning:
                    if last_assistant_step is not None:
                        last_assistant_step.reasoning_content = _merge_ordered_text(
                            last_assistant_step.reasoning_content or "", reasoning
                        )
                    else:
                        if not pending_reasoning:
                            pending_reasoning_timestamp = timestamp
                        pending_reasoning = _merge_ordered_text(
                            pending_reasoning, reasoning
                        )
                continue

            if event_type == "assistant.message":
                tool_calls = _tool_calls(data.get("toolRequests"))
                api_call_id = _string_or_none(
                    data.get("apiCallId") or data.get("api_call_id")
                )
                usage = (
                    _resolve_usage(api_call_id)
                    if api_call_id
                    else None
                )
                prompt_tokens = _optional_int(data.get("inputTokens"))
                if prompt_tokens is None and usage is not None:
                    prompt_tokens = usage.input_tokens
                completion_tokens = _optional_int(data.get("outputTokens"))
                cached_tokens = usage.cache_read_tokens if usage is not None else None
                assistant_event_id = api_call_id or _string_or_none(
                    data.get("modelCallId") or data.get("messageId")
                )
                reasoning_content = _merge_ordered_text(
                    pending_reasoning,
                    _flatten_content(data.get("reasoningText")),
                )
                pending_reasoning = ""
                pending_reasoning_timestamp = None
                message = _flatten_content(data.get("content"))
                if assistant_event_id and (
                    existing_step := assistant_steps_by_id.get(assistant_event_id)
                ):
                    _merge_assistant_event(
                        existing_step,
                        message=message,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_tokens=cached_tokens,
                        phase=_string_or_none(data.get("phase")),
                    )
                    for tool_call in tool_calls:
                        if tool_call.tool_call_id:
                            scope.call_owners[tool_call.tool_call_id] = existing_step
                    last_assistant_step = existing_step
                    continue

                step = append_step(
                    Step(
                        step_id=1,
                        timestamp=timestamp,
                        source="agent",
                        model_name=_string_or_none(data.get("model")) or current_model,
                        reasoning_effort=current_effort,
                        message=message,
                        reasoning_content=reasoning_content or None,
                        tool_calls=tool_calls or None,
                        metrics=_step_metrics(
                            prompt_tokens, completion_tokens, cached_tokens
                        ),
                        llm_call_count=1,
                        extra=_assistant_extra(api_call_id, data),
                    )
                )
                if assistant_event_id:
                    assistant_steps_by_id[assistant_event_id] = step
                last_assistant_step = step
                for tool_call in tool_calls:
                    if tool_call.tool_call_id:
                        scope.call_owners[tool_call.tool_call_id] = step
                continue

            if event_type == "tool.user_requested":
                # The user dispatched this tool themselves, so no inference
                # produced it: ATIF models that as an agent step with
                # `llm_call_count` 0 carrying the real call.
                call_id = str(data.get("toolCallId") or "")
                if call_id and call_id not in scope.call_owners:
                    scope.call_owners[call_id] = append_step(
                        _dispatch_step(data, call_id, timestamp, user_requested=True)
                    )
                continue

            if event_type == "tool.execution_start":
                # Tools the model asked for arrive on `assistant.message`
                # first; anything else was dispatched without an inference
                # (e.g. a user-invoked tool), which ATIF models as an agent
                # step with `llm_call_count` 0.
                call_id = str(data.get("toolCallId") or "")
                if not call_id:
                    continue
                if call_id in scope.call_owners:
                    _enrich_tool_call(scope, call_id, _tool_call_extra(data))
                elif call_id not in requested_call_ids:
                    scope.call_owners[call_id] = append_step(
                        _dispatch_step(
                            data,
                            call_id,
                            timestamp,
                            user_requested=data.get("isUserRequested") is True,
                        )
                    )
                    end_turn()
                continue

            if event_type == "tool.execution_complete":
                content = _stringify_tool_result(data.get("result"))
                if data.get("error") is not None:
                    error = _stringify_tool_result(data.get("error"))
                    content = error if not content else f"{content}\n{error}"
                _attach_observation(
                    scope,
                    data.get("toolCallId"),
                    content,
                    timestamp,
                    success=data.get("success"),
                    extra=_tool_result_extra(data),
                )
                continue

            if event_type == "abort":
                if reason := _flatten_content(data.get("reason")):
                    scope.abort_reasons.append(reason)
                continue

            if event_type == "session.error":
                if error_detail := _session_error(data):
                    scope.errors.append(error_detail)
                continue

            if event_type == "message":
                source = "agent" if event.get("role") == "assistant" else "user"
                is_agent = source == "agent"
                append_step(
                    Step(
                        step_id=1,
                        timestamp=timestamp,
                        source=source,
                        message=_flatten_content(event.get("content")),
                        model_name=(
                            _string_or_none(event.get("model")) or current_model
                            if is_agent
                            else None
                        ),
                        reasoning_effort=current_effort if is_agent else None,
                        llm_call_count=1 if is_agent else None,
                    )
                )
                last_assistant_step = scope.steps[-1] if is_agent else None
                continue

            if event_type == "tool_use":
                call_id = str(event.get("id") or "")
                step = append_step(
                    Step(
                        step_id=1,
                        timestamp=timestamp,
                        source="agent",
                        model_name=_string_or_none(event.get("model")) or current_model,
                        reasoning_effort=current_effort,
                        message="",
                        tool_calls=[
                            ToolCall(
                                tool_call_id=call_id,
                                function_name=str(event.get("name") or ""),
                                arguments=_normalize_arguments(event.get("input")),
                            )
                        ],
                        llm_call_count=1,
                    )
                )
                if call_id:
                    scope.call_owners[call_id] = step
                last_assistant_step = None
                continue

            if event_type == "tool_result":
                _attach_observation(
                    scope,
                    event.get("tool_use_id"),
                    _flatten_content(event.get("content")),
                    timestamp,
                )
                last_assistant_step = None

        if pending_reasoning:
            append_step(
                Step(
                    step_id=1,
                    timestamp=pending_reasoning_timestamp,
                    source="agent",
                    model_name=current_model,
                    reasoning_effort=current_effort,
                    message="",
                    reasoning_content=pending_reasoning,
                    llm_call_count=1,
                )
            )

        return scope


def find_copilot_session_events(
    agent_logs_dir: Path, session_id: str | None = None
) -> Path | None:
    """Locate the native session event stream inside the agent log directory.

    The run passes an explicit `--session-id`, so the matching directory is
    authoritative. Modification times are only a fallback for streams produced
    by a session id this process never saw (e.g. a resumed run).
    """
    session_root = agent_logs_dir / "copilot-home" / "session-state"
    if session_id:
        candidate = session_root / session_id / "events.jsonl"
        if candidate.is_file():
            return candidate

    candidates: list[tuple[float, Path]] = []
    for path in session_root.glob("**/events.jsonl"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], str(candidate[1])))[1]


_TURN_BOUNDARY_EVENTS = frozenset(
    {
        "assistant.turn_start",
        "user.message",
        "system.message",
        "tool.execution_complete",
        "tool.user_requested",
        "session.compaction_complete",
        "abort",
    }
)
"""Events after which Copilot CLI may reuse an `apiCallId` for a new call."""


def _reconcile_usage_reports(
    usage_by_call: Mapping[
        tuple[str | None, int, str], tuple[int | None, int | None, int | None, int | None]
    ],
    message_usage_by_call: Mapping[
        tuple[str | None, int, str], tuple[int | None, int | None]
    ],
) -> list[
    tuple[
        tuple[int | None, int | None, int | None, int | None] | None,
        tuple[int | None, int | None] | None,
    ]
]:
    """Pair the two streams' reports so one inference is counted once.

    The captured `assistant.usage` stream and the persisted `assistant.message`
    stream describe the same inferences but do not always identify them the
    same way: stdout usage is not always tagged with the agent that made the
    call, and ~30% of persisted messages carry no `apiCallId` at all. Matching
    on the full key alone therefore left two entries for one call, which the
    caller then summed -- doubling the reported tokens.

    Each usage report is claimed at most once, so a genuinely reused
    `chatcmpl-N` id still yields two pairs and still sums.
    """
    pairs: list[
        tuple[
            tuple[int | None, int | None, int | None, int | None] | None,
            tuple[int | None, int | None] | None,
        ]
    ] = []
    claimed: set[tuple[str | None, int, str]] = set()

    def claim(candidates: Iterable[tuple[str | None, int, str]]) -> bool:
        for key in candidates:
            if key in usage_by_call and key not in claimed:
                claimed.add(key)
                pairs.append((usage_by_call[key], message_usage))
                return True
        return False

    for message_key, message_usage in message_usage_by_call.items():
        agent_id, turn, call_key = message_key
        if claim([message_key]):
            continue
        if call_key.startswith("call:"):
            # The call id identifies the inference on its own; an untagged
            # report is the same call reported without its agent.
            if claim(
                key
                for key in usage_by_call
                if key[2] == call_key and key[0] in (agent_id, None)
            ):
                continue
        elif claim(
            key
            for key in usage_by_call
            if key[1] == turn and key[0] in (agent_id, None)
        ):
            # The message carries no call id, so the turn it landed in is the
            # only thing that can identify which report describes it.
            continue
        pairs.append((None, message_usage))
    for key, usage in usage_by_call.items():
        if key not in claimed:
            pairs.append((usage, None))
    return pairs


def _parse_session_metrics(events: list[dict[str, Any]]) -> _SessionMetrics:
    shutdowns: list[dict[str, Any]] = []
    seen_shutdowns: set[str] = set()
    summarization_count = 0
    peak_context_tokens = 0
    checkpoint_nano_aiu: float | None = None
    reported_premium_requests: int | None = None
    usage_by_call: dict[
        tuple[str | None, int, str],
        tuple[int | None, int | None, int | None, int | None],
    ] = {}
    message_usage_by_call: dict[
        tuple[str | None, int, str], tuple[int | None, int | None]
    ] = {}
    turn_ordinals: dict[str | None, int] = {}

    for event in events:
        event_type = event.get("type")
        data = event.get("data")
        data = data if isinstance(data, dict) else {}
        agent_id = _subagent_id(event)
        if event_type in _TURN_BOUNDARY_EVENTS:
            # `apiCallId` is reused once a turn ends, so usage has to be
            # accumulated per turn or two real calls collapse into one.
            turn_ordinals[agent_id] = turn_ordinals.get(agent_id, 0) + 1
        turn = turn_ordinals.get(agent_id, 0)
        if event_type == "session.shutdown":
            shutdown_fingerprint = _json_fingerprint(data)
            if shutdown_fingerprint in seen_shutdowns:
                continue
            seen_shutdowns.add(shutdown_fingerprint)
            shutdowns.append(data)
            peak_context_tokens = max(
                peak_context_tokens,
                _optional_int(data.get("currentTokens")) or 0,
                (_optional_int(data.get("systemTokens")) or 0)
                + (_optional_int(data.get("conversationTokens")) or 0)
                + (_optional_int(data.get("toolDefinitionsTokens")) or 0),
            )
        elif event_type == "session.usage_checkpoint":
            # Checkpoints carry the cumulative AIU total, so they survive a
            # session that pier had to kill before it could shut down.
            nano_aiu = data.get("totalNanoAiu")
            if isinstance(nano_aiu, (int, float)) and not isinstance(nano_aiu, bool):
                checkpoint_nano_aiu = max(checkpoint_nano_aiu or 0.0, float(nano_aiu))
            reported_premium_requests = _max_optional(
                reported_premium_requests,
                _optional_int(data.get("totalPremiumRequests")),
            )
        elif event_type == "result":
            # The stdout stream closes with a `result` summary that the
            # persisted stream never contains, and it carries its payload at the
            # event root rather than under `data`. Its premium request count is
            # the same session total `session.shutdown` reports, so it is a
            # fallback for a killed session rather than something to add on top.
            usage = data.get("usage") or event.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            reported_premium_requests = _max_optional(
                reported_premium_requests,
                _optional_int(usage.get("premiumRequests")),
            )
        elif event_type == "session.compaction_start":
            total = (
                (_optional_int(data.get("systemTokens")) or 0)
                + (_optional_int(data.get("conversationTokens")) or 0)
                + (_optional_int(data.get("toolDefinitionsTokens")) or 0)
            )
            peak_context_tokens = max(peak_context_tokens, total)
        elif event_type == "session.compaction_complete":
            # `success` is a required field on this event; only completed
            # compactions are summarizations. A failed attempt must not count,
            # but its pre-compaction context size is still a real peak.
            if data.get("success") is not False:
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
        elif event_type == "assistant.usage":
            usage = data.get("usage")
            usage = usage if isinstance(usage, dict) else data
            input_tokens = _optional_int(usage.get("inputTokens"))
            cache_read_tokens = _optional_int(usage.get("cacheReadTokens"))
            cache_write_tokens = _optional_int(usage.get("cacheWriteTokens"))
            output_tokens = _optional_int(usage.get("outputTokens"))
            api_call_id = _string_or_none(
                data.get("apiCallId")
                or data.get("api_call_id")
                or data.get("providerCallId")
                or usage.get("apiCallId")
            )
            usage_key = (
                agent_id,
                turn,
                f"call:{api_call_id}" if api_call_id else _event_identity(event),
            )
            previous = usage_by_call.get(usage_key, (None, None, None, None))
            usage_by_call[usage_key] = (
                _max_optional(previous[0], input_tokens),
                _max_optional(previous[1], cache_read_tokens),
                _max_optional(previous[2], cache_write_tokens),
                _max_optional(previous[3], output_tokens),
            )
        elif event_type == "assistant.message":
            input_tokens = _optional_int(data.get("inputTokens"))
            output_tokens = _optional_int(data.get("outputTokens"))
            api_call_id = _string_or_none(
                data.get("apiCallId") or data.get("api_call_id")
            )
            message_id = _string_or_none(
                data.get("modelCallId") or data.get("messageId")
            )
            message_key = (
                agent_id,
                turn,
                f"call:{api_call_id}"
                if api_call_id
                else (
                    f"message:{message_id}"
                    if message_id
                    else f"event:{_json_fingerprint(event)}"
                ),
            )
            previous_input, previous_output = message_usage_by_call.get(
                message_key, (None, None)
            )
            message_usage_by_call[message_key] = (
                _max_optional(previous_input, input_tokens),
                _max_optional(previous_output, output_tokens),
            )

    if not shutdowns:
        pairs = _reconcile_usage_reports(usage_by_call, message_usage_by_call)
        usage_input, has_usage_input = _sum_optional(
            usage[0] for usage, _ in pairs if usage is not None
        )
        usage_cache_read, has_usage_cache_read = _sum_optional(
            usage[1] for usage, _ in pairs if usage is not None
        )
        usage_cache_write, has_usage_cache_write = _sum_optional(
            usage[2] for usage, _ in pairs if usage is not None
        )
        message_input, has_message_input = _sum_optional(
            message[0] for _, message in pairs if message is not None
        )
        output_tokens, has_output = _sum_optional(
            _max_optional(
                usage[3] if usage is not None else None,
                message[1] if message is not None else None,
            )
            for usage, message in pairs
        )
        has_usage_prompt = (
            has_usage_input or has_usage_cache_read or has_usage_cache_write
        )
        if has_usage_prompt:
            # Per call: prefer the usage event's inputTokens; fall back to the
            # message input for any call whose usage event arrived without it
            # (e.g. a partially captured stream after a timeout). inputTokens is
            # the cache-inclusive prompt total (OpenAI-style prompt_tokens), so
            # cached reads/writes are already counted within it and must not be
            # added again.
            prompt_tokens, _ = _sum_optional(
                (
                    usage[0]
                    if usage is not None and usage[0] is not None
                    else (message[0] if message is not None else None)
                )
                for usage, message in pairs
            )
        else:
            prompt_tokens = message_input
        return _SessionMetrics(
            input_tokens=(
                prompt_tokens if has_usage_prompt or has_message_input else None
            ),
            cache_read_tokens=(usage_cache_read if has_usage_cache_read else None),
            output_tokens=output_tokens if has_output else None,
            peak_context_tokens=peak_context_tokens or None,
            summarization_count=summarization_count,
            aiu=_nano_aiu_to_aiu(checkpoint_nano_aiu),
            premium_requests=reported_premium_requests,
        )

    prompt_total = 0
    cache_read = 0
    output = 0
    for shutdown in shutdowns:
        # `tokenDetails` is the billing record -- it is what `totalNanoAiu` is
        # derived from -- and it counts every call the session made, including
        # the compaction/summarization calls that `modelMetrics` omits. Those
        # calls are large and frequent in long-horizon runs, so preferring
        # `modelMetrics` here under-reported a compacting session's tokens by
        # as much as a third. `modelMetrics` remains the fallback for streams
        # that carry no `tokenDetails` at all.
        usage = _shutdown_token_details_usage(shutdown) or _shutdown_model_usage(
            shutdown
        )
        if usage is None:
            continue
        prompt_total += usage[0]
        cache_read += usage[1]
        output += usage[3]
    nano_aiu = sum(
        float(value)
        for shutdown in shutdowns
        if isinstance(value := shutdown.get("totalNanoAiu"), (int, float))
    )
    # A resumed session shuts down more than once, and each shutdown reports the
    # requests billed for its own run, so these sum the way AIU does.
    shutdown_premium_requests, has_premium_requests = _sum_optional(
        _optional_int(shutdown.get("totalPremiumRequests")) for shutdown in shutdowns
    )

    return _SessionMetrics(
        input_tokens=prompt_total,
        cache_read_tokens=cache_read,
        output_tokens=output,
        peak_context_tokens=peak_context_tokens or None,
        summarization_count=summarization_count,
        aiu=_nano_aiu_to_aiu(nano_aiu or checkpoint_nano_aiu),
        premium_requests=(
            shutdown_premium_requests
            if has_premium_requests
            else reported_premium_requests
        ),
    )


def _nano_aiu_to_aiu(nano_aiu: float | None) -> float | None:
    return round(nano_aiu / 1_000_000_000, 6) if nano_aiu else None


def _shutdown_token_details_usage(
    shutdown: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Session totals from `tokenDetails`, or None if it reports nothing.

    The `tokenDetails` billing categories are disjoint -- the "input" bucket
    excludes cached reads and writes -- so the cache-inclusive prompt total is
    the sum of the three input-side categories.
    """
    token_details = shutdown.get("tokenDetails")
    if not isinstance(token_details, dict):
        return None

    def count(token_type: str) -> int | None:
        details = token_details.get(token_type)
        if not isinstance(details, dict):
            return None
        return _optional_int(details.get("tokenCount"))

    buckets = {
        name: count(name) for name in ("input", "cache_read", "cache_write", "output")
    }
    if all(value is None for value in buckets.values()):
        return None
    resolved = {name: value or 0 for name, value in buckets.items()}
    return (
        resolved["input"] + resolved["cache_read"] + resolved["cache_write"],
        resolved["cache_read"],
        resolved["cache_write"],
        resolved["output"],
    )


def _shutdown_model_usage(
    shutdown: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    model_metrics = shutdown.get("modelMetrics")
    if isinstance(model_metrics, dict):
        metrics = model_metrics.values()
    elif isinstance(model_metrics, list):
        metrics = model_metrics
    else:
        return None

    totals = [0, 0, 0, 0]
    has_usage = False
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        usage = metric.get("usage")
        if not isinstance(usage, dict):
            continue
        values = (
            _optional_int(usage.get("inputTokens")),
            _optional_int(usage.get("cacheReadTokens")),
            _optional_int(usage.get("cacheWriteTokens")),
            _optional_int(usage.get("outputTokens")),
        )
        if any(value is not None for value in values):
            has_usage = True
        for index, value in enumerate(values):
            totals[index] += value or 0

    return (totals[0], totals[1], totals[2], totals[3]) if has_usage else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL event stream defensively.

    Copilot CLI streams are written incrementally by a process that pier may
    kill on timeout, so the file can end mid-line, contain a BOM, CRLF line
    endings or bytes that are not valid UTF-8. None of that may cost us the
    rest of the trajectory, and the largest observed stream is tens of
    megabytes, so lines are decoded one at a time instead of all at once.
    """
    if not path.is_file():
        return []

    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return events
    return events


def _timestamp_epoch(timestamp: Any) -> float | None:
    """Parse an upstream timestamp into a comparable instant.

    Ordering by the raw string happens to work while every timestamp is UTC,
    but a stream that ever reports a local offset would sort `+01:00` before
    `Z` for the same instant, so the comparison is done on the instant itself.
    """
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _combine_event_streams(
    *event_streams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the persisted and captured streams into one chronological stream.

    The two streams overlap but neither is a superset: prompt tokens only ever
    reach stdout, while the session file is the only durable record. Merging
    them in timestamp order (rather than concatenating) keeps a usage report
    inside the turn it describes, which is what attributes it to the right
    API call when a provider reuses call ids across turns.
    """
    ordered: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for event_stream in event_streams:
        kept: list[dict[str, Any]] = []
        instants: list[float | None] = []
        for event in event_stream:
            identity = _event_identity(event)
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(event)
            instants.append(_timestamp_epoch(event.get("timestamp")))
        # An event without a usable timestamp inherits its neighbour's so that
        # it stays next to the events it was emitted between; a captured
        # stream that starts mid-session borrows from the first event that
        # does carry one rather than sorting to the front of the session.
        last: float | None = None
        for index, instant in enumerate(instants):
            if instant is None:
                instants[index] = last
            else:
                last = instant
        first = next((instant for instant in instants if instant is not None), None)
        ordered.extend(
            (first if instant is None else instant, event)
            if first is not None
            else (float("-inf"), event)
            for instant, event in zip(instants, kept, strict=True)
        )
    ordered.sort(key=lambda item: item[0])
    return [event for _, event in ordered]


_TOOL_CALL_EXTRA_KEYS = (
    ("intentionSummary", "intention_summary"),
    ("toolTitle", "tool_title"),
    ("mcpServerName", "mcp_server_name"),
    ("mcpToolName", "mcp_tool_name"),
    ("shellToolInfo", "shell_tool_info"),
)


def _tool_call_extra(data: dict[str, Any]) -> dict[str, Any]:
    """Collect the descriptive metadata Copilot CLI attaches to a tool call."""
    extra: dict[str, Any] = {}
    for key, extra_key in _TOOL_CALL_EXTRA_KEYS:
        value = data.get(key)
        if value not in (None, "", [], {}):
            extra[extra_key] = value
    return extra


def _tool_calls(value: Any) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[ToolCall] = []
    for request in value:
        if not isinstance(request, dict):
            continue
        extra = _tool_call_extra(request)
        calls.append(
            ToolCall(
                tool_call_id=str(request.get("toolCallId") or request.get("id") or ""),
                function_name=str(request.get("name") or ""),
                arguments=_normalize_arguments(request.get("arguments")),
                extra=extra or None,
            )
        )
    return calls


_TOOL_RESULT_EXTRA_KEYS = (
    ("toolTelemetry", "tool_telemetry"),
    ("isUserRequested", "user_requested"),
)


def _tool_result_extra(data: dict[str, Any]) -> dict[str, Any]:
    """Keep the execution metadata Copilot CLI reports alongside a result."""
    extra: dict[str, Any] = {}
    for key, extra_key in _TOOL_RESULT_EXTRA_KEYS:
        value = data.get(key)
        if value not in (None, "", [], {}):
            extra[extra_key] = value
    return extra


def _dispatch_step(
    data: dict[str, Any],
    call_id: str,
    timestamp: str | None,
    *,
    user_requested: bool,
) -> Step:
    """Build the `llm_call_count` 0 step for a tool dispatched without an LLM.

    `user_requested` is only asserted when Copilot said so: an unrequested
    `tool.execution_start` is just as likely to be a resumed session whose
    `assistant.message` was never persisted, and claiming the user typed it
    would be fabricating provenance.
    """
    extra = _tool_call_extra(data)
    return Step(
        step_id=1,
        timestamp=timestamp,
        source="agent",
        message="",
        tool_calls=[
            ToolCall(
                tool_call_id=call_id,
                function_name=str(data.get("toolName") or ""),
                arguments=_normalize_arguments(data.get("arguments")),
                extra=extra or None,
            )
        ],
        llm_call_count=0,
        extra={"user_requested": True} if user_requested else None,
    )


_COMPACTION_EXTRA_KEYS = (
    ("preCompactionTokens", "pre_compaction_tokens"),
    ("preCompactionMessagesLength", "pre_compaction_messages"),
    ("checkpointNumber", "checkpoint_number"),
    ("trigger", "trigger"),
    ("tokenLimit", "token_limit"),
)


def _compaction_extra(data: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {"compaction": True}
    for key, extra_key in _COMPACTION_EXTRA_KEYS:
        value = data.get(key)
        if value is not None:
            extra[extra_key] = value
    return extra


def _collect_permissions(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index permission requests and decisions by the tool call they gate.

    `permission.requested` / `permission.completed` are always emitted on the
    root stream — they carry no `agentId` even when the call belongs to a
    subagent — so they are collected across the whole session and applied to
    whichever scope actually owns the call.
    """
    permissions: dict[str, dict[str, Any]] = {}
    pending: dict[str, str] = {}
    for event in events:
        event_type = event.get("type")
        if event_type not in ("permission.requested", "permission.completed"):
            continue
        data = event.get("data")
        data = data if isinstance(data, dict) else {}
        request_id = str(data.get("requestId") or "")
        if event_type == "permission.requested":
            request = data.get("permissionRequest")
            request = request if isinstance(request, dict) else {}
            call_id = str(request.get("toolCallId") or "")
            if not call_id:
                continue
            if request_id:
                pending[request_id] = call_id
            if kind := _string_or_none(request.get("kind")):
                permissions.setdefault(call_id, {})["permission_kind"] = kind
            continue
        call_id = str(data.get("toolCallId") or pending.get(request_id) or "")
        result = data.get("result")
        result = result if isinstance(result, dict) else {}
        if not call_id or not (decision := _string_or_none(result.get("kind"))):
            continue
        permission = permissions.setdefault(call_id, {})
        permission["permission_decision"] = decision
        if feedback := _flatten_content(result.get("feedback")):
            permission["permission_feedback"] = feedback
    return permissions


def _enrich_tool_call(scope: _Scope, call_id: str, extra: dict[str, Any]) -> None:
    """Merge late-arriving metadata into an already-requested tool call.

    `tool.execution_start` and the `permission.*` pair describe a call the
    model asked for on an earlier `assistant.message`, so the metadata belongs
    on the existing `ToolCall` rather than on a step of its own.
    """
    owner = scope.call_owners.get(call_id)
    if owner is None:
        return
    extra = {key: value for key, value in extra.items() if value is not None}
    if not extra:
        return
    for call in owner.tool_calls or []:
        if call.tool_call_id != call_id:
            continue
        merged = dict(call.extra or {})
        merged.update(extra)
        call.extra = merged
        return


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


_TOOL_RESULT_TEXT_KEYS = ("content", "output", "stdout", "text", "message")
_TOOL_RESULT_DISPLAY_KEYS = ("detailedContent", "displayContent")


def _merge_result_text(text: str, detail: Any) -> str:
    """Fold a rendering variant into the result text without duplicating it.

    Copilot CLI often repeats the payload under `detailedContent` and
    `displayContent`, but just as often `content` is a one-line summary and
    the variant holds the output the agent actually saw, so a variant that is
    neither a copy nor a subset of what we already have is kept as well.
    """
    if not isinstance(detail, str) or not detail or detail == text:
        return text
    if not text or detail.startswith(text) or text in detail:
        return detail
    if detail in text:
        return text
    return f"{text}\n{detail}"


def _stringify_tool_result(result: Any) -> str:
    """Render a tool result as observation text without duplicating it."""
    if not isinstance(result, dict):
        return _flatten_content(result)

    for key in _TOOL_RESULT_TEXT_KEYS:
        value = result.get(key)
        if not isinstance(value, str):
            continue
        text = value
        # Display variants are excluded from the remainder whichever text key
        # won, so they have to be merged here or they are dropped outright.
        for display_key in _TOOL_RESULT_DISPLAY_KEYS:
            text = _merge_result_text(text, result.get(display_key))
        remainder = {
            k: v
            for k, v in result.items()
            if k != key and k not in _TOOL_RESULT_DISPLAY_KEYS
        }
        if remainder:
            return f"{text}\n{json.dumps(remainder, ensure_ascii=False)}"
        return text
    return json.dumps(result, ensure_ascii=False)


def _attach_observation(
    scope: _Scope,
    call_id: Any,
    content: str,
    timestamp: str | None,
    *,
    success: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    call_id_text = str(call_id or "")
    result_extra = dict(extra or {})
    if isinstance(success, bool):
        # Copilot reports `success` for the *harness*: a `bash` call whose
        # command exits non-zero is still a successful execution. Claiming
        # `is_error: false` would therefore suppress the viewer's own
        # content heuristic for genuinely failed commands, so only the
        # negative signal is promoted -- matching `cursor_cli` and
        # `claude_code`, which likewise set `is_error` only when true. The
        # raw field is preserved verbatim alongside it.
        result_extra["copilot_success"] = success
        if not success:
            result_extra["is_error"] = True
    owner = scope.call_owners.get(call_id_text) if call_id_text else None
    if owner is None:
        # Copilot CLI can report a completion for a call that never appeared in
        # `toolRequests` (e.g. after a resume). ATIF models an unattributable
        # result as an observation with no `source_call_id`, which keeps the
        # output out of the assistant's mouth.
        if call_id_text:
            result_extra["copilot_tool_call_id"] = call_id_text
        scope.steps.append(
            Step(
                step_id=len(scope.steps) + 1,
                timestamp=timestamp,
                source="agent",
                message="",
                llm_call_count=0,
                observation=Observation(
                    results=[
                        ObservationResult(
                            content=content or None,
                            extra=result_extra or None,
                        )
                    ]
                ),
            )
        )
        return

    if owner.observation is None:
        owner.observation = Observation(results=[])
    elif any(
        result.source_call_id == call_id_text for result in owner.observation.results
    ):
        return
    owner.observation.results.append(
        ObservationResult(
            source_call_id=call_id_text,
            content=content or None,
            extra=result_extra or None,
        )
    )


def _merge_assistant_event(
    step: Step,
    *,
    message: str,
    reasoning_content: str,
    tool_calls: list[ToolCall],
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None = None,
    phase: str | None = None,
) -> None:
    if message:
        existing_message = step.message if isinstance(step.message, str) else ""
        if not existing_message:
            step.message = message
        elif message.startswith(existing_message):
            step.message = message
        elif not existing_message.startswith(message):
            step.message = f"{existing_message}\n{message}"

    if reasoning_content:
        step.reasoning_content = _merge_ordered_text(
            step.reasoning_content or "", reasoning_content
        )

    if tool_calls:
        existing_ids = {call.tool_call_id for call in step.tool_calls or []}
        step.tool_calls = [
            *(step.tool_calls or []),
            *(call for call in tool_calls if call.tool_call_id not in existing_ids),
        ]

    if prompt_tokens is not None or completion_tokens is not None or cached_tokens:
        metrics = step.metrics or Metrics()
        if prompt_tokens is not None:
            metrics.prompt_tokens = max(metrics.prompt_tokens or 0, prompt_tokens)
        if completion_tokens is not None:
            metrics.completion_tokens = max(
                metrics.completion_tokens or 0, completion_tokens
            )
        if cached_tokens is not None:
            metrics.cached_tokens = max(metrics.cached_tokens or 0, cached_tokens)
        step.metrics = metrics

    if phase:
        extra = dict(step.extra or {})
        phases = extra.get("phases")
        phases = list(phases) if isinstance(phases, list) else []
        if phase not in phases:
            phases.append(phase)
        extra["phases"] = phases
        step.extra = extra


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


def _sum_optional(values: Iterable[int | None]) -> tuple[int, bool]:
    total = 0
    found = False
    for value in values:
        if value is not None:
            total += value
            found = True
    return total, found


def _merge_ordered_text(existing: str, new: str) -> str:
    existing = existing.strip()
    new = new.strip()
    if not existing:
        return new
    if not new or new == existing or existing.startswith(new):
        return existing
    if new.startswith(existing):
        return new

    max_overlap = min(len(existing), len(new))
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(new[:overlap]) and _overlap_on_boundaries(
            existing, new, overlap
        ):
            return existing + new[overlap:]
    return f"{existing}\n\n{new}"


def _overlap_on_boundaries(existing: str, new: str, overlap: int) -> bool:
    # Only treat a shared substring as a genuine continuation (rather than a
    # coincidental mid-word collision such as "Use cat" + "category") when the
    # overlap aligns to whitespace boundaries on both sides.
    trailing = new[overlap:]
    if trailing and not trailing[0].isspace():
        return False
    leading_index = len(existing) - overlap - 1
    return leading_index < 0 or existing[leading_index].isspace()


def _subagent_id(event: dict[str, Any]) -> str | None:
    """Return the delegated agent that owns an event, if any.

    Copilot CLI interleaves subagent work into the parent stream, tagging it
    with the subagent's `agentId`. `subagent.*` events themselves are emitted
    by the parent, so their subject is carried in `data`.
    """
    agent_id = event.get("agentId")
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    return None


def _record_subagent(
    subagents: dict[str, _Subagent], agent_id: str, event: dict[str, Any]
) -> _Subagent:
    """Fold a `subagent.*` lifecycle event into the delegation record."""
    info = subagents.setdefault(agent_id, _Subagent(agent_id=agent_id))
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type.startswith("subagent."):
        return info

    data = event.get("data")
    data = data if isinstance(data, dict) else {}
    info.tool_call_id = str(data.get("toolCallId") or "") or info.tool_call_id
    info.name = _string_or_none(data.get("agentName")) or info.name
    info.display_name = (
        _string_or_none(data.get("agentDisplayName")) or info.display_name
    )
    info.model = _string_or_none(data.get("model")) or info.model
    info.description = _string_or_none(data.get("agentDescription")) or info.description
    info.tools = data.get("tools") if isinstance(data.get("tools"), list) else info.tools
    info.total_tool_calls = _first_not_none(
        _optional_int(data.get("totalToolCalls")), info.total_tool_calls
    )
    info.total_tokens = _first_not_none(
        _optional_int(data.get("totalTokens")), info.total_tokens
    )
    info.duration_ms = _first_not_none(
        _optional_int(data.get("durationMs")), info.duration_ms
    )
    if error := _flatten_content(data.get("error")):
        info.error = error
    if event_type == "subagent.started":
        info.status = info.status or "started"
    elif event_type == "subagent.completed":
        info.status = "completed"
    elif event_type == "subagent.failed":
        info.status = "failed"
    return info


def _attach_subagent_references(
    scope: _Scope,
    subagents: dict[str, _Subagent],
    owner_id: str | None = None,
) -> None:
    """Link each delegating tool call to the trajectory it produced."""
    for info in subagents.values():
        if not info.trajectory_id or info.agent_id == owner_id:
            # A subagent that reports its own spawning call owns it in its own
            # scope; referencing itself from there would point at a trajectory
            # embedded somewhere else entirely.
            continue
        # Copilot CLI uses the delegating tool call id as the subagent's
        # `agentId`, so it stays resolvable even when the lifecycle events
        # that spell it out are missing from a truncated stream.
        call_id = info.tool_call_id or info.agent_id
        owner = scope.call_owners.get(call_id)
        if owner is None:
            continue
        reference = SubagentTrajectoryRef(
            trajectory_id=info.trajectory_id,
            extra=info.reference_extra(),
        )
        if owner.observation is None:
            owner.observation = Observation(results=[])
        for result in owner.observation.results:
            if result.source_call_id == call_id:
                result.subagent_trajectory_ref = [reference]
                break
        else:
            # A run killed mid-delegation never reports the delegating call's
            # completion, which would leave the embedded trajectory
            # unreachable; an empty result keeps the reference resolvable
            # without inventing output the tool never returned.
            owner.observation.results.append(
                ObservationResult(
                    source_call_id=call_id,
                    subagent_trajectory_ref=[reference],
                )
            )


def _usage_by_api_call(
    events: list[dict[str, Any]],
) -> dict[tuple[str | None, int, str], _CallUsage]:
    """Index `assistant.usage` reports by the API call they describe.

    Persisted event streams never carry prompt tokens on `assistant.message`;
    they only appear on the ephemeral stdout `assistant.usage` events. Reports
    are keyed by agent and turn as well, because `apiCallId` is only unique
    within one turn of one conversation: OpenAI-compatible servers hand out
    sequential `chatcmpl-N` ids that a parent and its subagents both use and
    that repeat once a tool round-trip has completed.
    """
    usage_by_call: dict[tuple[str | None, int, str], _CallUsage] = {}
    turn_ordinals: dict[str | None, int] = {}
    for event in events:
        agent_id = _subagent_id(event)
        if event.get("type") in _TURN_BOUNDARY_EVENTS:
            turn_ordinals[agent_id] = turn_ordinals.get(agent_id, 0) + 1
        if event.get("type") != "assistant.usage":
            continue
        data = event.get("data")
        data = data if isinstance(data, dict) else {}
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else data
        api_call_id = _string_or_none(
            data.get("apiCallId")
            or data.get("api_call_id")
            or data.get("providerCallId")
            or usage.get("apiCallId")
        )
        if not api_call_id:
            continue
        entry = usage_by_call.setdefault(
            (agent_id, turn_ordinals.get(agent_id, 0), api_call_id), _CallUsage()
        )
        entry.input_tokens = _max_optional(
            entry.input_tokens, _optional_int(usage.get("inputTokens"))
        )
        entry.cache_read_tokens = _max_optional(
            entry.cache_read_tokens,
            _optional_int(usage.get("cacheReadTokens")),
        )
        entry.output_tokens = _max_optional(
            entry.output_tokens, _optional_int(usage.get("outputTokens"))
        )
    return usage_by_call


def _scoped_usage(
    usage_by_call: dict[tuple[str | None, int, str], _CallUsage],
    agent_id: str | None,
    turn_ordinal: int,
    api_call_id: str,
    claimed: set[tuple[str | None, int, str]],
) -> _CallUsage | None:
    """Resolve a usage report, preferring the one reported for this agent.

    Every report describes exactly one inference, so `claimed` records the ones
    already spent: without it a partially captured stdout stream would charge
    the same prompt tokens to several steps and the per-step metrics would sum
    to more than `final_metrics`.

    Stdout usage events are not always tagged with the agent that made the
    call. An untagged report is bucketed under the *root's* turn counter, which
    a subagent scope does not share, so the turn is only trusted when the
    report is tagged for this very scope; otherwise the id has to identify the
    call on its own, and an ambiguous id resolves to nothing rather than to a
    guess.
    """
    exact = (agent_id, turn_ordinal, api_call_id)
    if exact not in claimed and usage_by_call.get(exact) is not None:
        claimed.add(exact)
        return usage_by_call[exact]
    owners: tuple[str | None, ...] = (
        (agent_id,) if agent_id is None else (agent_id, None)
    )
    for owner in owners:
        matches = [
            key
            for key in usage_by_call
            if key[0] == owner and key[2] == api_call_id and key not in claimed
        ]
        if len(matches) == 1:
            claimed.add(matches[0])
            return usage_by_call[matches[0]]
    return None


def _requested_tool_call_ids(events: list[dict[str, Any]]) -> set[str]:
    """Collect every tool call the model asked for in this scope.

    `assistant.message` always precedes the matching `tool.execution_start`,
    but collecting the ids up front keeps the step builder single-pass and
    tolerant of resumed sessions where the request was never persisted.
    """
    call_ids: set[str] = set()
    for event in events:
        if event.get("type") != "assistant.message":
            continue
        data = event.get("data")
        requests = data.get("toolRequests") if isinstance(data, dict) else None
        if not isinstance(requests, list):
            continue
        for request in requests:
            if not isinstance(request, dict):
                continue
            call_id = str(request.get("toolCallId") or request.get("id") or "")
            if call_id:
                call_ids.add(call_id)
    return call_ids


_USER_MESSAGE_EXTRA_KEYS = (("delivery", "delivery"), ("source", "message_source"))


def _user_message_extra(data: dict[str, Any]) -> dict[str, Any] | None:
    """Preserve how a user message reached the agent (steering, schedule, …)."""
    extra: dict[str, Any] = {}
    for key, extra_key in _USER_MESSAGE_EXTRA_KEYS:
        if value := _string_or_none(data.get(key)):
            extra[extra_key] = value
    attachments = data.get("attachments")
    if isinstance(attachments, list) and attachments:
        extra["attachments"] = attachments
    return extra or None


def _assistant_extra(api_call_id: str | None, data: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if api_call_id:
        extra["api_call_id"] = api_call_id
    if phase := _string_or_none(data.get("phase")):
        extra["phases"] = [phase]
    return extra


def _step_metrics(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None,
) -> Metrics | None:
    if prompt_tokens is None and completion_tokens is None and cached_tokens is None:
        return None
    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )


def _session_error(data: dict[str, Any]) -> dict[str, Any] | None:
    error: dict[str, Any] = {}
    for key in ("errorType", "message", "code", "statusCode", "source", "fatal", "stack"):
        value = data.get(key)
        if value is not None:
            error[key] = value
    # An error shape this version of the CLI does not use yet is still worth
    # keeping verbatim rather than stringifying into something unparseable.
    return error or (dict(data) or None)


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _event_identity(event: dict[str, Any]) -> str:
    event_id = event.get("id")
    if isinstance(event_id, str) and event_id:
        return f"id:{event_id}"
    return f"event:{_json_fingerprint(event)}"


def _first_not_none(value: int | None, fallback: int | None) -> int | None:
    """Prefer a freshly reported value, keeping a legitimate zero."""
    return fallback if value is None else value


def _max_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None

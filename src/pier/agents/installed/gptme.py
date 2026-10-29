"""gptme installed agent for Pier.

Runs the ``gptme`` CLI in non-interactive mode and converts its line-delimited
JSON conversation output to an ATIF trajectory.

gptme emits one ``Message.to_dict()`` object per line on stdout when invoked
with ``--output-format json``. Tool calls are embedded in assistant content;
with ``--tool-format tool`` they use the native ``@tool(call_id): {json}``
syntax, and tool results are ``system`` rows carrying the matching ``call_id``.
That makes the conversion here lossless for tool identity without depending on
gptme's own parser at conversion time (the conversion runs on the Pier host,
where gptme is not installed).
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from typing import Any, ClassVar

from pier.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
)
from pier.agents.network import allowlist_from_urls
from pier.agents.utils import get_api_key_var_names_from_model_name
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
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)

#: Native gptme tool-call header, mirroring ``gptme.tools.base.toolcall_re``.
_TOOLCALL_RE = re.compile(
    r"^@([\w.]+)\(([\w\-:\.]+)\):\s*({.*)", re.MULTILINE | re.DOTALL
)

_UV_INSTALLER = "https://astral.sh/uv/0.7.13/install.sh"


def _find_json_end(text: str, start: int) -> int | None:
    """Return the index just past the JSON object beginning at ``start``."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _codeblock_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of fenced code blocks so examples are skipped."""
    ranges: list[tuple[int, int]] = []
    block_start: int | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if block_start is None:
                block_start = offset
            else:
                ranges.append((block_start, offset + len(line)))
                block_start = None
        offset += len(line)
    if block_start is not None:
        ranges.append((block_start, len(text)))
    return ranges


def iter_tool_calls(content: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Yield ``(tool_name, call_id, arguments)`` for native-format calls in content."""
    calls: list[tuple[str, str, dict[str, Any]]] = []
    excluded = _codeblock_ranges(content)
    search_from = 0
    while match := _TOOLCALL_RE.search(content, search_from):
        position = match.start()
        block_end = next(
            (end for start, end in excluded if start <= position < end), None
        )
        if block_end is not None:
            search_from = block_end
            continue
        json_start = match.start(3)
        json_end = _find_json_end(content, json_start)
        if json_end is None:
            break
        search_from = json_end
        try:
            arguments = json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        calls.append((match.group(1), match.group(2), arguments))
    return calls


_REASONING_OPEN = re.compile(
    r"^\s*(?:<\s*think(?:ing)?\s*>|Thinking)\s*", re.IGNORECASE
)
_REASONING_CLOSE = re.compile(r"<\s*/\s*think(?:ing)?\s*>")


def _split_reasoning(content: str) -> tuple[str | None, str]:
    """Split gptme's inline reasoning block out of the message.

    gptme renders thinking as a ``<think>``/``<thinking>`` block, and some
    providers emit a bare ``Thinking`` header instead. The closing tag is
    authoritative; only recognized opening markers are stripped so ordinary
    text before a closing tag is never silently discarded.
    """
    close = _REASONING_CLOSE.search(content)
    if close is None:
        return None, content
    head = content[: close.start()]
    if not _REASONING_OPEN.match(head):
        return None, content
    head = _REASONING_OPEN.sub("", head, count=1)
    return (head.strip() or None), content[close.end() :].lstrip()


def _step_metrics(metadata: dict[str, Any]) -> Metrics | None:
    usage = metadata.get("usage") or {}
    values: dict[str, Any] = {}
    for source, target in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("cache_read_tokens", "cached_tokens"),
    ):
        if isinstance(usage.get(source), int):
            values[target] = usage[source]
    if isinstance(metadata.get("cost"), (int, float)):
        values["cost_usd"] = float(metadata["cost"])
    return Metrics(**values) if values else None


def convert_messages(
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    version: str,
    model_name: str | None = None,
) -> Trajectory:
    """Convert gptme ``Message.to_dict()`` rows into an ATIF v1.7 trajectory.

    Missing usage is omitted rather than reported as zero, and unknown or
    duplicate call IDs are dropped instead of being attributed to the wrong
    step.
    """
    steps: list[Step] = []
    call_owners: dict[str, Step] = {}
    active_action: Step | None = None

    for row in messages:
        # gptme wraps each JSONL row as {"type": "message", ...}; ignore other
        # event kinds rather than misreading them as conversation turns.
        if row.get("type", "message") != "message":
            continue
        role = row.get("role")
        content = row.get("content") or ""
        call_id = row.get("call_id")

        if role == "system" and (call_id or active_action):
            target = call_owners.get(call_id) if call_id else active_action
            if target is not None:
                result = ObservationResult(
                    content=content, source_call_id=call_id if call_id else None
                )
                if target.observation is None:
                    target.observation = Observation(results=[])
                target.observation.results.append(result)
                if call_id:
                    active_action = None
                continue
            # Unknown call_id: not a tool result for any recorded call. Keep the
            # text as a plain system step instead of dropping it.

        active_action = None
        if role not in ("system", "user", "assistant"):
            continue

        step = Step(
            step_id=len(steps) + 1,
            source="agent" if role == "assistant" else role,
            message=content,
            timestamp=row.get("timestamp"),
        )
        if role == "assistant":
            metadata = row.get("metadata") or {}
            reasoning, content = _split_reasoning(content)
            step.message = content
            step.reasoning_content = reasoning
            calls = [
                ToolCall(
                    tool_call_id=identifier,
                    function_name=tool_name,
                    arguments=arguments,
                )
                for tool_name, identifier, arguments in iter_tool_calls(content)
            ]
            if calls:
                step.tool_calls = calls
                for call in calls:
                    call_owners.setdefault(call.tool_call_id, step)
                active_action = step
            if isinstance(metadata.get("model"), str):
                step.model_name = metadata["model"]
            step.metrics = _step_metrics(metadata)
        steps.append(step)

    if not steps:
        raise ValueError("gptme conversation contained no convertible messages")

    peak = peak_context_tokens_from_steps(steps)
    totals: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }
    cost = 0.0
    for step in steps:
        if step.metrics is None:
            continue
        for field in totals:
            value = getattr(step.metrics, field)
            if value is not None:
                totals[field] += value
        if step.metrics.cost_usd is not None:
            cost += step.metrics.cost_usd

    final_metrics = FinalMetrics(
        total_prompt_tokens=totals["prompt_tokens"] or None,
        total_completion_tokens=totals["completion_tokens"] or None,
        total_cached_tokens=totals["cached_tokens"] or None,
        total_cost_usd=cost or None,
        total_steps=len(steps),
        extra=extra_with_context_metrics(
            None, peak_context_tokens=peak, summarization_count=None
        ),
    )

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(name="gptme", version=version, model_name=model_name),
        steps=steps,
        notes=(
            "Converted from gptme --output-format json. Tool identity is "
            "explicit (call_id); missing usage is unknown, not zero."
        ),
        final_metrics=final_metrics,
    )


class Gptme(BaseInstalledAgent):
    """The gptme CLI agent (https://gptme.org)."""

    SUPPORTS_ATIF: bool = True

    _DEFAULT_PROVIDER_DOMAINS: ClassVar[dict[str, list[str]]] = {
        "anthropic": ["api.anthropic.com"],
        "deepseek": ["api.deepseek.com"],
        "google": [".googleapis.com"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
    }

    @staticmethod
    def name() -> str:
        return AgentName.GPTME.value

    def get_version_command(self) -> str | None:
        return '. "$HOME/.local/bin/env"; gptme --version'

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"=={self._version}" if self._version else ""
        root_run = (
            "if command -v apt-get &>/dev/null; then"
            "  apt-get update && apt-get install -y curl git;"
            " elif command -v apk &>/dev/null; then"
            "  apk add --no-cache curl bash git python3;"
            " elif command -v dnf &>/dev/null; then"
            "  dnf install -y curl git python3;"
            " fi"
        )
        agent_run = (
            "set -euo pipefail\n"
            f"curl -LsSf {_UV_INSTALLER} | sh\n"
            'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then\n'
            '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc"\n'
            "fi\n"
            '. "$HOME/.local/bin/env"\n'
            f"uv tool install gptme{version_spec}\n"
            '. "$HOME/.local/bin/env"; gptme --version\n'
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(run=root_run, user="root"),
                InstallStep(run=agent_run, user="agent"),
            ],
            verification_command='. "$HOME/.local/bin/env"; gptme --version',
        )

    def network_allowlist(self) -> NetworkAllowlist:
        urls: list[str] = []
        for key in (
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "ANTHROPIC_BASE_URL",
            "OPENROUTER_API_BASE",
        ):
            if value := self._get_env(key):
                urls.append(value)
        provider = self.model_name.split("/", 1)[0] if self.model_name else ""
        return allowlist_from_urls(
            urls, default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider, [])
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        conversation_path = self.logs_dir / "gptme.jsonl"
        if not conversation_path.exists():
            self.logger.debug(f"gptme conversation log {conversation_path} missing")
            return
        rows: list[dict[str, Any]] = []
        for line in conversation_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        try:
            trajectory = convert_messages(
                rows,
                session_id=str(uuid.uuid4()),
                version=self.version() or "unknown",
                model_name=self.model_name,
            )
        except ValueError as exc:
            self.logger.debug(f"Failed to convert gptme trajectory: {exc}")
            return
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2), encoding="utf-8"
        )
        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        model = self.model_name
        if not model:
            raise ValueError("gptme requires a model name (provider/model)")

        env = self.build_process_env({"GPTME_NON_INTERACTIVE": "1"})
        try:
            for key in get_api_key_var_names_from_model_name(model):
                if value := self._get_env(key):
                    env[key] = value
        except ValueError as exc:
            raise ValueError(
                f"Unable to determine API key for model {model}: {exc}"
            ) from exc

        agent_logs = EnvironmentPaths.agent_dir
        command = (
            '. "$HOME/.local/bin/env"; '
            "gptme --non-interactive --output-format json --tool-format tool "
            f"--model {shlex.quote(model)} "
            f"{shlex.quote(instruction)} "
            # gptme appends piped stdin to the prompt; keep the container's
            # inherited stdin out of the task instruction.
            f"< /dev/null 2> {agent_logs}/gptme.stderr "
            f"| tee {agent_logs}/gptme.jsonl"
        )
        await self.exec_as_agent(environment, command=command, env=env, cwd="/app")

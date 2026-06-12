import json
import shlex
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from pier.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.utils.pricing import resolve_cache_read_rate, resolve_model_pricing
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json


class CursorSystemEvent(BaseModel):
    type: Literal["system"]
    subtype: str
    apiKeySource: str | None = None
    cwd: str | None = None
    session_id: str
    model: str | None = None
    permissionMode: str | None = None


class CursorMessageContent(BaseModel):
    type: Literal["text"]
    text: str


class CursorMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[CursorMessageContent]


class CursorUserMessage(BaseModel):
    type: Literal["user"]
    message: CursorMessage
    session_id: str


class CursorAssistantMessage(BaseModel):
    type: Literal["assistant"]
    message: CursorMessage
    session_id: str
    model_call_id: str | None = None
    timestamp_ms: int | None = None


class CursorThinkingBlock(BaseModel):
    type: Literal["thinking"]
    subtype: str | None = None
    text: str | None = None
    session_id: str
    timestamp_ms: int | None = None


class CursorToolCall(BaseModel):
    type: Literal["tool_call"]
    subtype: Literal["started", "completed"]
    call_id: str
    tool_call: dict[str, Any]
    session_id: str
    model_call_id: str | None = None
    timestamp_ms: int | None = None


class CursorUsage(BaseModel):
    inputTokens: int = 0
    outputTokens: int = 0
    cacheReadTokens: int = 0
    cacheWriteTokens: int = 0
    totalCost: float | None = None
    cost: float | None = None

    def reported_cost_usd(self) -> float | None:
        if self.totalCost is not None:
            return self.totalCost
        if self.cost is not None:
            return self.cost
        return None


class CursorResult(BaseModel):
    type: Literal["result"]
    subtype: str
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool
    result: str
    usage: CursorUsage | None = None
    session_id: str
    request_id: str | None = None


class CursorInteractionQuery(BaseModel):
    type: Literal["interaction_query"]
    subtype: str | None = None
    query_type: str | None = None
    session_id: str
    timestamp_ms: int | None = None
    query: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


CursorEvent = TypeAdapter(
    Annotated[
        CursorSystemEvent
        | CursorUserMessage
        | CursorAssistantMessage
        | CursorThinkingBlock
        | CursorToolCall
        | CursorResult
        | CursorInteractionQuery,
        Field(discriminator="type"),
    ]
)


class CursorCli(BaseInstalledAgent):
    """
    Cursor CLI installed agent.

    Parses JSON lines emitted by ``cursor-agent --output-format=stream-json`` into
    ATIF. The event shape follows Cursor's CLI output format docs.
    """

    SUPPORTS_ATIF: bool = True
    _OUTPUT_FILENAME = "cursor-cli.txt"

    # Per-million-token USD rates from https://cursor.com/docs/models-and-pricing
    # (API pool table for Composer; Auto pool for auto). Converted to per-token below.
    _CURSOR_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
        "composer-2.5": {
            "input": 0.5,
            "output": 2.5,
            "cache_read": 0.2,
            "cache_write": 0.5,
        },
        "composer-2.5-fast": {
            "input": 3.0,
            "output": 15.0,
            "cache_read": 0.5,
            "cache_write": 3.0,
        },
        "composer-2": {
            "input": 0.5,
            "output": 2.5,
            "cache_read": 0.2,
            "cache_write": 0.5,
        },
        "composer-2-fast": {
            "input": 1.5,
            "output": 7.5,
            "cache_read": 0.35,
            "cache_write": 1.5,
        },
        "composer-1.5": {
            "input": 3.5,
            "output": 17.5,
            "cache_read": 0.35,
            "cache_write": 3.5,
        },
        "composer-1": {
            "input": 1.25,
            "output": 10.0,
            "cache_read": 0.125,
            "cache_write": 1.25,
        },
        "auto": {
            "input": 1.25,
            "output": 6.0,
            "cache_read": 0.25,
            "cache_write": 1.25,
        },
    }
    CLI_FLAGS = [
        CliFlag("mode", cli="--mode", type="enum", choices=["plan", "ask"]),
    ]

    @staticmethod
    def name() -> str:
        return AgentName.CURSOR_CLI.value

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; cursor-agent --version'

    def network_allowlist(self) -> NetworkAllowlist:
        return allowlist_from_urls(
            [],
            default_domains=[
                "cursor.com",
                "api.cursor.sh",
                "api2.cursor.sh",
                "api3.cursor.sh",
                ".cursor.sh",
            ],
        )

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=("apt-get update && apt-get install -y curl ca-certificates"),
                ),
                InstallStep(
                    user="agent",
                    run=(
                        "set -euo pipefail; "
                        "curl https://cursor.com/install -fsS | bash && "
                        'export PATH="$HOME/.local/bin:$PATH" && '
                        "cursor-agent --version"
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    @staticmethod
    def _millis_to_iso(timestamp_ms: int | None) -> str | None:
        if timestamp_ms is None:
            return None
        try:
            return datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    def _parse_stdout(self) -> list[dict[str, Any]]:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in output_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return events

    @classmethod
    def _model_slug(cls, model_name: str) -> str:
        return model_name.split("/", 1)[-1].lower()

    @classmethod
    def _cursor_builtin_pricing(cls, model_name: str) -> dict[str, float] | None:
        """Return per-token rates for known Cursor/Composer models, if any."""
        rates = cls._CURSOR_PRICING_PER_MILLION.get(cls._model_slug(model_name))
        if rates is None:
            return None
        return {key: value / 1_000_000 for key, value in rates.items()}

    @staticmethod
    def _cost_from_token_rates(
        usage_totals: dict[str, int],
        rates: dict[str, float],
    ) -> float:
        input_rate = rates["input"]
        output_rate = rates["output"]
        cache_read_rate = rates.get("cache_read", input_rate)
        cache_write_rate = rates.get("cache_write", input_rate)
        return (
            usage_totals.get("inputTokens", 0) * input_rate
            + usage_totals.get("cacheReadTokens", 0) * cache_read_rate
            + usage_totals.get("cacheWriteTokens", 0) * cache_write_rate
            + usage_totals.get("outputTokens", 0) * output_rate
        )

    def _resolve_pricing_rates(self) -> tuple[dict[str, float], str] | None:
        """Resolve per-token rates from built-in Cursor pricing or LiteLLM."""
        if not self.model_name:
            return None

        builtin = self._cursor_builtin_pricing(self.model_name)
        if builtin is not None:
            return builtin, "cursor_pricing"

        pricing = resolve_model_pricing(self.model_name, logger=self.logger)
        if pricing is None:
            self.logger.warning(
                "No pricing entry for model '%s'; leaving cursor-cli cost_usd as None",
                self.model_name,
            )
            return None

        # Cursor trajectories use prompt caching; refuse to silently bill cache
        # reads at the full input rate when the discounted rate is unknown.
        cache_read_rate = resolve_cache_read_rate(
            pricing, used_cached_tokens=True, logger=self.logger
        )
        if cache_read_rate is None:
            return None  # left unpriced; resolve_cache_read_rate logged why
        cache_write_rate = (
            pricing.cache_creation_input_token_cost
            if pricing.cache_creation_input_token_cost is not None
            else pricing.input_cost_per_token
        )

        return (
            {
                "input": pricing.input_cost_per_token,
                "output": pricing.output_cost_per_token,
                "cache_read": cache_read_rate,
                "cache_write": cache_write_rate,
            },
            pricing.source,
        )

    def _compute_cost_from_usage_totals(
        self,
        usage_totals: dict[str, int],
    ) -> tuple[float, str] | None:
        """Estimate USD cost from token usage when the CLI omits dollar cost.

        Uses built-in Cursor/Composer rates first, then LiteLLM's pricing table.
        Returns None rather than $0 when pricing is unavailable.
        """
        resolved = self._resolve_pricing_rates()
        if resolved is None:
            return None
        rates, source = resolved
        return self._cost_from_token_rates(usage_totals, rates), source

    def _apply_result_event(
        self,
        event: CursorResult,
        final_metrics: FinalMetrics,
    ) -> None:
        extra: dict[str, Any] = dict(final_metrics.extra or {})
        extra["duration_ms"] = extra.get("duration_ms", 0) + event.duration_ms
        extra["duration_api_ms"] = (
            extra.get("duration_api_ms", 0) + event.duration_api_ms
        )
        if event.request_id:
            extra["request_id"] = event.request_id
        if event.is_error:
            extra["is_error"] = True
            extra["result_subtype"] = event.subtype

        if event.usage is not None:
            usage_totals: dict[str, int] = dict(
                extra.get(
                    "usage_totals",
                    {
                        "inputTokens": 0,
                        "outputTokens": 0,
                        "cacheReadTokens": 0,
                        "cacheWriteTokens": 0,
                    },
                )
            )
            usage_totals["inputTokens"] += event.usage.inputTokens
            usage_totals["outputTokens"] += event.usage.outputTokens
            usage_totals["cacheReadTokens"] += event.usage.cacheReadTokens
            usage_totals["cacheWriteTokens"] += event.usage.cacheWriteTokens
            extra["usage_totals"] = usage_totals

            final_metrics.total_prompt_tokens = (
                (final_metrics.total_prompt_tokens or 0)
                + event.usage.inputTokens
                + event.usage.cacheReadTokens
                + event.usage.cacheWriteTokens
            )
            final_metrics.total_completion_tokens = (
                final_metrics.total_completion_tokens or 0
            ) + event.usage.outputTokens
            final_metrics.total_cached_tokens = (
                final_metrics.total_cached_tokens or 0
            ) + event.usage.cacheReadTokens

            reported_cost = event.usage.reported_cost_usd()
            if reported_cost is not None:
                final_metrics.total_cost_usd = (
                    final_metrics.total_cost_usd or 0.0
                ) + reported_cost
                extra["cost_source"] = "cursor_cli"

        final_metrics.extra = extra

    def _finalize_cost_metrics(self, final_metrics: FinalMetrics) -> None:
        """Fill total_cost_usd from token usage when the CLI did not report cost."""
        if final_metrics.total_cost_usd is not None:
            return
        extra = final_metrics.extra or {}
        usage_totals = extra.get("usage_totals")
        if not isinstance(usage_totals, dict):
            return
        estimated = self._compute_cost_from_usage_totals(usage_totals)
        if estimated is None:
            return
        cost, source = estimated
        final_metrics.total_cost_usd = cost
        final_metrics.extra = {**extra, "cost_source": source}

    @staticmethod
    def _message_text(message: CursorMessage) -> str:
        return "".join(part.text for part in message.content).strip()

    @staticmethod
    def _drain_thinking(thinking: list[CursorThinkingBlock]) -> str | None:
        reasoning_content = "".join(block.text or "" for block in thinking).strip()
        thinking.clear()
        return reasoning_content or None

    @staticmethod
    def _normalize_tool_result_content(result: Any) -> str | None:
        if result is None:
            return None
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _apply_tool_call_event(event: CursorToolCall, step: Step) -> None:
        if event.subtype != "completed":
            return
        if step.tool_calls is None:
            step.tool_calls = []
        if step.observation is None:
            step.observation = Observation(results=[])

        for tool_name, tool_call in event.tool_call.items():
            if not isinstance(tool_call, dict):
                continue
            args = tool_call.get("args")
            if not isinstance(args, dict):
                args = {}
            step.tool_calls.append(
                ToolCall(
                    tool_call_id=event.call_id,
                    function_name=tool_name,
                    arguments=args,
                )
            )
            step.observation.results.append(
                ObservationResult(
                    source_call_id=event.call_id,
                    content=CursorCli._normalize_tool_result_content(
                        tool_call.get("result")
                    ),
                )
            )

    def _convert_events_to_trajectory(
        self,
        events: list[dict[str, Any]],
    ) -> Trajectory | None:
        session_id: str | None = None
        steps: list[Step] = []
        step_id = 1
        model_call_steps: dict[str, Step] = {}
        thinking: list[CursorThinkingBlock] = []
        final_metrics = FinalMetrics()

        for event_dict in events:
            try:
                event = CursorEvent.validate_python(event_dict)
            except ValidationError as exc:
                self.logger.debug(
                    "Skipping unsupported Cursor CLI event type %r: %s",
                    event_dict.get("type"),
                    exc,
                )
                continue

            match event:
                case CursorSystemEvent():
                    session_id = event.session_id
                case CursorUserMessage():
                    steps.append(
                        Step(
                            step_id=step_id,
                            source="user",
                            message=self._message_text(event.message),
                        )
                    )
                    step_id += 1
                case CursorAssistantMessage():
                    reasoning_content = self._drain_thinking(thinking)
                    step = Step(
                        step_id=step_id,
                        timestamp=self._millis_to_iso(event.timestamp_ms),
                        source="agent",
                        model_name=self.model_name,
                        message=self._message_text(event.message),
                        reasoning_content=reasoning_content,
                        llm_call_count=1,
                    )
                    steps.append(step)
                    step_id += 1
                    if event.model_call_id:
                        model_call_steps[event.model_call_id] = step
                case CursorThinkingBlock():
                    if event.text:
                        thinking.append(event)
                case CursorToolCall():
                    step = (
                        model_call_steps.get(event.model_call_id)
                        if event.model_call_id
                        else None
                    )
                    if step is None:
                        reasoning_content = self._drain_thinking(thinking)
                        step = Step(
                            step_id=step_id,
                            timestamp=self._millis_to_iso(event.timestamp_ms),
                            source="agent",
                            model_name=self.model_name,
                            message="",
                            reasoning_content=reasoning_content,
                            llm_call_count=1,
                        )
                        steps.append(step)
                        step_id += 1
                        if event.model_call_id:
                            model_call_steps[event.model_call_id] = step
                    elif thinking:
                        reasoning_content = self._drain_thinking(thinking)
                        if reasoning_content:
                            step.reasoning_content = (
                                f"{step.reasoning_content}\n\n{reasoning_content}"
                                if step.reasoning_content
                                else reasoning_content
                            )
                    self._apply_tool_call_event(event, step)
                case CursorResult():
                    self._apply_result_event(event, final_metrics)
                case CursorInteractionQuery():
                    continue

        if thinking:
            steps.append(
                Step(
                    step_id=step_id,
                    source="agent",
                    model_name=self.model_name,
                    message="",
                    reasoning_content=self._drain_thinking(thinking),
                    llm_call_count=1,
                )
            )
            step_id += 1

        if not steps:
            return None

        self._finalize_cost_metrics(final_metrics)
        final_metrics.total_steps = len(steps)
        final_metrics.extra = extra_with_context_metrics(
            final_metrics.extra,
            peak_context_tokens=peak_context_tokens_from_steps(steps),
            summarization_count=None,
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id or "unknown",
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        if not events:
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception:
            self.logger.exception("Failed to convert Cursor CLI events to trajectory")
            return

        if not trajectory:
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            trajectory_path.write_text(
                format_trajectory_json(trajectory.to_json_dict())
            )
            self.logger.debug("Wrote Cursor CLI trajectory to %s", trajectory_path)
        except OSError as exc:
            self.logger.debug(
                "Failed to write trajectory file %s: %s",
                trajectory_path,
                exc,
            )

        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    def _build_register_mcp_servers_command(self) -> str | None:
        if not self.mcp_servers:
            return None
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "command": server.command,
                    "args": server.args,
                }
            else:
                servers[server.name] = {"url": server.url}
        config = shlex.quote(json.dumps({"mcpServers": servers}, indent=2))
        return f"mkdir -p ~/.cursor && printf '%s\n' {config} > ~/.cursor/mcp.json"

    def _build_register_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        skills_dir = shlex.quote(self.skills_dir)
        return (
            f"if [ -d {skills_dir} ]; then "
            "mkdir -p ~/.cursor/skills && "
            f"cp -r {skills_dir}/* ~/.cursor/skills/ 2>/dev/null || true; "
            "fi"
        )

    @staticmethod
    def _no_internet_cli_config() -> dict[str, Any]:
        return {
            "permissions": {
                "allow": [
                    "Shell(**)",
                    "Read(**)",
                    "Write(**)",
                    "Mcp(**)",
                ],
                "deny": [],
            },
            "approvalMode": "allowlist",
            "webFetchDomainAllowlist": [],
        }

    def _build_no_internet_cli_config_command(self) -> str:
        config = shlex.quote(json.dumps(self._no_internet_cli_config(), indent=2))
        return f"mkdir -p ~/.cursor && printf '%s\n' {config} > ~/.cursor/cli-config.json"

    @staticmethod
    def _should_disable_web_tools(environment: BaseEnvironment) -> bool:
        return not environment.task_env_config.allow_internet

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("Cursor CLI requires agent.model_name")

        model = self.model_name.split("/", 1)[-1]
        env = self.build_process_env(
            {"CURSOR_API_KEY": self._get_env("CURSOR_API_KEY")}
        )
        if "CURSOR_API_KEY" not in env:
            raise ValueError(
                "CURSOR_API_KEY is required for cursor-cli. "
                "Set it in the process environment or pass --env-file .env."
            )

        disable_web_tools = self._should_disable_web_tools(environment)
        for setup_command in (
            self._build_no_internet_cli_config_command() if disable_web_tools else None,
            self._build_register_mcp_servers_command(),
            self._build_register_skills_command(),
        ):
            if setup_command:
                await self.exec_as_agent(environment, command=setup_command, env=env)

        cli_flags = self.build_cli_flags()
        extra_flags = f"{cli_flags} " if cli_flags else ""
        escaped_instruction = shlex.quote(instruction)
        escaped_model = shlex.quote(model)
        permission_flag = "--trust" if disable_web_tools else "--force"

        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"cursor-agent {permission_flag} --print --output-format=stream-json "
                f"{extra_flags}--model {escaped_model} -- {escaped_instruction} "
                f"2>&1 | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env=env,
        )

import json
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import yaml

from pier.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
    CliFlag,
)
from pier.agents.network import allowlist_from_urls, collect_url_values
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
from pier.utils.logger import logger


def _normalize_content(raw_content: Any) -> str:
    """Normalize message content which may be a string, list of parts, or None."""
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for part in raw_content:
            if isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(raw_content)


def _iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    return None


def _message_timestamp(
    message: dict[str, Any], *, prefer_created: bool = False
) -> str | None:
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    fields = (
        ("created_at", "timestamp", "completed_at")
        if prefer_created
        else ("completed_at", "timestamp", "created_at")
    )
    for field in fields:
        timestamp = _iso_timestamp(message.get(field))
        if timestamp:
            return timestamp
    return _iso_timestamp(extra.get("timestamp"))


def _first_message_timestamp(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        timestamp = _message_timestamp(message, prefer_created=True)
        if timestamp:
            return timestamp
    return None


def _add_observation_to_last_agent_step(
    steps: list[Step],
    content: str,
    _logger: Any,
    message_index: int,
    timestamp: str | None = None,
) -> None:
    """Add observation content to the most recent agent step."""
    if steps and steps[-1].source == "agent":
        prev_step = steps[-1]
        if timestamp:
            prev_step.timestamp = timestamp
        if prev_step.observation and prev_step.observation.results:
            prev_step.observation.results.append(ObservationResult(content=content))
        else:
            prev_step.observation = Observation(
                results=[ObservationResult(content=content)]
            )
    else:
        _logger.warning(f"Message at index {message_index} has no preceding agent step")


def _build_step_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    prompt_tokens_details: dict[str, Any],
    completion_tokens_details: dict[str, Any],
    total_cost_usd: float,
    total_completion_tokens: int,
    step_cost_usd: float | None = None,
) -> Metrics | None:
    """Build metrics for an individual step."""
    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    step_cost = step_cost_usd
    if (
        step_cost is None
        and total_cost_usd > 0
        and total_completion_tokens > 0
        and completion_tokens > 0
    ):
        step_cost = (completion_tokens / total_completion_tokens) * total_cost_usd

    extra_metrics: dict[str, Any] = {}
    reasoning_tokens = completion_tokens_details.get("reasoning_tokens") or 0
    text_tokens = completion_tokens_details.get("text_tokens")
    if text_tokens is None and completion_tokens > 0 and reasoning_tokens > 0:
        text_tokens = max(0, completion_tokens - reasoning_tokens)
    if prompt_tokens_details:
        extra_metrics["prompt_tokens_details"] = prompt_tokens_details
    if completion_tokens_details:
        extra_metrics["completion_tokens_details"] = completion_tokens_details
    if text_tokens is not None:
        extra_metrics["text_tokens"] = text_tokens

    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens if cached_tokens > 0 else None,
        cost_usd=step_cost if step_cost and step_cost > 0 else None,
        extra=extra_metrics if extra_metrics else None,
    )


def _parse_tool_calls(
    message: dict[str, Any], step_id: int
) -> list[ToolCall] | None:
    """Parse tool calls from an assistant message into ATIF ToolCall objects."""
    message_tool_calls = message.get("tool_calls")
    if not message_tool_calls:
        return None

    tool_calls: list[ToolCall] = []
    for tc in message_tool_calls:
        tc_id = tc.get("id", f"call_{step_id}_{len(tool_calls) + 1}")
        func = tc.get("function") or {}
        func_name = func.get("name", "bash")
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                arguments = {"command": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"command": str(raw_args)}
        tool_calls.append(
            ToolCall(
                tool_call_id=tc_id,
                function_name=func_name,
                arguments=arguments,
            )
        )

    return tool_calls if tool_calls else None


def _reasoning_from_message(message: dict[str, Any]) -> str | None:
    if reasoning_content := _normalize_content(message.get("reasoning_content")):
        return reasoning_content
    if reasoning := _normalize_content(message.get("reasoning")):
        return reasoning

    thinking_parts: list[str] = []
    for block in message.get("thinking_blocks") or []:
        if isinstance(block, dict) and block.get("thinking"):
            thinking_parts.append(str(block["thinking"]))

    return "\n".join(thinking_parts) if thinking_parts else None


def _usage_from_message(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") or {}
    response_data = extra.get("response") or {}
    usage = response_data.get("usage") or message.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def _cost_from_usage(usage: dict[str, Any]) -> float | None:
    cost = usage.get("cost")
    if isinstance(cost, int | float) and cost > 0:
        return float(cost)

    cost_details = usage.get("cost_details")
    if not isinstance(cost_details, dict):
        return None
    upstream_cost = cost_details.get("upstream_inference_cost")
    if isinstance(upstream_cost, int | float) and upstream_cost > 0:
        return float(upstream_cost)
    return None


def _response_output_text_reasoning_and_tool_calls(
    message: dict[str, Any], step_id: int
) -> tuple[str, str | None, list[ToolCall] | None]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in message.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
        elif item_type == "reasoning":
            for part in item.get("summary") or item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    reasoning_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    reasoning_parts.append(part)
        elif item_type == "function_call":
            raw_args = item.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"command": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {"command": str(raw_args)}
            tool_calls.append(
                ToolCall(
                    tool_call_id=item.get("call_id")
                    or item.get("id")
                    or f"call_{step_id}_{len(tool_calls) + 1}",
                    function_name=item.get("name") or "bash",
                    arguments=arguments,
                )
            )

    return "\n".join(text_parts), "\n".join(reasoning_parts) or None, tool_calls or None


def convert_mini_swe_agent_to_atif(
    mini_swe_agent_trajectory: dict[str, Any],
    session_id: str,
) -> Trajectory:
    """
    Convert mini-swe-agent v2 trajectory format to ATIF format.

    Expects the v2 native tool-calling format where assistant messages
    contain a ``tool_calls`` array and tool results use ``role: "tool"``.

    Args:
        mini_swe_agent_trajectory: The mini-swe-agent trajectory data
        session_id: The session ID for the ATIF trajectory

    Returns:
        Trajectory: The converted ATIF trajectory
    """
    _logger = logger.getChild(__name__)

    # Extract metadata
    info = mini_swe_agent_trajectory.get("info") or {}
    config = info.get("config") or {}
    model_config = config.get("model") or {}
    agent_config = config.get("agent") or {}

    model_name = model_config.get("model_name") or "unknown"
    mini_version = info.get("mini_version") or "unknown"
    trajectory_format = mini_swe_agent_trajectory.get("trajectory_format", "unknown")

    messages = mini_swe_agent_trajectory.get("messages") or []
    session_start_timestamp = _first_message_timestamp(messages)

    steps: list[Step] = []
    step_id = 1

    # Track cumulative token counts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_text_tokens = 0
    total_cost_usd = (info.get("model_stats") or {}).get("instance_cost") or 0.0
    total_usage_cost = 0.0

    # First pass: count total completion tokens for cost apportioning
    for message in messages:
        usage = _usage_from_message(message)
        total_completion_tokens += (
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        total_usage_cost += _cost_from_usage(usage) or 0.0

    if total_cost_usd <= 0 and total_usage_cost > 0:
        total_cost_usd = total_usage_cost

    # Process messages
    for i, message in enumerate(messages):
        role = message.get("role")
        content = _normalize_content(message.get("content"))
        timestamp = _message_timestamp(message) or session_start_timestamp

        # Extract token usage
        usage = _usage_from_message(message)
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = (
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        prompt_tokens_details = (
            usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        )
        if not isinstance(prompt_tokens_details, dict):
            prompt_tokens_details = {}
        completion_tokens_details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        if not isinstance(completion_tokens_details, dict):
            completion_tokens_details = {}
        cached_tokens = (
            prompt_tokens_details.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        )
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens") or 0
        text_tokens = completion_tokens_details.get("text_tokens")
        if text_tokens is None and completion_tokens > 0 and reasoning_tokens > 0:
            text_tokens = max(0, completion_tokens - reasoning_tokens)
        step_cost_usd = _cost_from_usage(usage)

        total_prompt_tokens += prompt_tokens
        total_cached_tokens += cached_tokens
        total_reasoning_tokens += reasoning_tokens
        total_text_tokens += text_tokens or 0

        if role == "system":
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="system",
                    message=content,
                )
            )
            step_id += 1

        elif role == "user":
            if i == 1:
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=timestamp,
                        source="user",
                        message=content,
                    )
                )
                step_id += 1
            else:
                _add_observation_to_last_agent_step(
                    steps, content, _logger, i, timestamp
                )

        elif role == "tool":
            _add_observation_to_last_agent_step(steps, content, _logger, i, timestamp)

        elif role == "assistant":
            tool_calls = _parse_tool_calls(message, step_id)
            reasoning = _reasoning_from_message(message)

            metrics = _build_step_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                prompt_tokens_details=prompt_tokens_details,
                completion_tokens_details=completion_tokens_details,
                total_cost_usd=total_cost_usd,
                total_completion_tokens=total_completion_tokens,
                step_cost_usd=step_cost_usd,
            )

            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model_name,
                    message=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    llm_call_count=1,
                )
            )
            step_id += 1

        elif message.get("object") == "response":
            (
                response_content,
                reasoning,
                tool_calls,
            ) = _response_output_text_reasoning_and_tool_calls(message, step_id)

            metrics = _build_step_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                prompt_tokens_details=prompt_tokens_details,
                completion_tokens_details=completion_tokens_details,
                total_cost_usd=total_cost_usd,
                total_completion_tokens=total_completion_tokens,
                step_cost_usd=step_cost_usd,
            )

            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model_name,
                    message=response_content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    llm_call_count=1,
                )
            )
            step_id += 1

        elif message.get("type") == "function_call_output":
            output = message.get("output")
            if not isinstance(output, str):
                output = _normalize_content(output)
            _add_observation_to_last_agent_step(steps, output, _logger, i, timestamp)

    # Build final metrics
    final_extra: dict[str, Any] = {}
    if total_reasoning_tokens > 0:
        final_extra["total_reasoning_tokens"] = total_reasoning_tokens
    if total_text_tokens > 0:
        final_extra["total_text_tokens"] = total_text_tokens
    final_extra = extra_with_context_metrics(
        final_extra if final_extra else None,
        peak_context_tokens=peak_context_tokens_from_steps(steps),
        summarization_count=None,
    )

    final_metrics = FinalMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached_tokens if total_cached_tokens > 0 else None,
        total_cost_usd=total_cost_usd if total_cost_usd > 0 else None,
        total_steps=len(steps),
        extra=final_extra,
    )

    agent = Agent(
        name="mini-swe-agent",
        version=mini_version,
        model_name=model_name,
        extra={
            "original_format": trajectory_format,
            "agent_config": agent_config,
        },
    )

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=agent,
        steps=steps,
        final_metrics=final_metrics,
        notes="Converted from mini-swe-agent trajectory format to ATIF",
    )


def convert_and_save_trajectory(
    mini_swe_agent_trajectory_path: Path,
    atif_trajectory_path: Path,
    session_id: str,
) -> Trajectory:
    """
    Convert mini-swe-agent trajectory file to ATIF format and save it.

    Args:
        mini_swe_agent_trajectory_path: Path to mini-swe-agent trajectory.json
        atif_trajectory_path: Path to save the ATIF trajectory.json
        session_id: The session ID for the ATIF trajectory
    """
    _logger = logger.getChild(__name__)

    try:
        mini_swe_agent_trajectory = json.loads(
            mini_swe_agent_trajectory_path.read_text()
        )

        atif_trajectory = convert_mini_swe_agent_to_atif(
            mini_swe_agent_trajectory,
            session_id,
        )

        atif_trajectory_path.write_text(
            json.dumps(atif_trajectory.to_json_dict(), indent=2)
        )

        _logger.info(
            f"Successfully converted trajectory to ATIF format: {atif_trajectory_path}"
        )
        return atif_trajectory

    except Exception as e:
        _logger.error(f"Failed to convert trajectory: {e}")
        raise


class MiniSweAgent(BaseInstalledAgent):
    """
    The Mini SWE Agent uses the mini-swe-agent tool to solve tasks.
    """

    SUPPORTS_ATIF: bool = True

    CLI_FLAGS: ClassVar[list[CliFlag]] = []
    _LITELLM_MODEL_COST_MAP_URL = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )
    _DEFAULT_PROVIDER_DOMAINS: dict[str, list[str]] = {
        "anthropic": ["api.anthropic.com"],
        "bedrock": [".amazonaws.com"],
        "deepseek": ["api.deepseek.com"],
        "gemini": [".googleapis.com"],
        "google": [".googleapis.com"],
        "groq": ["api.groq.com"],
        "mistral": ["api.mistral.ai"],
        "openai": ["api.openai.com"],
        "openrouter": ["openrouter.ai"],
        "vertex_ai": [".googleapis.com"],
        "xai": ["api.x.ai"],
    }

    def __init__(
        self,
        cost_limit: str | int | float | None = 0,
        reasoning_effort: str | None = None,
        model_class: str | None = "auto",
        model_kwargs: dict[str, Any] | None = None,
        set_cache_control: str | None = None,
        config_yaml: str | None = None,
        config_file: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cost_limit = cost_limit
        self._reasoning_effort = reasoning_effort
        self._model_class = model_class
        self._model_kwargs = model_kwargs or {}
        self._set_cache_control = set_cache_control
        self._config_yaml = config_yaml
        if config_file:
            self._config_yaml = Path(config_file).read_text()

    @staticmethod
    def name() -> str:
        return AgentName.MINI_SWE_AGENT.value

    def get_version_command(self) -> str | None:
        return (
            '. "$HOME/.local/bin/env"; uv tool list 2>/dev/null | grep mini-swe-agent'
        )

    def parse_version(self, stdout: str) -> str:
        # Output: "mini-swe-agent v0.1.2"
        import re

        match = re.search(r"(\d+\.\d+\S*)", stdout)
        return match.group(1) if match else stdout.strip()

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"=={self._version}" if self._version else ""
        root_run = (
            "if command -v apt-get &>/dev/null; then"
            "  apt-get update && apt-get install -y curl build-essential git;"
            " elif command -v apk &>/dev/null; then"
            "  apk add --no-cache curl bash build-base git python3 py3-pip;"
            " elif command -v yum &>/dev/null; then"
            "  yum install -y curl git gcc make;"
            " elif command -v dnf &>/dev/null; then"
            "  dnf install -y curl git gcc make;"
            " else"
            '  echo "Warning: No known package manager found, assuming build tools are available" >&2;'
            " fi"
        )
        agent_run = f"""
set -euo pipefail
curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
source "$HOME/.local/bin/env"
uv tool install mini-swe-agent{version_spec}

python_bin="$(head -n 1 "$(command -v mini-swe-agent)" | sed 's/^#!//')"
"$python_bin" <<'PY'
import json
import sys
import urllib.request
from importlib.resources import files

url = "{self._LITELLM_MODEL_COST_MAP_URL}"
path = files("litellm").joinpath("model_prices_and_context_window_backup.json")

try:
    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict) or len(data) < 1000:
        raise ValueError(
            "unexpected LiteLLM model cost map shape: "
            f"{{type(data).__name__}}, "
            f"{{len(data) if isinstance(data, dict) else 'n/a'}} entries"
        )
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
except Exception as exc:
    print(
        f"Warning: failed to refresh LiteLLM model cost map backup: {{exc}}",
        file=sys.stderr,
    )
PY

mini-swe-agent --help
"""
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(
                    user="agent",
                    env={"LITELLM_LOCAL_MODEL_COST_MAP": "true"},
                    run=agent_run,
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        urls: list[str] = []
        for key in (
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "ANTHROPIC_BASE_URL",
            "GEMINI_API_BASE",
            "OPENROUTER_API_BASE",
        ):
            if value := self._get_env(key):
                urls.append(value)

        if self._config_yaml:
            try:
                parsed = yaml.safe_load(self._config_yaml) or {}
            except yaml.YAMLError:
                parsed = {}
            urls.extend(collect_url_values(parsed))

        provider = None
        if self.model_name and "/" in self.model_name:
            provider = self.model_name.split("/", 1)[0]

        return allowlist_from_urls(
            urls,
            default_domains=self._DEFAULT_PROVIDER_DOMAINS.get(provider or "", []),
        )

    @property
    def _mini_swe_agent_trajectory_path(self) -> PurePosixPath:
        """Path where mini-swe-agent writes its own trajectory format."""
        return EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json"

    @property
    def _atif_trajectory_path(self) -> PurePosixPath:
        """Path where we write the ATIF-formatted trajectory."""
        return EnvironmentPaths.agent_dir / "trajectory.json"

    @property
    def _model_class_override(self) -> str | None:
        if self._model_class != "auto":
            return self._model_class
        if self.model_name and self.model_name.startswith("openai/"):
            return "litellm_response"
        if self.model_name and self.model_name.startswith("openrouter/"):
            return "openrouter"
        return None

    @property
    def _run_model_name(self) -> str:
        if (
            self._model_class_override in {"openrouter", "openrouter_response"}
            and self.model_name
            and self.model_name.startswith("openrouter/")
        ):
            return self.model_name.removeprefix("openrouter/")
        return self.model_name or ""

    def _model_kwargs_config_flags(self) -> str:
        return "".join(
            f"-c model.model_kwargs.{key}={shlex.quote(json.dumps(value))} "
            for key, value in self._model_kwargs.items()
        )

    def _build_config_flags(self, *, custom_config_path: str | None = None) -> str:
        config_flags = "-c mini.yaml "

        if self._cost_limit is not None:
            config_flags += f"-c agent.cost_limit={shlex.quote(str(self._cost_limit))} "

        if custom_config_path:
            config_flags += f"-c {custom_config_path} "

        if model_class := self._model_class_override:
            config_flags += f"-c model.model_class={shlex.quote(model_class)} "

        if self._reasoning_effort:
            config_flags += (
                f"-c model.model_kwargs.reasoning_effort="
                f"{shlex.quote(self._reasoning_effort)} "
            )

        if self._set_cache_control:
            config_flags += (
                f"-c model.set_cache_control="
                f"{shlex.quote(self._set_cache_control)} "
            )

        config_flags += self._model_kwargs_config_flags()

        return config_flags

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Read the mini-swe-agent trajectory
        mini_trajectory_path = self.logs_dir / "mini-swe-agent.trajectory.json"

        if not mini_trajectory_path.exists():
            self.logger.debug(
                f"Mini-swe-agent trajectory file {mini_trajectory_path} does not exist"
            )
            return

        # Convert mini-swe-agent trajectory to ATIF format
        atif_trajectory_path = self.logs_dir / "trajectory.json"
        session_id = str(uuid.uuid4())
        try:
            atif_trajectory = convert_and_save_trajectory(
                mini_swe_agent_trajectory_path=mini_trajectory_path,
                atif_trajectory_path=atif_trajectory_path,
                session_id=session_id,
            )
            if atif_trajectory.final_metrics:
                populate_context_from_final_metrics(
                    context, atif_trajectory.final_metrics
                )
        except Exception as e:
            self.logger.debug(f"Failed to convert trajectory to ATIF format: {e}")

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        augmented_instruction = instruction
        if self.mcp_servers:
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for s in self.mcp_servers:
                if s.transport == "stdio":
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            augmented_instruction = instruction + mcp_info

        escaped_instruction = shlex.quote(augmented_instruction)

        run_model_name = self._run_model_name
        if not run_model_name or "/" not in run_model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        env = self.build_process_env(
            {
                "LITELLM_LOCAL_MODEL_COST_MAP": "true",
                "MSWEA_CONFIGURED": "true",  # Disable interactive setup
                "MSWEA_COST_TRACKING": "ignore_errors",  # Ignore unknown model costs
            }
        )

        if self._get_env("MSWEA_API_KEY"):
            env["MSWEA_API_KEY"] = self._get_env("MSWEA_API_KEY") or ""
        else:
            try:
                api_key_vars = get_api_key_var_names_from_model_name(self.model_name)
                for api_key_var in api_key_vars:
                    if self._get_env(api_key_var):
                        env[api_key_var] = self._get_env(api_key_var) or ""
                    else:
                        raise ValueError(
                            f"Unset API variable for model {self.model_name}. "
                            f"Please set {api_key_var} or MSWEA_API_KEY environment variable"
                        )
            except ValueError as e:
                raise ValueError(
                    f"Unable to determine API key for model {self.model_name}: {e}. "
                    "Please set MSWEA_API_KEY environment variable as fallback"
                )

        # Pass through common API base configurations if present
        if self._get_env("OPENAI_API_BASE"):
            env["OPENAI_API_BASE"] = self._get_env("OPENAI_API_BASE") or ""
        if self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = self._get_env("OPENAI_BASE_URL") or ""

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        custom_config_path = None

        # Write custom config into the container if provided
        if self._config_yaml:
            custom_config_path = "/tmp/mswea-config/custom.yaml"
            heredoc_marker = f"MSWEA_CONFIG_EOF_{uuid.uuid4().hex[:8]}"
            write_config_cmd = (
                f"mkdir -p /tmp/mswea-config\n"
                f"cat > '{custom_config_path}' << '{heredoc_marker}'\n"
                f"{self._config_yaml}\n"
                f"{heredoc_marker}\n"
            )
            await self.exec_as_agent(environment, command=write_config_cmd, env=env)

        config_flags = self._build_config_flags(custom_config_path=custom_config_path)

        await self.exec_as_agent(
            environment,
            command=(
                '. "$HOME/.local/bin/env"; '
                f"mini-swe-agent --yolo --model={run_model_name} --task={escaped_instruction} "
                f"--output={self._mini_swe_agent_trajectory_path} {extra_flags}"
                f"{config_flags}"
                f"--exit-immediately 2>&1 </dev/null | tee /logs/agent/mini-swe-agent.txt"
            ),
            env=env,
        )

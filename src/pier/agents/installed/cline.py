"""Pier harness for the Cline CLI.

Cline's ``--json`` mode emits newline-delimited extension messages.  The
adapter intentionally consumes that public stream rather than depending on
Cline's private on-disk session format, which keeps trials portable across
the CLI, VS Code, and standalone distributions.
"""

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pier.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
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
    SUPPORTS_ATIF = True
    _OUTPUT_FILENAME = "cline.txt"

    @staticmethod
    def name() -> str:
        return AgentName.CLINE.value

    def get_version_command(self) -> str | None:
        return "cline --version"

    def install_spec(self) -> AgentInstallSpec:
        package = f"@{self._version}" if self._version else ""
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="agent",
                    run=f"set -euo pipefail; npm install --global cline{package}; cline --version",
                )
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        domains: set[str] = set()
        provider = (self.model_name or "").split("/", 1)[0]
        defaults = {
            "cline": ["app.cline.bot", "api.cline.bot"],
            "anthropic": ["api.anthropic.com"],
            "openai": ["api.openai.com"],
            "openrouter": ["openrouter.ai"],
            "gemini": ["generativelanguage.googleapis.com"],
        }
        domains.update(defaults.get(provider, []))
        for key in ("CLINE_API_URL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
            value = self._get_env(key)
            if value:
                hostname = urlparse(
                    value if "://" in value else f"https://{value}"
                ).hostname
                if hostname:
                    domains.add(hostname)
        return NetworkAllowlist(domains=sorted(domains or {"api.cline.bot"}))

    def _parse_stdout(self) -> list[dict[str, Any]]:
        path = self.logs_dir / self._OUTPUT_FILENAME
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

    @staticmethod
    def _timestamp(event: dict[str, Any]) -> str | None:
        value = event.get("ts", event.get("timestamp"))
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
            return str(value)
        except (OSError, ValueError, OverflowError):
            return None

    @staticmethod
    def _text(event: dict[str, Any]) -> str:
        text = event.get("text", "")
        if isinstance(text, str):
            return text
        return json.dumps(text, ensure_ascii=False)

    def _convert_events_to_trajectory(self, events: list[dict[str, Any]]) -> Trajectory | None:
        if not events:
            return None

        steps: list[Step] = []
        prompt_tokens = output_tokens = cached_tokens = 0
        cost = 0.0
        session_id = "unknown"
        for event in events:
            session_id = str(event.get("taskId", event.get("session_id", session_id)))
            category = event.get("type")
            subtype = event.get("say", event.get("ask"))
            source = "agent" if category == "say" else "user" if category == "ask" else "system"
            if source == "system" or not subtype:
                continue

            text = self._text(event)
            reasoning = event.get("reasoning")
            tool_calls: list[ToolCall] = []
            observations: list[ObservationResult] = []
            raw_tool = event.get("tool") or event.get("tool_call")
            if isinstance(raw_tool, dict):
                call_id = str(raw_tool.get("id", raw_tool.get("tool_use_id", f"tool-{len(steps) + 1}")))
                tool_name = str(raw_tool.get("name", raw_tool.get("function", subtype)))
                arguments = raw_tool.get("input", raw_tool.get("arguments", {}))
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                tool_calls.append(ToolCall(tool_call_id=call_id, function_name=tool_name, arguments=arguments))
                if "result" in raw_tool:
                    observations.append(ObservationResult(source_call_id=call_id, content=str(raw_tool["result"])))

            usage = event.get("usage") or event.get("metrics") or {}
            if not isinstance(usage, dict):
                usage = {}
            prompt = usage.get("inputTokens", usage.get("prompt_tokens", 0)) or 0
            completion = usage.get("outputTokens", usage.get("completion_tokens", 0)) or 0
            cached = usage.get("cacheReadTokens", usage.get("cached_tokens", 0)) or 0
            if usage:
                prompt_tokens += int(prompt) + int(cached)
                output_tokens += int(completion)
                cached_tokens += int(cached)
                cost += float(usage.get("costUsd", usage.get("cost_usd", 0)) or 0)

            metrics = Metrics(
                prompt_tokens=int(prompt) + int(cached),
                completion_tokens=int(completion),
                cached_tokens=int(cached) if cached else None,
                cost_usd=(
                    float(usage.get("costUsd", usage.get("cost_usd", 0)))
                    if usage.get("costUsd", usage.get("cost_usd")) is not None
                    else None
                ),
            ) if usage else None
            kwargs: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "timestamp": self._timestamp(event),
                "source": source,
                "message": text,
                "llm_call_count": 1 if source == "agent" else None,
            }
            if source == "agent":
                kwargs["model_name"] = self.model_name
                if reasoning:
                    kwargs["reasoning_content"] = str(reasoning)
                if tool_calls:
                    kwargs["tool_calls"] = tool_calls
                if observations:
                    kwargs["observation"] = Observation(results=observations)
                if metrics:
                    kwargs["metrics"] = metrics
            steps.append(Step(**kwargs))

        if not steps:
            return None
        final = FinalMetrics(
            total_prompt_tokens=prompt_tokens or None,
            total_completion_tokens=output_tokens or None,
            total_cached_tokens=cached_tokens or None,
            total_cost_usd=cost or None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=None,
            ),
        )
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory = self._convert_events_to_trajectory(self._parse_stdout())
        if not trajectory:
            return
        (self.logs_dir / "trajectory.json").write_text(
            format_trajectory_json(trajectory.to_json_dict()), encoding="utf-8"
        )
        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        provider, model = (
            self.model_name.split("/", 1)
            if self.model_name and "/" in self.model_name
            else ("cline", self.model_name or "")
        )
        env = self.build_process_env()
        for key in (
            "CLINE_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "CLINE_API_URL",
            "ANTHROPIC_BASE_URL",
            "OPENAI_BASE_URL",
        ):
            if value := self._get_env(key):
                env[key] = value
        args = [
            "cline",
            "--json",
            "--auto-approve",
            "true",
            "--provider",
            provider,
        ]
        if model:
            args.extend(["--model", model])
        args.extend(["--", instruction])
        command = (
            " ".join(shlex.quote(arg) for arg in args)
            + " 2>&1 | tee /logs/agent/cline.txt"
        )
        result = await self.exec_as_agent(environment, command=command, env=env)
        if result.return_code != 0:
            raise NonZeroAgentExitCodeError(f"Cline exited with code {result.return_code}")
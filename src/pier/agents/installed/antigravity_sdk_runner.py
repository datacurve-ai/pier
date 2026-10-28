"""Run google-antigravity 0.1.9 and persist its stream as ATIF v1.7."""

import argparse
import asyncio
import enum
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _value(value: Any) -> Any:
    """Return a JSON-safe representation of an SDK value."""
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _enum_value(value: Any) -> str:
    return str(_value(value))


def _content(value: Any) -> str:
    normalized = _value(value)
    return (
        normalized
        if isinstance(normalized, str)
        else json.dumps(normalized, ensure_ascii=False)
    )


def _tool_name(tool: Any) -> str:
    """Preserve the MCP server name in ATIF's flat function namespace."""
    name = _enum_value(tool.name)
    server = getattr(tool, "server_name", None)
    return f"mcp_{server}_{name}" if server else name


def _result_content(result: Any) -> str:
    value = {"error": result.error} if getattr(result, "error", None) else result.result
    return _content(value)


def resolve_skill_paths(raw_paths: str | None) -> list[str] | None:
    """Parse configured skill paths and expand them as the sandbox agent user."""
    if not raw_paths:
        return None
    try:
        parsed = json.loads(raw_paths)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(
        isinstance(path, str) for path in parsed
    ):
        return None
    return [str(Path(path).expanduser()) for path in parsed]


def thinking_level(effort: str, levels: Any) -> Any:
    """Map a validated effort value onto Antigravity's supported levels."""
    mapping = {
        "minimal": levels.MINIMAL,
        "low": levels.LOW,
        "medium": levels.MEDIUM,
        "high": levels.HIGH,
    }
    if effort not in mapping:
        raise ValueError(f"invalid REASONING_EFFORT value: {effort}")
    return mapping[effort]


def is_complete_response(step: Any) -> bool:
    return bool(getattr(step, "is_complete_response", False))


class AtifCollector:
    """Aggregate cumulative SDK events into one ATIF step per model call."""

    def __init__(
        self,
        instruction: str,
        normalized_model: str,
        model_source: Any,
        reasoning_effort: str | None = None,
    ) -> None:
        self.normalized_model = normalized_model
        self.model_source = model_source
        self.reasoning_effort = reasoning_effort
        self.steps: list[dict[str, Any]] = [
            {
                "step_id": 1,
                "timestamp": None,
                "source": "user",
                "message": instruction,
            }
        ]
        self.complete_response_seen = False
        self._current: dict[str, Any] | None = None
        self._current_closed = True
        self._step_groups: dict[str, dict[str, Any]] = {}
        self._content: dict[str, str] = {}
        self._thinking: dict[str, str] = {}
        self._seen_tool_results: set[str] = set()
        self._tool_output_snapshots: set[str] = set()
        self._tool_groups: dict[str, dict[str, Any]] = {}
        self._tool_metadata: dict[str, tuple[str, str | None]] = {}
        self._unmatched_results: list[Any] = []
        self._usage_by_step: dict[str, Any] = {}

    @staticmethod
    def _step_key(step: Any) -> str:
        if sdk_id := getattr(step, "id", ""):
            return sdk_id
        return ":".join(
            (
                str(getattr(step, "step_index", 0)),
                _enum_value(getattr(step, "type", "")),
                _enum_value(getattr(step, "source", "")),
            )
        )

    def _new_agent_step(self) -> dict[str, Any]:
        step: dict[str, Any] = {
            "step_id": len(self.steps) + 1,
            "timestamp": None,
            "source": "agent",
            "message": "",
            "model_name": self.normalized_model,
            "llm_call_count": 1,
        }
        if self.reasoning_effort:
            step["reasoning_effort"] = self.reasoning_effort
        self.steps.append(step)
        return step

    def record_step(self, step: Any) -> None:
        """Merge one cumulative SDK update into its model-response group."""
        step_key = self._step_key(step)
        self.complete_response_seen |= is_complete_response(step)
        calls = getattr(step, "tool_calls", None) or []
        usage = getattr(step, "usage_metadata", None)
        starts_call = bool(
            calls
            or getattr(step, "thinking", "")
            or is_complete_response(step)
            or usage
        )
        group = self._step_groups.get(step_key)
        if group is None and (
            starts_call
            or (
                step.source == self.model_source
                and self._current is not None
                and not self._current_closed
            )
        ):
            if self._current is None or self._current_closed:
                self._current = self._new_agent_step()
                self._current_closed = False
            group = self._step_groups[step_key] = self._current
        elif group is None:
            if (
                step.source != self.model_source
                and _enum_value(getattr(step, "status", "")) == "DONE"
                and getattr(step, "content", "")
            ):
                system_step = {
                    "step_id": len(self.steps) + 1,
                    "timestamp": None,
                    "source": "system",
                    "message": step.content,
                }
                self._step_groups[step_key] = system_step
                self._content[step_key] = system_step["message"]
                self.steps.append(system_step)
            return

        if group["source"] != "agent" and starts_call:
            group.update(
                {
                    "source": "agent",
                    "model_name": self.normalized_model,
                    "llm_call_count": 1,
                }
            )
            if self.reasoning_effort:
                group["reasoning_effort"] = self.reasoning_effort
            self._current = group
            self._current_closed = False

        content = getattr(step, "content", "") or ""
        if content or step_key not in self._content:
            self._content[step_key] = content
        thinking = getattr(step, "thinking", "") or ""
        if thinking or step_key not in self._thinking:
            self._thinking[step_key] = thinking
        group["message"] = "\n".join(
            value
            for key, value in self._content.items()
            if self._step_groups[key] is group and value
        )
        reasoning = "\n".join(
            value
            for key, value in self._thinking.items()
            if self._step_groups[key] is group and value
        )
        if reasoning:
            group["reasoning_content"] = reasoning

        for index, call in enumerate(calls):
            call_id = call.id or f"{step_key}:tool:{index}"
            self._tool_groups[call_id] = group
            self._tool_metadata[call_id] = (
                _enum_value(call.name),
                getattr(call, "server_name", None),
            )
            tool_calls = group.setdefault("tool_calls", [])
            arguments = _value(call.args or {})
            converted = {
                "tool_call_id": call_id,
                "function_name": _tool_name(call),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
            existing = next(
                (item for item in tool_calls if item["tool_call_id"] == call_id),
                None,
            )
            if existing is None:
                tool_calls.append(converted)
            elif call_id not in self._seen_tool_results:
                existing.update(converted)
            if getattr(call, "output", None) is not None:
                self._tool_output_snapshots.add(call_id)
                self._add_observation(call_id, _content(call.output))
        self._match_results()

        if usage:
            self._usage_by_step[step_key] = usage
            group["metrics"] = {
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count,
                "cached_tokens": usage.cached_content_token_count,
            }
            if group is self._current:
                self._current_closed = True

    def record_tool_result(self, result: Any) -> None:
        if not self._attach_result(result):
            self._unmatched_results.append(result)

    def _match_results(self) -> None:
        unmatched = self._unmatched_results
        self._unmatched_results = []
        for result in unmatched:
            if not self._attach_result(result):
                self._unmatched_results.append(result)

    def _attach_result(self, result: Any) -> bool:
        call_id = getattr(result, "id", None)
        if not call_id:
            result_name = _enum_value(result.name)
            result_server = getattr(result, "server_name", None)
            candidates = [
                candidate
                for candidate, (name, server) in self._tool_metadata.items()
                if candidate not in self._seen_tool_results
                and candidate not in self._tool_output_snapshots
                and name == result_name
                and (result_server is None or server == result_server)
            ]
            if len(candidates) != 1:
                return False
            call_id = candidates[0]
        if call_id not in self._tool_groups:
            return False
        if call_id in self._seen_tool_results:
            return True
        self._seen_tool_results.add(call_id)
        self._add_observation(call_id, _result_content(result), authoritative=True)
        return True

    def _add_observation(
        self, call_id: str, content: str, *, authoritative: bool = False
    ) -> None:
        step = self._tool_groups.get(call_id)
        if step is None:
            return
        if not authoritative and call_id in self._seen_tool_results:
            return
        observation = step.setdefault("observation", {"results": []})
        existing = next(
            (
                item
                for item in observation["results"]
                if item["source_call_id"] == call_id
            ),
            None,
        )
        if existing is None:
            observation["results"].append(
                {"source_call_id": call_id, "content": content}
            )
        else:
            existing["content"] = content

    def totals(self) -> tuple[int, int, int]:
        prompt = sum(
            item.prompt_token_count or 0 for item in self._usage_by_step.values()
        )
        completion = sum(
            item.candidates_token_count or 0 for item in self._usage_by_step.values()
        )
        cached = sum(
            item.cached_content_token_count or 0
            for item in self._usage_by_step.values()
        )
        return prompt, completion, cached


async def run_agent(args: Any) -> None:
    from google.antigravity import (
        Agent,
        GeminiAPIEndpoint,
        GeminiModelOptions,
        LocalAgentConfig,
        ModelTarget,
        ModelType,
        ThinkingLevel,
    )
    from google.antigravity.hooks import hooks, policy
    from google.antigravity.types import (
        McpStdioServer,
        McpStreamableHttpServer,
        StepSource,
    )

    logging.getLogger().setLevel(logging.ERROR)

    model = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")
    if not model:
        raise ValueError("MODEL_NAME environment variable must be set")
    normalized_model = model.split("/", 1)[-1]

    mcp_servers = []
    if raw_mcp_servers := os.environ.get("MCP_SERVERS_JSON"):
        for mcp in json.loads(raw_mcp_servers):
            transport = mcp.get("transport", "stdio")
            if transport == "stdio":
                mcp_servers.append(
                    McpStdioServer(
                        name=mcp.get("name"),
                        command=mcp.get("command"),
                        args=mcp.get("args", []),
                    )
                )
            elif transport == "streamable-http":
                mcp_servers.append(
                    McpStreamableHttpServer(name=mcp.get("name"), url=mcp.get("url"))
                )
            else:
                raise ValueError(
                    f"Unsupported MCP transport for Antigravity 0.1.9: {transport}"
                )

    reasoning_effort = os.environ.get("REASONING_EFFORT", "medium").lower()
    level = thinking_level(reasoning_effort, ThinkingLevel)
    model_target = ModelTarget(
        name=normalized_model,
        types=[ModelType.TEXT],
        endpoint=GeminiAPIEndpoint(
            api_key=api_key,
            options=GeminiModelOptions(thinking_level=level),
        ),
    )

    skills_paths = resolve_skill_paths(os.environ.get("SKILLS_PATHS_JSON"))

    trajectory_path = Path(args.trajectory_path)
    raw_events_path = Path(args.logs_dir) / "antigravity-sdk-events.jsonl"
    raw_events_path.parent.mkdir(parents=True, exist_ok=True)
    raw_events_path.write_text("")
    collector = AtifCollector(
        args.instruction,
        normalized_model,
        StepSource.MODEL,
        reasoning_effort,
    )
    agent_version = version("google-antigravity")

    def checkpoint() -> None:
        _atomic_json_write(
            trajectory_path,
            build_atif_trajectory(
                collector.steps,
                *collector.totals(),
                agent_version=agent_version,
                model_name=normalized_model,
                incomplete=not collector.complete_response_seen,
                raw_events_path=raw_events_path.name,
            ),
        )

    checkpoint()

    @hooks.post_tool_call
    async def record_tool_result(result: Any) -> None:
        _append_jsonl(raw_events_path, "tool_result", result)
        collector.record_tool_result(result)
        checkpoint()

    config = LocalAgentConfig(
        models=[model_target],
        api_key=api_key,
        mcp_servers=mcp_servers,
        policies=[policy.allow_all()],
        hooks=[record_tool_result],
        skills_paths=skills_paths,
        workspaces=["/"],
    )

    print(f"Starting Antigravity SDK agent: {args.instruction[:200]}...")
    print(f"Using model: {normalized_model}")
    if mcp_servers:
        print(f"MCP servers: {[server.name for server in mcp_servers]}")

    async with Agent(config) as agent:
        await agent.conversation.send(args.instruction)
        async for step in agent.conversation.receive_steps():
            if step.source != StepSource.USER:
                _append_jsonl(raw_events_path, "step", step)
                collector.record_step(step)
                checkpoint()

    if not collector.complete_response_seen:
        raise RuntimeError(
            "Antigravity SDK became idle without an is_complete_response event; "
            f"raw events saved to {raw_events_path}"
        )
    print(f"Agent completed. Trajectory saved to {trajectory_path}")


def build_atif_trajectory(
    steps: list[dict[str, Any]],
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_cached_tokens: int,
    agent_version: str = "0.1.9",
    model_name: str | None = None,
    incomplete: bool = False,
    raw_events_path: str | None = None,
) -> dict[str, Any]:
    """Build an augmented ATIF v1.7 trajectory from collected steps."""
    for index, step in enumerate(steps):
        step["step_id"] = index + 1
    prompt_sizes = [
        metrics["prompt_tokens"]
        for step in steps
        if isinstance((metrics := step.get("metrics")), dict)
        and isinstance(metrics.get("prompt_tokens"), int)
    ]
    final_metrics: dict[str, Any] = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_cost_usd": None,
        "total_steps": len(steps),
    }
    if prompt_sizes:
        final_metrics["extra"] = {"peak_context_tokens": max(prompt_sizes)}

    trajectory: dict[str, Any] = {
        "schema_version": "ATIF-v1.7",
        "session_id": os.environ.get("SESSION_ID", "pier-session"),
        "agent": {
            "name": "antigravity-sdk",
            "version": agent_version,
            "model_name": model_name,
        },
        "steps": steps,
        "final_metrics": final_metrics,
    }
    if incomplete:
        trajectory["extra"] = {
            "incomplete": True,
            "reason": "missing_is_complete_response",
            "raw_events_path": raw_events_path,
        }
    return trajectory


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, kind: str, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(
            json.dumps({"kind": kind, "value": _value(value)}, ensure_ascii=False)
            + "\n"
        )


def main() -> None:
    if "--version" in sys.argv:
        print(version("google-antigravity"))
        return
    parser = argparse.ArgumentParser(description="Run Google Antigravity SDK agent")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--trajectory-path", required=True)
    asyncio.run(run_agent(parser.parse_args()))


if __name__ == "__main__":
    main()

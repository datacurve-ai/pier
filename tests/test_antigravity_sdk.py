import enum
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pier.agents.factory import AgentFactory
from pier.agents.installed.antigravity_sdk import AntigravitySDK
from pier.agents.installed.antigravity_sdk_runner import (
    AtifCollector,
    _append_jsonl,
    _atomic_json_write,
    build_atif_trajectory,
    resolve_skill_paths,
    thinking_level,
)
from pier.environments.base import BaseEnvironment, ExecResult
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName
from pier.models.task.config import MCPServerConfig
from pier.models.trajectories import Trajectory


class FakeEnvironment:
    session_id = "trial-session"

    def __init__(self, *, agent_install_spec: Any = None) -> None:
        self.agent_install_spec = agent_install_spec
        self.exec_calls: list[dict[str, Any]] = []
        self.uploaded: list[tuple[Path | str, str]] = []

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        return ExecResult(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploaded.append((source_path, target_path))


def _agent(tmp_path: Path, **kwargs: Any) -> AntigravitySDK:
    return AntigravitySDK(
        logs_dir=tmp_path,
        model_name="google/gemini-3.6-flash",
        extra_env={"GEMINI_API_KEY": "test-key"},
        **kwargs,
    )


def _step(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": "step-1",
        "step_index": 1,
        "type": "TEXT_RESPONSE",
        "source": "model",
        "status": "DONE",
        "content": "",
        "thinking": "",
        "tool_calls": [],
        "usage_metadata": None,
        "is_complete_response": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _usage(prompt: int = 10, completion: int = 2, cached: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        cached_content_token_count=cached,
    )


def test_agent_is_registered_and_uses_pinned_runtime(tmp_path: Path) -> None:
    agent = AgentFactory.create_agent_from_name(
        AgentName.ANTIGRAVITY_SDK,
        logs_dir=tmp_path,
        model_name="google/gemini-3.6-flash",
    )

    assert isinstance(agent, AntigravitySDK)
    assert agent.SUPPORTS_ATIF
    spec = agent.install_spec()
    assert spec.agent_name == "antigravity-sdk"
    assert spec.version == "0.1.9"
    assert "uv venv --python 3.12 /installed-agent/venv" in spec.steps[0].run
    assert "google-antigravity==0.1.9" in spec.steps[0].run
    assert "mcp==1.27.2" in spec.steps[0].run
    assert "protobuf==7.35.1" in spec.steps[0].run
    assert "https://astral.sh/uv/0.7.13/install.sh" in spec.steps[0].run
    assert "sha256sum -c -" in spec.steps[0].run
    assert (
        "curl -LsSf https://astral.sh/uv/0.7.13/install.sh |" not in spec.steps[0].run
    )
    assert "UV_PYTHON_INSTALL_DIR=/installed-agent/python" in spec.steps[0].run
    assert "chmod -R a+rX /installed-agent" in spec.steps[0].run
    assert "--require-hashes" in spec.steps[0].run
    assert "--no-deps" in spec.steps[0].run
    assert "-r /installed-agent/antigravity_sdk_requirements.lock" in spec.steps[0].run
    assert spec.metadata["requirements-lock"] == "antigravity_sdk_requirements.lock"
    assert spec.steps[-1].user == "agent"
    assert "google-antigravity" in spec.steps[-1].run


@pytest.mark.asyncio
async def test_setup_uploads_runner_after_preinstalled_runtime(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    environment = FakeEnvironment(agent_install_spec=agent.install_spec())

    await agent.setup(cast(BaseEnvironment, environment))

    assert len(environment.uploaded) == 1
    source, target = environment.uploaded[0]
    assert Path(source).name == "antigravity_sdk_runner.py"
    assert target == "/installed-agent/antigravity_sdk_runner.py"
    assert environment.exec_calls[0]["command"] == "mkdir -p /installed-agent"
    assert "chmod a+r" in environment.exec_calls[-1]["command"]


@pytest.mark.asyncio
async def test_run_passes_model_mcp_skills_and_session(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        reasoning_effort="high",
        skills_dir="/opt/task-skills",
        skill_paths=["/opt/shared-skills"],
        mcp_servers=[
            MCPServerConfig(
                name="repo",
                transport="stdio",
                command="python",
                args=["server.py"],
            ),
            MCPServerConfig(
                name="docs",
                transport="streamable-http",
                url="https://mcp.example.com/api",
            ),
        ],
    )
    environment = FakeEnvironment()

    await agent.run(
        "Fix the user's bug", cast(BaseEnvironment, environment), AgentContext()
    )

    call = environment.exec_calls[-1]
    assert "/installed-agent/venv/bin/python" in call["command"]
    assert "Fix the user'\"'\"'s bug" in call["command"]
    env = call["env"]
    assert env["MODEL_NAME"] == "google/gemini-3.6-flash"
    assert env["REASONING_EFFORT"] == "high"
    assert env["SESSION_ID"] == "trial-session"
    assert json.loads(env["SKILLS_PATHS_JSON"]) == [
        "/opt/task-skills",
        "/opt/shared-skills",
    ]
    assert json.loads(env["MCP_SERVERS_JSON"]) == [
        {
            "name": "repo",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
        },
        {
            "name": "docs",
            "transport": "streamable-http",
            "url": "https://mcp.example.com/api",
        },
    ]


def test_invalid_version_and_reasoning_effort_fail_early(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires google-antigravity==0.1.9"):
        _agent(tmp_path, version="0.2.0")
    with pytest.raises(ValueError, match="Invalid reasoning_effort"):
        _agent(tmp_path, reasoning_effort="maximum")
    with pytest.raises(ValueError, match="Invalid reasoning_effort"):
        _agent(tmp_path, reasoning_effort="xhigh")


def test_none_reasoning_effort_uses_medium_default(tmp_path: Path) -> None:
    agent = _agent(tmp_path, reasoning_effort=None)

    assert agent._reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_run_rejects_sse_mcp_without_silently_dropping_it(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        mcp_servers=[
            MCPServerConfig(name="legacy", transport="sse", url="https://mcp.test/sse")
        ],
    )

    with pytest.raises(ValueError, match="does not support MCP SSE"):
        await agent.run(
            "Fix it", cast(BaseEnvironment, FakeEnvironment()), AgentContext()
        )


def test_populate_context_validates_trajectory_and_adds_known_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    trajectory = build_atif_trajectory(
        steps=[
            {"step_id": 1, "timestamp": None, "source": "user", "message": "Fix it"},
            {
                "step_id": 2,
                "timestamp": None,
                "source": "agent",
                "message": "Done",
                "model_name": "gemini-3.6-flash",
                "llm_call_count": 1,
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cached_tokens": 10,
                },
            },
        ],
        total_prompt_tokens=100,
        total_completion_tokens=20,
        total_cached_tokens=10,
        model_name="gemini-3.6-flash",
    )
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
    monkeypatch.setattr(agent, "_compute_cost_from_pricing", lambda *_: 0.25)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 100
    assert context.n_output_tokens == 20
    assert context.n_cache_tokens == 10
    assert context.cost_usd == 0.25
    assert context.peak_context_tokens == 100
    assert context.n_agent_steps == 1
    saved = Trajectory.model_validate_json((tmp_path / "trajectory.json").read_text())
    assert saved.final_metrics is not None
    assert saved.final_metrics.total_cost_usd == 0.25


def test_skill_paths_expand_as_sandbox_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/home/agent")

    assert resolve_skill_paths(
        json.dumps(["~/.claude/skills", "/opt/task-skills"])
    ) == ["/home/agent/.claude/skills", "/opt/task-skills"]
    assert resolve_skill_paths("not-json") is None
    assert resolve_skill_paths(json.dumps({"path": "~/.claude/skills"})) is None


def test_default_skill_paths_include_pier_opencode_location() -> None:
    assert "~/.config/opencode/skills" in AntigravitySDK.DEFAULT_SKILL_PATHS


def test_thinking_level_supports_only_sdk_efforts() -> None:
    levels = SimpleNamespace(MINIMAL="minimal", LOW="low", MEDIUM="medium", HIGH="high")

    assert thinking_level("medium", levels) == "medium"
    with pytest.raises(ValueError, match="REASONING_EFFORT"):
        thinking_level("xhigh", levels)


def test_collector_groups_cumulative_text_thinking_tools_and_usage() -> None:
    collector = AtifCollector("Investigate", "gemini-3.6-flash", "model", "high")
    collector.record_step(_step(id="thinking", type="THINKING", thinking="Reasoning"))
    call = SimpleNamespace(
        id="call-1", name="run_command", server_name=None, args={"command": "pwd"}
    )
    collector.record_step(
        _step(id="tool", type="TOOL_CALL", content="Running", tool_calls=[call])
    )
    collector.record_tool_result(
        SimpleNamespace(
            id="call-1",
            name="run_command",
            server_name=None,
            result="/workspace",
            error=None,
        )
    )
    collector.record_step(
        _step(
            id="text",
            content="Done",
            is_complete_response=True,
            usage_metadata=_usage(),
        )
    )

    assert collector.complete_response_seen
    assert collector.totals() == (10, 2, 3)
    response = collector.steps[1]
    assert response["message"] == "Running\nDone"
    assert response["reasoning_content"] == "Reasoning"
    assert response["reasoning_effort"] == "high"
    assert response["llm_call_count"] == 1
    assert response["observation"]["results"][0]["content"] == "/workspace"
    Trajectory.model_validate(
        build_atif_trajectory(
            collector.steps,
            *collector.totals(),
            model_name="gemini-3.6-flash",
        )
    )


def test_collector_keeps_only_latest_cumulative_model_text() -> None:
    collector = AtifCollector("Fix it", "gemini-3.6-flash", "model")

    collector.record_step(_step(id="draft", content="Cumulative draft"))
    collector.record_step(
        _step(
            id="draft",
            content="Final answer",
            is_complete_response=True,
            usage_metadata=_usage(),
        )
    )

    assert [step["message"] for step in collector.steps] == ["Fix it", "Final answer"]
    assert collector.complete_response_seen


def test_empty_cumulative_update_does_not_erase_reasoning_chunk() -> None:
    collector = AtifCollector("Fix it", "gemini-3.6-flash", "model")
    collector.record_step(_step(id="thinking-1", thinking="First thought"))
    collector.record_step(_step(id="thinking-2", thinking="Second thought"))
    collector.record_step(_step(id="thinking-1", thinking=""))

    assert collector.steps[1]["reasoning_content"] == ("First thought\nSecond thought")


def test_system_originated_mcp_call_remains_an_agent_action() -> None:
    collector = AtifCollector("Fix it", "gemini-3.6-flash", "model")
    collector.record_step(
        _step(
            id="mcp-step",
            source="system",
            type="TOOL_CALL",
            content="User replied",
            tool_calls=[
                SimpleNamespace(
                    id="mcp-1",
                    name="mcp_user_message_user",
                    server_name=None,
                    args={"question": "What output do you expect?"},
                    output="hello, world",
                )
            ],
        )
    )

    trajectory = Trajectory.model_validate(
        build_atif_trajectory(collector.steps, 0, 0, 0)
    )
    mcp_step = trajectory.steps[1]
    assert mcp_step.source == "agent"
    assert mcp_step.tool_calls is not None
    assert mcp_step.tool_calls[0].function_name == "mcp_user_message_user"
    assert mcp_step.observation is not None
    assert mcp_step.observation.results[0].content == "hello, world"


def test_cumulative_system_note_is_promoted_before_agent_fields_are_added() -> None:
    collector = AtifCollector("Fix it", "gemini-3.6-flash", "model", "high")
    collector.record_step(
        _step(id="system-step", source="system", content="Preparing", status="DONE")
    )
    collector.record_step(
        _step(
            id="system-step",
            source="system",
            type="TOOL_CALL",
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    name="run_command",
                    server_name=None,
                    args={"command": "pytest"},
                    output="passed",
                )
            ],
            usage_metadata=_usage(),
        )
    )

    trajectory = Trajectory.model_validate(
        build_atif_trajectory(collector.steps, *collector.totals())
    )
    promoted = trajectory.steps[1]
    assert promoted.source == "agent"
    assert promoted.message == "Preparing"
    assert promoted.reasoning_effort == "high"
    assert promoted.tool_calls is not None
    assert promoted.metrics is not None


def test_collector_deduplicates_mcp_calls_and_matches_idless_results() -> None:
    collector = AtifCollector("Investigate", "gemini-3.6-flash", "model")
    call = SimpleNamespace(
        id="call-1",
        name="message_user",
        server_name="user",
        args={"question": "What happened?"},
    )
    dispatch = _step(id="tool", type="TOOL_CALL", tool_calls=[call])

    collector.record_step(dispatch)
    collector.record_step(dispatch)
    collector.record_tool_result(
        SimpleNamespace(
            id=None,
            name="message_user",
            server_name="user",
            result="It timed out.",
            error=None,
        )
    )

    response = collector.steps[1]
    assert response["tool_calls"] == [
        {
            "tool_call_id": "call-1",
            "function_name": "mcp_user_message_user",
            "arguments": {"question": "What happened?"},
        }
    ]
    assert response["observation"]["results"] == [
        {"source_call_id": "call-1", "content": "It timed out."}
    ]


def test_collector_updates_cumulative_tool_output_and_prefers_hook_result() -> None:
    collector = AtifCollector("Investigate", "gemini-3.6-flash", "model")
    call = SimpleNamespace(
        id="call-1",
        name="run_command",
        server_name=None,
        args={"command": "pytest"},
        output="",
    )
    dispatch = _step(id="tool", type="TOOL_CALL", tool_calls=[call])

    collector.record_step(dispatch)
    call.output = "partial"
    collector.record_step(dispatch)
    collector.record_tool_result(
        SimpleNamespace(
            id="call-1",
            name="run_command",
            server_name=None,
            result="complete",
            error=None,
        )
    )
    call.output = "late snapshot"
    collector.record_step(dispatch)

    assert collector.steps[1]["observation"]["results"] == [
        {"source_call_id": "call-1", "content": "complete"}
    ]


def test_idless_result_does_not_replace_another_calls_snapshot() -> None:
    collector = AtifCollector("Investigate", "gemini-3.6-flash", "model")
    collector.record_step(
        _step(
            id="first-tool",
            type="TOOL_CALL",
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    name="run_command",
                    server_name=None,
                    args={"command": "first"},
                    output="first result",
                )
            ],
        )
    )
    collector.record_step(
        _step(
            id="second-tool",
            type="TOOL_CALL",
            tool_calls=[
                SimpleNamespace(
                    id="call-2",
                    name="run_command",
                    server_name=None,
                    args={"command": "second"},
                    output=None,
                )
            ],
        )
    )

    collector.record_tool_result(
        SimpleNamespace(
            id=None,
            name="run_command",
            server_name=None,
            result="second result",
            error=None,
        )
    )

    assert collector.steps[1]["observation"]["results"] == [
        {"source_call_id": "call-1", "content": "first result"},
        {"source_call_id": "call-2", "content": "second result"},
    ]


def test_collector_normalizes_tool_arguments_to_json_safe_dict(
    tmp_path: Path,
) -> None:
    class Mode(enum.Enum):
        FAST = "fast"

    collector = AtifCollector("Investigate", "gemini-3.6-flash", "model")
    collector.record_step(
        _step(
            id="tool",
            type="TOOL_CALL",
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    name="run_command",
                    server_name=None,
                    args={"mode": Mode.FAST},
                ),
                SimpleNamespace(
                    id="call-2",
                    name="run_command",
                    server_name=None,
                    args=Mode.FAST,
                ),
            ],
        )
    )

    tool_calls = collector.steps[1]["tool_calls"]
    assert tool_calls[0]["arguments"] == {"mode": "fast"}
    assert tool_calls[1]["arguments"] == {}
    _atomic_json_write(
        tmp_path / "trajectory.json",
        build_atif_trajectory(collector.steps, 0, 0, 0),
    )


def test_live_persistence_and_incomplete_marker(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    trajectory_path = tmp_path / "trajectory.json"
    events_path.write_text("")
    _append_jsonl(events_path, "step", _step(content="Working"))
    value = build_atif_trajectory(
        [{"step_id": 1, "source": "user", "message": "Fix it"}],
        0,
        0,
        0,
        incomplete=True,
        raw_events_path=events_path.name,
    )
    _atomic_json_write(trajectory_path, value)

    assert json.loads(events_path.read_text())["kind"] == "step"
    saved = json.loads(trajectory_path.read_text())
    assert saved["extra"] == {
        "incomplete": True,
        "reason": "missing_is_complete_response",
        "raw_events_path": "events.jsonl",
    }
    assert saved["final_metrics"]["total_cost_usd"] is None
    assert not trajectory_path.with_suffix(".json.tmp").exists()


def test_network_allowlist_includes_google_and_remote_mcp(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        mcp_servers=[
            MCPServerConfig(
                name="docs",
                transport="streamable-http",
                url="https://mcp.example.com/api",
            )
        ],
    )

    assert set(agent.network_allowlist().domains) == {
        ".googleapis.com",
        "mcp.example.com",
    }

import json
from pathlib import Path

import pytest

from pier.agents.installed.grok_build import GrokBuild


@pytest.fixture(autouse=True)
def _no_ambient_grok_env(monkeypatch):
    """Auth mode and flag fallbacks read os.environ; keep the tests hermetic."""
    for key in (
        "XAI_API_KEY",
        "GROK_CLI_CHAT_PROXY_BASE_URL",
        "GROK_XAI_API_BASE_URL",
        "GROK_BUILD_MAX_TURNS",
        "GROK_BUILD_EFFORT_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_agent(tmp_path: Path, **kwargs) -> GrokBuild:
    """Construct a GrokBuild with a stub session auth file (hermetic in CI)."""
    auth_file = tmp_path / "grok-auth.json"
    auth_file.write_text("{}")
    kwargs.setdefault("auth_file", str(auth_file))
    return GrokBuild(logs_dir=tmp_path, **kwargs)


def _write_stream(logs_dir: Path, lines: list[dict]) -> None:
    path = logs_dir / "grok-build.txt"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


_INIT_EVENT = {
    "type": "system",
    "subtype": "init",
    "session_id": "0199-test-session",
    "model": "grok-4.6",
    "cwd": "/app",
}

_TOOL_CALL_EVENT = {
    "type": "assistant",
    "session_id": "0199-test-session",
    "message": {
        "id": "msg_0",
        "role": "assistant",
        "model": "grok-4.6",
        "content": [
            {"type": "thinking", "thinking": "Create the file."},
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "search_replace",
                "input": {"path": "hello.txt", "content": "hi"},
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 0,
        },
        "stop_reason": "tool_use",
    },
}

_TOOL_RESULT_EVENT = {
    "type": "user",
    "session_id": "0199-test-session",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "ok"}],
    },
}

_FINAL_EVENT = {
    "type": "assistant",
    "session_id": "0199-test-session",
    "message": {
        "id": "msg_1",
        "role": "assistant",
        "model": "grok-4.6",
        "content": [{"type": "text", "text": "Done."}],
        "usage": {"input_tokens": 120, "output_tokens": 5},
        "stop_reason": "end_turn",
    },
}

_RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "session_id": "0199-test-session",
    "total_cost_usd": 0.5,
    "num_turns": 2,
}

_FULL_STREAM = [
    _INIT_EVENT,
    _TOOL_CALL_EVENT,
    _TOOL_RESULT_EVENT,
    _FINAL_EVENT,
    _RESULT_EVENT,
]


def test_install_spec_pins_version(tmp_path: Path):
    agent = _make_agent(tmp_path, version="1.0.3")

    spec = agent.install_spec()

    assert "install.sh | bash -s 1.0.3" in spec.steps[-1].run
    assert spec.version == "1.0.3"


def test_install_spec_unpinned_omits_version_arg(tmp_path: Path):
    agent = _make_agent(tmp_path)

    assert "install.sh | bash &&" in agent.install_spec().steps[-1].run


def test_install_spec_alpine_includes_coreutils_for_stdbuf(tmp_path: Path):
    agent = _make_agent(tmp_path)

    assert "coreutils" in agent.install_spec().steps[0].run


def test_default_cli_flags(tmp_path: Path):
    agent = _make_agent(tmp_path)

    flags = agent.build_cli_flags()

    assert "--disable-web-search" in flags
    assert "--no-plan" in flags
    assert "--no-subagents" in flags
    assert "--max-turns" not in flags


def test_subagents_can_be_enabled(tmp_path: Path):
    agent = _make_agent(tmp_path, no_subagents=False)

    assert "--no-subagents" not in agent.build_cli_flags()


def test_reasoning_effort_and_max_turns_flags(tmp_path: Path):
    agent = _make_agent(tmp_path, reasoning_effort="xhigh", max_turns=50)

    flags = agent.build_cli_flags()

    assert "--reasoning-effort xhigh" in flags
    assert "--max-turns 50" in flags


def test_invalid_reasoning_effort_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="reasoning_effort"):
        _make_agent(tmp_path, reasoning_effort="ultra")


def test_claude_only_kwargs_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="max_budget_usd"):
        _make_agent(tmp_path, max_budget_usd="5")


def test_memory_dir_rejected_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="memory"):
        _make_agent(tmp_path, memory_dir="/some/memory")


def test_task_supplied_skills_warn_but_construct(tmp_path: Path):
    # skills_dir/mcp_servers can be injected from task configs; they must not
    # abort the trial at construction.
    agent = _make_agent(tmp_path, skills_dir="/task/skills")

    assert agent.skills_dir == "/task/skills"


def test_missing_auth_raises_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        GrokBuild(logs_dir=tmp_path, auth_file=str(tmp_path / "missing-auth.json"))


def test_api_key_mode_needs_no_auth_file(tmp_path: Path):
    agent = GrokBuild(
        logs_dir=tmp_path,
        auth_file=str(tmp_path / "missing-auth.json"),
        extra_env={"XAI_API_KEY": "xai-test"},
    )

    assert agent._session_auth_path() is None


def test_resolved_model_strips_provider_prefix(tmp_path: Path):
    assert (
        _make_agent(tmp_path, model_name="xai/grok-4.6")._resolved_model() == "grok-4.6"
    )
    assert _make_agent(tmp_path, model_name="grok-4.5")._resolved_model() == "grok-4.5"
    assert _make_agent(tmp_path)._resolved_model() is None


def test_network_allowlist_session_mode(tmp_path: Path):
    agent = _make_agent(tmp_path)

    domains = agent.network_allowlist().domains

    assert "cli-chat-proxy.grok.com" in domains
    assert "auth.x.ai" in domains
    assert "x.ai" in domains
    assert "storage.googleapis.com" in domains
    assert "api.x.ai" not in domains


def test_network_allowlist_api_key_mode(tmp_path: Path):
    agent = _make_agent(tmp_path, extra_env={"XAI_API_KEY": "xai-test"})

    assert "api.x.ai" in agent.network_allowlist().domains


def test_network_allowlist_skips_placeholder_urls(tmp_path: Path):
    agent = _make_agent(
        tmp_path, extra_env={"GROK_CLI_CHAT_PROXY_BASE_URL": "${PROXY_URL}"}
    )

    domains = agent.network_allowlist().domains

    assert "${PROXY_URL}" not in domains
    assert "cli-chat-proxy.grok.com" in domains


def test_stream_converts_to_atif_trajectory(tmp_path: Path):
    agent = _make_agent(tmp_path, model_name="grok-4.6")
    _write_stream(tmp_path, _FULL_STREAM)

    events, session_id = agent._load_stream_events()
    assert session_id == "0199-test-session"
    # init and result events are not message events
    assert len(events) == 3
    # the inherited parser reads the result event from the stream file
    assert agent._parse_total_cost_from_stream_json() == 0.5

    events.insert(
        0, {"type": "user", "message": {"role": "user", "content": "Create hello.txt"}}
    )
    trajectory = agent._convert_raw_events_to_trajectory(events, session_id)

    assert trajectory is not None
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[0].message == "Create hello.txt"

    tool_steps = [s for s in trajectory.steps if s.tool_calls]
    assert tool_steps and tool_steps[0].tool_calls[0].function_name == "search_replace"
    assert tool_steps[0].observation.results[0].content == "ok"

    agent_steps = [s for s in trajectory.steps if s.source == "agent"]
    assert all(s.model_name == "grok-4.6" for s in agent_steps)

    final = trajectory.final_metrics
    assert final is not None
    assert final.total_completion_tokens == 25
    assert final.total_cost_usd == 0.5


def test_cost_uses_first_result_event(tmp_path: Path):
    agent = _make_agent(tmp_path, model_name="grok-4.6")
    trailing_error_result = {
        "type": "result",
        "subtype": "error_during_execution",
        "total_cost_usd": 0.0,
    }
    _write_stream(tmp_path, [*_FULL_STREAM, trailing_error_result])

    assert agent._parse_total_cost_from_stream_json() == 0.5


def test_non_string_timestamps_are_dropped(tmp_path: Path):
    agent = _make_agent(tmp_path, model_name="grok-4.6")
    numeric_ts_event = dict(_FINAL_EVENT, timestamp=1755100000000)
    _write_stream(tmp_path, [_INIT_EVENT, numeric_ts_event, _RESULT_EVENT])

    events, session_id = agent._load_stream_events()

    assert "timestamp" not in events[0]
    # the converter's string sort must not raise on the sanitized events
    assert agent._convert_raw_events_to_trajectory(events, session_id) is not None


def test_subagent_events_keep_stream_order(tmp_path: Path):
    # Subagent events must not be tagged isSidechain: the inherited converter
    # hoists sidechains ahead of the mainline, which would reorder the
    # timestamp-less grok stream.
    agent = _make_agent(tmp_path, model_name="grok-4.6")
    subagent_event = dict(_FINAL_EVENT, parent_tool_use_id="call-parent-1")
    _write_stream(tmp_path, [_INIT_EVENT, _TOOL_CALL_EVENT, subagent_event])

    events, session_id = agent._load_stream_events()

    assert all(event.get("isSidechain") is None for event in events)
    trajectory = agent._convert_raw_events_to_trajectory(list(events), session_id)
    assert trajectory.steps[0].tool_calls, "mainline tool call must stay first"


def test_prompt_echo_is_not_duplicated(tmp_path: Path):
    from pier.models.agent.context import AgentContext

    agent = _make_agent(tmp_path, model_name="grok-4.6")
    echo_event = {
        "type": "user",
        "session_id": "0199-test-session",
        "message": {"role": "user", "content": "Create hello.txt"},
    }
    _write_stream(tmp_path, [_INIT_EVENT, echo_event, _FINAL_EVENT, _RESULT_EVENT])
    agent._instruction_used = "Create hello.txt"

    agent.populate_context_post_run(AgentContext())

    written = json.loads((tmp_path / "trajectory.json").read_text())
    user_steps = [
        s
        for s in written["steps"]
        if s["source"] == "user" and s.get("message") == "Create hello.txt"
    ]
    assert len(user_steps) == 1


def test_malformed_stream_lines_are_skipped(tmp_path: Path):
    agent = _make_agent(tmp_path, model_name="grok-4.6")
    path = tmp_path / "grok-build.txt"
    path.write_text(
        json.dumps(_INIT_EVENT)
        + "\nnot json\n{truncated\n"
        + json.dumps(_FINAL_EVENT)
        + "\n"
        + json.dumps(_RESULT_EVENT)
        + "\n"
    )

    events, session_id = agent._load_stream_events()

    assert session_id == "0199-test-session"
    assert len(events) == 1


def test_populate_context_writes_trajectory(tmp_path: Path):
    from pier.models.agent.context import AgentContext

    agent = _make_agent(tmp_path, model_name="grok-4.6")
    _write_stream(tmp_path, _FULL_STREAM)
    agent._instruction_used = "Create hello.txt"

    context = AgentContext()
    agent.populate_context_post_run(context)

    written = json.loads((tmp_path / "trajectory.json").read_text())
    assert written["agent"]["name"] == "grok-build"
    assert written["session_id"] == "0199-test-session"
    assert context.n_output_tokens == 25

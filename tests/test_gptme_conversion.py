import json
from pathlib import Path

import pytest

from pier.agents.installed.gptme import Gptme, convert_messages, iter_tool_calls
from pier.models.trajectories import Trajectory

SESSION = "11111111-1111-1111-1111-111111111111"


def _row(role: str, content: str, **extra) -> dict:
    return {
        "role": role,
        "content": content,
        "timestamp": "2026-10-17T00:00:00+00:00",
        **extra,
    }


def test_iter_tool_calls_parses_native_format():
    content = (
        "Running it.\n"
        '@shell(call-1): {"command": "printf hi", "timeout": 10}\n'
        '@save(call-2): {"path": "a.txt", "content": "x"}\n'
    )

    calls = iter_tool_calls(content)

    assert [name for name, _, _ in calls] == ["shell", "save"]
    assert calls[0][1] == "call-1"
    assert calls[0][2]["command"] == "printf hi"


def test_iter_tool_calls_ignores_fenced_examples():
    content = '```\n@shell(call-1): {"command": "rm -rf /"}\n```\n'

    assert iter_tool_calls(content) == []


def test_convert_messages_links_results_by_call_id():
    rows = [
        _row("user", "do the thing"),
        _row(
            "assistant",
            '@shell(call-1): {"command": "printf hi"}',
            metadata={"model": "openrouter/deepseek/deepseek-v4.1-flash-1017"},
        ),
        _row("system", "hi", call_id="call-1"),
        _row("assistant", "done"),
    ]

    trajectory = convert_messages(rows, session_id=SESSION, version="0.35.1")

    assert [step.source for step in trajectory.steps] == [
        "user",
        "agent",
        "agent",
    ]
    action = trajectory.steps[1]
    assert action.tool_calls is not None
    assert action.tool_calls[0].tool_call_id == "call-1"
    assert action.observation is not None
    assert action.observation.results[0].source_call_id == "call-1"
    assert action.observation.results[0].content == "hi"


def test_convert_messages_keeps_usage_and_cost():
    rows = [
        _row("user", "hi"),
        _row(
            "assistant",
            "hello",
            metadata={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_tokens": 3,
                },
                "cost": 0.001,
            },
        ),
    ]

    trajectory = convert_messages(rows, session_id=SESSION, version="0.35.1")
    step = trajectory.steps[1]

    assert step.metrics is not None
    assert (step.metrics.prompt_tokens, step.metrics.completion_tokens) == (10, 4)
    assert step.metrics.cached_tokens == 3
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 10
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.001)


def test_convert_messages_omits_missing_usage():
    trajectory = convert_messages(
        [_row("user", "hi"), _row("assistant", "hello")],
        session_id=SESSION,
        version="0.35.1",
    )

    assert trajectory.steps[1].metrics is None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens is None


def test_convert_messages_rejects_empty_conversation():
    with pytest.raises(ValueError):
        convert_messages([], session_id=SESSION, version="0.35.1")


def test_trajectory_json_roundtrips_through_schema(tmp_path: Path):
    rows = [
        _row("user", "hi"),
        _row("assistant", '@shell(call-1): {"command": "printf hi"}'),
        _row("system", "hi", call_id="call-1"),
    ]
    trajectory = convert_messages(rows, session_id=SESSION, version="0.35.1")

    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory.to_json_dict()), encoding="utf-8")

    assert Trajectory.model_validate(json.loads(path.read_text())).session_id == SESSION


def test_gptme_agent_registered_and_installs(tmp_path: Path):
    agent = Gptme(
        logs_dir=tmp_path, model_name="openrouter/deepseek/deepseek-v4.1-flash"
    )

    assert Gptme.name() == "gptme"
    assert agent.SUPPORTS_ATIF is True
    spec = agent.install_spec()
    assert spec.agent_name == "gptme"
    assert "uv tool install gptme" in spec.steps[-1].run
    assert "openrouter.ai" in set(agent.network_allowlist().domains)


def test_populate_context_reads_conversation_log(tmp_path: Path):
    rows = [
        _row("user", "hi"),
        _row(
            "assistant",
            "hello",
            metadata={"usage": {"input_tokens": 7, "output_tokens": 2}, "cost": 0.002},
        ),
    ]
    (tmp_path / "gptme.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    agent = Gptme(
        logs_dir=tmp_path, model_name="openrouter/deepseek/deepseek-v4.1-flash"
    )

    from pier.models.agent.context import AgentContext

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert (tmp_path / "trajectory.json").exists()
    assert context.n_input_tokens == 7
    assert context.n_output_tokens == 2
    assert context.cost_usd == pytest.approx(0.002)


def test_convert_messages_ignores_non_message_events():
    rows = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "tool_start", "role": "assistant", "content": "noise"},
        {"type": "message", "role": "assistant", "content": "hello"},
    ]

    trajectory = convert_messages(rows, session_id=SESSION, version="0.35.1")

    assert [step.message for step in trajectory.steps] == ["hi", "hello"]


def test_convert_messages_splits_inline_reasoning():
    rows = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": " thinking\nlet me think\n</think>\nOK"},
    ]

    step = convert_messages(rows, session_id=SESSION, version="0.35.1").steps[1]

    assert step.reasoning_content == "let me think"
    assert step.message == "OK"


def test_unmatched_call_id_kept_as_system_step():
    rows = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "done"},
        {"role": "system", "content": "Task complete.", "call_id": "chatcmpl-tool-x"},
    ]

    trajectory = convert_messages(rows, session_id=SESSION, version="0.35.1")

    assert trajectory.steps[-1].source == "system"
    assert trajectory.steps[-1].message == "Task complete."


def test_convert_messages_splits_bare_thinking_header():
    # Some providers emit a bare "Thinking" header instead of a think tag.
    rows = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Thinking\npondering\n</think" + "ing>\nOK"},
    ]

    step = convert_messages(rows, session_id=SESSION, version="0.35.1").steps[1]

    assert step.reasoning_content == "pondering"
    assert step.message == "OK"


def test_gptme_resolves_through_the_agent_factory(tmp_path: Path):
    from pier.agents.factory import AgentFactory
    from pier.models.agent.name import AgentName

    agent = AgentFactory.create_agent_from_name(
        AgentName.GPTME,
        logs_dir=tmp_path,
        model_name="openrouter/deepseek/deepseek-v4.1-flash",
    )

    assert isinstance(agent, Gptme)
    assert agent.import_path() == "pier.agents.installed.gptme:Gptme"

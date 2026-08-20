from pathlib import Path

import pytest

from pier.agents.factory import AgentFactory
from pier.agents.installed.cline import Cline
from pier.models.agent.name import AgentName


def test_cline_is_registered(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.CLINE, logs_dir=tmp_path, model_name="openrouter/qwen/qwen3-coder"
    )
    assert isinstance(agent, Cline)
    assert agent.name() == "cline"


def test_cline_install_and_network_defaults(tmp_path: Path):
    agent = Cline(logs_dir=tmp_path, model_name="anthropic/claude-sonnet-4")
    assert "npm install --global cline" in agent.install_spec().steps[0].run
    assert "api.anthropic.com" in agent.network_allowlist().domains


def test_cline_converts_json_stream_to_atif(tmp_path: Path):
    agent = Cline(logs_dir=tmp_path, model_name="cline/claude-sonnet-4")
    events = [
        {"type": "say", "say": "text", "text": "Inspecting the repository", "ts": 1000},
        {
            "type": "say",
            "say": "tool",
            "text": "Reading README",
            "reasoning": "Need project context",
            "tool": {
                "id": "tool-1",
                "name": "read_file",
                "input": {"path": "README.md"},
                "result": "# Pier",
            },
            "usage": {
                "inputTokens": 100,
                "outputTokens": 20,
                "cacheReadTokens": 5,
                "costUsd": 0.01,
            },
            "taskId": "task-1",
        },
    ]
    trajectory = agent._convert_events_to_trajectory(events)
    assert trajectory is not None
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.session_id == "task-1"
    assert trajectory.steps[1].tool_calls[0].function_name == "read_file"
    assert trajectory.steps[1].observation.results[0].content == "# Pier"
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 20
    assert trajectory.final_metrics.total_cached_tokens == 5
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.01)
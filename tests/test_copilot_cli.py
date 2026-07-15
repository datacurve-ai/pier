import json
from pathlib import Path

from pier.agents.factory import AgentFactory
from pier.agents.installed.copilot_cli import CopilotCli
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName


def test_copilot_cli_is_registered_with_current_flags(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.COPILOT_CLI,
        logs_dir=tmp_path,
        model_name="gpt-5.4",
        reasoning_effort="minimal",
        context_tier="long_context",
    )

    assert isinstance(agent, CopilotCli)
    assert agent.name() == "copilot-cli"
    assert "--effort minimal" in agent.build_cli_flags()
    assert "--context long_context" in agent.build_cli_flags()
    assert "--allow-all-tools" in agent.build_cli_flags()
    assert "--no-ask-user" in agent.build_cli_flags()
    assert "--context-tier" not in agent.build_cli_flags()


def test_copilot_cli_install_spec_and_allowlist(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, version="1.0.71-2")

    spec = agent.install_spec()
    domains = set(agent.network_allowlist().domains)

    assert spec.agent_name == "copilot-cli"
    assert spec.version == "1.0.71-2"
    assert any("https://gh.io/copilot-install" in step.run for step in spec.steps)
    assert any("musl-based images" in step.run for step in spec.steps)
    assert spec.verification_command is not None
    assert "copilot --version" in spec.verification_command
    assert {".github.com", ".githubcopilot.com", ".githubusercontent.com", "gh.io"} <= domains
    bare = {domain for domain in domains if not domain.startswith(".")}
    assert not any(f".{domain}" in domains for domain in bare)


def test_copilot_cli_version_parser(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path)

    assert agent.parse_version("GitHub Copilot CLI 1.0.71-2.\n") == "1.0.71-2"
    assert agent.parse_version("dev-build") == "dev-build"


def test_copilot_cli_auth_uses_official_precedence(tmp_path: Path):
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "GITHUB_TOKEN": "github",
            "GH_TOKEN": "gh",
            "COPILOT_GITHUB_TOKEN": "copilot",
        },
    )

    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "copilot"}


def test_copilot_cli_run_command_preserves_pipeline_status(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path)

    command = agent._build_run_command(
        setup="mkdir -p /logs/agent",
        instruction="fix it",
        flag_text="--session-id 1234 --allow-all-tools",
        jsonl_path="/logs/agent/copilot-cli.jsonl",
        output_path="/logs/agent/copilot-cli.txt",
    )

    assert "bash -lc" in command
    assert "--output-format json" in command
    assert "tee /logs/agent/command-0/stdout.txt" in command
    assert "exit ${PIPESTATUS[0]}" in command
    assert "cp -a" not in command


def test_copilot_cli_native_session_root_is_mounted():
    assert CopilotCli._COPILOT_HOME.as_posix() == "/logs/agent/copilot-home"


def test_copilot_cli_registers_skills_under_copilot_home(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, skills_dir="/skills")

    command = agent._build_register_skills_command()

    assert '"$COPILOT_HOME/skills"' in command
    assert "~/.copilot/skills" not in command


def test_copilot_cli_converts_native_events_to_atif(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    trajectory = agent._convert_events_to_trajectory(_session_events())

    assert trajectory is not None
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.session_id == "session-1"
    assert trajectory.agent.name == "copilot-cli"
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[1].source == "agent"
    assert sum(step.source == "agent" for step in trajectory.steps) == 1
    assert trajectory.steps[1].tool_calls is not None
    assert trajectory.steps[1].tool_calls[0].function_name == "view"
    assert trajectory.steps[1].observation is not None
    assert trajectory.steps[1].observation.results[0].content == "# Pier"
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 130
    assert trajectory.final_metrics.total_cached_tokens == 20
    assert trajectory.final_metrics.total_completion_tokens == 5
    assert trajectory.final_metrics.extra == {
        "copilot_aiu": 0.25,
        "peak_context_tokens": 2000,
        "summarization_count": 1,
    }


def test_copilot_cli_timeout_metrics_deduplicate_api_calls(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "data": {
                "apiCallId": "api-1",
                "content": "Working",
                "inputTokens": 100,
                "outputTokens": 5,
            },
        },
        {
            "type": "assistant.message",
            "data": {
                "apiCallId": "api-1",
                "content": "",
                "inputTokens": 100,
                "outputTokens": 5,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert len(trajectory.steps) == 1
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 5


def test_copilot_cli_populates_context_from_native_session(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events_path = (
        tmp_path
        / "copilot-home"
        / "session-state"
        / "session-1"
        / "events.jsonl"
    )
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(json.dumps(event) for event in _session_events()) + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert (tmp_path / "trajectory.json").exists()
    assert context.n_input_tokens == 130
    assert context.n_cache_tokens == 20
    assert context.n_output_tokens == 5
    assert context.peak_context_tokens == 2000
    assert context.summarization_count == 1
    assert context.n_agent_steps == 1
    assert context.metadata == {
        "copilot_session_events": str(
            Path("copilot-home")
            / "session-state"
            / "session-1"
            / "events.jsonl"
        ),
        "copilot_aiu": 0.25,
    }


def _session_events() -> list[dict]:
    return [
        {
            "type": "session.start",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"sessionId": "session-1", "selectedModel": "gpt-5.4"},
        },
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"content": "Fix the bug"},
        },
        {
            "type": "assistant.turn_start",
            "timestamp": "2026-01-01T00:00:01.100Z",
            "data": {"turnId": "0"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {
                "apiCallId": "api-1",
                "model": "gpt-5.4",
                "content": "I'll inspect the repository.",
                "outputTokens": 5,
                "toolRequests": [
                    {
                        "toolCallId": "tool-1",
                        "name": "view",
                        "arguments": {"path": "README.md"},
                    }
                ],
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.600Z",
            "data": {
                "apiCallId": "api-1",
                "model": "gpt-5.4",
                "content": "",
                "outputTokens": 5,
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:01.700Z",
            "data": {
                "toolCallId": "tool-1",
                "success": True,
                "result": {"content": "# Pier"},
            },
        },
        {
            "type": "session.compaction_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"preCompactionTokens": 2000},
        },
        {
            "type": "session.shutdown",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "totalNanoAiu": 250_000_000,
                "currentTokens": 1500,
                "tokenDetails": {
                    "input": {"tokenCount": 100},
                    "cache_read": {"tokenCount": 20},
                    "cache_write": {"tokenCount": 10},
                    "output": {"tokenCount": 5},
                },
            },
        },
    ]

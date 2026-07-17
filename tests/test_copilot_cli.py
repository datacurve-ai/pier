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


def test_copilot_cli_shutdown_metrics_aggregate_models_without_token_details(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "id": "00000000-0000-4000-8000-000000000001",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": None,
            "data": {
                "messageId": "message-1",
                "content": "Done",
                "outputTokens": 9,
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000002",
            "type": "session.shutdown",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "parentId": "00000000-0000-4000-8000-000000000001",
            "data": {
                "modelMetrics": {
                    "gpt-5.4": {
                        "usage": {
                            "inputTokens": 100,
                            "outputTokens": 5,
                            "cacheReadTokens": 20,
                            "cacheWriteTokens": 10,
                        }
                    },
                    "gpt-5-mini": {
                        "usage": {
                            "inputTokens": 7,
                            "outputTokens": 4,
                            "cacheReadTokens": 3,
                            "cacheWriteTokens": 2,
                        }
                    },
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert len(trajectory.steps) == 1
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 142
    assert trajectory.final_metrics.total_cached_tokens == 23
    assert trajectory.final_metrics.total_completion_tokens == 9


def test_copilot_cli_shutdown_metrics_keep_token_details_fallback(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "data": {"messageId": "message-1", "content": "Done"},
        },
        {
            "type": "session.shutdown",
            "data": {
                "tokenDetails": {
                    "input": {"tokenCount": 100},
                    "cache_read": {"tokenCount": 20},
                    "cache_write": {"tokenCount": 10},
                    "output": {"tokenCount": 5},
                }
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 130
    assert trajectory.final_metrics.total_cached_tokens == 20
    assert trajectory.final_metrics.total_completion_tokens == 5


def test_copilot_cli_shutdown_metrics_do_not_double_count_compatibility_data(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "data": {"messageId": "message-1", "content": "Done"},
        },
        {
            "type": "session.shutdown",
            "data": {
                "modelMetrics": {
                    "gpt-5.4": {
                        "usage": {
                            "inputTokens": 100,
                            "cacheReadTokens": 20,
                            "cacheWriteTokens": 10,
                            "outputTokens": 5,
                        }
                    }
                },
                "tokenDetails": {
                    "input": {"tokenCount": 1000},
                    "cache_read": {"tokenCount": 200},
                    "cache_write": {"tokenCount": 100},
                    "output": {"tokenCount": 50},
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 130
    assert trajectory.final_metrics.total_cached_tokens == 20
    assert trajectory.final_metrics.total_completion_tokens == 5


def test_copilot_cli_timeout_metrics_combine_and_deduplicate_usage(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    persisted_events = [
        {
            "id": "00000000-0000-4000-8000-000000000011",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": None,
            "data": {
                "messageId": "message-1",
                "apiCallId": "api-1",
                "content": "Working",
                "outputTokens": 5,
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000012",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "parentId": "00000000-0000-4000-8000-000000000011",
            "data": {
                "messageId": "message-2",
                "apiCallId": "api-2",
                "content": "Still working",
                "outputTokens": 7,
            },
        },
    ]
    captured_events = [
        {
            "id": "00000000-0000-4000-8000-000000000010",
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:00.900Z",
            "parentId": None,
            "ephemeral": True,
            "data": {
                "model": "gpt-5.4",
                "inputTokens": 100,
                "outputTokens": 5,
                "cacheReadTokens": 20,
                "cacheWriteTokens": 10,
                "apiCallId": "api-1",
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000013",
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.900Z",
            "parentId": "00000000-0000-4000-8000-000000000011",
            "ephemeral": True,
            "data": {
                "model": "gpt-5.4",
                "inputTokens": 50,
                "cacheReadTokens": 5,
                "cacheWriteTokens": 2,
                "apiCallId": "api-2",
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000014",
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:02.900Z",
            "parentId": "00000000-0000-4000-8000-000000000012",
            "ephemeral": True,
            "data": {
                "model": "gpt-5.4",
                "inputTokens": 25,
                "outputTokens": 11,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "apiCallId": "api-3",
            },
        },
    ]
    events_path = (
        tmp_path / "copilot-home" / "session-state" / "session-timeout" / "events.jsonl"
    )
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(json.dumps(event) for event in persisted_events) + "\n",
        encoding="utf-8",
    )
    (tmp_path / CopilotCli._JSONL_FILENAME).write_text(
        "\n".join(
            json.dumps(event)
            for event in [captured_events[0], *captured_events, persisted_events[0]]
        )
        + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 212
    assert context.n_cache_tokens == 25
    assert context.n_output_tokens == 23
    assert context.n_agent_steps == 2


def test_copilot_cli_excludes_tagged_subagent_events_from_root_trajectory(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "id": "00000000-0000-4000-8000-000000000020",
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "parentId": None,
            "data": {"content": "Fix the bug"},
        },
        {
            "id": "00000000-0000-4000-8000-000000000021",
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "parentId": "00000000-0000-4000-8000-000000000020",
            "agentId": "subagent-1",
            "data": {"content": "Inspect the parser"},
        },
        {
            "id": "00000000-0000-4000-8000-000000000022",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": "00000000-0000-4000-8000-000000000021",
            "agentId": "subagent-1",
            "data": {
                "messageId": "subagent-message-1",
                "content": "I found the issue",
                "toolRequests": [
                    {
                        "toolCallId": "subagent-tool-1",
                        "name": "view",
                        "arguments": {"path": "parser.py"},
                    }
                ],
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000023",
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "parentId": "00000000-0000-4000-8000-000000000022",
            "agentId": "subagent-1",
            "data": {
                "toolCallId": "subagent-tool-1",
                "success": True,
                "result": {"content": "subagent output"},
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000024",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "parentId": "00000000-0000-4000-8000-000000000023",
            "data": {
                "messageId": "root-message-1",
                "content": "Fixed the bug",
                "outputTokens": 3,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert [step.message for step in trajectory.steps] == [
        "Fix the bug",
        "Fixed the bug",
    ]
    assert sum(step.source == "agent" for step in trajectory.steps) == 1
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_steps == 2

    events_path = (
        tmp_path
        / "copilot-home"
        / "session-state"
        / "session-subagents"
        / "events.jsonl"
    )
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_agent_steps == 1


def test_copilot_cli_preserves_ordered_native_reasoning_without_duplicates(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "id": "00000000-0000-4000-8000-000000000030",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "parentId": None,
            "ephemeral": True,
            "data": {"reasoningId": "reasoning-1", "content": "Inspect first."},
        },
        {
            "id": "00000000-0000-4000-8000-000000000031",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:00.600Z",
            "parentId": "00000000-0000-4000-8000-000000000030",
            "ephemeral": True,
            "data": {"reasoningId": "reasoning-1", "content": "Inspect first."},
        },
        {
            "id": "00000000-0000-4000-8000-000000000032",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:00.700Z",
            "parentId": "00000000-0000-4000-8000-000000000031",
            "ephemeral": True,
            "data": {
                "reasoningId": "reasoning-2",
                "content": "Then patch.",
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000033",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": "00000000-0000-4000-8000-000000000032",
            "data": {
                "messageId": "message-1",
                "content": "Done",
                "reasoningText": "Inspect first.\n\nThen patch.",
                "outputTokens": 3,
            },
        },
        {
            "id": "00000000-0000-4000-8000-000000000034",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:01.100Z",
            "parentId": "00000000-0000-4000-8000-000000000033",
            "ephemeral": True,
            "data": {
                "reasoningId": "reasoning-3",
                "content": "Then patch.\n\nFinally verify.",
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert len(trajectory.steps) == 1
    assert (
        trajectory.steps[0].reasoning_content
        == "Inspect first.\n\nThen patch.\n\nFinally verify."
    )


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


def test_copilot_cli_timeout_after_compaction_start_reports_peak_context_tokens(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "data": {"content": "Fix the bug"},
        },
        {
            "type": "assistant.message",
            "data": {"messageId": "message-1", "content": "Working on it", "outputTokens": 5},
        },
        {
            "type": "session.compaction_start",
            "data": {
                "systemTokens": 500,
                "conversationTokens": 1200,
                "toolDefinitionsTokens": 300,
            },
        },
        # No session.compaction_complete — timed out before compaction finished
        {
            "type": "assistant.usage",
            "data": {"inputTokens": 100, "outputTokens": 5, "apiCallId": "api-1"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    # peak_context_tokens = systemTokens(500) + conversationTokens(1200) + toolDefinitionsTokens(300)
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 2000


def test_copilot_cli_extra_env_reserved_keys_are_normalized(tmp_path: Path):
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "COPILOT_HOME": "/custom/home",
            "COPILOT_GITHUB_TOKEN": "",
            "SOME_OTHER_VAR": "value",
        },
    )

    assert "COPILOT_HOME" not in agent._extra_env
    assert "COPILOT_GITHUB_TOKEN" not in agent._extra_env
    assert agent._extra_env.get("SOME_OTHER_VAR") == "value"


def test_copilot_cli_timeout_uses_message_input_for_calls_missing_from_usage_stream(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "data": {"content": "Fix the bug"},
        },
        # Call api-1: usage event has input tokens
        {
            "type": "assistant.usage",
            "data": {
                "inputTokens": 100,
                "cacheReadTokens": 10,
                "outputTokens": 5,
                "apiCallId": "api-1",
            },
        },
        {
            "type": "assistant.message",
            "data": {
                "apiCallId": "api-1",
                "content": "Working",
                "inputTokens": 95,
                "outputTokens": 5,
            },
        },
        # Call api-2: usage event has no inputTokens (timeout cut it short),
        # but assistant.message carries inputTokens
        {
            "type": "assistant.usage",
            "data": {
                "cacheReadTokens": 5,
                "outputTokens": 3,
                "apiCallId": "api-2",
            },
        },
        {
            "type": "assistant.message",
            "data": {
                "apiCallId": "api-2",
                "content": "Still working",
                "inputTokens": 80,
                "outputTokens": 3,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    # total_prompt_tokens includes all input + cache tokens (cache is part of the prompt context).
    # input: 100 (api-1 usage) + 80 (api-2 message fallback) = 180
    # cache: 10 (api-1 cache_read) + 5 (api-2 cache_read) = 15
    # total: 180 + 15 = 195
    assert trajectory.final_metrics.total_prompt_tokens == 195
    assert trajectory.final_metrics.total_cached_tokens == 15


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
                "modelMetrics": {
                    "gpt-5.4": {
                        "usage": {
                            "inputTokens": 100,
                            "cacheReadTokens": 20,
                            "cacheWriteTokens": 10,
                            "outputTokens": 5,
                        }
                    }
                },
            },
        },
    ]

import json
import os
from pathlib import Path

import pytest

from atif_assertions import assert_one_step_per_api_call, assert_valid_atif
from pier.agents.installed.copilot_cli import (
    CopilotCli,
    _combine_event_streams,
    _read_jsonl,
    _stringify_tool_result,
    find_copilot_session_events,
)
from pier.models.agent.context import AgentContext

_FIXTURES = Path(__file__).parent / "fixtures" / "copilot_cli"


def test_copilot_cli_converts_native_events_to_atif(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    trajectory = agent._convert_events_to_trajectory(_session_events())

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
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
    assert trajectory.final_metrics.total_prompt_tokens == 100
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert len(trajectory.steps) == 1
    assert trajectory.final_metrics is not None
    # inputTokens per model is cache-inclusive: gpt-5.4 100 + gpt-5-mini 7 = 107.
    assert trajectory.final_metrics.total_prompt_tokens == 107
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    # modelMetrics usage wins over the legacy tokenDetails compatibility shape,
    # and its cache-inclusive inputTokens (100) is not double-counted.
    assert trajectory.final_metrics.total_prompt_tokens == 100
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

    assert context.n_input_tokens == 175
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
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
    assert context.n_input_tokens == 100
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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    # peak_context_tokens = systemTokens(500) + conversationTokens(1200) + toolDefinitionsTokens(300)
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 2000


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
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    # inputTokens is the cache-inclusive prompt total, so cached reads are not
    # added again. Missing api-2 usage input falls back to the message input.
    # input: 100 (api-1 usage) + 80 (api-2 message fallback) = 180
    # cache: 10 (api-1 cache_read) + 5 (api-2 cache_read) = 15 (subset of input)
    assert trajectory.final_metrics.total_prompt_tokens == 180
    assert trajectory.final_metrics.total_cached_tokens == 15


def test_copilot_cli_shutdown_input_tokens_are_cache_inclusive(tmp_path: Path):
    # In Copilot CLI 1.0.71, modelMetrics[*].usage.inputTokens is the
    # OpenAI-style prompt_tokens value, which already includes cacheReadTokens
    # and cacheWriteTokens. Adding the cache breakdowns on top double-counts.
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "id": "00000000-0000-4000-8000-000000000041",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": None,
            "data": {"messageId": "message-1", "content": "Done", "outputTokens": 5},
        },
        {
            "id": "00000000-0000-4000-8000-000000000042",
            "type": "session.shutdown",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "parentId": "00000000-0000-4000-8000-000000000041",
            "data": {
                "modelMetrics": {
                    "gpt-5.4": {
                        "usage": {
                            "inputTokens": 100,
                            "outputTokens": 5,
                            "cacheReadTokens": 20,
                            "cacheWriteTokens": 10,
                        }
                    }
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    # inputTokens (100) already accounts for the 20 cached + 10 cache-write
    # tokens, so the prompt total is 100 and the cached subset is 20.
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_cached_tokens == 20
    assert trajectory.final_metrics.total_completion_tokens == 5


def test_copilot_cli_failed_compaction_is_not_counted_as_summarization(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    def _events(success: bool) -> list[dict]:
        return [
            {
                "type": "assistant.message",
                "data": {"messageId": "m", "content": "Working", "outputTokens": 1},
            },
            {
                "type": "session.compaction_complete",
                "data": {"success": success, "preCompactionTokens": 2000},
            },
        ]

    failed = agent._convert_events_to_trajectory(_events(success=False))
    assert failed is not None
    assert failed.final_metrics is not None
    assert failed.final_metrics.extra["summarization_count"] == 0

    succeeded = agent._convert_events_to_trajectory(_events(success=True))
    assert succeeded is not None
    assert succeeded.final_metrics is not None
    assert succeeded.final_metrics.extra["summarization_count"] == 1


def test_copilot_cli_distinct_reasoning_blocks_are_not_merged_on_partial_overlap(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "id": "00000000-0000-4000-8000-000000000051",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "parentId": None,
            "ephemeral": True,
            "data": {"reasoningId": "reasoning-1", "content": "Use cat"},
        },
        {
            "id": "00000000-0000-4000-8000-000000000052",
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:00.600Z",
            "parentId": "00000000-0000-4000-8000-000000000051",
            "ephemeral": True,
            "data": {"reasoningId": "reasoning-2", "content": "category names matter"},
        },
        {
            "id": "00000000-0000-4000-8000-000000000053",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "parentId": "00000000-0000-4000-8000-000000000052",
            "data": {"messageId": "message-1", "content": "Done", "outputTokens": 1},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert len(trajectory.steps) == 1
    # "Use cat" ends with "cat", which is a prefix of "category"; these are two
    # distinct reasoning blocks and must not be stitched into "Use category".
    assert (
        trajectory.steps[0].reasoning_content == "Use cat\n\ncategory names matter"
    )


def test_copilot_cli_usage_checkpoint_supplies_aiu_without_shutdown(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working", "outputTokens": 2},
        },
        {
            "type": "session.usage_checkpoint",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"totalNanoAiu": 120_000_000},
        },
        {
            "type": "session.usage_checkpoint",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"totalNanoAiu": 480_000_000},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    # Checkpoints report a running cumulative total, so the latest wins.
    assert trajectory.final_metrics.extra["copilot_aiu"] == 0.48


def test_copilot_cli_shutdown_aiu_wins_over_usage_checkpoints(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.usage_checkpoint",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"totalNanoAiu": 120_000_000},
        },
        {
            "type": "session.shutdown",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"totalNanoAiu": 500_000_000, "currentTokens": 10},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["copilot_aiu"] == 0.5


def test_copilot_cli_peak_context_uses_shutdown_component_sum(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.shutdown",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "currentTokens": 900,
                "systemTokens": 400,
                "conversationTokens": 500,
                "toolDefinitionsTokens": 300,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 1200


def test_copilot_cli_returns_no_trajectory_without_convertible_events(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    assert agent._convert_events_to_trajectory([]) is None
    assert (
        agent._convert_events_to_trajectory(
            [
                {
                    "type": "session.start",
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "data": {"sessionId": "s-1"},
                }
            ]
        )
        is None
    )


def test_copilot_cli_read_jsonl_survives_hostile_streams(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b'{"type": "user.message", "data": {"content": "first"}}\r\n'
        + b"\n"
        + b'{"type": "assistant.message", "data": {"content": "caf\xe9"}}\n'
        + b"not json at all\n"
        + b'["not", "a", "dict"]\n'
        + b'{"type": "assistant.message", "data": {"conten'
    )

    events = _read_jsonl(path)

    assert [event["type"] for event in events] == [
        "user.message",
        "assistant.message",
    ]
    assert events[0]["data"]["content"] == "first"
    assert events[1]["data"]["content"].startswith("caf")


def test_copilot_cli_read_jsonl_tolerates_missing_paths(tmp_path: Path):
    assert _read_jsonl(tmp_path / "absent.jsonl") == []
    assert _read_jsonl(tmp_path) == []


def test_copilot_cli_tool_result_display_copies_are_not_duplicated(tmp_path: Path):
    assert (
        _stringify_tool_result(
            {
                "content": "ok",
                "detailedContent": "ok",
                "displayContent": "ok",
            }
        )
        == "ok"
    )


def test_copilot_cli_tool_result_keeps_detailed_content_extension(tmp_path: Path):
    assert (
        _stringify_tool_result({"content": "head", "detailedContent": "head\ntail"})
        == "head\ntail"
    )


def test_copilot_cli_tool_result_keeps_unknown_keys(tmp_path: Path):
    result = _stringify_tool_result({"content": "ok", "exitCode": 1})

    assert result.startswith("ok\n")
    assert json.loads(result.split("\n", 1)[1]) == {"exitCode": 1}


def test_copilot_cli_tool_result_falls_back_to_json(tmp_path: Path):
    assert json.loads(_stringify_tool_result({"files": ["a", "b"]})) == {
        "files": ["a", "b"]
    }
    assert _stringify_tool_result(None) == ""
    assert _stringify_tool_result(["a", "b"]) == "a\nb"


def test_copilot_cli_session_discovery_prefers_the_requested_session(tmp_path: Path):
    root = tmp_path / "copilot-home" / "session-state"
    wanted = root / "wanted" / "events.jsonl"
    newer = root / "newer" / "events.jsonl"
    for path in (wanted, newer):
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    os.utime(newer, (2_000_000_000, 2_000_000_000))

    assert find_copilot_session_events(tmp_path, session_id="wanted") == wanted
    assert find_copilot_session_events(tmp_path) == newer


def test_copilot_cli_session_discovery_falls_back_when_session_is_missing(
    tmp_path: Path,
):
    root = tmp_path / "copilot-home" / "session-state"
    existing = root / "other" / "events.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}\n", encoding="utf-8")

    assert find_copilot_session_events(tmp_path, session_id="absent") == existing
    assert find_copilot_session_events(tmp_path / "empty") is None


def test_copilot_cli_reads_the_session_started_by_this_run(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    agent._session_id = "ours"
    root = tmp_path / "copilot-home" / "session-state"
    for name, message in (("ours", "our answer"), ("theirs", "their answer")):
        path = root / name / "events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "type": "assistant.message",
                    "timestamp": "2026-01-01T00:00:01.000Z",
                    "data": {"apiCallId": "api-1", "content": message},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    os.utime(root / "theirs" / "events.jsonl", (2_000_000_000, 2_000_000_000))
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["message"] == "our answer"
    assert context.metadata is not None
    assert context.metadata["copilot_session_events"] == (
        "copilot-home/session-state/ours/events.jsonl"
    )


def test_copilot_cli_post_run_is_a_no_op_without_events(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    context = AgentContext(metadata={"existing": "value"})

    agent.populate_context_post_run(context)

    assert not (tmp_path / "trajectory.json").exists()
    assert context.metadata == {"existing": "value"}
    assert context.n_agent_steps is None


def test_copilot_cli_post_run_falls_back_to_the_captured_stream(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    (tmp_path / "copilot-cli.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant.message",
                "timestamp": "2026-01-01T00:00:01.000Z",
                "data": {"apiCallId": "api-1", "content": "captured"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["message"] == "captured"
    assert context.metadata is None


def test_copilot_cli_opaque_reasoning_never_reaches_the_trajectory(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Answer",
                "reasoningOpaque": "opaque-blob",
                "encryptedContent": "encrypted-blob",
            },
        }
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    payload = json.dumps(trajectory.to_json_dict())
    assert "opaque-blob" not in payload
    assert "encrypted-blob" not in payload
    assert trajectory.steps[0].reasoning_content is None


def test_copilot_cli_trailing_reasoning_becomes_its_own_step(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Running",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "tool-1", "result": {"content": "ok"}},
        },
        {
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"content": "Deciding what to do next"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert len(trajectory.steps) == 2
    trailing = trajectory.steps[1]
    assert trailing.message == ""
    assert trailing.reasoning_content == "Deciding what to do next"
    assert trailing.timestamp == "2026-01-01T00:00:03.000Z"


def test_copilot_cli_reasoning_before_a_message_belongs_to_that_message(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.reasoning",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"content": "Planning the fix"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"apiCallId": "api-1", "content": "Here is the fix"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].reasoning_content == "Planning the fix"
    assert trajectory.steps[0].message == "Here is the fix"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "session_basic",
        "session_subagents",
        "session_compaction",
        "session_timeout",
        "session_model_change",
        "session_system_events",
    ],
)
def test_copilot_cli_sanitized_fixtures_convert_to_valid_atif(
    tmp_path: Path, fixture_name: str
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / f"{fixture_name}.jsonl")
    assert events, f"fixture {fixture_name} is empty"

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert any(step.source == "agent" for step in trajectory.steps)


def test_copilot_cli_subagent_fixture_links_every_delegation(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / "session_subagents.jsonl")

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    assert trajectory.subagent_trajectories
    references = [
        ref
        for step in trajectory.steps
        if step.observation is not None
        for result in step.observation.results
        for ref in result.subagent_trajectory_ref or []
    ]
    assert references
    embedded = {sub.trajectory_id for sub in trajectory.subagent_trajectories}
    assert {ref.trajectory_id for ref in references} <= embedded


def test_copilot_cli_timeout_fixture_recovers_from_the_truncated_tail(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    path = _FIXTURES / "session_timeout.jsonl"
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]

    events = _read_jsonl(path)

    assert len(events) == len(raw_lines) - 1
    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)
    assert trajectory is not None
    assert_valid_atif(trajectory)


def test_copilot_cli_compaction_fixture_reports_the_pre_compaction_peak(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / "session_compaction.jsonl")

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    # Both compactions in this session failed, so none of them summarized the
    # context, but the context they tried to compact is still the real peak.
    assert trajectory.final_metrics.extra["summarization_count"] == 0
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 107_296


def test_copilot_cli_system_events_fixture_reports_the_abort(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / "session_system_events.jsonl")

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    assert len(trajectory.final_metrics.extra["abort_reasons"]) == 1
    assert any(step.source == "system" for step in trajectory.steps)


def test_copilot_cli_model_change_fixture_tracks_every_model(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / "session_model_change.jsonl")

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    models = {
        step.model_name for step in trajectory.steps if step.source == "agent"
    }
    assert len(models) > 1
    assert None not in models


def test_copilot_cli_fixture_conversion_is_deterministic(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _read_jsonl(_FIXTURES / "session_model_change.jsonl")

    first = agent._convert_events_to_trajectory(events, metrics_events=events)
    second = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert first is not None and second is not None
    assert first.to_json_dict() == second.to_json_dict()


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
            "data": {"success": True, "preCompactionTokens": 2000},
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


def test_copilot_cli_embeds_subagent_trajectory_and_links_parent_tool_call(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    trajectory = agent._convert_events_to_trajectory(_subagent_events())

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    subagent = trajectory.subagent_trajectories[0]
    assert subagent.trajectory_id == "subagent-task-1"
    assert subagent.agent.name == "copilot-cli:general-purpose"
    assert subagent.agent.model_name == "claude-haiku-4.5"
    assert [step.message for step in subagent.steps] == [
        "Inspecting the parser",
        "The parser drops trailing commas",
    ]
    assert subagent.final_metrics is not None
    assert subagent.final_metrics.total_completion_tokens == 11
    assert subagent.final_metrics.extra == {
        "copilot_total_tokens": 190_174,
        "copilot_total_tool_calls": 8,
        "copilot_duration_ms": 77_963,
    }

    delegating_step = trajectory.steps[1]
    assert delegating_step.tool_calls is not None
    assert delegating_step.tool_calls[0].function_name == "task"
    assert delegating_step.observation is not None
    reference = delegating_step.observation.results[0].subagent_trajectory_ref
    assert reference is not None
    assert [ref.trajectory_id for ref in reference] == ["subagent-task-1"]
    assert reference[0].extra == {
        "agent_id": "task-1",
        "agent_name": "general-purpose",
        "agent_display_name": "General Purpose Agent",
        "model": "claude-haiku-4.5",
        "status": "completed",
    }


def test_copilot_cli_root_totals_count_subagent_tokens_exactly_once(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")

    trajectory = agent._convert_events_to_trajectory(_subagent_events())

    assert trajectory is not None
    assert trajectory.final_metrics is not None
    # Copilot CLI bills subagent inference to the session, so the root totals
    # are session-wide (7 root + 11 subagent output tokens). The embedded
    # trajectory reports the same 11 tokens for its own accounting and they are
    # never added on top of the session total.
    assert trajectory.final_metrics.total_completion_tokens == 18
    assert trajectory.subagent_trajectories is not None
    subagent_metrics = trajectory.subagent_trajectories[0].final_metrics
    assert subagent_metrics is not None
    assert subagent_metrics.total_completion_tokens == 11
    assert trajectory.final_metrics.total_steps == len(trajectory.steps)


def test_copilot_cli_failed_subagent_reports_error_on_reference(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        event
        for event in _subagent_events()
        if event.get("type") != "subagent.completed"
    ]
    events.append(
        {
            "type": "subagent.failed",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "agentId": "task-1",
            "data": {
                "toolCallId": "task-1",
                "agentName": "general-purpose",
                "error": "AbortError: This operation was aborted",
            },
        }
    )

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[1].observation is not None
    reference = trajectory.steps[1].observation.results[0].subagent_trajectory_ref
    assert reference is not None
    assert reference[0].extra["status"] == "failed"
    assert reference[0].extra["error"] == "AbortError: This operation was aborted"


def test_copilot_cli_links_subagent_without_lifecycle_events(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        event
        for event in _subagent_events()
        if not str(event.get("type", "")).startswith("subagent.")
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    assert trajectory.steps[1].observation is not None
    reference = trajectory.steps[1].observation.results[0].subagent_trajectory_ref
    assert reference is not None
    assert reference[0].extra == {"agent_id": "task-1"}


def test_copilot_cli_orphan_subagent_events_stay_unreferenced(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Fix the bug"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Done"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "ghost-1",
            "data": {"apiCallId": "api-2", "content": "Orphaned work"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    assert len(trajectory.subagent_trajectories) == 1
    assert all(step.observation is None for step in trajectory.steps)


def test_copilot_cli_system_messages_become_system_steps(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "system.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Session resumed from a previous run"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Understood"},
        },
        {
            "type": "system.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"content": ""},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert [step.source for step in trajectory.steps] == ["system", "agent"]
    assert trajectory.steps[0].message == "Session resumed from a previous run"
    assert trajectory.steps[0].model_name is None
    assert trajectory.steps[0].llm_call_count is None


def test_copilot_cli_abort_and_session_error_land_in_final_metrics(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "abort",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"reason": "user_interrupt"},
        },
        {
            "type": "session.error",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"message": "rate limit exceeded", "code": "429", "fatal": False},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["abort_reasons"] == ["user_interrupt"]
    assert trajectory.final_metrics.extra["session_errors"] == [
        {"message": "rate limit exceeded", "code": "429", "fatal": False}
    ]


@pytest.mark.parametrize(
    "event_type",
    [
        "hook.execution_start",
        "hook.execution_complete",
        "permission.requested",
        "permission.granted",
        "system.notification",
        "session.info",
        "session.warning",
        "session.mode_changed",
        "session.permissions_changed",
        "session.plan_changed",
        "session.task_complete",
        "subagent.selected",
        "skill.invoked",
        "tool.user_requested",
        "copilot.event_from_the_future",
    ],
)
def test_copilot_cli_ignores_unhandled_event_types(tmp_path: Path, event_type: str):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Fix the bug"},
        },
        {
            "type": event_type,
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"content": "noise", "message": "noise"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"apiCallId": "api-1", "content": "Done"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert [step.source for step in trajectory.steps] == ["user", "agent"]


def test_copilot_cli_converts_legacy_stdout_message_tool_events(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "session.start",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {
                "sessionId": "legacy-session",
                "selectedModel": "gpt-5.4",
                "reasoningEffort": "medium",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "role": "user",
            "content": [{"text": "Fix legacy output"}],
        },
        {
            "type": "message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "role": "assistant",
            "content": {"text": "I will inspect"},
        },
        {
            "type": "tool_use",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "id": "legacy-tool-1",
            "name": "bash",
            "input": '{"command": "cat file"}',
        },
        {
            "type": "tool_result",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "tool_use_id": "legacy-tool-1",
            "content": [{"text": "legacy output"}],
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.session_id == "legacy-session"
    assert [step.source for step in trajectory.steps] == ["user", "agent", "agent"]
    assert [step.message for step in trajectory.steps] == [
        "Fix legacy output",
        "I will inspect",
        "",
    ]
    assistant_step = trajectory.steps[1]
    assert assistant_step.model_name == "gpt-5.4"
    assert assistant_step.reasoning_effort == "medium"
    assert assistant_step.llm_call_count == 1
    tool_step = trajectory.steps[2]
    assert tool_step.tool_calls is not None
    assert tool_step.tool_calls[0].tool_call_id == "legacy-tool-1"
    assert tool_step.tool_calls[0].function_name == "bash"
    assert tool_step.tool_calls[0].arguments == {"command": "cat file"}
    assert tool_step.observation is not None
    assert tool_step.observation.results[0].content == "legacy output"


def test_copilot_cli_tracks_model_and_effort_changes(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="github/gpt-5.4")
    events = [
        {
            "type": "session.start",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"sessionId": "s-1", "selectedModel": "gpt-5.4"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "First"},
        },
        {
            "type": "session.model_change",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "newModel": "claude-sonnet-4.5",
                "reasoningEffort": "high",
                "contextTier": "long_context",
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "api-2", "content": "Second"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "data": {
                "apiCallId": "api-3",
                "model": "gpt-5.4-mini",
                "content": "Third",
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert [step.model_name for step in trajectory.steps] == [
        "gpt-5.4",
        "claude-sonnet-4.5",
        "gpt-5.4-mini",
    ]
    assert [step.reasoning_effort for step in trajectory.steps] == [
        None,
        "high",
        "high",
    ]


def test_copilot_cli_uses_configured_reasoning_effort_before_any_change(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4", reasoning_effort="low")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "First"},
        }
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].reasoning_effort == "low"


def test_copilot_cli_tool_only_assistant_message_has_no_fabricated_text(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "tool-1", "result": {"content": "ok"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].message == ""


def test_copilot_cli_unrequested_tool_execution_is_a_zero_call_step(
    tmp_path: Path,
):
    """A dispatch nothing requested is recorded without inventing its origin.

    The request may simply never have been persisted (a resumed session), so
    the step says only that no inference produced it.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "toolCallId": "tool-9",
                "toolName": "bash",
                "arguments": {"command": "ls"},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "tool-9", "result": {"content": "README.md"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.llm_call_count == 0
    assert step.metrics is None
    assert step.extra is None
    assert step.tool_calls is not None
    assert step.tool_calls[0].arguments == {"command": "ls"}
    assert step.observation is not None
    assert step.observation.results[0].content == "README.md"


def test_copilot_cli_flagged_tool_execution_records_the_user_as_dispatcher(
    tmp_path: Path,
):
    """`isUserRequested` is Copilot's own word for it, so it is trusted."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "toolCallId": "tool-9",
                "toolName": "bash",
                "arguments": {"command": "ls"},
                "isUserRequested": True,
            },
        }
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].extra == {"user_requested": True}


def test_copilot_cli_requested_tool_execution_start_does_not_duplicate_call(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Listing",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"toolCallId": "tool-1", "toolName": "bash", "arguments": {}},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "tool-1", "result": {"content": "ok"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].tool_calls is not None
    assert len(trajectory.steps[0].tool_calls) == 1


def test_copilot_cli_never_completed_tool_call_has_no_observation(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Running",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"toolCallId": "tool-1", "toolName": "bash", "arguments": {}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].observation is None


def test_copilot_cli_duplicate_tool_completion_is_recorded_once(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Running",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "tool-1", "result": {"content": "ok"}},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"toolCallId": "tool-1", "result": {"content": "ok"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].observation is not None
    assert len(trajectory.steps[0].observation.results) == 1


def test_copilot_cli_orphan_tool_completion_becomes_an_unattributed_observation(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Hello"},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "ghost-1", "result": {"content": "stray output"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    orphan = trajectory.steps[1]
    assert orphan.source == "agent"
    assert orphan.llm_call_count == 0
    # Tool output must never be presented as something the agent said.
    assert orphan.message == ""
    assert orphan.observation is not None
    result = orphan.observation.results[0]
    assert result.source_call_id is None
    assert result.content == "stray output"
    assert result.extra == {"copilot_tool_call_id": "ghost-1"}


def test_copilot_cli_tool_error_is_appended_to_the_observation(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Running",
                "toolRequests": [
                    {"toolCallId": "tool-1", "name": "bash", "arguments": {}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "tool-1",
                "success": False,
                "result": None,
                "error": {"message": "command not found", "code": "ENOENT"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].observation is not None
    content = trajectory.steps[0].observation.results[0].content
    assert content is not None
    assert "command not found" in content
    assert "ENOENT" in content


def test_copilot_cli_merges_streaming_messages_sharing_an_api_call_id(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "messageId": "m-1",
                "content": "Let me",
                "outputTokens": 2,
                "phase": "commentary",
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "apiCallId": "api-1",
                "messageId": "m-2",
                "content": "Let me check the parser.",
                "outputTokens": 6,
                "phase": "final_answer",
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.message == "Let me check the parser."
    assert step.llm_call_count == 1
    assert step.metrics is not None
    assert step.metrics.completion_tokens == 6
    assert step.extra == {
        "api_call_id": "api-1",
        "phases": ["commentary", "final_answer"],
    }


def test_copilot_cli_distinct_messages_sharing_an_api_call_id_are_concatenated(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "First thought"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"apiCallId": "api-1", "content": "Unrelated conclusion"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.steps[0].message == "First thought\nUnrelated conclusion"


def test_copilot_cli_messages_without_api_call_id_are_separate_steps(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"messageId": "m-1", "content": "First"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"messageId": "m-2", "content": "Second"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert [step.message for step in trajectory.steps] == ["First", "Second"]


def test_copilot_cli_prompt_tokens_come_from_the_usage_stream(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Hello", "outputTokens": 4},
        }
    ]
    metrics_events = events + [
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.100Z",
            "data": {
                "apiCallId": "api-1",
                "inputTokens": 1200,
                "cacheReadTokens": 1000,
                "outputTokens": 4,
            },
        }
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 1200
    assert trajectory.steps[0].metrics.cached_tokens == 1000
    assert trajectory.steps[0].metrics.completion_tokens == 4


def test_copilot_cli_usage_is_not_shared_across_colliding_api_call_ids(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    # OpenAI-compatible servers hand out sequential ids, so a parent session
    # and its subagent can both report "chatcmpl-1" for different calls.
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Delegating",
                "toolRequests": [
                    {"toolCallId": "task-1", "name": "task", "arguments": {}}
                ],
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"apiCallId": "chatcmpl-1", "content": "Delegated work"},
        },
    ]
    metrics_events = events + [
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.100Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 100},
        },
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:02.100Z",
            "agentId": "task-1",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 900},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 100
    assert trajectory.subagent_trajectories is not None
    subagent_step = trajectory.subagent_trajectories[0].steps[0]
    assert subagent_step.metrics is not None
    assert subagent_step.metrics.prompt_tokens == 900


def test_copilot_cli_untagged_usage_still_reaches_subagent_steps(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-root", "content": "Delegating"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"apiCallId": "api-sub", "content": "Delegated work"},
        },
    ]
    metrics_events = events + [
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:02.100Z",
            "data": {"apiCallId": "api-sub", "inputTokens": 750},
        }
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    subagent_step = trajectory.subagent_trajectories[0].steps[0]
    assert subagent_step.metrics is not None
    assert subagent_step.metrics.prompt_tokens == 750


def test_copilot_cli_user_message_records_delivery_and_source(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {
                "content": "Run the suite",
                "delivery": "steering",
                "source": "schedule-3",
                "attachments": [{"type": "image", "name": "screenshot.png"}],
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "On it"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].extra == {
        "delivery": "steering",
        "message_source": "schedule-3",
        "attachments": [{"type": "image", "name": "screenshot.png"}],
    }


def test_copilot_cli_user_message_falls_back_to_transformed_content(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "", "transformedContent": "Expanded prompt"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "On it"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].message == "Expanded prompt"
    assert trajectory.steps[0].extra is None


def _subagent_events() -> list[dict]:
    return [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Investigate the parser"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Delegating the investigation.",
                "outputTokens": 7,
                "toolRequests": [
                    {
                        "toolCallId": "task-1",
                        "name": "task",
                        "arguments": {"prompt": "Investigate the parser"},
                    }
                ],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:01.100Z",
            "agentId": "task-1",
            "data": {
                "toolCallId": "task-1",
                "agentName": "general-purpose",
                "agentDisplayName": "General Purpose Agent",
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "api-sub-1",
                "model": "claude-haiku-4.5",
                "content": "Inspecting the parser",
                "outputTokens": 4,
                "toolRequests": [
                    {
                        "toolCallId": "sub-tool-1",
                        "name": "view",
                        "arguments": {"path": "parser.py"},
                    }
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"toolCallId": "sub-tool-1", "result": {"content": "def parse():"}},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.500Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "api-sub-2",
                "model": "claude-haiku-4.5",
                "content": "The parser drops trailing commas",
                "outputTokens": 7,
            },
        },
        {
            "type": "subagent.completed",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "agentId": "task-1",
            "data": {
                "toolCallId": "task-1",
                "agentName": "general-purpose",
                "agentDisplayName": "General Purpose Agent",
                "model": "claude-haiku-4.5",
                "totalToolCalls": 8,
                "totalTokens": 190174,
                "durationMs": 77963,
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:03.500Z",
            "data": {
                "toolCallId": "task-1",
                "result": {"content": "The parser drops trailing commas"},
            },
        },
    ]


def test_copilot_cli_subagent_abort_is_reported_on_its_own_trajectory(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _subagent_events()
    events.insert(
        -1,
        {
            "type": "abort",
            "timestamp": "2026-01-01T00:00:02.900Z",
            "agentId": "task-1",
            "data": {"reason": "subagent_timeout"},
        },
    )

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert "abort_reasons" not in trajectory.final_metrics.extra
    assert trajectory.subagent_trajectories is not None
    subagent_metrics = trajectory.subagent_trajectories[0].final_metrics
    assert subagent_metrics is not None
    assert subagent_metrics.extra is not None
    assert subagent_metrics.extra["abort_reasons"] == ["subagent_timeout"]


def test_copilot_cli_unknown_session_error_shape_is_preserved(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.error",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"unexpectedField": {"detail": "future shape"}},
        },
        {
            "type": "session.error",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["session_errors"] == [
        {"unexpectedField": {"detail": "future shape"}}
    ]


def test_copilot_cli_reused_api_call_id_across_turns_stays_two_steps(tmp_path: Path):
    """Locally hosted OpenAI-compatible servers reuse sequential call ids."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="qwen3-coder")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Reading the file",
                "outputTokens": 40,
                "toolRequests": [
                    {"toolCallId": "t1", "name": "view", "arguments": {"path": "a.py"}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "success": True, "result": {"content": "ok"}},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Now writing",
                "outputTokens": 55,
                "toolRequests": [
                    {"toolCallId": "t2", "name": "edit", "arguments": {"path": "a.py"}}
                ],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "data": {"toolCallId": "t2", "success": True, "result": {"content": "ok"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 2
    assert agent_steps[0].message == "Reading the file"
    assert agent_steps[1].message == "Now writing"
    assert [call.tool_call_id for call in agent_steps[0].tool_calls or []] == ["t1"]
    assert [call.tool_call_id for call in agent_steps[1].tool_calls or []] == ["t2"]
    assert sum(step.llm_call_count or 0 for step in agent_steps) == 2
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_completion_tokens == 95


def test_copilot_cli_streamed_chunks_of_one_call_still_merge(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Reading"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Reading the file",
                "outputTokens": 12,
                "toolRequests": [{"toolCallId": "t1", "name": "view"}],
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].message == "Reading the file"
    assert trajectory.steps[0].llm_call_count == 1


def test_copilot_cli_keeps_detailed_content_that_replaces_the_summary(tmp_path: Path):
    assert (
        _stringify_tool_result(
            {"content": "Ran 3 tests.", "detailedContent": "FAILED test_a: boom"}
        )
        == "Ran 3 tests.\nFAILED test_a: boom"
    )


def test_copilot_cli_does_not_duplicate_repeated_tool_result_renderings(tmp_path: Path):
    assert _stringify_tool_result({"content": "same", "detailedContent": "same"}) == (
        "same"
    )
    assert (
        _stringify_tool_result({"content": "head", "detailedContent": "head and tail"})
        == "head and tail"
    )
    assert (
        _stringify_tool_result(
            {
                "content": "head",
                "detailedContent": "head and tail",
                "displayContent": "head",
            }
        )
        == "head and tail"
    )


def test_copilot_cli_keeps_disjoint_display_content(tmp_path: Path):
    assert (
        _stringify_tool_result({"content": "summary", "displayContent": "rendered box"})
        == "summary\nrendered box"
    )


def test_copilot_cli_user_dispatched_tool_is_a_deterministic_step(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"content": "!git status"},
        },
        {
            "type": "tool.user_requested",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "u1",
                "toolName": "local_shell",
                "arguments": {"command": "git status"},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "toolCallId": "u1",
                "success": True,
                "isUserRequested": True,
                "result": {"content": "On branch main"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    dispatch = trajectory.steps[1]
    assert dispatch.source == "agent"
    assert dispatch.llm_call_count == 0
    # The shell output must not be attributed to the model as speech.
    assert dispatch.message == ""
    assert dispatch.extra == {"user_requested": True}
    assert dispatch.tool_calls is not None
    assert dispatch.tool_calls[0].function_name == "local_shell"
    assert dispatch.tool_calls[0].arguments == {"command": "git status"}
    assert dispatch.observation is not None
    result = dispatch.observation.results[0]
    assert result.source_call_id == "u1"
    assert result.content == "On branch main"
    assert result.extra == {"copilot_success": True, "user_requested": True}


def test_copilot_cli_records_tool_failure_as_is_error(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Reading",
                "toolRequests": [{"toolCallId": "t1", "name": "view"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "t1",
                "success": False,
                "error": {"message": "Path does not exist", "code": "failure"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[0].observation
    assert observation is not None
    assert observation.results[0].extra == {"copilot_success": False, "is_error": True}


def test_copilot_cli_success_never_claims_the_command_worked(tmp_path: Path):
    """`success` reports the harness, not the command.

    A `bash` call whose command exits non-zero is still a *successful*
    execution to Copilot. Emitting `is_error: false` for it would override the
    viewer's own content heuristic and hide the failure, so only the negative
    signal is promoted; the raw field is kept verbatim beside it.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Building",
                "toolRequests": [{"toolCallId": "t1", "name": "bash"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "t1",
                "success": True,
                "result": {"content": "error: build failed with exit code 1"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[0].observation
    assert observation is not None
    assert observation.results[0].extra == {"copilot_success": True}


def test_copilot_cli_missing_success_flag_leaves_no_error_verdict(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Reading",
                "toolRequests": [{"toolCallId": "t1", "name": "view"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "result": {"content": "ok"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[0].observation
    assert observation is not None
    assert observation.results[0].extra is None


def test_copilot_cli_compaction_summary_becomes_a_system_step(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.compaction_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "success": True,
                "summaryContent": "<overview>The user asked for X</overview>",
                "preCompactionTokens": 136920,
                "preCompactionMessagesLength": 30,
                "checkpointNumber": 1,
                "trigger": "threshold",
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "api-2", "content": "Continuing"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    summary = trajectory.steps[1]
    assert summary.source == "system"
    assert summary.message == "<overview>The user asked for X</overview>"
    assert summary.extra == {
        "compaction": True,
        "pre_compaction_tokens": 136920,
        "pre_compaction_messages": 30,
        "checkpoint_number": 1,
        "trigger": "threshold",
    }
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["summarization_count"] == 1
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 136920


def test_copilot_cli_failed_compaction_adds_no_system_step(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.compaction_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "success": False,
                "error": "Compaction failed: empty response",
                "summaryContent": "half written",
                "preCompactionTokens": 107296,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert [step.source for step in trajectory.steps] == ["agent"]
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["summarization_count"] == 0
    assert trajectory.final_metrics.extra["peak_context_tokens"] == 107296


def test_copilot_cli_permission_decision_is_recorded_on_the_tool_call(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Removing the file",
                "toolRequests": [{"toolCallId": "t1", "name": "bash"}],
            },
        },
        {
            "type": "permission.requested",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "requestId": "req-1",
                "permissionRequest": {
                    "kind": "shell",
                    "toolCallId": "t1",
                    "fullCommandText": "rm -rf build",
                },
            },
        },
        {
            "type": "permission.completed",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "requestId": "req-1",
                "toolCallId": "t1",
                "result": {
                    "kind": "denied-interactively-by-user",
                    "feedback": "No, keep the build directory",
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    tool_calls = trajectory.steps[0].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].extra == {
        "permission_kind": "shell",
        "permission_decision": "denied-interactively-by-user",
        "permission_feedback": "No, keep the build directory",
    }


def test_copilot_cli_permission_completion_resolves_via_request_id(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Fetching",
                "toolRequests": [{"toolCallId": "t1", "name": "fetch"}],
            },
        },
        {
            "type": "permission.requested",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "requestId": "req-1",
                "permissionRequest": {"kind": "url", "toolCallId": "t1"},
            },
        },
        {
            "type": "permission.completed",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"requestId": "req-1", "result": {"kind": "approved"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    tool_calls = trajectory.steps[0].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].extra == {
        "permission_kind": "url",
        "permission_decision": "approved",
    }


def test_copilot_cli_keeps_tool_call_metadata(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Searching",
                "toolRequests": [
                    {
                        "toolCallId": "t1",
                        "name": "web_search",
                        "intentionSummary": "Look up the ATIF spec",
                        "toolTitle": "Searching the web",
                        "mcpServerName": "github-mcp-server",
                        "mcpToolName": "web_search",
                    }
                ],
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "t1",
                "toolName": "web_search",
                "shellToolInfo": {"possiblePaths": ["/repo"]},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "toolCallId": "t1",
                "success": True,
                "result": {"content": "found"},
                "toolTelemetry": {"properties": {"resultCount": 3}},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    tool_calls = trajectory.steps[0].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].extra == {
        "intention_summary": "Look up the ATIF spec",
        "tool_title": "Searching the web",
        "mcp_server_name": "github-mcp-server",
        "mcp_tool_name": "web_search",
        "shell_tool_info": {"possiblePaths": ["/repo"]},
    }
    observation = trajectory.steps[0].observation
    assert observation is not None
    assert observation.results[0].extra == {
        "copilot_success": True,
        "tool_telemetry": {"properties": {"resultCount": 3}},
    }


def test_copilot_cli_execution_start_never_duplicates_a_requested_call(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Reading",
                "toolRequests": [{"toolCallId": "t1", "name": "view"}],
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "toolName": "view"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert len(trajectory.steps) == 1


def test_copilot_cli_selected_agents_are_reported(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "subagent.selected",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "data": {"agentName": "Squad", "agentDisplayName": "Squad", "tools": ["*"]},
        },
        {
            "type": "subagent.selected",
            "timestamp": "2026-01-01T00:00:00.600Z",
            "data": {"agentName": "Squad"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["copilot_selected_agents"] == ["Squad"]


def test_copilot_cli_subagent_description_reaches_the_reference(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _subagent_events()
    for event in events:
        if event["type"] == "subagent.started":
            event["data"]["agentDescription"] = "Full-capability agent in a subprocess"

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[1].observation
    assert observation is not None
    reference = observation.results[0].subagent_trajectory_ref
    assert reference is not None
    assert reference[0].extra is not None
    assert reference[0].extra["agent_description"] == (
        "Full-capability agent in a subprocess"
    )


def test_copilot_cli_subagent_zero_totals_are_not_dropped(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = _subagent_events()
    for event in events:
        if event["type"] == "subagent.completed":
            event["data"]["totalToolCalls"] = 0
            event["data"]["totalTokens"] = 0
            event["data"]["durationMs"] = 0

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    metrics = trajectory.subagent_trajectories[0].final_metrics
    assert metrics is not None
    assert metrics.extra is not None
    assert metrics.extra["copilot_total_tool_calls"] == 0
    assert metrics.extra["copilot_total_tokens"] == 0
    assert metrics.extra["copilot_duration_ms"] == 0


def test_copilot_cli_subagent_without_a_completion_is_still_referenced(tmp_path: Path):
    """A run killed mid-delegation never reports the delegating call's result."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        event
        for event in _subagent_events()
        if not (
            event["type"] == "tool.execution_complete"
            and event["data"].get("toolCallId") == "task-1"
        )
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[1].observation
    assert observation is not None
    result = observation.results[0]
    assert result.source_call_id == "task-1"
    # No output was produced, and none may be invented.
    assert result.content is None
    assert result.subagent_trajectory_ref is not None
    assert result.subagent_trajectory_ref[0].trajectory_id == "subagent-task-1"


def test_copilot_cli_nested_subagent_nests_inside_its_delegator(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Delegating",
                "toolRequests": [{"toolCallId": "outer", "name": "task"}],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "agentId": "outer",
            "data": {"toolCallId": "outer", "agentName": "general-purpose"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "outer",
            "data": {
                "apiCallId": "api-2",
                "content": "Delegating further",
                "toolRequests": [{"toolCallId": "inner", "name": "task"}],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:02.200Z",
            "agentId": "inner",
            "data": {"toolCallId": "inner", "agentName": "explore"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "agentId": "inner",
            "data": {"apiCallId": "api-3", "content": "Exploring"},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "agentId": "outer",
            "data": {
                "toolCallId": "inner",
                "success": True,
                "result": {"content": "inner done"},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:05.000Z",
            "data": {
                "toolCallId": "outer",
                "success": True,
                "result": {"content": "outer done"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    assert [t.trajectory_id for t in trajectory.subagent_trajectories] == [
        "subagent-outer"
    ]
    outer = trajectory.subagent_trajectories[0]
    assert outer.subagent_trajectories is not None
    assert [t.trajectory_id for t in outer.subagent_trajectories] == ["subagent-inner"]
    inner_observation = outer.steps[0].observation
    assert inner_observation is not None
    reference = inner_observation.results[0].subagent_trajectory_ref
    assert reference is not None
    assert reference[0].trajectory_id == "subagent-inner"


def test_copilot_cli_session_totals_keep_colliding_ids_apart(tmp_path: Path):
    """Without a shutdown event the totals are summed from per-call usage."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="qwen3-coder")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Delegating",
                "inputTokens": 100,
                "outputTokens": 10,
                "toolRequests": [{"toolCallId": "task-1", "name": "task"}],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"toolCallId": "task-1", "agentName": "explore"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Working",
                "inputTokens": 900,
                "outputTokens": 90,
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "toolCallId": "task-1",
                "success": True,
                "result": {"content": "done"},
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 1000
    assert trajectory.final_metrics.total_completion_tokens == 100


def test_copilot_cli_keeps_the_full_session_error_shape(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "session.error",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "errorType": "RateLimit",
                "message": "429 Too Many Requests",
                "statusCode": 429,
                "stack": "Error: 429\n  at request",
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra["session_errors"] == [
        {
            "errorType": "RateLimit",
            "message": "429 Too Many Requests",
            "statusCode": 429,
            "stack": "Error: 429\n  at request",
        }
    ]


def test_copilot_cli_reads_nested_usage_payloads(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working", "outputTokens": 12},
        }
    ]
    metrics_events = [
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {
                "providerCallId": "api-1",
                "usage": {
                    "inputTokens": 4096,
                    "cacheReadTokens": 2048,
                    "outputTokens": 12,
                },
            },
        }
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    metrics = trajectory.steps[0].metrics
    assert metrics is not None
    assert metrics.prompt_tokens == 4096
    assert metrics.cached_tokens == 2048


def test_copilot_cli_tolerates_non_dict_event_payloads(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {"type": "assistant.message", "timestamp": "2026-01-01T00:00:01.000Z"},
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": ["not", "a", "dict"],
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "api-1", "content": "Still fine"},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[-1].message == "Still fine"


def _reused_call_id_events() -> list[dict]:
    return [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Reading",
                "outputTokens": 10,
                "toolRequests": [{"toolCallId": "t1", "name": "view"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "success": True, "result": {"content": "ok"}},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "Done", "outputTokens": 20},
        },
    ]


def test_copilot_cli_combines_event_streams_chronologically(tmp_path: Path):
    """Prompt tokens only reach stdout; they must land in the right turn."""
    persisted = _reused_call_id_events()
    captured = [
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 100},
        },
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:03.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 900},
        },
    ]

    combined = _combine_event_streams(persisted, captured)

    assert [event["timestamp"] for event in combined] == [
        "2026-01-01T00:00:01.000Z",
        "2026-01-01T00:00:01.500Z",
        "2026-01-01T00:00:02.000Z",
        "2026-01-01T00:00:03.000Z",
        "2026-01-01T00:00:03.500Z",
    ]

    agent = CopilotCli(logs_dir=tmp_path, model_name="qwen3-coder")
    trajectory = agent._convert_events_to_trajectory(
        persisted, metrics_events=combined
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert [step.metrics.prompt_tokens for step in agent_steps] == [100, 900]
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 1000


def test_copilot_cli_unambiguous_usage_survives_stream_skew(tmp_path: Path):
    """A call id used once needs no turn agreement between the two streams."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Reading"},
        },
    ]
    metrics_events = [
        *events,
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.500Z",
            "data": {"content": "go"},
        },
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:09.000Z",
            "data": {"apiCallId": "api-1", "inputTokens": 4096},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 4096


def test_copilot_cli_combined_streams_keep_untimestamped_events_in_place(
    tmp_path: Path,
):
    persisted = [
        {"type": "session.start", "timestamp": "2026-01-01T00:00:00.000Z", "data": {}},
        {"type": "assistant.usage", "id": "no-clock", "data": {"apiCallId": "api-1"}},
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:05.000Z",
            "data": {"apiCallId": "api-1"},
        },
    ]

    combined = _combine_event_streams(persisted, [])

    assert [event.get("id") or event["type"] for event in combined] == [
        "session.start",
        "no-clock",
        "assistant.message",
    ]


def test_copilot_cli_does_not_double_count_usage_reported_on_two_streams(
    tmp_path: Path,
):
    """One call reported by both streams is one call, whatever turn it lands in.

    The captured stdout stream can start mid-session, so its first events have
    no turn boundary in front of them and sit in turn 0 while the persisted
    `assistant.message` for the very same call sits in a later turn. Keying
    the reconciliation by turn made the two independent entries and summed
    them, doubling the session totals.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Hi"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Hello",
                "inputTokens": 100,
                "outputTokens": 10,
            },
        },
    ]
    metrics_events = [
        {
            "type": "assistant.usage",
            "data": {"apiCallId": "api-1", "inputTokens": 100, "outputTokens": 10},
        },
        *events,
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 10


def test_copilot_cli_reused_call_id_in_two_turns_still_sums(tmp_path: Path):
    """Reconciling across streams must not collapse a genuinely reused id."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Hi"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Looking",
                "inputTokens": 100,
                "outputTokens": 10,
                "toolRequests": [{"toolCallId": "t1", "name": "bash"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "result": {"content": "ok"}},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Done",
                "inputTokens": 900,
                "outputTokens": 90,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events, metrics_events=events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 1000
    assert trajectory.final_metrics.total_completion_tokens == 100


def test_copilot_cli_never_charges_one_usage_report_to_two_steps(tmp_path: Path):
    """A report describes one inference, so a reused id must not spend it twice.

    With a partially captured stdout stream only the first of two calls
    sharing an id has a report; charging it to both made the per-step metrics
    sum to twice the session total.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Hi"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Looking",
                "toolRequests": [{"toolCallId": "t1", "name": "bash"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "t1", "result": {"content": "ok"}},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "Done"},
        },
    ]
    metrics_events = [
        events[0],
        events[1],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 500},
        },
        events[2],
        events[3],
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[1].metrics is not None
    assert trajectory.steps[1].metrics.prompt_tokens == 500
    assert trajectory.steps[2].metrics is None or (
        trajectory.steps[2].metrics.prompt_tokens is None
    )
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 500


def test_copilot_cli_untagged_usage_never_crosses_an_agent_boundary(tmp_path: Path):
    """An untagged report is bucketed under the root's turn counter.

    A subagent does not share that counter, so resolving one against its own
    turn ordinal handed the parent's tokens to the child (and vice versa) when
    both scopes reused a `chatcmpl-N` id.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {"content": "Delegate"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-1",
                "content": "Delegating",
                "toolRequests": [
                    {"toolCallId": "task-1", "name": "task", "arguments": {}}
                ],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"agentName": "reviewer"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "agentId": "task-1",
            "data": {"apiCallId": "chatcmpl-1", "content": "Reviewed"},
        },
        {
            "type": "subagent.completed",
            "timestamp": "2026-01-01T00:00:05.000Z",
            "agentId": "task-1",
            "data": {},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:06.000Z",
            "data": {"toolCallId": "task-1", "result": {"content": "reviewed"}},
        },
    ]
    metrics_events = [
        *events[:2],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 1000},
        },
        events[2],
        # The root's turn advances before the subagent's report arrives, so
        # the two untagged reports for the reused id land in different buckets.
        {
            "type": "user.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"content": "keep going"},
        },
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:03.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 9000},
        },
        *events[3:],
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.steps[1].metrics is not None
    assert trajectory.steps[1].metrics.prompt_tokens == 1000
    assert trajectory.subagent_trajectories is not None
    subagent = trajectory.subagent_trajectories[0]
    assert subagent.steps[0].metrics is not None
    assert subagent.steps[0].metrics.prompt_tokens == 9000
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 10000


def test_copilot_cli_applies_permissions_to_a_subagent_tool_call(tmp_path: Path):
    """Permission events are root-level even for calls a subagent owns.

    Copilot never tags them with an `agentId`, so looking them up in the
    emitting scope alone dropped the decision -- including the denials -- for
    every tool a delegated agent ran.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Delegating",
                "toolRequests": [
                    {"toolCallId": "task-1", "name": "task", "arguments": {}}
                ],
            },
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"agentName": "reviewer"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "api-2",
                "content": "Removing",
                "toolRequests": [{"toolCallId": "child-1", "name": "bash"}],
            },
        },
        {
            "type": "permission.requested",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "data": {
                "requestId": "req-1",
                "permissionRequest": {"toolCallId": "child-1", "kind": "shell"},
            },
        },
        {
            "type": "permission.completed",
            "timestamp": "2026-01-01T00:00:05.000Z",
            "data": {
                "requestId": "req-1",
                "result": {"kind": "denied", "feedback": "too destructive"},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:06.000Z",
            "agentId": "task-1",
            "data": {"toolCallId": "child-1", "success": False, "result": {}},
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:07.000Z",
            "data": {"toolCallId": "task-1", "result": {"content": "stopped"}},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    child_calls = trajectory.subagent_trajectories[0].steps[0].tool_calls
    assert child_calls is not None
    assert child_calls[0].extra == {
        "permission_kind": "shell",
        "permission_decision": "denied",
        "permission_feedback": "too destructive",
    }


def test_copilot_cli_user_dispatched_tool_ends_the_api_turn(tmp_path: Path):
    """A user-dispatched tool is a turn boundary, so the id may be reused."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "First"},
        },
        {
            "type": "tool.user_requested",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"toolCallId": "u1", "toolName": "bash", "arguments": "{}"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:04.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "Second"},
        },
    ]
    metrics_events = [
        events[0],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 100},
        },
        *events[1:],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:04.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 900},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert [step.message for step in trajectory.steps] == ["First", "", "Second"]
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 100
    assert trajectory.steps[2].metrics is not None
    assert trajectory.steps[2].metrics.prompt_tokens == 900


def test_copilot_cli_compaction_ends_the_api_turn(tmp_path: Path):
    """Compaction rewrites the conversation, so the next call is a new turn."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "First"},
        },
        {
            "type": "session.compaction_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {"success": True, "summaryContent": "Summary so far"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "data": {"apiCallId": "chatcmpl-1", "content": "Second"},
        },
    ]
    metrics_events = [
        events[0],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 100},
        },
        events[1],
        events[2],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:03.500Z",
            "data": {"apiCallId": "chatcmpl-1", "inputTokens": 900},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert_one_step_per_api_call(trajectory)
    assert [step.source for step in trajectory.steps] == ["agent", "system", "agent"]
    assert trajectory.steps[0].metrics is not None
    assert trajectory.steps[0].metrics.prompt_tokens == 100
    assert trajectory.steps[2].metrics is not None
    assert trajectory.steps[2].metrics.prompt_tokens == 900
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 1000


def test_copilot_cli_merges_streams_by_instant_not_by_string(tmp_path: Path):
    """A local offset must not sort ahead of the `Z` it is simultaneous with."""
    persisted = [
        {
            "id": "later",
            "type": "assistant.message",
            "timestamp": "2026-01-01T09:00:00.000Z",
            "data": {},
        }
    ]
    captured = [
        {
            "id": "earlier",
            "type": "assistant.usage",
            "timestamp": "2026-01-01T09:30:00.000+01:00",
            "data": {},
        }
    ]

    combined = _combine_event_streams(persisted, captured)

    assert [event["id"] for event in combined] == ["earlier", "later"]


def test_copilot_cli_captured_stream_prefix_stays_with_its_first_event(
    tmp_path: Path,
):
    """A stdout stream starting mid-session must not sort to turn zero.

    Untimestamped leading events used to inherit the empty string and sort
    ahead of the whole session, landing a usage report in a turn that had not
    happened yet.
    """
    persisted = [
        {
            "id": "start",
            "type": "session.start",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "data": {},
        },
        {
            "id": "message",
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:09.000Z",
            "data": {},
        },
    ]
    captured = [
        {"id": "no-clock", "type": "assistant.usage", "data": {}},
        {
            "id": "clocked",
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:05.000Z",
            "data": {},
        },
    ]

    combined = _combine_event_streams(persisted, captured)

    assert [event["id"] for event in combined] == [
        "start",
        "no-clock",
        "clocked",
        "message",
    ]


def test_copilot_cli_subagent_never_becomes_its_own_parent(tmp_path: Path):
    """A self-referential delegation must not delete the trajectory."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"apiCallId": "api-1", "content": "Working"},
        },
        {
            "type": "subagent.started",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {"agentName": "reviewer", "toolCallId": "task-1"},
        },
        # The subagent itself reports the call that spawned it, so its own
        # scope owns `task-1` and it would adopt itself.
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:03.000Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "api-2",
                "content": "Reviewing",
                "toolRequests": [{"toolCallId": "task-1", "name": "task"}],
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.subagent_trajectories is not None
    assert len(trajectory.subagent_trajectories) == 1


def test_copilot_cli_keeps_display_variants_of_a_non_content_result(
    tmp_path: Path,
):
    """A rendering variant must survive whichever text key carried the result."""
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "api-1",
                "content": "Running",
                "toolRequests": [{"toolCallId": "t1", "name": "bash"}],
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "data": {
                "toolCallId": "t1",
                "result": {
                    "output": "3 files changed",
                    "detailedContent": "a.py\nb.py\nc.py",
                },
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert_valid_atif(trajectory)
    observation = trajectory.steps[0].observation
    assert observation is not None
    assert observation.results[0].content == "3 files changed\na.py\nb.py\nc.py"


def test_copilot_cli_untagged_usage_matches_a_tagged_message(tmp_path: Path):
    """Stdout usage need not carry the `agentId` the persisted message does.

    Reconciling on the full key left the untagged report and the tagged
    message as two entries for one inference, doubling its output tokens.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {
                "apiCallId": "chatcmpl-7",
                "content": "Delegating",
                "toolRequests": [
                    {"toolCallId": "task-1", "name": "task", "arguments": {}}
                ],
            },
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "agentId": "task-1",
            "data": {
                "apiCallId": "chatcmpl-9",
                "content": "Reviewed",
                "outputTokens": 10,
            },
        },
    ]
    metrics_events = [
        events[0],
        events[1],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:02.500Z",
            "data": {
                "apiCallId": "chatcmpl-9",
                "inputTokens": 100,
                "outputTokens": 10,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 10


def test_copilot_cli_message_without_a_call_id_matches_its_turn(tmp_path: Path):
    """A message with no `apiCallId` still has to reconcile with its report.

    Roughly a third of real `assistant.message` events carry no `apiCallId`,
    so identity matching alone left them counted twice.
    """
    agent = CopilotCli(logs_dir=tmp_path, model_name="gpt-5.4")
    events = [
        {
            "type": "assistant.message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "data": {"messageId": "message-1", "content": "Working", "outputTokens": 5},
        }
    ]
    metrics_events = [
        events[0],
        {
            "type": "assistant.usage",
            "timestamp": "2026-01-01T00:00:01.500Z",
            "data": {"apiCallId": "api-1", "inputTokens": 100, "outputTokens": 5},
        },
    ]

    trajectory = agent._convert_events_to_trajectory(
        events, metrics_events=metrics_events
    )

    assert trajectory is not None
    assert_valid_atif(trajectory)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 5

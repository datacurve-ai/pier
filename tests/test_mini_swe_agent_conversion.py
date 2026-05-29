from pathlib import Path

from pier.agents.installed.mini_swe_agent import (
    MiniSweAgent,
    convert_mini_swe_agent_to_atif,
)


def test_mini_swe_install_uses_patched_litellm(tmp_path: Path):
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openai/gpt-5.5")

    spec = agent.install_spec()
    run = spec.steps[-1].run

    assert spec.steps[-1].env == {"LITELLM_LOCAL_MODEL_COST_MAP": "true"}
    assert MiniSweAgent._PATCHED_LITELLM_PACKAGE in run
    assert "--reinstall-package litellm" in run
    assert MiniSweAgent._LITELLM_MODEL_COST_MAP_URL in run
    assert "BerriAI/litellm/main" not in run
    assert "model_prices_and_context_window_backup.json" in run
    assert "claude-opus-4-8" in run


def test_mini_swe_cost_limit_zero_is_config_override(tmp_path: Path):
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openai/gpt-5.5")

    config_flags = agent._build_config_flags()

    assert "-c mini.yaml" in config_flags
    assert "-c agent.cost_limit=0" in config_flags
    assert "--cost-limit" not in agent.build_cli_flags()


def test_mini_swe_openai_uses_responses_model_class(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        reasoning_effort="xhigh",
    )

    config_flags = agent._build_config_flags()

    assert "-c model.model_class=litellm_response" in config_flags
    assert "-c model.model_kwargs.reasoning_effort=xhigh" in config_flags


def test_mini_swe_model_class_can_be_overridden(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        model_class="litellm",
    )

    assert "-c model.model_class=litellm " in agent._build_config_flags()


def test_mini_swe_model_class_can_be_disabled(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        model_class=None,
    )

    assert "model.model_class" not in agent._build_config_flags()


def test_mini_swe_openrouter_uses_native_model_class(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openrouter/minimax/minimax-m2.7",
    )

    assert agent._run_model_name == "minimax/minimax-m2.7"
    assert "-c model.model_class=openrouter" in agent._build_config_flags()


def test_mini_swe_model_kwargs_pass_through_nested_values(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5-mini",
        model_class="litellm_response_toolcall",
        model_kwargs={
            "drop_params": True,
            "reasoning": {"effort": "high", "summary": "auto"},
        },
    )

    config_flags = agent._build_config_flags()

    assert "-c model.model_class=litellm_response_toolcall" in config_flags
    assert "-c model.model_kwargs.drop_params=true" in config_flags
    assert (
        """-c model.model_kwargs.reasoning='{"effort": "high", "summary": "auto"}'"""
        in config_flags
    )


def test_mini_swe_set_cache_control_can_be_enabled(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openrouter/qwen/qwen3.6-plus",
        set_cache_control="default_end",
    )

    assert "-c model.set_cache_control=default_end" in agent._build_config_flags()


def test_convert_openai_responses_usage_to_atif_metrics():
    trajectory = convert_mini_swe_agent_to_atif(
        {
            "trajectory_format": "mini-swe-agent-1.1",
            "info": {
                "mini_version": "2.2.8",
                "model_stats": {"instance_cost": 0.03, "api_calls": 1},
                "config": {"model": {"model_name": "openai/gpt-5.5"}},
            },
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "object": "response",
                    "model": "gpt-5.5-2026-04-23",
                    "created_at": 1_000,
                    "completed_at": 1_012,
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {"cached_tokens": 25},
                        "output_tokens": 40,
                        "output_tokens_details": {"reasoning_tokens": 30},
                    },
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "I will inspect the workspace.",
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "bash",
                            "arguments": '{"command": "pwd"}',
                        },
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "/app\n",
                    "extra": {
                        "returncode": 0,
                        "exception_info": "",
                        "timestamp": 1_015,
                    },
                },
            ],
        },
        "session",
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 100
    assert trajectory.final_metrics.total_completion_tokens == 40
    assert trajectory.final_metrics.total_cached_tokens == 25
    assert trajectory.final_metrics.total_cost_usd == 0.03
    assert trajectory.final_metrics.extra == {
        "total_reasoning_tokens": 30,
        "total_text_tokens": 10,
        "peak_context_tokens": 100,
    }

    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 1
    assert trajectory.steps[0].timestamp == "1970-01-01T00:16:40+00:00"
    assert trajectory.steps[1].timestamp == "1970-01-01T00:16:40+00:00"
    assert agent_steps[0].timestamp == "1970-01-01T00:16:55+00:00"
    assert agent_steps[0].metrics is not None
    assert agent_steps[0].metrics.prompt_tokens == 100
    assert agent_steps[0].metrics.completion_tokens == 40
    assert agent_steps[0].metrics.cached_tokens == 25
    assert agent_steps[0].metrics.cost_usd == 0.03
    assert agent_steps[0].metrics.extra is not None
    assert agent_steps[0].metrics.extra["text_tokens"] == 10
    assert agent_steps[0].message == "I will inspect the workspace."
    assert agent_steps[0].reasoning_content is None
    assert agent_steps[0].tool_calls is not None
    assert agent_steps[0].tool_calls[0].arguments == {"command": "pwd"}
    assert agent_steps[0].observation is not None
    assert "/app" in agent_steps[0].observation.results[0].content


def test_convert_chat_message_keeps_reasoning_separate_from_visible_content():
    trajectory = convert_mini_swe_agent_to_atif(
        {
            "trajectory_format": "mini-swe-agent-1.1",
            "info": {
                "mini_version": "2.2.8",
                "model_stats": {"instance_cost": 0.01, "api_calls": 1},
                "config": {
                    "model": {"model_name": "anthropic/claude-opus-4-7"}
                },
            },
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "I will create the file.",
                    "reasoning_content": "The task asks for hello.txt.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "touch hello.txt"}',
                            },
                        }
                    ],
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 80,
                                "completion_tokens": 20,
                            }
                        }
                    },
                },
            ],
        },
        "session",
    )

    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0].message == "I will create the file."
    assert agent_steps[0].reasoning_content == "The task asks for hello.txt."


def test_convert_openrouter_byok_uses_upstream_cost_details():
    trajectory = convert_mini_swe_agent_to_atif(
        {
            "trajectory_format": "mini-swe-agent-1.1",
            "info": {
                "mini_version": "2.2.8",
                "model_stats": {"instance_cost": 0.0, "api_calls": 1},
                "config": {
                    "model": {"model_name": "moonshotai/kimi-k2.6"}
                },
            },
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "pwd"}',
                            },
                        }
                    ],
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 1000,
                                "completion_tokens": 100,
                                "cost": 0,
                                "is_byok": True,
                                "prompt_tokens_details": {
                                    "cached_tokens": 800,
                                    "cache_write_tokens": 0,
                                },
                                "cost_details": {
                                    "upstream_inference_cost": 0.00123,
                                },
                            }
                        }
                    },
                },
            ],
        },
        "session",
    )

    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_cost_usd == 0.00123
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert agent_steps[0].metrics is not None
    assert agent_steps[0].metrics.cost_usd == 0.00123

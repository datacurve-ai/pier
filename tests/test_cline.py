import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pier.agents.factory import AgentFactory
from pier.agents.installed.cline import Cline
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName


def test_cline_is_registered(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.CLINE,
        logs_dir=tmp_path,
        model_name="openrouter/qwen/qwen3-coder",
    )
    assert isinstance(agent, Cline)
    assert agent.name() == "cline"


def test_cline_install_bootstraps_node_and_supports_checked_tarball(tmp_path: Path):
    checksum = "a" * 64
    agent = Cline(
        logs_dir=tmp_path,
        model_name="anthropic/claude-sonnet-4",
        tarball_url="https://example.com/cline.tgz",
        tarball_sha256=checksum,
    )

    spec = agent.install_spec()
    assert len(spec.steps) == 2
    assert "apt-get install" in spec.steps[0].run
    assert "nvm install 22" in spec.steps[1].run
    assert "https://example.com/cline.tgz" in spec.steps[1].run
    assert checksum in spec.steps[1].run
    assert "sha256sum -c" in spec.steps[1].run
    assert "api.anthropic.com" in agent.network_allowlist().domains


def test_cline_rejects_invalid_tarball_checksum(tmp_path: Path):
    with pytest.raises(ValueError, match="64-character SHA-256"):
        Cline(
            logs_dir=tmp_path,
            model_name="openrouter/qwen/qwen3-coder",
            tarball_url="https://example.com/cline.tgz",
            tarball_sha256="not-a-checksum",
        )


def test_cline_builds_current_headless_cli_command(tmp_path: Path):
    agent = Cline(
        logs_dir=tmp_path,
        model_name="openrouter:openai/gpt-5.6-sol",
        extra_env={"API_KEY": "test-key"},
        **{
            "reasoning-effort": "high",
            "max-consecutive-mistakes": "6",
        },
    )

    command, env = agent._build_run_command("Create /app/out.txt with `ok`")

    assert "cline --json --auto-approve true --provider openrouter" in command
    assert '--key "$PIER_CLINE_API_KEY"' in command
    assert "--model openai/gpt-5.6-sol" in command
    assert "--thinking high" in command
    assert "--retries 6" in command
    assert "--max-consecutive-mistakes" not in command
    assert "-- 'Create /app/out.txt with `ok`'" in command
    assert env["OPENROUTER_API_KEY"] == "test-key"
    assert env["PIER_CLINE_API_KEY"] == "test-key"
    assert env["CLINE_WRITE_PROMPT_ARTIFACTS"] == "1"
    assert "test-key" not in command


def test_cline_requires_provider_key(tmp_path: Path):
    agent = Cline(
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5.6-sol",
        extra_env={},
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key in ("OPENROUTER_API_KEY", "API_KEY"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY or API_KEY"):
            agent._build_run_command("Solve the task")


@pytest.mark.asyncio
async def test_cline_preserves_sessions_after_run(tmp_path: Path):
    agent = Cline(
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5.6-sol",
        extra_env={"OPENROUTER_API_KEY": "test-key"},
    )
    environment = MagicMock()
    environment.agent_process_env.side_effect = lambda env: env or {}
    environment.exec = AsyncMock(
        return_value=ExecResult(return_code=0, stdout="", stderr="")
    )

    await agent.run("Solve the task", environment, AgentContext())

    commands = [call.kwargs["command"] for call in environment.exec.call_args_list]
    assert len(commands) == 2
    assert "cline --json --auto-approve true" in commands[0]
    assert "cp -r ~/.cline/data/sessions/*" in commands[1]


def test_cline_converts_native_session_to_atif(tmp_path: Path):
    session_dir = tmp_path / "sessions" / "session-from-directory"
    session_dir.mkdir(parents=True)
    (session_dir / "session-from-directory.messages.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Inspect the repository",
                        "ts": 1000,
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "text": "Need project context"},
                            {"type": "text", "text": "Reading README"},
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "read_file",
                                "input": {"path": "README.md"},
                            },
                        ],
                        "ts": 2000,
                        "modelInfo": {"id": "openai/gpt-5.6-sol"},
                        "metrics": {
                            "inputTokens": 100,
                            "outputTokens": 20,
                            "cacheReadTokens": 5,
                            "cacheWriteTokens": 2,
                            "cost": 0.01,
                        },
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "# Pier",
                            }
                        ],
                        "ts": 3000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    agent = Cline(
        logs_dir=tmp_path,
        model_name="openrouter/openai/gpt-5.6-sol",
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == "session-from-directory"
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "read_file"
    assert trajectory["steps"][1]["observation"]["results"][0]["content"] == "# Pier"
    assert trajectory["steps"][1]["reasoning_content"] == "Need project context"
    assert trajectory["steps"][1]["llm_call_count"] == 1
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100
    assert trajectory["final_metrics"]["total_completion_tokens"] == 20
    assert trajectory["final_metrics"]["total_cached_tokens"] == 5
    assert trajectory["final_metrics"]["total_cost_usd"] == pytest.approx(0.01)
    assert trajectory["final_metrics"]["extra"]["peak_context_tokens"] == 100
    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 5
    assert context.n_output_tokens == 20
    assert context.cost_usd == pytest.approx(0.01)


def test_cline_uses_session_id_from_document(tmp_path: Path):
    agent = Cline(logs_dir=tmp_path, model_name="cline/claude-sonnet-4")
    trajectory = agent._convert_session_to_trajectory(
        {
            "sessionId": "session-from-document",
            "messages": [{"role": "user", "content": "hello"}],
        },
        session_id="session-from-directory",
    )
    assert trajectory.session_id == "session-from-document"

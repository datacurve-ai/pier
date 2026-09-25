from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.base import NonZeroAgentExitCodeError
from pier.agents.installed.codex import (
    Codex,
    _is_model_capacity_failure,
    _root_thread_id,
)


CAPACITY_EVENTS = """\
{"type":"error","message":"Selected model is at capacity. Please try a different model."}
{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a different model."}}
"""


def test_detects_only_exact_model_capacity_errors():
    assert _is_model_capacity_failure(CAPACITY_EVENTS)
    assert not _is_model_capacity_failure(
        '{"type":"turn.failed","error":{"message":"Authentication failed"}}'
    )
    assert not _is_model_capacity_failure("Selected model is at capacity")


def test_root_thread_id_uses_first_started_thread():
    output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"root-thread"}',
            '{"type":"thread.started","thread_id":"child-thread"}',
        ]
    )

    assert _root_thread_id(output) == "root-thread"


@pytest.mark.asyncio
async def test_capacity_failure_resumes_root_thread(monkeypatch, tmp_path):
    agent = Codex(logs_dir=tmp_path, model_name="openai/gpt-test")
    agent.exec_as_agent = AsyncMock(
        side_effect=[
            NonZeroAgentExitCodeError("capacity"),
            SimpleNamespace(stdout=CAPACITY_EVENTS),
            SimpleNamespace(
                stdout='{"type":"thread.started","thread_id":"root-thread"}\n'
            ),
            SimpleNamespace(stdout=""),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("pier.agents.installed.codex.asyncio.sleep", sleep)

    await agent._run_with_capacity_resume(
        environment=object(),
        initial_command="codex exec initial",
        resume_command_prefix="codex exec resume --json ",
        env={"CODEX_HOME": "/tmp/codex-home"},
    )

    sleep.assert_awaited_once_with(30)
    resume_command = agent.exec_as_agent.await_args_list[-1].kwargs["command"]
    assert "codex exec resume --json root-thread" in resume_command
    assert "tee -a /logs/agent/codex.txt" in resume_command
    assert "--last" not in resume_command


@pytest.mark.asyncio
async def test_non_capacity_failure_is_not_retried(monkeypatch, tmp_path):
    agent = Codex(logs_dir=tmp_path, model_name="openai/gpt-test")
    failure = NonZeroAgentExitCodeError("failed")
    agent.exec_as_agent = AsyncMock(
        side_effect=[
            failure,
            SimpleNamespace(
                stdout=(
                    '{"type":"turn.failed","error":{"message":'
                    '"Authentication failed"}}\n'
                )
            ),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("pier.agents.installed.codex.asyncio.sleep", sleep)

    with pytest.raises(NonZeroAgentExitCodeError) as exc_info:
        await agent._run_with_capacity_resume(
            environment=object(),
            initial_command="codex exec initial",
            resume_command_prefix="codex exec resume --json ",
            env={},
        )

    assert exc_info.value is failure
    sleep.assert_not_awaited()

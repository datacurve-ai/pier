"""Tests for the optional ``[[verifier.collect]]`` task hooks.

Collect hooks run inside the agent environment after the agent finishes and
immediately before artifact collection, so tasks can materialize artifacts
(e.g. capture the agent's change set as /logs/artifacts/model.patch for a
separate verifier).
"""

import asyncio
import functools
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pier.models.task.config import StepConfig, VerifierCollectConfig, VerifierConfig
from pier.trial.trial import Trial


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _fake_trial(hooks: list[VerifierCollectConfig]) -> SimpleNamespace:
    environment = AsyncMock()
    environment.exec.return_value = SimpleNamespace(return_code=0)
    return SimpleNamespace(
        _task=SimpleNamespace(
            config=SimpleNamespace(verifier=VerifierConfig(collect=hooks))
        ),
        _environment=environment,
        _logger=logging.getLogger(__name__),
    )


@run_async
async def test_no_hooks_is_a_noop() -> None:
    trial = _fake_trial([])
    await Trial._run_collect_hooks(trial)
    trial._environment.exec.assert_not_awaited()


@run_async
async def test_hook_is_executed() -> None:
    hook = VerifierCollectConfig(
        command="echo hi > /logs/artifacts/out.txt",
        timeout_sec=120.0,
        user="root",
    )
    trial = _fake_trial([hook])
    await Trial._run_collect_hooks(trial)
    trial._environment.exec.assert_awaited_once_with(
        command="echo hi > /logs/artifacts/out.txt",
        timeout_sec=120,
        user="root",
    )


@run_async
async def test_step_hooks_run_after_task_hooks() -> None:
    task_hook = VerifierCollectConfig(command="task-hook")
    step_hook = VerifierCollectConfig(command="step-hook")
    trial = _fake_trial([task_hook])
    step_cfg = StepConfig(name="grade", verifier=VerifierConfig(collect=[step_hook]))
    await Trial._run_collect_hooks(trial, step_cfg)
    commands = [
        call.kwargs["command"] for call in trial._environment.exec.await_args_list
    ]
    assert commands == ["task-hook", "step-hook"]


@run_async
async def test_sidecar_hook_is_skipped() -> None:
    hook = VerifierCollectConfig(command="pg_dump app", service="postgres")
    trial = _fake_trial([hook])
    await Trial._run_collect_hooks(trial)
    trial._environment.exec.assert_not_awaited()


@run_async
async def test_nonzero_exit_does_not_raise() -> None:
    trial = _fake_trial([VerifierCollectConfig(command="exit 3")])
    trial._environment.exec.return_value = SimpleNamespace(return_code=3)
    await Trial._run_collect_hooks(trial)  # must not raise


@run_async
async def test_exec_exception_is_swallowed() -> None:
    trial = _fake_trial([VerifierCollectConfig(command="boom")])
    trial._environment.exec.side_effect = RuntimeError("env died")
    await Trial._run_collect_hooks(trial)  # must not raise

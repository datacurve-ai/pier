"""The shared verifier's directories must be cleared before verification.

``/logs/verifier`` is created and chmod 777'd at environment start, so for a
shared verifier it stays writable for the whole agent phase. If it is not
cleared before the verifier runs, a file left behind there — a stale reward
from an earlier attempt, or one the agent wrote itself — is what
``Verifier.verify`` reads back as the trial's score.

``_run_steps`` already resets per step; these cover the single-step path.
"""

import asyncio
import functools
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pier.models.task.config import (
    EnvironmentConfig,
    TaskConfig,
    VerifierConfig,
    VerifierEnvironmentMode,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.trial.trial import Trial


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _fake_trial(verifier: VerifierConfig) -> SimpleNamespace:
    environment = AsyncMock()
    environment.env_paths = EnvironmentPaths()
    return SimpleNamespace(
        _task=SimpleNamespace(
            config=TaskConfig(
                name="demo/t",
                version="1.0.0",
                verifier=verifier,
            )
        ),
        _environment=environment,
        _logger=logging.getLogger(__name__),
    )


@run_async
async def test_shared_verifier_dirs_are_reset() -> None:
    trial = _fake_trial(VerifierConfig())
    await Trial._reset_shared_verifier_dirs(trial)

    env_paths = EnvironmentPaths()
    trial._environment.reset_dirs.assert_awaited_once_with(
        remove_dirs=[env_paths.verifier_dir, env_paths.tests_dir],
        create_dirs=[env_paths.verifier_dir, env_paths.tests_dir],
        chmod_dirs=[env_paths.verifier_dir],
    )


@run_async
async def test_planted_reward_does_not_survive_into_verification() -> None:
    """The reset must remove the directory, not just recreate it.

    An agent writing /logs/verifier/reward.txt during its own turn is the case
    this guards; `rm -rf` on the dir is what actually drops that file.
    """
    trial = _fake_trial(VerifierConfig())
    await Trial._reset_shared_verifier_dirs(trial)

    kwargs = trial._environment.reset_dirs.await_args.kwargs
    assert EnvironmentPaths().verifier_dir in kwargs["remove_dirs"]


@run_async
async def test_separate_verifier_environment_is_left_alone() -> None:
    """Separate mode gets a fresh environment that empties itself, and the
    agent environment may already be stopped by this point."""
    trial = _fake_trial(
        VerifierConfig(
            environment_mode=VerifierEnvironmentMode.SEPARATE,
            environment=EnvironmentConfig(),
        )
    )
    await Trial._reset_shared_verifier_dirs(trial)
    trial._environment.reset_dirs.assert_not_awaited()


@run_async
async def test_separate_mode_inferred_from_environment_block() -> None:
    """environment_mode is optional; a [verifier.environment] block implies it."""
    trial = _fake_trial(VerifierConfig(environment=EnvironmentConfig()))
    await Trial._reset_shared_verifier_dirs(trial)
    trial._environment.reset_dirs.assert_not_awaited()


@run_async
async def test_run_verification_resets_before_verifying() -> None:
    """The reset must be wired into _run_verification, and must land before
    the verifier runs — not after it has already read the reward file."""
    order: list[str] = []
    trial = _fake_trial(VerifierConfig())
    trial._invoke_hooks = AsyncMock()
    trial.result = SimpleNamespace(verifier=None)
    # bind the real reset so this exercises the wiring, not a stub of it
    trial._reset_shared_verifier_dirs = functools.partial(
        Trial._reset_shared_verifier_dirs, trial
    )
    trial._environment.reset_dirs = AsyncMock(
        side_effect=lambda **_: order.append("reset")
    )
    trial._verify_with_retry = AsyncMock(side_effect=lambda: order.append("verify"))

    await Trial._run_verification(trial)

    assert order == ["reset", "verify"]

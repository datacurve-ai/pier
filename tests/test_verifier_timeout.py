import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pier.models.job.config import RetryConfig
from pier.models.task.config import (
    VerifierConfig as TaskVerifierConfig,
)
from pier.models.task.task import Task
from pier.models.trial.config import (
    EnvironmentConfig,
    TaskConfig as TrialTaskConfig,
    TrialConfig,
    VerifierConfig as TrialVerifierConfig,
)
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import StepResult, TimingInfo, TrialResult
from pier.models.verifier.result import VerifierResult
from pier.trial.trial import (
    Trial,
    VerifierArtifactTransferTimeoutError,
    VerifierEnvironmentStartTimeoutError,
    VerifierEnvironmentTimeoutError,
    VerifierInfrastructureTimeoutError,
    VerifierRewardDownloadTimeoutError,
    VerifierRewardParseTimeoutError,
    VerifierRewardTimeoutError,
    VerifierTestExecutionTimeoutError,
    VerifierTimeoutError,
)
from pier.verifier.verifier import Verifier


def _create_dummy_task_dir(tmpdir: Path) -> Path:
    task_dir = tmpdir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text("[environment]\nos = 'linux'\n")
    (task_dir / "instruction.md").write_text("Test task instruction\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


# ============================================================================
# 1. Exception Hierarchy Tests
# ============================================================================


def test_verifier_timeout_exception_hierarchy():
    """Verify timeout subphases inherit from VerifierTimeoutError and asyncio.TimeoutError."""
    assert issubclass(VerifierTimeoutError, asyncio.TimeoutError)
    assert issubclass(VerifierEnvironmentTimeoutError, VerifierTimeoutError)
    assert issubclass(VerifierArtifactTransferTimeoutError, VerifierTimeoutError)
    assert issubclass(VerifierTestExecutionTimeoutError, VerifierTimeoutError)
    assert issubclass(VerifierRewardParseTimeoutError, VerifierTimeoutError)

    # Aliases
    assert VerifierEnvironmentStartTimeoutError is VerifierEnvironmentTimeoutError
    assert VerifierInfrastructureTimeoutError is VerifierArtifactTransferTimeoutError
    assert VerifierRewardTimeoutError is VerifierRewardParseTimeoutError
    assert VerifierRewardDownloadTimeoutError is VerifierRewardParseTimeoutError


# ============================================================================
# 2. Config & Backward Compatibility Tests
# ============================================================================


def test_verifier_config_defaults_and_max_attempts():
    """Verify default max_attempts and custom values."""
    cfg = TrialVerifierConfig()
    assert cfg.max_attempts == 2

    cfg_single = TrialVerifierConfig(max_attempts=1)
    assert cfg_single.max_attempts == 1

    # max_retries backward compatibility
    cfg_retries_0 = TrialVerifierConfig.model_validate({"max_retries": 0})
    assert cfg_retries_0.max_attempts == 1

    cfg_retries_2 = TrialVerifierConfig.model_validate({"max_retries": 2})
    assert cfg_retries_2.max_attempts == 3

    # TaskVerifierConfig
    task_vcfg = TaskVerifierConfig()
    assert task_vcfg.max_attempts is None
    assert "currently wraps environment startup" in TaskVerifierConfig.model_fields["timeout_sec"].description


def test_retry_config_excludes_candidate_test_timeout():
    """Verify candidate test timeout and reward parse timeouts are excluded from outer job retries."""
    retry_cfg = RetryConfig()
    assert "VerifierTestExecutionTimeoutError" in retry_cfg.exclude_exceptions
    assert "VerifierRewardParseTimeoutError" in retry_cfg.exclude_exceptions
    assert "VerifierEnvironmentTimeoutError" not in retry_cfg.exclude_exceptions
    assert "VerifierArtifactTransferTimeoutError" not in retry_cfg.exclude_exceptions


# ============================================================================
# 3. Verifier Subphase Timeout Tests
# ============================================================================


@pytest.mark.asyncio
async def test_verifier_test_execution_timeout():
    """Candidate test execution command timing out raises VerifierTestExecutionTimeoutError."""
    mock_env = MagicMock()
    mock_env.env_paths.tests_dir = Path("/tests")
    mock_env.env_paths.verifier_dir = Path("/verifier")
    mock_env.capabilities.mounted = True

    async def mock_exec(command, *args, **kwargs):
        if "chmod" in command:
            return
        await asyncio.sleep(0.2)

    mock_env.exec = AsyncMock(side_effect=mock_exec)

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_paths = TrialPaths(Path(tmpdir))
        task_dir = _create_dummy_task_dir(Path(tmpdir))

        task = Task(task_dir)
        verifier = Verifier(
            task=task,
            trial_paths=trial_paths,
            environment=mock_env,
            skip_tests_upload=True,
        )

        with pytest.raises(VerifierTestExecutionTimeoutError):
            await verifier.verify(timeout_sec=0.05)


@pytest.mark.asyncio
async def test_verifier_upload_timeout():
    """Test upload timing out raises VerifierArtifactTransferTimeoutError."""
    mock_env = MagicMock()
    mock_env.env_paths.tests_dir = Path("/tests")
    mock_env.env_paths.verifier_dir = Path("/verifier")

    async def mock_upload(*args, **kwargs):
        await asyncio.sleep(0.2)

    mock_env.upload_dir = AsyncMock(side_effect=mock_upload)

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_paths = TrialPaths(Path(tmpdir))
        task_dir = _create_dummy_task_dir(Path(tmpdir))

        task = Task(task_dir)
        verifier = Verifier(
            task=task,
            trial_paths=trial_paths,
            environment=mock_env,
            skip_tests_upload=False,
        )

        with pytest.raises(VerifierArtifactTransferTimeoutError):
            await verifier.verify(timeout_sec=0.05)


# ============================================================================
# 4. Trial Retry Behavior & Durable State Tests
# ============================================================================


@pytest.mark.asyncio
async def test_trial_candidate_test_timeout_is_not_retried():
    """A candidate test execution timeout must NOT be retried even if max_attempts=2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = _create_dummy_task_dir(Path(tmpdir))
        task = Task(task_dir)
        trials_dir = Path(tmpdir) / "trials"
        trial_cfg = TrialConfig(
            trial_name="test_trial",
            trials_dir=trials_dir,
            task=TrialTaskConfig(path=task_dir),
            environment=EnvironmentConfig(type="docker"),
            verifier=TrialVerifierConfig(max_attempts=2),
        )

        trial = Trial(trial_cfg, _task=task)
        try:
            trial._result = TrialResult(
                trial_name="test_trial",
                task_name="task",
                task_id=trial_cfg.task.get_task_id(),
                started_at=TimingInfo().started_at,
                config=trial_cfg,
                task_checksum="dummy",
                trial_uri=trial._trial_paths.trial_dir.as_uri(),
                agent_info=trial._agent.to_agent_info(),
                source="test",
            )
            trial.result.verifier = TimingInfo()

            verify_mock = AsyncMock(
                side_effect=VerifierTestExecutionTimeoutError("Candidate test timed out")
            )
            trial._verify_once = verify_mock

            with pytest.raises(VerifierTestExecutionTimeoutError):
                await trial._verify_with_retry()

            # Candidate test timeout should NOT be retried!
            assert verify_mock.call_count == 1
            assert trial.result.verifier_attempt == 1
            assert trial.result.max_verifier_attempts == 2

            # Check progress.json
            progress_path = trial._trial_paths.progress_path
            assert progress_path.exists()
            progress_data = json.loads(progress_path.read_text())
            assert progress_data["verifier_attempt"] == 1
            assert progress_data["max_verifier_attempts"] == 2
        finally:
            trial._close_logger_handler()


@pytest.mark.asyncio
async def test_trial_infrastructure_timeout_retried_up_to_max_attempts():
    """Infrastructure timeouts (e.g. environment startup or artifact transfer) ARE retried up to max_attempts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = _create_dummy_task_dir(Path(tmpdir))
        task = Task(task_dir)
        trials_dir = Path(tmpdir) / "trials"
        trial_cfg = TrialConfig(
            trial_name="test_trial",
            trials_dir=trials_dir,
            task=TrialTaskConfig(path=task_dir),
            environment=EnvironmentConfig(type="docker"),
            verifier=TrialVerifierConfig(max_attempts=2),
        )

        trial = Trial(trial_cfg, _task=task)
        try:
            trial._result = TrialResult(
                trial_name="test_trial",
                task_name="task",
                task_id=trial_cfg.task.get_task_id(),
                started_at=TimingInfo().started_at,
                config=trial_cfg,
                task_checksum="dummy",
                trial_uri=trial._trial_paths.trial_dir.as_uri(),
                agent_info=trial._agent.to_agent_info(),
                source="test",
            )
            trial.result.verifier = TimingInfo()

            # First attempt fails with infra timeout, second succeeds
            verify_mock = AsyncMock(
                side_effect=[
                    VerifierArtifactTransferTimeoutError("Transfer timed out"),
                    VerifierResult(rewards={"reward": 1.0}),
                ]
            )
            trial._verify_once = verify_mock

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await trial._verify_with_retry()

            assert verify_mock.call_count == 2
            assert trial.result.verifier_attempt == 2
            assert trial.result.max_verifier_attempts == 2
            assert trial.result.verifier.attempt == 2
            assert trial.result.verifier.max_attempts == 2
            assert trial.result.verifier_result.rewards == {"reward": 1.0}
        finally:
            trial._close_logger_handler()


@pytest.mark.asyncio
async def test_trial_max_attempts_one_disables_retries():
    """Setting max_attempts=1 disables all retries even on infrastructure timeouts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = _create_dummy_task_dir(Path(tmpdir))
        task = Task(task_dir)
        trials_dir = Path(tmpdir) / "trials"
        trial_cfg = TrialConfig(
            trial_name="test_trial",
            trials_dir=trials_dir,
            task=TrialTaskConfig(path=task_dir),
            environment=EnvironmentConfig(type="docker"),
            verifier=TrialVerifierConfig(max_attempts=1),
        )

        trial = Trial(trial_cfg, _task=task)
        try:
            trial._result = TrialResult(
                trial_name="test_trial",
                task_name="task",
                task_id=trial_cfg.task.get_task_id(),
                started_at=TimingInfo().started_at,
                config=trial_cfg,
                task_checksum="dummy",
                trial_uri=trial._trial_paths.trial_dir.as_uri(),
                agent_info=trial._agent.to_agent_info(),
                source="test",
            )
            trial.result.verifier = TimingInfo()

            verify_mock = AsyncMock(
                side_effect=VerifierEnvironmentTimeoutError("Env startup timed out")
            )
            trial._verify_once = verify_mock

            with pytest.raises(VerifierEnvironmentTimeoutError):
                await trial._verify_with_retry()

            assert verify_mock.call_count == 1
            assert trial.result.verifier_attempt == 1
            assert trial.result.max_verifier_attempts == 1
        finally:
            trial._close_logger_handler()


# ============================================================================
# 5. StepResult & TimingInfo Models
# ============================================================================


def test_step_result_and_timing_info_attempts():
    timing = TimingInfo(attempt=1, max_attempts=2)
    assert timing.attempt == 1
    assert timing.max_attempts == 2

    step = StepResult(
        step_name="step1",
        verifier_attempt=1,
        max_verifier_attempts=2,
    )
    assert step.step_name == "step1"
    assert step.verifier_attempt == 1
    assert step.max_verifier_attempts == 2


@pytest.mark.asyncio
async def test_verifier_reward_download_timeout():
    """Verifier reward download timing out raises VerifierRewardParseTimeoutError."""
    mock_env = MagicMock()
    mock_env.env_paths.tests_dir = Path("/tests")
    mock_env.env_paths.verifier_dir = Path("/verifier")
    mock_env.capabilities.mounted = False

    async def mock_exec(command, *args, **kwargs):
        return

    async def mock_download(*args, **kwargs):
        await asyncio.sleep(0.2)

    mock_env.exec = AsyncMock(side_effect=mock_exec)
    mock_env.download_dir = AsyncMock(side_effect=mock_download)

    with tempfile.TemporaryDirectory() as tmpdir:
        trial_paths = TrialPaths(Path(tmpdir))
        task_dir = _create_dummy_task_dir(Path(tmpdir))

        task = Task(task_dir)
        verifier = Verifier(
            task=task,
            trial_paths=trial_paths,
            environment=mock_env,
            skip_tests_upload=True,
        )

        with pytest.raises(VerifierRewardParseTimeoutError):
            await verifier.verify(timeout_sec=0.05)


@pytest.mark.asyncio
async def test_trial_durable_result_json_and_progress_json():
    """Verify result.json and progress.json contain verifier_attempt and max_verifier_attempts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = _create_dummy_task_dir(Path(tmpdir))
        task = Task(task_dir)
        trials_dir = Path(tmpdir) / "trials"
        trial_cfg = TrialConfig(
            trial_name="test_trial",
            trials_dir=trials_dir,
            task=TrialTaskConfig(path=task_dir),
            environment=EnvironmentConfig(type="docker"),
            verifier=TrialVerifierConfig(max_attempts=3),
        )

        trial = Trial(trial_cfg, _task=task)
        try:
            trial._result = TrialResult(
                trial_name="test_trial",
                task_name="task",
                task_id=trial_cfg.task.get_task_id(),
                started_at=TimingInfo().started_at,
                config=trial_cfg,
                task_checksum="dummy",
                trial_uri=trial._trial_paths.trial_dir.as_uri(),
                agent_info=trial._agent.to_agent_info(),
                source="test",
            )
            trial.result.verifier_attempt = 2
            trial.result.max_verifier_attempts = 3
            trial._stop_agent_environment = AsyncMock()

            await trial._cleanup_and_finalize()

            assert trial._trial_paths.result_path.exists()
            result_data = json.loads(trial._trial_paths.result_path.read_text())
            assert result_data["verifier_attempt"] == 2
            assert result_data["max_verifier_attempts"] == 3

            assert trial._trial_paths.progress_path.exists()
            progress_data = json.loads(trial._trial_paths.progress_path.read_text())
            assert progress_data["phase"] == "completed"
            assert progress_data["verifier_attempt"] == 2
            assert progress_data["max_verifier_attempts"] == 3
        finally:
            trial._close_logger_handler()


def test_cli_verifier_max_attempts_option():
    """Verify --verifier-max-attempts CLI option is exposed in pier run --help."""
    from typer.testing import CliRunner
    from pier.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--verifier-max-attempts" in result.output



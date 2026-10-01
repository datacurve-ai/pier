from pathlib import Path
from uuid import uuid4

from pier.models.job.result import JobStats
from pier.models.trial.result import AgentInfo, TrialResult
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import (
    AgentConfig, EnvironmentConfig, 
    TaskConfig, TrialConfig, VerifierConfig
)
from pier.models.agent.context import AgentContext


def _make_trial_result(*, n_agent_steps: int | None = None, reward: float | None = None) -> TrialResult:
    agent_result = AgentContext(n_agent_steps=n_agent_steps) if n_agent_steps is not None else None
    return TrialResult(
        id=uuid4(),
        task_name="task",
        trial_name="trial__abc",
        trial_uri="file:///tmp/trial",
        task_id=LocalTaskId(path=Path("/tmp/task")),
        task_checksum="abc",
        config=TrialConfig(
            trial_name="trial__abc",
            trials_dir=Path("/tmp"),
            task=TaskConfig(path=Path("/tmp/task")),
            agent=AgentConfig(name="oracle"),
            environment=EnvironmentConfig(),
            verifier=VerifierConfig(),
        ),
        agent_info=AgentInfo(name="oracle", version="1"),
        agent_result=agent_result,
    )


def test_increment_increases_completed_count():
    stats = JobStats()
    stats.increment(_make_trial_result())
    assert stats.n_completed_trials == 1


def test_remove_trial_reverses_increment():
    stats = JobStats()
    trial = _make_trial_result()
    stats.increment(trial)
    stats.remove_trial(trial)
    assert stats.n_completed_trials == 0


def test_increment_accumulates_agent_steps():
    stats = JobStats()
    stats.increment(_make_trial_result(n_agent_steps=10))
    stats.increment(_make_trial_result(n_agent_steps=5))
    assert stats.n_agent_steps == 15


def test_increment_skips_none_agent_steps():
    stats = JobStats()
    stats.increment(_make_trial_result(n_agent_steps=None))
    assert stats.n_agent_steps is None


def test_remove_trial_subtracts_agent_steps():
    stats = JobStats()
    trial = _make_trial_result(n_agent_steps=10)
    stats.increment(trial)
    stats.remove_trial(trial)
    assert stats.n_agent_steps == 0


def test_agent_steps_not_negative_after_remove():
    stats = JobStats()
    trial = _make_trial_result(n_agent_steps=3)
    stats.increment(trial)
    stats.remove_trial(trial)
    stats.remove_trial(trial)  # double-remove should not go below 0
    assert stats.n_agent_steps == 0


def test_old_result_json_loads_without_agent_steps():
    old_stats = {"n_completed_trials": 5, "n_errored_trials": 1}
    stats = JobStats.model_validate(old_stats)
    assert stats.n_agent_steps is None  # backward compat: field absent = None


def test_increment_mixed_none_and_non_none_agent_steps():
    stats = JobStats()
    stats.increment(_make_trial_result(n_agent_steps=10))
    stats.increment(_make_trial_result(n_agent_steps=None)) # Should not reset
    stats.increment(_make_trial_result(n_agent_steps=5))
    assert stats.n_agent_steps == 15


def test_update_trial_replaces_agent_steps():
    stats = JobStats()
    original = _make_trial_result(n_agent_steps=10)
    retry = _make_trial_result(n_agent_steps=4)
    stats.increment(original)
    stats.update_trial(retry, previous_result=original)
    assert stats.n_agent_steps == 4

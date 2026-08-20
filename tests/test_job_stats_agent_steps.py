"""Tests for JobStats.n_agent_steps accumulation across trials."""

from pathlib import Path

from pier.models.job.result import JobStats
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import TaskConfig, TrialConfig
from pier.models.trial.result import AgentInfo, TrialResult


def _make_trial_result(
    trial_name: str = "task__abc123",
    n_agent_steps: int | None = None,
) -> TrialResult:
    """Create a minimal TrialResult for testing."""
    task_id = LocalTaskId(path=Path("/tmp/task"))
    task_config = TaskConfig(path=Path("/tmp/task"), source="test")
    config = TrialConfig(
        task=task_config,
        trial_name=trial_name,
    )
    return TrialResult(
        task_name="task",
        trial_name=trial_name,
        trial_uri="file:///tmp/task",
        task_id=task_id,
        source="test",
        task_checksum="abc123",
        config=config,
        agent_info=AgentInfo(name="test-agent", version="1"),
        n_agent_steps=n_agent_steps,
    )


def test_increment_accumulates_agent_steps() -> None:
    stats = JobStats()
    stats.increment(_make_trial_result("trial_1", n_agent_steps=5))
    stats.increment(_make_trial_result("trial_2", n_agent_steps=10))
    assert stats.n_agent_steps == 15


def test_increment_skips_none_agent_steps() -> None:
    stats = JobStats()
    stats.increment(_make_trial_result("trial_1", n_agent_steps=None))
    assert stats.n_agent_steps is None


def test_increment_mixed_none_and_non_none_agent_steps() -> None:
    stats = JobStats()
    stats.increment(_make_trial_result("trial_1", n_agent_steps=None))
    stats.increment(_make_trial_result("trial_2", n_agent_steps=15))
    stats.increment(_make_trial_result("trial_3", n_agent_steps=None))
    assert stats.n_agent_steps == 15


def test_remove_trial_subtracts_agent_steps() -> None:
    stats = JobStats()
    trial_1 = _make_trial_result("trial_1", n_agent_steps=10)
    trial_2 = _make_trial_result("trial_2", n_agent_steps=5)
    stats.increment(trial_1)
    stats.increment(trial_2)
    assert stats.n_agent_steps == 15

    stats.remove_trial(trial_1)
    assert stats.n_agent_steps == 5


def test_agent_steps_not_negative_after_remove() -> None:
    stats = JobStats()
    trial = _make_trial_result("trial_1", n_agent_steps=5)
    stats.increment(trial)
    # Simulate removing more than accumulated (edge case)
    big_trial = _make_trial_result("trial_2", n_agent_steps=100)
    stats.increment(big_trial)
    stats.remove_trial(big_trial)
    assert stats.n_agent_steps == 5
    stats.remove_trial(trial)
    assert stats.n_agent_steps == 0


def test_remove_trial_with_none_steps_leaves_stats_unchanged() -> None:
    stats = JobStats()
    trial_with_steps = _make_trial_result("trial_1", n_agent_steps=10)
    trial_without_steps = _make_trial_result("trial_2", n_agent_steps=None)
    stats.increment(trial_with_steps)
    stats.increment(trial_without_steps)
    assert stats.n_agent_steps == 10

    stats.remove_trial(trial_without_steps)
    assert stats.n_agent_steps == 10


def test_update_trial_replaces_agent_steps() -> None:
    stats = JobStats()
    old_trial = _make_trial_result("trial_1", n_agent_steps=10)
    new_trial = _make_trial_result("trial_1", n_agent_steps=4)
    stats.increment(old_trial)
    assert stats.n_agent_steps == 10

    stats.update_trial(new_trial, previous_result=old_trial)
    assert stats.n_agent_steps == 4


def test_from_trial_results_accumulates_agent_steps() -> None:
    trials = [
        _make_trial_result("trial_1", n_agent_steps=5),
        _make_trial_result("trial_2", n_agent_steps=10),
        _make_trial_result("trial_3", n_agent_steps=None),
    ]
    stats = JobStats.from_trial_results(trials)
    assert stats.n_agent_steps == 15
    assert stats.n_completed_trials == 3


def test_old_result_json_loads_without_agent_steps() -> None:
    """Backward compatibility: legacy data without n_agent_steps loads fine."""
    stats = JobStats.model_validate({"n_completed_trials": 5})
    assert stats.n_agent_steps is None

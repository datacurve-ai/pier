from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from pier.models.agent.context import AgentContext
from pier.models.job.result import JobResult, JobStats
from pier.models.task.id import LocalTaskId
from pier.models.trial.config import TaskConfig, TrialConfig
from pier.models.trial.result import AgentInfo, StepResult, TrialResult
from pier.models.verifier.result import VerifierResult
from pier.viewer.server import create_app


def trial(*contexts: AgentContext | None) -> TrialResult:
    return TrialResult(
        task_name="task",
        trial_name=str(uuid4()),
        trial_uri="file:///task",
        task_id=LocalTaskId(path=Path("/task")),
        task_checksum="checksum",
        config=TrialConfig(task=TaskConfig(path=Path("/task"))),
        agent_info=AgentInfo(name="mini-swe-agent", version="2.4.6"),
        step_results=[
            StepResult(step_name=str(i), agent_result=context)
            for i, context in enumerate(contexts)
        ],
        verifier_result=VerifierResult(rewards={"reward": 1}),
    )


def usage(value: int | None) -> AgentContext:
    return AgentContext(
        n_input_tokens=value,
        n_cache_tokens=value,
        n_output_tokens=value,
        cost_usd=value,
        n_agent_steps=1,
    )


def totals(stats: JobStats) -> tuple[int | None, int | None, int | None, float | None]:
    saved = JobStats.model_validate_json(stats.model_dump_json())
    return (
        saved.n_input_tokens,
        saved.n_cache_tokens,
        saved.n_output_tokens,
        saved.cost_usd,
    )


@pytest.mark.parametrize("missing_first", [False, True])
@pytest.mark.parametrize(
    "field", ["n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"]
)
def test_trial_totals_require_every_context_to_report_each_counter(
    field, missing_first
):
    partial = usage(2).model_copy(update={field: None})
    contexts = [partial, usage(3)] if missing_first else [usage(3), partial]
    expected = tuple(
        None if name == field else 5
        for name in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")
    )
    result = trial(*contexts)

    assert result.compute_token_cost_totals() == expected
    assert result.agent_step_count() == 2


@pytest.mark.parametrize("value", [None, 0, 3])
def test_trial_totals_preserve_explicit_zero_and_skip_non_agent_steps(value):
    result = trial(usage(value), None, usage(value))

    assert (
        result.compute_token_cost_totals()
        == (None if value is None else 2 * value,) * 4
    )
    assert result.agent_step_count() == 2


def test_empty_trial_has_no_usage():
    assert trial().compute_token_cost_totals() == (None,) * 4


@pytest.mark.parametrize("missing_first", [False, True])
@pytest.mark.parametrize(
    "field", ["n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"]
)
def test_job_totals_require_every_trial_to_report_each_counter(field, missing_first):
    partial = trial(usage(2).model_copy(update={field: None}))
    complete = trial(usage(3))
    trials = [partial, complete] if missing_first else [complete, partial]
    expected = tuple(
        None if name == field else 5
        for name in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")
    )

    stats = JobStats.from_trial_results(trials)

    assert totals(stats) == expected
    assert (
        stats.n_completed_trials == stats.evals["mini-swe-agent__adhoc"].n_trials == 2
    )


@pytest.mark.parametrize("value", [None, 0, 3])
def test_job_totals_preserve_explicit_zero(value):
    stats = JobStats.from_trial_results([trial(usage(value)), trial(usage(value))])

    assert totals(stats) == (None if value is None else 2 * value,) * 4


def test_job_usage_recovers_when_missing_trial_is_retried():
    complete, missing, another_missing = (
        trial(usage(3)),
        trial(usage(None)),
        trial(usage(None)),
    )
    stats = JobStats.from_trial_results([complete, missing, another_missing])
    job = JobResult(
        id=uuid4(), started_at=datetime.now(), n_total_trials=3, stats=stats
    )
    stats = job.stats
    assert totals(stats) == (None,) * 4

    stats.remove_trial(missing)
    assert totals(stats) == (None,) * 4
    stats.update_trial(trial(usage(2)), previous_result=another_missing)
    assert totals(stats) == (5,) * 4
    assert (
        stats.n_completed_trials == stats.evals["mini-swe-agent__adhoc"].n_trials == 2
    )

    stats.remove_trial(complete)
    assert totals(stats) == (2,) * 4


def test_removing_last_trial_restores_missing_usage():
    result = trial(usage(3))
    stats = JobStats.from_trial_results([result])

    stats.remove_trial(result)

    assert totals(stats) == (None,) * 4
    stats.increment(trial(usage(0)))
    assert totals(stats) == (0,) * 4


@pytest.mark.parametrize("cost, expected", [(None, None), (0, 3), (2, 5)])
def test_viewer_reports_only_complete_cost_totals(tmp_path, cost, expected):
    for result in [trial(usage(3)), trial(usage(cost))]:
        trial_dir = tmp_path / "job" / result.trial_name
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(result.model_dump_json())

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/jobs/job/heatmap")

    assert response.status_code == 200
    cell = next(iter(next(iter(response.json()["cells"].values())).values()))
    assert cell["total_cost_usd"] == expected
    assert cell["avg_cost_usd"] == (3 if cost is None else (3 + cost) / 2)
    assert cell["n_trials"] == 2


@pytest.mark.parametrize("cached, expected", [(None, None), (0, 3), (2, 1)])
def test_viewer_does_not_invent_uncached_input_when_cache_is_missing(
    tmp_path, cached, expected
):
    result = trial(AgentContext(n_input_tokens=3, n_cache_tokens=cached))
    trial_dir = tmp_path / "job" / result.trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(result.model_dump_json())

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/jobs/job/trials")

    assert response.status_code == 200
    assert response.json()["items"][0]["input_tokens"] == expected

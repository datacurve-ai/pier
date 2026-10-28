import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.models.agent.context import AgentContext
from pier.models.job.result import JobResult, JobStats
from pier.models.task.id import LocalTaskId
from pier.models.trajectories.final_metrics import FinalMetrics
from pier.models.trial.config import TaskConfig, TrialConfig
from pier.models.trial.result import AgentInfo, TrialResult
from pier.utils.trajectory_metrics import populate_context_from_final_metrics
from pier.viewer.server import create_app


@pytest.mark.parametrize(
    "tokens", [(None, None, None), (0, 0, 0), (100, 25, 40), (100, None, 40)]
)
def test_final_metrics_preserve_counter_presence_in_context(tokens):
    context = AgentContext(
        n_input_tokens=1, n_cache_tokens=1, n_output_tokens=1, n_agent_steps=3
    )
    metrics = FinalMetrics(
        total_prompt_tokens=tokens[0],
        total_cached_tokens=tokens[1],
        total_completion_tokens=tokens[2],
        total_cost_usd=None,
        extra={"peak_context_tokens": 0, "summarization_count": 0},
    )

    populate_context_from_final_metrics(context, metrics)

    saved = json.loads(context.model_dump_json())
    assert (
        saved["n_input_tokens"],
        saved["n_cache_tokens"],
        saved["n_output_tokens"],
    ) == tokens
    assert saved["cost_usd"] is None
    assert saved["peak_context_tokens"] == saved["summarization_count"] == 0
    assert saved["n_agent_steps"] == 3


@pytest.mark.parametrize("tokens", [None, 0, 25])
def test_mini_trajectory_usage_survives_final_result_serialization(
    tmp_path: Path, tokens
):
    usage = (
        None
        if tokens is None
        else {
            "prompt_tokens": tokens,
            "completion_tokens": tokens,
            "prompt_tokens_details": {"cached_tokens": tokens},
            "cost": tokens,
        }
    )
    (tmp_path / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "answer",
                        "extra": {"response": {"usage": usage}},
                    }
                ],
            }
        )
    )
    context = AgentContext()
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="provider/model")

    agent.populate_context_post_run(context)

    result = TrialResult(
        task_name="task",
        trial_name="trial",
        trial_uri=tmp_path.as_uri(),
        task_id=LocalTaskId(path=tmp_path),
        task_checksum="checksum",
        config=TrialConfig(task=TaskConfig(path=tmp_path)),
        agent_info=AgentInfo(name="mini-swe-agent", version="2.4.6"),
        agent_result=context,
    )
    saved = json.loads(result.model_dump_json())["agent_result"]
    assert (
        saved["n_input_tokens"],
        saved["n_cache_tokens"],
        saved["n_output_tokens"],
    ) == (tokens, tokens, tokens)
    assert TrialResult.model_validate_json(
        result.model_dump_json()
    ).compute_token_cost_totals() == (tokens, tokens, tokens, tokens)
    atif = json.loads((tmp_path / "trajectory.json").read_text())
    assert (
        atif["steps"][0]["llm_call_count"] == atif["final_metrics"]["total_steps"] == 1
    )

    job_dir = tmp_path / "jobs" / "job"
    trial_dir = job_dir / "trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(result.model_dump_json())
    job = JobResult(
        id=uuid4(),
        started_at=datetime.now(),
        n_total_trials=1,
        stats=JobStats.from_trial_results([result]),
    )
    (job_dir / "result.json").write_text(job.model_dump_json())

    with TestClient(create_app(tmp_path / "jobs")) as client:
        response = client.get("/api/jobs/job")
    assert response.status_code == 200
    saved_stats = response.json()["stats"]
    assert (
        saved_stats["n_input_tokens"],
        saved_stats["n_cache_tokens"],
        saved_stats["n_output_tokens"],
        saved_stats["cost_usd"],
    ) == (tokens,) * 4
    assert saved_stats["n_completed_trials"] == 1

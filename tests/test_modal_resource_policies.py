import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pier.environments import modal as modal_environment
from pier.environments.factory import EnvironmentFactory
from pier.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from pier.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from pier.models.trial.paths import TrialPaths


def policies(cpu, memory):
    return {"cpu_enforcement_policy": cpu, "memory_enforcement_policy": memory}


@pytest.mark.parametrize(
    ("settings", "overrides", "expected"),
    [
        ({}, {}, {"cpu": (4, 4), "memory": 16384}),
        ({"kwargs": policies("ignore", "limit")}, {}, {"memory": (128, 16384)}),
        (policies("ignore", "limit"), {}, {"memory": (128, 16384)}),
        ({}, policies("limit", "limit"), {"cpu": (0.125, 4), "memory": (128, 16384)}),
        (
            {"kwargs": policies("ignore", "ignore")},
            policies("request", "request"),
            {"cpu": 4, "memory": 16384},
        ),
    ],
    ids=["auto", "job-kwargs", "top-level", "caller-overrides", "caller-precedence"],
)
def test_start_applies_resource_policies_to_modal_requests(
    tmp_path, monkeypatch, settings, overrides, expected
):
    process = MagicMock()
    process.stdout.read.aio = process.stderr.read.aio = AsyncMock(return_value="")
    process.wait.aio = AsyncMock(return_value=0)
    sandbox = MagicMock()
    sandbox.filesystem.make_directory.aio = AsyncMock()
    sandbox.exec.aio = AsyncMock(return_value=process)
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(modal_environment.Image, "from_registry", MagicMock())
    monkeypatch.setattr(modal_environment.App.lookup, "aio", AsyncMock())
    monkeypatch.setattr(modal_environment.Sandbox.create, "aio", create)
    environment = EnvironmentFactory.create_environment_from_config(
        config=TrialEnvironmentConfig(type="modal", **settings),
        environment_dir=tmp_path,
        environment_name="resource-policy-task",
        session_id="resource-policy-trial",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=TaskEnvironmentConfig(
            docker_image="python:3.12-slim", cpus=4, memory_mb=16384
        ),
        **overrides,
    )

    asyncio.run(environment.start(force_build=False))

    request = create.call_args.kwargs
    assert {
        key: request[key] for key in ("cpu", "memory") if key in request
    } == expected

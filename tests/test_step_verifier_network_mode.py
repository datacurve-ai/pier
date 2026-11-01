import pytest
from pydantic import ValidationError

from pier.models.task.config import NetworkMode, TaskConfig
from pier.models.task.verifier_mode import resolve_effective_verifier_env_config


@pytest.mark.parametrize("task_network", ["public", "no-network"])
@pytest.mark.parametrize(
    "verifier_config",
    [
        'environment_mode = "separate"',
        'environment_mode = "separate"\n[verifier.environment]\n'
        'docker_image = "example/verifier:tag"\ncpus = 2',
        '[verifier.environment]\ndocker_image = "example/verifier:tag"\ncpus = 2',
    ],
)
def test_step_network_override_inherits_separate_verifier(
    task_network, verifier_config
):
    step_network = "no-network" if task_network == "public" else "public"
    cfg = TaskConfig.model_validate_toml(
        f"""
[environment]
network_mode = "{task_network}"
docker_image = "example/agent:tag"
cpus = 1
[verifier]
{verifier_config}
[[steps]]
name = "one"
[steps.verifier]
network_mode = "{step_network}"
[[steps]]
name = "two"
"""
    )

    step_env = resolve_effective_verifier_env_config(cfg, cfg.steps[0])
    sibling_env = resolve_effective_verifier_env_config(cfg, cfg.steps[1])
    assert step_env is cfg.steps[0].verifier.environment
    assert step_env is not None
    assert sibling_env is not None
    assert step_env is not sibling_env
    assert step_env.network_mode == NetworkMode(step_network)
    assert step_env.allow_internet is (step_network == "public")
    assert sibling_env.network_mode == NetworkMode(task_network)
    assert sibling_env.allow_internet is (task_network == "public")
    assert cfg.environment.allow_internet is (task_network == "public")
    assert step_env.docker_image == sibling_env.docker_image
    assert step_env.cpus == sibling_env.cpus
    if cfg.verifier.environment is not None:
        assert step_env.docker_image == "example/verifier:tag"
        assert step_env.cpus == 2
        assert cfg.verifier.environment.network_mode == NetworkMode(task_network)
    else:
        assert step_env.docker_image == "example/agent:tag"
        assert step_env.cpus == 1

    restored = TaskConfig.model_validate_toml(cfg.model_dump_toml())
    restored_env = resolve_effective_verifier_env_config(restored, restored.steps[0])
    assert restored_env == step_env


@pytest.mark.parametrize("task_verifier_mode", ["shared", "separate"])
def test_explicit_shared_step_rejects_conflicting_network(task_verifier_mode):
    with pytest.raises(ValidationError, match="shares that environment"):
        TaskConfig.model_validate_toml(
            f"""
[environment]
network_mode = "public"
[verifier]
environment_mode = "{task_verifier_mode}"
[[steps]]
name = "one"
[steps.verifier]
environment_mode = "shared"
network_mode = "no-network"
"""
        )


def test_explicit_separate_step_overrides_shared_task():
    cfg = TaskConfig.model_validate_toml(
        """
[environment]
network_mode = "public"
[verifier]
environment_mode = "shared"
[[steps]]
name = "one"
[steps.verifier]
environment_mode = "separate"
network_mode = "no-network"
"""
    )
    step_env = resolve_effective_verifier_env_config(cfg, cfg.steps[0])
    assert step_env is not None
    assert step_env.allow_internet is False
    assert cfg.environment.allow_internet is True

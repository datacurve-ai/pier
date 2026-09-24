import asyncio
import json
import logging

import pytest
import yaml

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    docker_run_command,
    proxy_environment,
    write_docker_proxy_compose,
)
from pier.environments.base import ExecResult
from pier.environments.docker.docker import DockerEnvironment
from pier.environments.modal import ModalEnvironment, _ModalDinD, _ModalDirect
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import TrialPaths


def _modal_vm_strategy(tmp_path, *, allow_internet, domains=()):
    env = ModalEnvironment.__new__(ModalEnvironment)
    env.environment_dir = tmp_path
    env.environment_name = "task"
    env.session_id = "task.1"
    env.task_env_config = EnvironmentConfig(allow_internet=allow_internet)
    env.network_allowlist = NetworkAllowlist(domains=list(domains))
    env.agent_install_spec = None
    env._persistent_env = {}
    env._vm_runtime = True
    env.logger = logging.getLogger("test")
    return _ModalDinD(env)


def test_docker_proxy_compose_does_not_inject_proxy_env_into_main(tmp_path):
    path = tmp_path / "docker-compose-egress-proxy.json"
    write_docker_proxy_compose(
        path=path,
        proxy_dir=tmp_path / "proxy",
        allowlist=type("Allowlist", (), {"domains": ["api.openai.com"]})(),
        token="secret",
    )

    compose = json.loads(path.read_text())
    main = compose["services"]["main"]
    assert "environment" not in main
    assert main["networks"] == ["pier-egress-internal"]
    assert EGRESS_PROXY_SERVICE in main["depends_on"]
    assert compose["networks"]["pier-egress-internal"]["internal"] is True
    assert "pier-egress-external" in compose["networks"]
    assert compose["services"][EGRESS_PROXY_SERVICE]["networks"] == [
        "pier-egress-internal",
        "pier-egress-external",
    ]


def test_docker_proxy_compose_preserves_main_task_networks(tmp_path):
    path = tmp_path / "docker-compose-egress-proxy.json"
    write_docker_proxy_compose(
        path=path,
        proxy_dir=tmp_path / "proxy",
        allowlist=type("Allowlist", (), {"domains": ["api.openai.com"]})(),
        token="secret",
        main_networks=["default"],
    )

    compose = json.loads(path.read_text())

    assert compose["services"]["main"]["networks"] == [
        "default",
        "pier-egress-internal",
    ]


def test_modal_vm_no_internet_keeps_task_networks_internal(tmp_path):
    strategy = _modal_vm_strategy(
        tmp_path, allow_internet=False, domains=["api.openai.com"]
    )
    config = {
        "services": {
            "main": {"networks": {"task-network": None}},
            "sidecar": {"networks": {"task-network": None}},
        },
        "networks": {"task-network": {"internal": True}},
    }

    overlay = yaml.safe_load(strategy._build_vm_network_overlay(config))

    assert overlay["networks"]["task-network"]["internal"] is True
    assert overlay["networks"]["default"]["internal"] is True
    assert "services" not in overlay
    assert strategy._compose_no_proxy_hosts(config) == ["main", "sidecar"]


def test_modal_vm_internet_adds_egress_to_main_only(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=True)
    config = {
        "services": {
            "main": {"networks": {"task-network": None}},
            "sidecar": {"networks": {"task-network": None}},
        },
        "networks": {"task-network": {"internal": True}},
    }

    overlay = yaml.safe_load(strategy._build_vm_network_overlay(config))

    assert overlay["services"] == {
        "main": {"networks": ["task-network", "pier-main-internet"]}
    }
    assert overlay["networks"] == {"pier-main-internet": {}}


def test_modal_vm_internet_preserves_implicit_default_network(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=True)
    config = {
        "services": {
            "main": {"networks": {"default": None}},
            "sidecar": {"networks": {"default": None}},
        },
        "networks": {"default": {}},
    }

    overlay = yaml.safe_load(strategy._build_vm_network_overlay(config))

    assert overlay["services"]["main"]["networks"] == [
        "default",
        "pier-main-internet",
    ]


def test_modal_vm_isolates_networks_from_resolved_includes(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {
        "services": {
            "main": {"networks": {"default": None}},
            "sidecar": {"networks": {"included-network": None}},
        },
        "networks": {"default": {}, "included-network": {}},
    }

    overlay = yaml.safe_load(strategy._build_vm_network_overlay(config))

    assert overlay["networks"] == {
        "default": {"internal": True},
        "included-network": {"internal": True},
    }


def test_modal_vm_rejects_unisolated_resolved_task_network(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {
        "services": {
            "main": {"networks": {"default": None}},
            "sidecar": {"networks": {"included-network": None}},
        },
        "networks": {
            "default": {"internal": True},
            "included-network": {},
        },
    }

    with pytest.raises(
        RuntimeError,
        match="sidecar is attached to non-internal network included-network",
    ):
        strategy._validate_vm_network_isolation(config)


def test_modal_vm_allows_only_proxy_on_external_network(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {
        "services": {
            "main": {"networks": {"default": None, "pier-egress-internal": None}},
            EGRESS_PROXY_SERVICE: {
                "networks": {
                    "pier-egress-internal": None,
                    "pier-egress-external": None,
                }
            },
        },
        "networks": {
            "default": {"internal": True},
            "pier-egress-internal": {"internal": True},
            "pier-egress-external": {},
        },
    }

    strategy._validate_vm_network_isolation(config)


def test_modal_vm_prepares_policy_from_resolved_config_before_start(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    source_config = {
        "services": {
            "main": {"networks": {"default": None}},
            "sidecar": {"networks": {"included-network": None}},
        },
        "networks": {"default": {}, "included-network": {}},
    }
    isolated_config = {
        **source_config,
        "networks": {
            "default": {"internal": True},
            "included-network": {"internal": True},
        },
    }
    resolve_calls = []
    uploads = {}

    async def resolve_config(*, include_vm_networking=True):
        resolve_calls.append(include_vm_networking)
        return isolated_config if include_vm_networking else source_config

    async def upload_text(filename, content):
        uploads[filename] = content

    strategy._resolve_compose_config = resolve_config
    strategy._upload_text = upload_text

    asyncio.run(strategy._prepare_vm_networking())

    overlay = yaml.safe_load(uploads[strategy._VM_NETWORK_COMPOSE])
    assert resolve_calls == [False, True]
    assert overlay["networks"]["included-network"]["internal"] is True


def test_modal_vm_proxy_bypasses_compose_aliases(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {
        "services": {
            "main": {"networks": {"task-network": None}},
            "sidecar": {
                "hostname": "sidecar-host",
                "networks": {"task-network": {"aliases": ["records.example.internal"]}},
            },
        },
        "networks": {"task-network": {"internal": True}},
    }

    env = proxy_environment(
        "secret",
        EGRESS_PROXY_SERVICE,
        EGRESS_PROXY_PORT,
        no_proxy_hosts=strategy._compose_no_proxy_hosts(config),
    )

    assert env["NO_PROXY"].split(",") == [
        "localhost",
        "127.0.0.1",
        "main",
        "records.example.internal",
        "sidecar",
        "sidecar-host",
    ]


def test_modal_vm_no_internet_rejects_network_mode_bypass(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {"services": {"main": {"network_mode": "host"}}, "networks": {}}

    with pytest.raises(ValueError, match="main=host"):
        strategy._build_vm_network_overlay(config)


def test_modal_vm_rejects_reserved_proxy_network_collision(tmp_path):
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    config = {
        "services": {"main": {"networks": {"pier-egress-external": None}}},
        "networks": {"pier-egress-external": {}},
    }

    with pytest.raises(ValueError, match="reserved by Pier"):
        strategy._build_vm_network_overlay(config)


def test_modal_vm_compose_advertises_preinstall_and_filtered_egress(
    tmp_path, monkeypatch
):
    import pier.environments.modal as modal_module

    monkeypatch.setattr(modal_module, "_HAS_MODAL", True)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "docker-compose.yaml").write_text("services:\n  main: {}\n")
    install = AgentInstallSpec(
        agent_name="test-agent",
        steps=[InstallStep(run="echo installed")],
    )

    env = ModalEnvironment(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="task.1",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(allow_internet=False),
        agent_install_spec=install,
        network_allowlist=NetworkAllowlist(domains=["api.openai.com"]),
        vm_runtime=True,
    )

    assert env.capabilities.disable_internet is True
    assert env.capabilities.filtered_egress is True
    assert env.capabilities.preinstall_agents is True
    assert env.agent_install_spec == install


def test_modal_vm_compose_derives_main_image_with_agent(tmp_path):
    (tmp_path / "docker-compose.yaml").write_text("services:\n  main:\n    build: .\n")
    strategy = _modal_vm_strategy(tmp_path, allow_internet=False)
    strategy._env.trial_paths = TrialPaths(tmp_path / "trial")
    strategy._env.default_user = "agent"
    strategy._env.agent_install_spec = AgentInstallSpec(
        agent_name="test-agent",
        cache_key="test-cache",
        steps=[InstallStep(run="echo installed")],
    )
    uploads = {}
    commands = []

    async def resolve_main():
        return {"build": {"context": str(tmp_path)}}

    async def upload_text(filename, content):
        uploads[filename] = content

    async def upload_dir(_source, _target):
        return None

    async def vm_exec(command, **_kwargs):
        commands.append(command)
        return ExecResult(return_code=0)

    strategy._resolve_main_compose_config = resolve_main
    strategy._upload_text = upload_text
    strategy._env._sdk_upload_dir = upload_dir
    strategy._vm_exec = vm_exec

    async def run():
        base_image = await strategy._prepare_agent_base_image()
        await strategy._build_agent_image(base_image, force_build=True)
        return base_image

    base_image = asyncio.run(run())

    dockerfile = tmp_path / "trial" / "modal-agent-build" / "Dockerfile"
    assert dockerfile.read_text().startswith(f"FROM {base_image}\n")
    assert "echo installed" in dockerfile.read_text()
    assert "--no-cache" in commands[0]
    assert "build: !reset null" in uploads[strategy._AGENT_IMAGE_COMPOSE]


def test_modal_failed_start_tears_down_created_sandbox():
    class FailingStrategy:
        def __init__(self):
            self.cleaned = False

        async def start(self, _force_build):
            raise RuntimeError("failed")

        async def _teardown_sandbox(self):
            self.cleaned = True

    env = ModalEnvironment.__new__(ModalEnvironment)
    env._strategy = FailingStrategy()

    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(env.start(force_build=False))

    assert env._strategy.cleaned is True


def test_docker_agent_process_env_adds_proxy_only_for_agent_commands():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._egress_proxy_env = proxy_environment(
        "secret", EGRESS_PROXY_SERVICE, EGRESS_PROXY_PORT
    )

    process_env = env.agent_process_env({"OPENAI_API_KEY": "test"})

    assert process_env["OPENAI_API_KEY"] == "test"
    assert process_env["HTTP_PROXY"].startswith("http://agent:secret@")


def test_modal_agent_process_env_adds_proxy_only_for_agent_commands():
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._egress_proxy_env = proxy_environment("secret", "r123.modal.host", 12345)

    process_env = env.agent_process_env({"OPENAI_API_KEY": "test"})

    assert process_env["OPENAI_API_KEY"] == "test"
    assert process_env["HTTP_PROXY"] == "http://agent:secret@r123.modal.host:12345"


def test_modal_direct_exec_uses_non_login_shell():
    class DummyModalEnv:
        def __init__(self):
            self.kwargs = None

        async def _sdk_exec(self, command, **kwargs):
            self.kwargs = kwargs
            return ExecResult(return_code=0)

    env = DummyModalEnv()
    result = asyncio.run(_ModalDirect(env).exec("echo ok"))

    assert result.return_code == 0
    assert "login" not in env.kwargs


def test_agent_dockerfile_install_uses_non_login_shell():
    assert docker_run_command("echo $PATH") == 'RUN ["/bin/bash", "-c", "echo $PATH"]'


def test_modal_dind_compose_exec_uses_non_login_shell():
    env = ModalEnvironment.__new__(ModalEnvironment)
    env.environment_name = "task"
    env.task_env_config = type(
        "TaskEnv",
        (),
        {"env": {}, "cpus": 1, "memory_mb": 1024, "docker_image": None},
    )()
    env._persistent_env = {}
    strategy = _ModalDinD(env)

    captured = []

    async def compose_exec(parts, timeout_sec=None):
        captured.append(parts)
        return ExecResult(return_code=0)

    strategy._compose_exec = compose_exec

    async def run():
        await strategy.exec("echo ok")

    asyncio.run(run())

    assert captured[0][-3:] == ["bash", "-c", "echo ok"]


def test_modal_exec_preserves_env_when_switching_user():
    class DummyStrategy:
        def __init__(self):
            self.command = None
            self.env = None

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.command = command
            self.env = env
            return ExecResult(return_code=0)

    env = ModalEnvironment.__new__(ModalEnvironment)
    env.default_user = None
    env._persistent_env = {}
    env.task_env_config = type("TaskEnv", (), {"workdir": None})()
    env._strategy = DummyStrategy()

    result = asyncio.run(
        ModalEnvironment.exec(env, "echo $PATH", user="agent", env={"PATH": "/custom"})
    )

    assert result.return_code == 0
    assert env._strategy.env == {"PATH": "/custom"}
    assert env._strategy.command == "su -m agent -s /bin/bash -c 'echo $PATH'"

import asyncio
import json
import os
import subprocess

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    docker_run_command,
    proxy_environment,
    proxy_policy_env,
    squid_bootstrap_command,
    write_docker_proxy_compose,
)
from pier.environments.base import ExecResult
from pier.environments.docker.docker import DockerEnvironment
from pier.environments.modal import ModalEnvironment, _ModalDinD, _ModalDirect
from pier.models.agent.network import NetworkAllowlist


def test_docker_proxy_compose_does_not_inject_proxy_env_into_main(tmp_path):
    path = tmp_path / "docker-compose-egress-proxy.json"
    write_docker_proxy_compose(
        path=path,
        proxy_dir=tmp_path / "proxy",
        allowlist=NetworkAllowlist(domains=["api.openai.com"]),
        token="secret",
    )

    compose = json.loads(path.read_text())
    main = compose["services"]["main"]
    assert "environment" not in main
    assert main["networks"] == ["pier-egress-internal"]
    assert EGRESS_PROXY_SERVICE in main["depends_on"]


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


def test_proxy_policy_passes_allowlisted_endpoints_to_squid():
    allowlist = NetworkAllowlist(domains=["192.0.2.10"], endpoints=["192.0.2.10:8000"])
    env = proxy_policy_env(allowlist, "secret")
    assert env["ALLOWLIST_DOMAINS"] == "192.0.2.10"
    assert env["ALLOWLIST_ENDPOINTS"] == "192.0.2.10:8000"


def test_squid_allows_allowlisted_endpoints_before_the_port_rules(tmp_path):
    script = squid_bootstrap_command()
    include = script.index("include /tmp/allowed_endpoints.conf")
    assert include < script.index("http_access deny !Safe_ports")
    assert include < script.index("http_access deny CONNECT !SSL_ports")

    # Run the part of the script that writes the endpoint rules.
    begin = script.index("i=0\n")
    end = script.index("done > /tmp/allowed_endpoints.conf") + len(
        "done > /tmp/allowed_endpoints.conf"
    )
    conf = tmp_path / "allowed_endpoints.conf"
    snippet = script[begin:end].replace("/tmp/allowed_endpoints.conf", str(conf))
    env = {
        "PATH": os.environ["PATH"],
        "ALLOWLIST_ENDPOINTS": "192.0.2.10:8000,models.example.com:8080",
    }
    subprocess.run(["bash", "-c", snippet], env=env, check=True)
    assert conf.read_text().splitlines() == [
        "acl allowed_endpoint_0_host dst 192.0.2.10",
        "acl allowed_endpoint_0_port port 8000",
        "http_access allow authenticated allowed_endpoint_0_host allowed_endpoint_0_port",
        "acl allowed_endpoint_1_host dstdomain models.example.com",
        "acl allowed_endpoint_1_port port 8080",
        "http_access allow authenticated allowed_endpoint_1_host allowed_endpoint_1_port",
    ]

    subprocess.run(
        ["bash", "-c", snippet], env={"PATH": os.environ["PATH"]}, check=True
    )
    assert conf.read_text() == ""

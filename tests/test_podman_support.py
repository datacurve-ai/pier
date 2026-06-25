"""Unit tests for opt-in podman / podman-compose support (PIER_DOCKER_CMD).

Each test asserts one of:
  (a) docker behavior is byte-identical when PIER_DOCKER_CMD is unset, or
  (b) the correct podman adaptation when it is set.
No container engine is required: subprocess / compose calls are mocked, and
DockerEnvironment is built with __new__ (only the attributes a method touches
are set), mirroring tests/test_filtered_egress_env.py.
"""

import asyncio
import functools
import json

import pytest

from pier.environments import agent_setup
from pier.environments.base import ExecResult
from pier.environments.docker import docker as docker_mod
from pier.environments.docker import write_mounts_compose_file
from pier.environments.docker.docker import (
    DockerEnvironment,
    _docker_cmd,
    _docker_engine,
    _is_podman,
)
from pier.models.trial.paths import EnvironmentPaths


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# --------------------------------------------------------------------------
# 1. Engine helpers
# --------------------------------------------------------------------------
def test_helpers_default_is_docker(monkeypatch):
    monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    assert _docker_cmd() == "docker"
    assert _is_podman() is False
    assert _docker_engine() == "docker"


def test_helpers_podman(monkeypatch):
    # the documented value is the engine name
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman")
    assert _docker_cmd() == "podman"
    assert _is_podman() is True
    assert _docker_engine() == "podman"
    # a trailing -compose is tolerated and normalized to the engine, so the
    # legacy `podman-compose` value still resolves to engine `podman`
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman-compose")
    assert _is_podman() is True
    assert _docker_engine() == "podman"


def test_helpers_path_and_variant_forms(monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_CMD", "/usr/bin/podman")
    assert _is_podman() is True
    assert _docker_engine() == "podman"
    # an absolute docker path must NOT be misclassified as podman
    monkeypatch.setenv("PIER_DOCKER_CMD", "/usr/local/bin/docker")
    assert _is_podman() is False
    assert _docker_engine() == "docker"
    # startswith("podman") intentionally catches podman-remote
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman-remote")
    assert _is_podman() is True


# --------------------------------------------------------------------------
# 2. write_mounts_compose_file (the :Z carrier; docker byte-identical)
# --------------------------------------------------------------------------
def test_mounts_compose_docker_byte_identical(tmp_path):
    mounts = [{"type": "bind", "source": "/h/v", "target": "/logs/verifier"}]
    p = write_mounts_compose_file(tmp_path / "m.json", mounts)
    expected = json.dumps({"services": {"main": {"volumes": mounts}}}, indent=2)
    assert p.read_text() == expected


def test_mounts_compose_serializes_selinux(tmp_path):
    mounts = [
        {
            "type": "bind",
            "source": "/h/v",
            "target": "/logs/verifier",
            "bind": {"selinux": "Z"},
        }
    ]
    p = write_mounts_compose_file(tmp_path / "m.json", mounts)
    vol = json.loads(p.read_text())["services"]["main"]["volumes"][0]
    assert vol["bind"] == {"selinux": "Z"}


# --------------------------------------------------------------------------
# 3. Egress-proxy default network (podman-only; no docker leak)
# --------------------------------------------------------------------------
def _proxy_networks(monkeypatch, tmp_path, cmd):
    if cmd is None:
        monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    else:
        monkeypatch.setenv("PIER_DOCKER_CMD", cmd)
    allowlist = type("Allowlist", (), {"domains": ["api.openai.com"]})()
    agent_setup.write_docker_proxy_compose(
        path=tmp_path / "c.json",
        proxy_dir=tmp_path / "proxy",
        allowlist=allowlist,
        token="secret",
    )
    return set(json.loads((tmp_path / "c.json").read_text())["networks"])


def test_egress_default_network_podman(monkeypatch, tmp_path):
    nets = _proxy_networks(monkeypatch, tmp_path, "podman")
    assert nets == {"pier-egress-internal", "default"}


def test_egress_no_default_network_docker(monkeypatch, tmp_path):
    nets = _proxy_networks(monkeypatch, tmp_path, None)
    assert nets == {"pier-egress-internal"}


# --------------------------------------------------------------------------
# 4. _default_log_mounts (:Z under podman; none under docker)
# --------------------------------------------------------------------------
def _env_with_paths(tmp_path):
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.trial_paths = type(
        "TP",
        (),
        {
            "verifier_dir": tmp_path / "v",
            "agent_dir": tmp_path / "a",
            "artifacts_dir": tmp_path / "art",
        },
    )()
    env._env_paths = EnvironmentPaths()
    return env


def test_default_log_mounts_podman_adds_selinux(monkeypatch, tmp_path):
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman")
    mounts = _env_with_paths(tmp_path)._default_log_mounts()
    assert len(mounts) == 3
    assert all(m["bind"] == {"selinux": "Z"} for m in mounts)


def test_default_log_mounts_docker_has_no_bind(monkeypatch, tmp_path):
    monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    mounts = _env_with_paths(tmp_path)._default_log_mounts()
    assert all("bind" not in m for m in mounts)


# --------------------------------------------------------------------------
# 4b. Trial._verifier_env_mounts (:Z under podman) — the separate-verifier-env
#     mount path that bypasses _default_log_mounts. Regression guard for the
#     SELinux fix that made the separate verifier env writable under podman.
# --------------------------------------------------------------------------
def _verifier_env_mounts(monkeypatch, tmp_path, cmd):
    if cmd is None:
        monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    else:
        monkeypatch.setenv("PIER_DOCKER_CMD", cmd)
    from pier.trial.trial import Trial

    t = Trial.__new__(Trial)
    t._environment = type(
        "E",
        (),
        {"env_paths": type("EP", (), {"for_os": lambda self, _os: EnvironmentPaths()})()},
    )()
    t._trial_paths = type("TP", (), {"verifier_dir": tmp_path / "v"})()
    return t._verifier_env_mounts(type("C", (), {"os": None})())


def test_verifier_env_mounts_podman_adds_selinux(monkeypatch, tmp_path):
    mounts = _verifier_env_mounts(monkeypatch, tmp_path, "podman")
    assert len(mounts) == 1
    assert mounts[0]["type"] == "bind"
    assert mounts[0]["bind"] == {"selinux": "Z"}


def test_verifier_env_mounts_docker_has_no_bind(monkeypatch, tmp_path):
    mounts = _verifier_env_mounts(monkeypatch, tmp_path, None)
    assert len(mounts) == 1
    assert "bind" not in mounts[0]


# --------------------------------------------------------------------------
# 5. _cp dispatch (compose cp on docker; native cp on podman)
# --------------------------------------------------------------------------
@run_async
async def test_cp_docker_uses_compose_cp(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(return_code=0)
    )
    env._resolve_main_container_id = AsyncMock()
    await env._cp("/h/x", "main:/c/x", check=True)
    env._run_docker_compose_command.assert_awaited_once_with(
        ["cp", "/h/x", "main:/c/x"], check=True
    )
    env._resolve_main_container_id.assert_not_awaited()


@run_async
async def test_cp_podman_uses_native_cp_with_prefix_rewrite(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setenv("PIER_DOCKER_CMD", "podman")
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._resolve_main_container_id = AsyncMock(return_value="abc123")

    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def fake_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    # download direction: main: prefix on the source -> <id>:
    await env._cp("main:/c/f", "/h/f", check=True)
    assert captured["argv"] == ("podman", "cp", "abc123:/c/f", "/h/f")
    # a host path that merely contains "main:" mid-string is NOT rewritten
    await env._cp("/home/main:keep", "main:/c/g", check=True)
    assert captured["argv"] == ("podman", "cp", "/home/main:keep", "abc123:/c/g")


# --------------------------------------------------------------------------
# 6. exec adds -T under podman only
# --------------------------------------------------------------------------
def _exec_env(monkeypatch, cmd):
    if cmd is None:
        monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    else:
        monkeypatch.setenv("PIER_DOCKER_CMD", cmd)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._resolve_user = lambda u: None
    env._merge_env = lambda e: None
    env.task_env_config = type("T", (), {"workdir": None})()
    env._platform = type(
        "P", (), {"exec_shell_args": staticmethod(lambda c: ["bash", "-c", c])}
    )()
    captured = {}

    async def fake_run(cmd_, check=False, timeout_sec=None):
        captured["cmd"] = cmd_
        return ExecResult(return_code=0)

    env._run_docker_compose_command = fake_run
    return env, captured


@run_async
async def test_exec_adds_T_under_podman(monkeypatch):
    env, captured = _exec_env(monkeypatch, "podman")
    await env.exec("true")
    assert captured["cmd"][:2] == ["exec", "-T"]


@run_async
async def test_exec_no_T_under_docker(monkeypatch):
    env, captured = _exec_env(monkeypatch, None)
    await env.exec("true")
    assert captured["cmd"][0] == "exec"
    assert "-T" not in captured["cmd"]


# --------------------------------------------------------------------------
# 7. compose prefix: --project-directory (docker) vs COMPOSE_PROJECT_DIR (podman)
# --------------------------------------------------------------------------
def _compose_env(monkeypatch, tmp_path, cmd):
    if cmd is None:
        monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    else:
        monkeypatch.setenv("PIER_DOCKER_CMD", cmd)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.session_id = "sess"
    env.environment_dir = tmp_path
    env._compose_task_env = {}
    env._persistent_env = {}
    env._windows_container_name = None
    env._env_vars = type(
        "EV", (), {"to_env_dict": lambda self, include_os_env=True: {}}
    )()
    monkeypatch.setattr(type(env), "_docker_compose_paths", property(lambda self: []))

    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def fake_exec(*args, env=None, **kwargs):
        captured["argv"] = list(args)
        captured["env"] = env
        return FakeProc()

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    return env, captured


@run_async
async def test_compose_prefix_podman(monkeypatch, tmp_path):
    env, captured = _compose_env(monkeypatch, tmp_path, "podman")
    await env._run_docker_compose_command(["up"])
    argv = captured["argv"]
    # symmetric with docker: `podman compose ...` (the `compose` subcommand)
    assert argv[:2] == ["podman", "compose"]
    # but no --project-directory (podman's provider rejects it); dir via env
    assert "--project-directory" not in argv
    assert captured["env"]["COMPOSE_PROJECT_DIR"] == str(
        tmp_path.resolve().absolute()
    )
    # the provider banner is silenced so it can't pollute captured exec/cp output
    assert captured["env"]["PODMAN_COMPOSE_WARNING_LOGS"] == "false"


@run_async
async def test_compose_prefix_podman_compose_value_normalizes(monkeypatch, tmp_path):
    # the legacy `podman-compose` value resolves to the same `podman compose ...`
    env, captured = _compose_env(monkeypatch, tmp_path, "podman-compose")
    await env._run_docker_compose_command(["up"])
    assert captured["argv"][:2] == ["podman", "compose"]


@run_async
async def test_compose_prefix_docker(monkeypatch, tmp_path):
    env, captured = _compose_env(monkeypatch, tmp_path, None)
    await env._run_docker_compose_command(["up"])
    argv = captured["argv"]
    assert argv[:2] == ["docker", "compose"]
    assert "--project-directory" in argv
    assert "COMPOSE_PROJECT_DIR" not in captured["env"]
    assert "PODMAN_COMPOSE_WARNING_LOGS" not in captured["env"]


# --------------------------------------------------------------------------
# 8. Windows tasks rejected under podman
# --------------------------------------------------------------------------
def test_windows_rejected_under_podman(monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman")
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._is_windows_container = True
    with pytest.raises(RuntimeError, match="not supported under podman"):
        env._validate_daemon_mode()


def test_windows_docker_does_not_raise_podman_message(monkeypatch):
    monkeypatch.delenv("PIER_DOCKER_CMD", raising=False)
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._is_windows_container = True
    # On a non-win32 host this raises the existing host-mismatch error, NOT the
    # podman one. (On win32 it would proceed; either way the podman message must
    # never appear under docker.)
    with pytest.raises(RuntimeError) as exc:
        env._validate_daemon_mode()
    assert "not supported under podman" not in str(exc.value)


# --------------------------------------------------------------------------
# 9. _resolve_main_container_id uses provider-independent compose labels
# --------------------------------------------------------------------------
@run_async
async def test_resolve_main_container_id_uses_docker_labels(monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_CMD", "podman")
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.session_id = "sess"

    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"abc123\n", b"")

    async def fake_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)
    cid = await env._resolve_main_container_id()
    assert cid == "abc123"
    argv = captured["argv"]
    assert argv[0] == "podman"
    assert "ps" in argv and "-q" in argv
    # the `com.docker.compose.*` labels are set by BOTH the podman-compose and
    # docker-compose providers, so the lookup is provider-independent.
    assert "label=com.docker.compose.project=sess" in argv
    assert "label=com.docker.compose.service=main" in argv

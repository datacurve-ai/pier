"""Regression tests for DockerEnvironment stream separation (issue #27).

``_run_docker_compose_command`` used to launch ``docker compose`` with
``stderr=STDOUT``, which merged the child's stderr into stdout and left
``ExecResult.stderr`` empty. These tests pin the fixed contract: stderr is
captured separately and mapped to ``ExecResult.stderr``.

They avoid a real Docker daemon by constructing a bare ``DockerEnvironment``
via ``__new__`` (matching the existing test style in this suite) and patching
``asyncio.create_subprocess_exec`` with a fake process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pier.environments.base import ExecResult
from pier.environments.docker.docker import DockerEnvironment


class _FakeProcess:
    """Minimal stand-in for the object create_subprocess_exec returns."""

    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def _bare_env(tmp_path: Path, monkeypatch) -> DockerEnvironment:
    # ``_docker_compose_paths`` is a computed property with no setter, so patch
    # it at the class level to an empty list for these unit tests.
    monkeypatch.setattr(
        DockerEnvironment,
        "_docker_compose_paths",
        property(lambda self: []),
    )

    env = DockerEnvironment.__new__(DockerEnvironment)
    env.session_id = "test-session"
    env.environment_name = "task"
    env.environment_dir = tmp_path
    env._compose_task_env = {}
    env._persistent_env = {}
    env._windows_container_name = None

    class _EnvVars:
        @staticmethod
        def to_env_dict(include_os_env: bool = True) -> dict[str, str]:
            return {}

    env._env_vars = _EnvVars()
    return env


def test_exec_keeps_stdout_and_stderr_separate(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProcess(stdout=b"out\n", stderr=b"err\n", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    env = _bare_env(tmp_path, monkeypatch)
    result = asyncio.run(env._run_docker_compose_command(["exec", "main", "true"]))

    # stderr must be piped separately, not folded into stdout.
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.PIPE
    assert captured["kwargs"]["stdout"] is asyncio.subprocess.PIPE

    assert isinstance(result, ExecResult)
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.return_code == 0


def test_exec_separation_preserves_nonzero_return_code(tmp_path, monkeypatch):
    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(stdout=b"", stderr=b"boom\n", returncode=2)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    env = _bare_env(tmp_path, monkeypatch)
    # check=False so a non-zero exit surfaces on the result instead of raising.
    result = asyncio.run(
        env._run_docker_compose_command(["exec", "main", "false"], check=False)
    )

    assert result.return_code == 2
    assert result.stderr == "boom\n"
    assert result.stdout is None


def test_failed_command_error_includes_separated_stderr(tmp_path, monkeypatch):
    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(
            stdout=b"partial out\n", stderr=b"real error\n", returncode=1
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    env = _bare_env(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(env._run_docker_compose_command(["up", "--detach"], check=True))

    message = str(excinfo.value)
    assert "real error" in message
    assert "partial out" in message

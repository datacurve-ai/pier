import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("e2b")

from e2b import ALL_TRAFFIC, Template

from pier.environments import e2b as e2b_module
from pier.environments.base import ExecResult
from pier.environments.e2b import E2BEnvironment
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import TrialPaths


@pytest.fixture(autouse=True)
def _reset_module_caches():
    e2b_module._IMAGE_ENV_CACHE.clear()
    e2b_module._IMAGE_ENV_LOCKS.clear()
    e2b_module._TEMPLATE_LOCKS.clear()
    yield
    e2b_module._IMAGE_ENV_CACHE.clear()
    e2b_module._IMAGE_ENV_LOCKS.clear()
    e2b_module._TEMPLATE_LOCKS.clear()


def _mock_image_config(env=None, workdir=None, user=None):
    return patch.object(
        E2BEnvironment,
        "_fetch_image_config",
        new=AsyncMock(
            return_value={"env": env or {}, "workdir": workdir, "user": user}
        ),
    )


def _make_env(
    tmp_path: Path,
    *,
    docker_image: str | None = "ubuntu:24.04",
    allow_internet: bool = False,
    domains: list[str] | None = None,
    install: AgentInstallSpec | None = None,
    default_user: str | int | None = None,
) -> E2BEnvironment:
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    return E2BEnvironment(
        environment_dir=env_dir,
        environment_name="datacurve/test-task",
        session_id="test-session",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=docker_image,
            cpus=2,
            memory_mb=8192,
            storage_mb=20480,
            allow_internet=allow_internet,
        ),
        agent_install_spec=install,
        network_allowlist=NetworkAllowlist(domains=domains or []),
        default_user=default_user,
    )


def _install_spec(command: str = "echo installed") -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name="mini-swe-agent",
        steps=[InstallStep(run=command, user="root")],
        verification_command="true",
    )


def test_template_name_is_deterministic_and_agent_sensitive(tmp_path):
    first = _make_env(tmp_path / "one", install=_install_spec())
    second = _make_env(tmp_path / "two", install=_install_spec())
    changed = _make_env(tmp_path / "three", install=_install_spec("echo changed"))

    assert first.template_name == second.template_name
    assert first.template_name != changed.template_name
    assert len(first.template_name) <= 128


def test_compose_tasks_are_rejected(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (env_dir / "docker-compose.yaml").write_text("services: {}\n")
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()

    with pytest.raises(ValueError, match="Docker Compose"):
        E2BEnvironment(
            environment_dir=env_dir,
            environment_name="task",
            session_id="s",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(docker_image="ubuntu:24.04"),
        )


def test_parse_image_ref_handles_tags_digests_and_registries():
    digest = "sha256:" + "a" * 64

    assert e2b_module._parse_image_ref("ubuntu") == (
        "registry-1.docker.io",
        "library/ubuntu",
        "latest",
    )
    assert e2b_module._parse_image_ref("ghcr.io/org/img:1.2") == (
        "ghcr.io",
        "org/img",
        "1.2",
    )
    assert e2b_module._parse_image_ref(f"ubuntu@{digest}") == (
        "registry-1.docker.io",
        "library/ubuntu",
        digest,
    )
    assert e2b_module._parse_image_ref(f"ghcr.io/org/img:1.2@{digest}") == (
        "ghcr.io",
        "org/img",
        digest,
    )
    assert e2b_module._parse_image_ref(f"localhost:5000/img:1.2@{digest}") == (
        "localhost:5000",
        "img",
        digest,
    )


def test_allowlist_maps_to_native_e2b_network_policy(tmp_path):
    env = _make_env(tmp_path, domains=[".amazonaws.com", "api.example.com"])

    assert env._sandbox_network_options() == {
        "allow_public_traffic": False,
        "allow_out": ["amazonaws.com", "*.amazonaws.com", "api.example.com"],
        "deny_out": [ALL_TRAFFIC],
    }


def test_isolated_task_blocks_egress_and_public_ingress(tmp_path):
    env = _make_env(tmp_path)

    assert env._sandbox_network_options() == {"allow_public_traffic": False}


def test_internet_tasks_disable_public_ingress(tmp_path):
    env = _make_env(tmp_path, allow_internet=True)

    assert env._sandbox_network_options() == {"allow_public_traffic": False}


@pytest.mark.asyncio
async def test_template_definition_bakes_agent_fingerprint_and_image_config(tmp_path):
    install = _install_spec()
    env = _make_env(tmp_path, install=install)

    with _mock_image_config(
        env={"PATH": "/opt/venv/bin:/usr/bin"}, workdir="/app", user="svc-user"
    ):
        serialized = Template.to_json(await env._template_definition())

    assert install.fingerprint() in serialized
    assert "echo installed" in serialized
    assert "image-config.json" in serialized
    assert "/opt/venv/bin:/usr/bin" in serialized
    assert "/app" in serialized
    assert "svc-user" in serialized


@pytest.mark.asyncio
async def test_agent_install_defaults_to_root_not_image_user(tmp_path):
    install = AgentInstallSpec(
        agent_name="mini-swe-agent",
        steps=[InstallStep(run="echo agent-step", user="agent")],
        verification_command="echo verify",
    )
    env = _make_env(tmp_path, install=install)

    with _mock_image_config(user="svc-user"):
        definition = json.loads(Template.to_json(await env._template_definition()))

    commands = {
        step["args"][0]: step["args"][1]
        for step in definition["steps"]
        if step["type"] == "RUN"
    }
    assert commands["echo agent-step"] == "root"
    assert commands["echo verify"] == "root"


@pytest.mark.asyncio
async def test_numeric_default_user_is_resolved_during_template_build(tmp_path):
    env = _make_env(tmp_path, install=_install_spec(), default_user=1000)

    with _mock_image_config():
        serialized = Template.to_json(await env._template_definition())

    assert "getent passwd 1000" in serialized
    assert "su -m" in serialized


@pytest.mark.asyncio
async def test_create_sandbox_uses_template_resources_and_network(tmp_path):
    env = _make_env(tmp_path, domains=["api.example.com"])
    sandbox = MagicMock()

    with patch(
        "pier.environments.e2b.AsyncSandbox.create",
        new=AsyncMock(return_value=sandbox),
    ) as create:
        await env._create_sandbox()

    assert create.await_args.kwargs["template"] == env.template_name
    assert create.await_args.kwargs["allow_internet_access"] is True
    assert create.await_args.kwargs["network"] == {
        "allow_public_traffic": False,
        "allow_out": ["api.example.com"],
        "deny_out": [ALL_TRAFFIC],
    }


@pytest.mark.asyncio
async def test_require_mode_fails_closed_when_template_missing(tmp_path):
    env = _make_env(tmp_path)
    env._template_mode = "required"

    with patch.object(env, "_template_exists", new=AsyncMock(return_value=False)):
        with pytest.raises(RuntimeError, match="Required E2B template"):
            await env._ensure_template(force_build=False)


@pytest.mark.asyncio
async def test_exec_defaults_to_image_user_and_workdir(tmp_path):
    env = _make_env(tmp_path)
    env._image_user = "appuser"
    env._image_workdir = "/app"
    result = MagicMock()
    result.stdout, result.stderr, result.exit_code = "", "", 0
    handle = MagicMock()
    handle.wait = AsyncMock(return_value=result)
    captured = ExecResult(stdout="", stderr="", return_code=0)

    with (
        patch.object(
            env, "_dispatch_command", new=AsyncMock(return_value=handle)
        ) as dispatch,
        patch.object(
            env, "_read_command_capture", new=AsyncMock(return_value=captured)
        ),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
    ):
        await env.exec("id")
        await env.exec("id", user="root")

    assert dispatch.await_args_list[0].kwargs["user"] == "appuser"
    assert dispatch.await_args_list[0].kwargs["cwd"] == "/app"
    assert dispatch.await_args_list[1].kwargs["user"] == "root"


@pytest.mark.asyncio
async def test_exec_keeps_secrets_out_of_command_text(tmp_path):
    env = _make_env(tmp_path)
    env._image_env = {
        "PATH": "/image/bin:/usr/bin",
        "IMAGE_TOKEN": "image-secret",
    }
    handle = MagicMock()
    handle.wait = AsyncMock(return_value=MagicMock(exit_code=0))
    captured = ExecResult(stdout="", stderr="", return_code=0)

    with (
        patch.object(
            env, "_dispatch_command", new=AsyncMock(return_value=handle)
        ) as dispatch,
        patch.object(
            env, "_read_command_capture", new=AsyncMock(return_value=captured)
        ),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
    ):
        await env.exec(
            "true",
            env={"API_TOKEN": "runtime-secret", "PATH": "/runtime/bin:/usr/bin"},
        )

    dispatched_command = dispatch.await_args.args[0]
    dispatched_env = dispatch.await_args.kwargs["env"]
    assert dispatched_env == {
        "PATH": "/runtime/bin:/usr/bin",
        "IMAGE_TOKEN": "image-secret",
        "API_TOKEN": "runtime-secret",
    }
    assert "export PATH=/runtime/bin:/usr/bin" in dispatched_command
    assert "runtime-secret" not in dispatched_command
    assert "image-secret" not in dispatched_command


@pytest.mark.asyncio
async def test_numeric_user_resolves_through_passwd_and_fails_closed(tmp_path):
    env = _make_env(tmp_path)
    env._sandbox = MagicMock()
    env._sandbox.commands.run = AsyncMock(return_value=MagicMock(stdout="appuser\n"))

    assert await env._resolve_e2b_user(1000) == "appuser"
    assert await env._resolve_e2b_user("1000") == "appuser"
    assert env._sandbox.commands.run.await_count == 1

    env._uid_name_cache.clear()
    env._sandbox.commands.run = AsyncMock(return_value=MagicMock(stdout=""))
    with pytest.raises(RuntimeError, match="no passwd entry"):
        await env._resolve_e2b_user(1000)


@pytest.mark.asyncio
async def test_reconnect_returns_durable_complete_output(tmp_path):
    env = _make_env(tmp_path)
    env._image_user = "appuser"
    first_handle = MagicMock(pid=42)
    first_handle.wait = AsyncMock(side_effect=OSError("stream dropped"))
    second_handle = MagicMock(pid=42)
    second_handle.wait = AsyncMock(return_value=MagicMock(exit_code=0))
    env._sandbox = MagicMock()
    env._sandbox.commands.connect = AsyncMock(return_value=second_handle)
    captured = ExecResult(stdout="before\nafter\n", stderr="warning\n", return_code=0)

    with (
        patch.object(
            env, "_dispatch_command", new=AsyncMock(return_value=first_handle)
        ),
        patch.object(
            env,
            "_read_command_capture",
            new=AsyncMock(side_effect=[None, captured]),
        ),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
        patch.object(e2b_module, "_COMMAND_STREAM_RETRYABLE", (OSError,)),
        patch("pier.environments.e2b.asyncio.sleep", new=AsyncMock()),
    ):
        result = await env.exec("printf 'before\\nafter\\n'")

    assert result == captured
    env._sandbox.commands.connect.assert_awaited_once_with(42, timeout=0)


@pytest.mark.asyncio
async def test_exec_preserves_sdk_result_when_capture_has_no_status(tmp_path):
    env = _make_env(tmp_path)
    handle = MagicMock(pid=42)
    handle.wait = AsyncMock(
        side_effect=e2b_module.CommandExitException(
            stdout="partial output\n",
            stderr="timed out\n",
            exit_code=124,
            error=None,
        )
    )

    with (
        patch.object(env, "_dispatch_command", new=AsyncMock(return_value=handle)),
        patch.object(env, "_read_command_capture", new=AsyncMock(return_value=None)),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()) as cleanup,
    ):
        result = await env.exec("sleep 60", timeout_sec=1)

    assert result == ExecResult(
        stdout="partial output\n", stderr="timed out\n", return_code=124
    )
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_dir_preserves_executables_symlinks_and_empty_dirs(tmp_path):
    import tarfile

    env = _make_env(tmp_path)
    env._sandbox = MagicMock()
    env._sandbox.files.remove = AsyncMock()
    source = tmp_path / "upload"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "helper.sh").write_text("#!/bin/sh\necho ok\n")
    (source / "helper.sh").chmod(0o755)
    (source / "link").symlink_to("helper.sh")

    async def inspect_archive(local, remote):
        with tarfile.open(local) as archive:
            assert archive.getmember("./empty").isdir()
            assert archive.getmember("./helper.sh").mode == 0o755
            assert archive.getmember("./link").issym()
            assert archive.getmember("./link").linkname == "helper.sh"

    with (
        patch.object(
            env, "upload_file", new=AsyncMock(side_effect=inspect_archive)
        ) as upload,
        patch.object(
            env, "exec", new=AsyncMock(return_value=ExecResult(return_code=0))
        ) as execute,
    ):
        await env.upload_dir(source, "/target")
    upload.assert_awaited_once()
    assert "tar " in execute.call_args.args[0]
    env._sandbox.files.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_file_streams_to_disk(tmp_path):
    class Stream:
        def __init__(self):
            self._chunks = iter([b"hello ", b"world"])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    env = _make_env(tmp_path)
    env._sandbox = MagicMock()
    env._sandbox.files.read = AsyncMock(return_value=Stream())
    target = tmp_path / "download" / "file.txt"

    await env.download_file("/remote/file.txt", target)

    assert target.read_bytes() == b"hello world"
    env._sandbox.files.read.assert_awaited_once_with(
        "/remote/file.txt", format="stream", user="root"
    )


@pytest.mark.parametrize(
    "ref",
    [
        "docker.io/ubuntu:24.04",
        "docker.io/library/ubuntu:24.04",
        "index.docker.io/ubuntu:24.04",
        "registry-1.docker.io/ubuntu:24.04",
    ],
)
def test_docker_hub_aliases_use_registry_api(ref):
    assert e2b_module._parse_image_ref(ref) == (
        "registry-1.docker.io",
        "library/ubuntu",
        "24.04",
    )


@pytest.mark.asyncio
async def test_prebuilt_image_ignores_multistage_dockerfile(tmp_path):
    env = _make_env(tmp_path)
    (env.environment_dir / "Dockerfile").write_text(
        "FROM scratch AS build\nFROM build\n"
    )
    env._validate_definition()
    with (
        _mock_image_config(),
        patch.object(
            Template,
            "from_dockerfile",
            side_effect=AssertionError("parsed unused Dockerfile"),
        ),
    ):
        definition = json.loads(Template.to_json(await env._template_definition()))
    assert definition["fromImage"] == "ubuntu:24.04"


@pytest.mark.asyncio
async def test_exec_deadline_terminates_command(tmp_path):
    env = _make_env(tmp_path)
    handle = MagicMock(pid=42)

    async def wait():
        await asyncio.sleep(60)

    handle.wait = AsyncMock(side_effect=wait)
    with (
        patch.object(env, "_dispatch_command", new=AsyncMock(return_value=handle)),
        patch.object(env, "_terminate_command", new=AsyncMock()) as terminate,
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
    ):
        guard = asyncio.timeout(1)
        async with guard:
            with pytest.raises(TimeoutError):
                await env.exec("sleep 60", timeout_sec=0.01)
        assert not guard.expired()
        terminate.assert_awaited_once_with(handle)


@pytest.mark.asyncio
async def test_cancelled_exec_terminates_command(tmp_path):
    env = _make_env(tmp_path)
    started = asyncio.Event()

    async def wait():
        started.set()
        await asyncio.Event().wait()

    handle = MagicMock(pid=42)
    handle.wait = AsyncMock(side_effect=wait)
    with (
        patch.object(env, "_dispatch_command", new=AsyncMock(return_value=handle)),
        patch.object(env, "_terminate_command", new=AsyncMock()) as terminate,
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
    ):
        task = asyncio.create_task(env.exec("sleep 60"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        terminate.assert_awaited_once_with(handle)


@pytest.mark.asyncio
async def test_upload_dir_rejects_missing_source(tmp_path):
    env = _make_env(tmp_path)
    env._sandbox = MagicMock()
    env._sandbox.files.write_files = AsyncMock()
    with pytest.raises(FileNotFoundError):
        await env.upload_dir(tmp_path / "absent", "/target")


@pytest.mark.asyncio
async def test_cancellation_during_dispatch_captures_handle_for_cleanup(tmp_path):
    env = _make_env(tmp_path)
    started = asyncio.Event()
    complete = asyncio.Event()
    handle = MagicMock(pid=42)

    async def dispatch(*args, **kwargs):
        started.set()
        await complete.wait()
        return handle

    with (
        patch.object(env, "_dispatch_command", side_effect=dispatch),
        patch.object(env, "_terminate_command", new=AsyncMock()) as terminate,
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
    ):
        task = asyncio.create_task(env.exec("sleep 60"))
        await started.wait()
        task.cancel()
        complete.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        terminate.assert_awaited_once_with(handle)


@pytest.mark.asyncio
async def test_capture_wrapper_stops_nested_children_on_term(tmp_path):
    # Exercise the actual shell wrapper, including a nested pipeline. SDK mocks
    # alone cannot prove that signalling the wrapper terminates its descendants.
    import shlex

    command = f"touch {shlex.quote(str(tmp_path / 'started'))}; sleep 0.5 | cat; touch {shlex.quote(str(tmp_path / 'late'))}"
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        E2BEnvironment._capture_command(command, str(tmp_path / "capture")),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(3):
            while not (tmp_path / "started").exists():
                await asyncio.sleep(0.01)
            proc.terminate()
            await proc.wait()
        await asyncio.sleep(0.6)
        assert not (tmp_path / "late").exists()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.parametrize(
    "dockerfile", [None, "FROM ubuntu:24.04\n", "FROM base AS builder\nFROM builder\n"]
)
def test_task_requires_prebuilt_image(tmp_path, dockerfile):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    if dockerfile is not None:
        (env_dir / "Dockerfile").write_text(dockerfile)
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    paths.mkdir()
    with pytest.raises(ValueError, match="docker_image"):
        E2BEnvironment(
            environment_dir=env_dir,
            environment_name="task",
            session_id="s",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["CANCELED", "UNAVAILABLE"])
async def test_sdk_pause_errors_reconnect_without_redispatch(tmp_path, code):
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from e2b import AsyncCommandHandle

    async def interrupted_events():
        raise ConnectError(getattr(Code, code), "sandbox paused during command")
        yield  # Make this an async event iterator like the SDK transport.

    env = _make_env(tmp_path)
    handle = AsyncCommandHandle(
        pid=42, handle_kill=AsyncMock(), events=interrupted_events()
    )
    resumed = MagicMock(pid=42)
    resumed.wait = AsyncMock(return_value=MagicMock(exit_code=0))
    env._sandbox = MagicMock()
    env._sandbox.commands.connect = AsyncMock(return_value=resumed)
    captured = ExecResult(stdout="before\nafter\n", stderr="", return_code=0)
    with (
        patch.object(
            env, "_dispatch_command", new=AsyncMock(return_value=handle)
        ) as dispatch,
        patch.object(
            env, "_read_command_capture", new=AsyncMock(side_effect=[None, captured])
        ),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
        patch.object(env, "_terminate_command", new=AsyncMock()) as terminate,
        patch("pier.environments.e2b.asyncio.sleep", new=AsyncMock()),
    ):
        assert (
            await env.exec("printf before; sleep 1; printf after", timeout_sec=10)
            == captured
        )
    dispatch.assert_awaited_once()
    env._sandbox.commands.connect.assert_awaited_once_with(42, timeout=0)
    terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_prebuilt_template_ignores_local_context(tmp_path):
    env = _make_env(tmp_path)
    name = env.template_name
    (env.environment_dir / "Dockerfile").unlink()
    env._validate_definition()
    with _mock_image_config():
        assert (
            json.loads(Template.to_json(await env._template_definition()))["fromImage"]
            == "ubuntu:24.04"
        )
    (env.environment_dir / "Dockerfile").write_bytes(b"unreadable Dockerfile \xff")
    assert env._build_template_name() == name
    changed = _make_env(tmp_path / "changed", docker_image="ubuntu:22.04")
    assert changed.template_name != name


@pytest.mark.asyncio
async def test_pause_recovery_cannot_extend_command_deadline(tmp_path):
    from e2b import TimeoutException

    env = _make_env(tmp_path)
    handle = MagicMock(pid=42)
    handle.wait = AsyncMock(side_effect=TimeoutException("sandbox paused"))
    with (
        patch.object(
            env, "_dispatch_command", new=AsyncMock(return_value=handle)
        ) as dispatch,
        patch.object(env, "_read_command_capture", new=AsyncMock(return_value=None)),
        patch.object(env, "_cleanup_command_capture", new=AsyncMock()),
        patch.object(env, "_terminate_command", new=AsyncMock()) as terminate,
    ):
        guard = asyncio.timeout(1)
        async with guard:
            with pytest.raises(TimeoutError):
                await env.exec("sleep 60", timeout_sec=0.02)
        assert not guard.expired()
    dispatch.assert_awaited_once()
    terminate.assert_awaited_once_with(handle)

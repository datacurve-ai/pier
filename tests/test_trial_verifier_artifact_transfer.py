"""Trial-level tests for artifact upload into verifier envs.

Vendored from harbor's tests/unit/test_trial_verifier_artifact_transfer.py and
the agent-env-stop lifecycle tests from tests/unit/test_trial_verifier_separate.py,
adapted to pier's Trial API (no pytest-asyncio, no with_default_user shim).
"""

import asyncio
import functools
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pier.environments.base import ExecResult
from pier.models.task.config import TaskOS
from pier.models.task.task import Task
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TrialConfig,
    VerifierConfig,
)
from pier.models.trial.config import TaskConfig as TrialTaskConfig
from pier.models.trial.paths import EnvironmentPaths, TrialPaths
from pier.models.trial.result import AgentInfo
from pier.trial.trial import Trial
from pier.verifier.verifier import DownloadVerifierDirError, Verifier


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _task_with_configured_artifacts(
    tmp: Path,
    artifacts: list[str] | str | None = None,
    *,
    separate: bool = True,
) -> Path:
    artifacts_toml = (
        "['/logs/agent/trajectory.json']"
        if artifacts is None
        else artifacts
        if isinstance(artifacts, str)
        else repr(artifacts)
    )
    task_dir = tmp / "task"
    task_dir.mkdir()
    verifier_mode = 'environment_mode = "separate"\n' if separate else ""
    (task_dir / "task.toml").write_text(
        f"artifacts = {artifacts_toml}\n"
        "[agent]\ntimeout_sec = 10.0\n"
        f"[verifier]\ntimeout_sec = 10.0\n{verifier_mode}"
        "[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def _make_env(mounted: bool) -> AsyncMock:
    env = AsyncMock()
    env.default_user = None
    env.capabilities.mounted = mounted
    env.task_os = TaskOS.LINUX
    env.env_paths = EnvironmentPaths()
    env.exec.return_value = ExecResult(stdout="/", stderr="", return_code=0)
    env.is_dir = AsyncMock(return_value=False)
    env.reset_dirs.return_value = None
    env.empty_dirs.return_value = None
    env.start.return_value = None
    env.stop.return_value = None
    env.upload_dir.return_value = None
    env.upload_file.return_value = None

    async def download_dir(source_dir, target_dir):
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "artifact.txt").write_text(source_dir)

    async def download_dir_with_exclusions(source_dir, target_dir, *, exclude):
        await download_dir(source_dir, target_dir)

    async def download_file(source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_path)

    env.download_dir.side_effect = download_dir
    env.download_dir_with_exclusions.side_effect = download_dir_with_exclusions
    env.download_file.side_effect = download_file
    return env


def _mock_agent() -> MagicMock:
    return MagicMock(
        name=lambda: "oracle",
        version=lambda: "1.0",
        SUPPORTS_WINDOWS=True,
        setup=AsyncMock(),
        run=AsyncMock(),
        to_agent_info=lambda: AgentInfo(name="oracle", version="1.0"),
        install_spec=lambda: None,
        network_allowlist=lambda: None,
    )


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("download_fails", [False, True])
@run_async
async def test_failed_verifier_retains_output_and_original_error(
    tmp_path, caplog, cancel, download_fails
):
    task = Task(_task_with_configured_artifacts(tmp_path))
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    environment = _make_env(mounted=False)
    started = asyncio.Event()

    async def execute(command, **kwargs):
        if command.startswith("chmod"):
            return ExecResult(stdout="", stderr="", return_code=0)
        started.set()
        if cancel:
            await asyncio.Future()
        raise RuntimeError("verifier execution failed")

    async def download(source_dir, target_dir):
        if download_fails:
            raise OSError("sandbox unavailable")
        (Path(target_dir) / "test-stdout.txt").write_text("last verifier output")

    environment.exec.side_effect = execute
    environment.download_dir.side_effect = download
    pending = asyncio.create_task(Verifier(task, paths, environment).verify())
    await started.wait()
    if cancel:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await pending
    if download_fails:
        assert "sandbox unavailable" in caplog.text
    else:
        assert paths.test_stdout_path.read_text() == "last verifier output"


@run_async
async def test_completed_verifier_reports_output_download_failure(tmp_path):
    task = Task(_task_with_configured_artifacts(tmp_path))
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    environment = _make_env(mounted=False)
    environment.download_dir.side_effect = OSError("sandbox unavailable")
    with pytest.raises(DownloadVerifierDirError):
        await Verifier(task, paths, environment).verify()


def _make_factory_recorder(
    agent_env: MagicMock, verifier_envs: list[MagicMock]
) -> tuple[MagicMock, list[dict]]:
    """Returns (patched_factory_fn, captured_call_records).

    The first call returns ``agent_env``, subsequent calls cycle through
    ``verifier_envs`` (one per separate verify pass).
    """
    calls: list[dict] = []
    call_index = [0]

    def fake_create(**kwargs):
        calls.append(kwargs)
        idx = call_index[0]
        call_index[0] += 1
        if idx == 0:
            return agent_env
        if idx - 1 < len(verifier_envs):
            return verifier_envs[idx - 1]
        raise AssertionError(
            f"Unexpected factory call #{idx}: {kwargs.get('session_id')}"
        )

    return fake_create, calls


async def _run_trial(task_dir, trials_dir, fake_create):
    config = TrialConfig(
        task=TrialTaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(type="docker", delete=False),
        verifier=VerifierConfig(),
    )
    with (
        patch(
            "pier.trial.trial.EnvironmentFactory.create_environment_from_config",
            side_effect=fake_create,
        ),
        patch(
            "pier.trial.execution.AgentFactory.create_agent_from_config",
            return_value=_mock_agent(),
        ),
    ):
        trial = await Trial.create(config)
        # Simulate the reward file being written into the verifier env's
        # mounted dir.
        trial._trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        trial._trial_paths.reward_text_path.write_text("1.0")
        await trial.run()
        return trial


async def _run(task_dir, trials_dir, agent_env, verifier_env):
    fake_create, _calls = _make_factory_recorder(agent_env, [verifier_env])
    return await _run_trial(task_dir, trials_dir, fake_create)


class TestVerifierArtifactUpload:
    @run_async
    async def test_shared_verifier_does_not_upload_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _task_with_configured_artifacts(Path(tmp), separate=False)
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=True)
            verifier_env = _make_env(mounted=False)

            await _run(task_dir, trials_dir, agent_env, verifier_env)

            verifier_env.start.assert_not_awaited()
            verifier_env.upload_dir.assert_not_awaited()
            verifier_env.upload_file.assert_not_awaited()

    @run_async
    async def test_separate_verifier_uploads_implicit_and_configured_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _task_with_configured_artifacts(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=True)
            verifier_env = _make_env(mounted=True)

            trial = await _run(task_dir, trials_dir, agent_env, verifier_env)

            verifier_env.upload_dir.assert_awaited_once_with(
                source_dir=trial._trial_paths.artifacts_dir,
                target_dir="/logs/artifacts",
            )
            verifier_env.upload_file.assert_awaited_once_with(
                source_path=trial._trial_paths.artifacts_dir / "trajectory.json",
                target_path="/logs/agent/trajectory.json",
            )

    @run_async
    async def test_non_mounted_verifier_gets_artifacts_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _task_with_configured_artifacts(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=True)
            verifier_env = _make_env(mounted=False)

            trial = await _run(task_dir, trials_dir, agent_env, verifier_env)

            verifier_env.upload_dir.assert_awaited_once_with(
                source_dir=trial._trial_paths.artifacts_dir,
                target_dir="/logs/artifacts",
            )
            verifier_env.upload_file.assert_awaited_once_with(
                source_path=trial._trial_paths.artifacts_dir / "trajectory.json",
                target_path="/logs/agent/trajectory.json",
            )

    @run_async
    async def test_agent_logs_uploaded_before_log_artifact_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _task_with_configured_artifacts(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=False)
            verifier_env = _make_env(mounted=False)
            events: list[tuple[str, str]] = []

            async def upload_dir(source_dir, target_dir):
                events.append(("agent_upload_dir", target_dir))

            async def download_file(source_path, target_path):
                events.append(("agent_download_file", source_path))
                target = Path(target_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source_path)

            async def verifier_upload_file(source_path, target_path):
                events.append(("verifier_upload_file", target_path))

            async def verifier_exec(*args, **kwargs):
                events.append(("verifier_exec", ""))
                return ExecResult(stdout="/", stderr="", return_code=0)

            agent_env.upload_dir.side_effect = upload_dir
            agent_env.download_file.side_effect = download_file
            verifier_env.upload_file.side_effect = verifier_upload_file
            verifier_env.exec.side_effect = verifier_exec

            await _run(task_dir, trials_dir, agent_env, verifier_env)

            assert events.index(("agent_upload_dir", "/logs/agent")) < events.index(
                ("agent_download_file", "/logs/agent/trajectory.json")
            )
            assert events.index(
                ("agent_download_file", "/logs/agent/trajectory.json")
            ) < events.index(("verifier_upload_file", "/logs/agent/trajectory.json"))
            assert events.index(
                ("verifier_upload_file", "/logs/agent/trajectory.json")
            ) < events.index(("verifier_exec", ""))

    @run_async
    async def test_directory_artifact_exclude_applies_to_collection_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            (task_dir / "task.toml").write_text(
                "artifacts = ["
                '{ source = "/logs/artifacts", exclude = ["*.pt", "cache"] }'
                "]\n"
                "[agent]\ntimeout_sec = 10.0\n"
                "[verifier]\ntimeout_sec = 10.0\n"
                "[verifier.environment]\n"
                "[environment]\n"
            )
            (task_dir / "instruction.md").write_text("Do nothing.\n")
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
            (task_dir / "tests").mkdir()
            (task_dir / "tests" / "Dockerfile").write_text("FROM ubuntu:24.04\n")

            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=False)
            agent_env.is_dir.return_value = True
            verifier_env = _make_env(mounted=False)

            trial = await _run(task_dir, trials_dir, agent_env, verifier_env)

            agent_env.download_dir_with_exclusions.assert_any_await(
                source_dir="/logs/artifacts",
                target_dir=trial._trial_paths.artifacts_dir,
                exclude=["*.pt", "cache"],
            )
            verifier_env.upload_dir.assert_awaited_once_with(
                source_dir=trial._trial_paths.artifacts_dir,
                target_dir="/logs/artifacts",
            )

    @run_async
    async def test_configured_artifact_uploads_destination_back_to_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _task_with_configured_artifacts(
                Path(tmp),
                artifacts=(
                    '[{ source = "/tmp/answer.json", '
                    'destination = "answers/final.json" }]'
                ),
            )
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()
            agent_env = _make_env(mounted=False)
            verifier_env = _make_env(mounted=False)

            trial = await _run(task_dir, trials_dir, agent_env, verifier_env)
            artifact_path = trial._trial_paths.artifacts_dir / "answers" / "final.json"

            agent_env.download_file.assert_any_await(
                source_path="/tmp/answer.json",
                target_path=artifact_path,
            )
            verifier_env.upload_file.assert_awaited_once_with(
                source_path=artifact_path,
                target_path="/tmp/answer.json",
            )


def _single_step_task_with_separate_verifier(tmp: Path) -> Path:
    task_dir = tmp / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 10.0\n"
        "[verifier]\ntimeout_sec = 10.0\n"
        "[verifier.environment]\n"  # implicit separate
        "[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    # No test.sh on host — the verifier image owns it.
    (tests_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN mkdir -p /tests\n"
        'RUN echo "#!/bin/bash\\necho 1 > /logs/verifier/reward.txt" > /tests/test.sh\n'
        "RUN chmod +x /tests/test.sh\n"
    )
    return task_dir


def _multi_step_task_with_step_tests(tmp: Path) -> Path:
    """Multi-step task where the grade step has its own verifier package."""
    task_dir = tmp / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[environment]\n\n"
        "[[steps]]\n"
        'name = "grade"\n'
        '[steps.verifier]\nenvironment_mode = "separate"\n'
    )
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    step_dir = task_dir / "steps" / "grade"
    step_dir.mkdir(parents=True)
    (step_dir / "instruction.md").write_text("Grade.\n")
    (step_dir / "tests").mkdir()
    (step_dir / "tests" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return task_dir


def _multi_step_task_all_separate(tmp: Path) -> Path:
    task_dir = tmp / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[environment]\n\n"
        "[[steps]]\n"
        'name = "build"\n'
        '[steps.verifier]\nenvironment_mode = "separate"\n'
        "[[steps]]\n"
        'name = "grade"\n'
        '[steps.verifier]\nenvironment_mode = "separate"\n'
    )
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (tests_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    for step in ("build", "grade"):
        step_dir = task_dir / "steps" / step
        step_dir.mkdir(parents=True)
        (step_dir / "instruction.md").write_text(f"Do {step}.\n")
    return task_dir


class TestSeparateVerifierAgentEnvLifecycle:
    """Artifacts are handed off first; then the agent env can stop early."""

    @run_async
    async def test_agent_env_stops_before_separate_verifier_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _single_step_task_with_separate_verifier(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()

            events: list[str] = []
            agent_env = _make_env(mounted=True)
            verifier_env = _make_env(mounted=True)

            async def stop_agent(delete: bool):
                events.append("agent_stop")

            async def verifier_start(force_build: bool):
                events.append("verifier_start")

            agent_env.stop.side_effect = stop_agent
            verifier_env.start.side_effect = verifier_start
            fake_create, _calls = _make_factory_recorder(agent_env, [verifier_env])

            await _run_trial(task_dir, trials_dir, fake_create)

            assert events.index("agent_stop") < events.index("verifier_start")

    @run_async
    async def test_final_separate_step_stops_agent_env_before_verifier_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _multi_step_task_with_step_tests(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()

            events: list[str] = []
            agent_env = _make_env(mounted=True)
            grade_env = _make_env(mounted=True)

            async def stop_agent(delete: bool):
                events.append("agent_stop")

            async def verifier_start(force_build: bool):
                events.append("verifier_start")

            agent_env.stop.side_effect = stop_agent
            grade_env.start.side_effect = verifier_start
            fake_create, _calls = _make_factory_recorder(agent_env, [grade_env])

            await _run_trial(task_dir, trials_dir, fake_create)

            assert events.index("agent_stop") < events.index("verifier_start")

    @run_async
    async def test_non_final_separate_step_keeps_agent_env_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _multi_step_task_all_separate(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()

            events: list[str] = []
            agent_env = _make_env(mounted=True)
            build_env = _make_env(mounted=True)
            grade_env = _make_env(mounted=True)

            async def stop_agent(delete: bool):
                events.append("agent_stop")

            async def build_start(force_build: bool):
                events.append("build_verifier_start")

            async def grade_start(force_build: bool):
                events.append("grade_verifier_start")

            agent_env.stop.side_effect = stop_agent
            build_env.start.side_effect = build_start
            grade_env.start.side_effect = grade_start
            fake_create, _calls = _make_factory_recorder(
                agent_env,
                [build_env, grade_env],
            )

            await _run_trial(task_dir, trials_dir, fake_create)

            assert events.index("build_verifier_start") < events.index("agent_stop")
            assert events.index("agent_stop") < events.index("grade_verifier_start")

    @run_async
    async def test_step_artifacts_uploaded_into_step_verifier_env(self):
        """Multi-step separate verification receives the step's artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _multi_step_task_with_step_tests(Path(tmp))
            trials_dir = Path(tmp) / "trials"
            trials_dir.mkdir()

            agent_env = _make_env(mounted=True)
            grade_env = _make_env(mounted=True)
            fake_create, _calls = _make_factory_recorder(agent_env, [grade_env])

            trial = await _run_trial(task_dir, trials_dir, fake_create)

            # Mounted agent env: the step collection pass targets the
            # trial-level artifacts dir, which is then uploaded into the
            # separate verifier env's convention dir.
            grade_env.upload_dir.assert_awaited_once_with(
                source_dir=trial._trial_paths.artifacts_dir,
                target_dir="/logs/artifacts",
            )

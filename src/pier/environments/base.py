import asyncio
import logging
import shlex
import tarfile
import tempfile
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePath, PurePosixPath
from typing import Literal

from pydantic import BaseModel

from pier.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from pier.environments.resource_policies import (
    validate_resource_capabilities,
    validate_resource_values,
)
from pier.models.agent.install import AgentInstallSpec
from pier.models.agent.network import NetworkAllowlist
from pier.models.task.config import EnvironmentConfig, HealthcheckConfig, TaskOS
from pier.models.trial.config import ResourceMode
from pier.models.trial.paths import EnvironmentPaths, TrialPaths
from pier.utils.env import resolve_env_vars
from pier.utils.logger import logger as global_logger
from pier.utils.scripts import quote_shell_arg

EnvironmentPath = str | PurePath
_TRANSFER_TAR_TEMPLATE = ".hb-transfer-{uuid}.tar.gz"
_ENV_TRANSFER_TAR_DIR = PurePosixPath("/tmp")


class HealthcheckError(RuntimeError):
    pass


class ExecResult(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    return_code: int


ExecOutputCallback = Callable[[str, str], Awaitable[None]]
ExecInputQueue = asyncio.Queue[str | None]


class BaseEnvironment(ABC):
    """
    The containerized environment the agent interacts with.
    Consists of 1+ container(s).

    Examples of types of environments: Docker, Apptainer, Containerd, Podman
    """

    environment_dir: Path
    environment_name: str
    session_id: str
    trial_paths: TrialPaths
    task_env_config: EnvironmentConfig
    logger: logging.Logger

    default_user: str | int | None

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        cpu_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        memory_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        suppress_override_warnings: bool = False,
        persistent_env: dict[str, str] | None = None,
        agent_install_spec: AgentInstallSpec | None = None,
        network_allowlist: NetworkAllowlist | None = None,
        default_user: str | int | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a BaseEnvironment from a directory path and name.

        Args:
            environment_dir: Path to the environment directory. The directory should
            contain the environment definition files (e.g. docker-compose.yaml).
            environment_name: The name of the environment. Typically <task_name>.
            session_id: The session ID for this instance of the environment. Typically
                the trial name, e.g. <task_name>__<trial_id>.
            trial_paths: The trial paths.
            task_env_config: The environment configuration from the task.
            logger: The logger to use for the environment.
        """
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.default_user = default_user

        self.task_env_config = task_env_config

        self._override_cpus = override_cpus
        self._override_memory_mb = override_memory_mb
        self._override_storage_mb = override_storage_mb
        self._override_gpus = override_gpus
        self._cpu_resource_mode = ResourceMode(cpu_enforcement_policy)
        self._memory_resource_mode = ResourceMode(memory_enforcement_policy)
        self._suppress_override_warnings = suppress_override_warnings
        self._persistent_env: dict[str, str] = persistent_env or {}
        self.agent_install_spec = agent_install_spec
        self.network_allowlist = network_allowlist or NetworkAllowlist()

        self.logger = (logger or global_logger).getChild(__name__)

        self._maybe_override_task_env_config()
        self._maybe_resolve_task_env()

        self._validate_definition()
        self._validate_resource_mode_support()
        self._validate_gpu_support()
        self._validate_internet_config()
        self._validate_agent_setup_options()
        self._validate_windows_support()

    @property
    def _uses_compose(self) -> bool:
        return False

    def _maybe_resolve_task_env(self):
        if self.task_env_config.env and not self._uses_compose:
            resolved = resolve_env_vars(self.task_env_config.env)
            self._persistent_env = {**resolved, **self._persistent_env}

    def _maybe_override_task_env_config(self):
        if self._override_cpus is not None:
            self.task_env_config.cpus = self._override_cpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    f"Overriding CPU count to {self._override_cpus} alters the "
                    "task from its intended configuration. This could disqualify you "
                    "from leaderboard submissions for some benchmarks."
                )
        if self._override_memory_mb is not None:
            self.task_env_config.memory_mb = self._override_memory_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    f"Overriding memory to {self._override_memory_mb} MB alters the "
                    "task from its intended configuration. This could disqualify you "
                    "from leaderboard submissions for some benchmarks."
                )
        if self._override_storage_mb is not None:
            self.task_env_config.storage_mb = self._override_storage_mb
            if not self._suppress_override_warnings:
                self.logger.warning(
                    f"Overriding storage to {self._override_storage_mb} MB alters the "
                    "task from its intended configuration. This could disqualify you "
                    "from leaderboard submissions for some benchmarks."
                )
        if self._override_gpus is not None:
            self.task_env_config.gpus = self._override_gpus
            if not self._suppress_override_warnings:
                self.logger.warning(
                    f"Overriding GPU count to {self._override_gpus} alters the "
                    "task from its intended configuration. This could disqualify you "
                    "from leaderboard submissions for some benchmarks."
                )

    def _resource_mode(self, resource: Literal["cpu", "memory"]) -> ResourceMode:
        return (
            getattr(self, "_cpu_resource_mode", ResourceMode.AUTO)
            if resource == "cpu"
            else getattr(self, "_memory_resource_mode", ResourceMode.AUTO)
        )

    def _resource_value(self, resource: Literal["cpu", "memory"]) -> int | None:
        if self._resource_mode(resource) == ResourceMode.IGNORE:
            return None
        if resource == "cpu":
            return self.task_env_config.cpus
        return self.task_env_config.memory_mb

    def _resource_request_value(
        self,
        resource: Literal["cpu", "memory"],
        *,
        auto_mode: ResourceMode,
    ) -> int | None:
        return self._resource_policy_value(
            resource,
            target=ResourceMode.REQUEST,
            auto_mode=auto_mode,
        )

    def _resource_limit_value(
        self,
        resource: Literal["cpu", "memory"],
        *,
        auto_mode: ResourceMode,
    ) -> int | None:
        return self._resource_policy_value(
            resource,
            target=ResourceMode.LIMIT,
            auto_mode=auto_mode,
        )

    def _resource_policy_value(
        self,
        resource: Literal["cpu", "memory"],
        *,
        target: ResourceMode,
        auto_mode: ResourceMode,
    ) -> int | None:
        value = self._resource_value(resource)
        if value is None:
            return None
        mode = self._resource_mode(resource)
        if mode == ResourceMode.AUTO:
            mode = auto_mode
        if mode == target or mode == ResourceMode.GUARANTEE:
            return value
        return None

    @property
    def _effective_cpus(self) -> int | None:
        return self._resource_value("cpu")

    @property
    def _effective_memory_mb(self) -> int | None:
        return self._resource_value("memory")

    @property
    def _effective_storage_mb(self) -> int | None:
        return self.task_env_config.storage_mb

    @property
    def _effective_gpus(self) -> int:
        return self.task_env_config.gpus or 0

    def _validate_resource_mode_support(self) -> None:
        resource_capabilities = type(self).resource_capabilities()
        if resource_capabilities is None:
            return

        environment_type = self.type()
        environment_label = str(getattr(environment_type, "value", environment_type))
        validate_resource_capabilities(
            environment_label=environment_label,
            resource_capabilities=resource_capabilities,
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
        )
        validate_resource_values(
            cpu_enforcement_policy=self._cpu_resource_mode,
            memory_enforcement_policy=self._memory_resource_mode,
            cpus=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
        )

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        """Resolve the effective user for a command.

        Returns ``user`` if explicitly provided, otherwise falls back to
        ``self.default_user``.  This allows the orchestrator to configure a
        default user (e.g. the task's agent user) on the environment once,
        so agent implementations don't need to thread a ``user`` parameter
        through every ``exec`` call.
        """
        return user if user is not None else self.default_user

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge persistent env vars with per-exec env vars.

        Per-exec env vars take precedence over persistent ones.
        """
        if not self._persistent_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        return merged or None

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Return environment variables for installed-agent commands.

        Filtered-egress environments override this to scope proxy variables to
        agent setup/run commands instead of verifier and task commands.
        """
        return env

    def _reset_dirs_command(
        self,
        *,
        remove_dirs: Sequence[EnvironmentPath],
        create_dirs: Sequence[EnvironmentPath],
        chmod_dirs: Sequence[EnvironmentPath] | None = None,
    ) -> str:
        """Build a shell command that resets environment directories."""
        q = lambda p: quote_shell_arg(p, self.task_os)  # noqa: E731

        if self.task_os == TaskOS.WINDOWS:
            commands = [
                f"if exist {q(path)} rmdir /S /Q {q(path)}" for path in remove_dirs
            ]
            commands.extend(f"mkdir {q(path)}" for path in create_dirs)
            return " & ".join(commands)

        remove_args = " ".join(q(path) for path in remove_dirs)
        create_args = " ".join(q(path) for path in create_dirs)
        command = f"rm -rf {remove_args} && mkdir -p {create_args}"
        if chmod_dirs:
            chmod_args = " ".join(q(path) for path in chmod_dirs)
            command += f" && chmod 777 {chmod_args}"
        return command

    def _empty_dirs_command(
        self,
        dirs: Sequence[EnvironmentPath],
        *,
        chmod: bool = True,
    ) -> str:
        """Build a shell command that empties directories without replacing roots."""
        q = lambda p: quote_shell_arg(p, self.task_os)  # noqa: E731

        if self.task_os == TaskOS.WINDOWS:
            commands: list[str] = []
            for path in dirs:
                path_str = str(path).rstrip("\\/")
                dir_probe = f"{path_str}\\NUL"
                children = f"{path_str}\\*"
                commands.extend(
                    [
                        f"if exist {q(path)} if not exist {q(dir_probe)} del /F /Q {q(path)}",
                        f"if not exist {q(dir_probe)} mkdir {q(path)}",
                        f"del /F /Q {q(children)} 2>NUL",
                        f'for /D %I in ({q(children)}) do rmdir /S /Q "%I"',
                    ]
                )
            return " & ".join(commands)

        commands = []
        for path in dirs:
            quoted = q(path)
            commands.extend(
                [
                    f"if [ -L {quoted} ] || {{ [ -e {quoted} ] && [ ! -d {quoted} ]; }}; then rm -rf {quoted}; fi",
                    f"mkdir -p {quoted}",
                    f"find {quoted} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +",
                ]
            )
            if chmod:
                commands.append(f"chmod 777 {quoted}")
        return " && ".join(commands)

    def _reset_dirs_user(self) -> str | None:
        """Use root only where that user exists and chmod is meaningful."""
        if self.task_os == TaskOS.WINDOWS:
            return None
        return "root"

    async def reset_dirs(
        self,
        *,
        remove_dirs: Sequence[EnvironmentPath],
        create_dirs: Sequence[EnvironmentPath],
        chmod_dirs: Sequence[EnvironmentPath] | None = None,
    ) -> ExecResult:
        """Remove and recreate environment directories using the target OS shell."""
        return await self.exec(
            self._reset_dirs_command(
                remove_dirs=remove_dirs,
                create_dirs=create_dirs,
                chmod_dirs=chmod_dirs,
            ),
            user=self._reset_dirs_user(),
        )

    async def empty_dirs(
        self,
        dirs: Sequence[EnvironmentPath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        """Ensure directories exist and are empty without replacing directory roots."""
        if not dirs:
            return None
        return await self.exec(
            self._empty_dirs_command(dirs, chmod=chmod),
            user=self._reset_dirs_user(),
        )

    @staticmethod
    @abstractmethod
    def type() -> str:
        # Returns str rather than EnvironmentType so that third-party
        # environments outside this repo can return arbitrary identifiers
        # without modifying the EnvironmentType enum.  Built-in environments
        # still return EnvironmentType members, which are str subclasses.
        """The environment type."""

    @property
    def env_paths(self) -> EnvironmentPaths:
        """Container-side environment paths. Override for non-POSIX containers."""
        return EnvironmentPaths.for_os(self.task_os)

    @property
    def task_os(self) -> TaskOS:
        """Target operating system declared by the task's [environment].os field."""
        return self.task_env_config.os

    _LEGACY_CAPABILITY_ATTRS: dict[str, str] = {
        "supports_gpus": "gpus",
        "can_disable_internet": "disable_internet",
        "is_mounted": "mounted",
    }

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        legacy = [name for name in cls._LEGACY_CAPABILITY_ATTRS if name in cls.__dict__]
        if legacy:
            warnings.warn(
                f"{cls.__name__} declares deprecated capability properties: "
                f"{', '.join(legacy)}. Override the `capabilities` property to "
                "return an `EnvironmentCapabilities` instance instead. The "
                "legacy properties will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        """The capabilities supported by this environment.

        Subclasses should override this property to return an
        ``EnvironmentCapabilities`` instance. Accessed during ``__init__``
        by the capability validators, so subclasses that derive
        capabilities from instance state must set up that state before
        calling ``super().__init__`` (see Modal's ``_compose_mode`` for
        an example).

        For backwards compatibility, this default implementation also
        reads the deprecated ``supports_gpus`` / ``can_disable_internet``
        / ``is_mounted`` properties if a subclass still declares them.
        Overriding this property takes precedence. The deprecation
        warning is emitted once at class definition via
        :meth:`__init_subclass__`.
        """
        kwargs: dict[str, bool] = {}
        for old_name, new_name in self._LEGACY_CAPABILITY_ATTRS.items():
            if hasattr(type(self), old_name):
                kwargs[new_name] = getattr(self, old_name)
        return EnvironmentCapabilities(**kwargs)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities | None:
        return None

    @classmethod
    def supports_exec_stdin(cls) -> bool:
        """Whether streamed exec processes can receive incremental stdin."""
        return False

    @abstractmethod
    def _validate_definition(self):
        """
        Validate that the necessary environment files are present.

        Raises:
            FileNotFoundError: If the necessary environment files are not present.
            [CustomError]: If the environment definition is invalid.
        """

    def _validate_gpu_support(self):
        """
        Validate that GPU requirements are supported by this environment.

        Raises:
            RuntimeError: If the task requires GPU but the environment doesn't support it.
        """
        if self._effective_gpus > 0 and not self.capabilities.gpus:
            raise RuntimeError(
                f"Task requires {self._effective_gpus} GPU(s) but {self.type()} "
                f"environment does not support GPU allocation. Please use a GPU-capable "
                f"environment type (e.g., Modal, Docker with nvidia-docker)."
            )

    def _validate_internet_config(self):
        """
        Validate that internet configuration is supported by this environment.

        Raises:
            ValueError: If internet isolation is requested but not supported.
        """
        if (
            not self.task_env_config.allow_internet
            and not self.capabilities.disable_internet
        ):
            raise ValueError(
                f"allow_internet=False is not supported by {self.type()} environment."
            )

    def _validate_agent_setup_options(self):
        if (
            self.agent_install_spec is not None
            and not self.capabilities.preinstall_agents
        ):
            raise ValueError(
                f"Agent preinstall is not supported by {self.type()} environment."
            )

        if (
            not self.task_env_config.allow_internet
            and self.network_allowlist.domains
            and not self.capabilities.filtered_egress
        ):
            raise ValueError(
                f"Filtered inference egress is not supported by {self.type()} environment."
            )

    def _validate_windows_support(self):
        """
        Validate that the target OS is supported by this environment.

        Raises:
            RuntimeError: If the task targets Windows but the environment
                cannot run Windows containers.
        """
        if self.task_env_config.os == TaskOS.WINDOWS and not self.capabilities.windows:
            raise RuntimeError(
                f"Task declares [environment].os = 'windows' but the "
                f"{self.type()} environment does not support Windows containers. "
                "Use an environment type that does (currently: docker)."
            )

    @classmethod
    def preflight(cls) -> None:
        """Check that required credentials/config are available before queueing trials.

        Called once before any trials are queued. Subclasses should override
        this to verify provider-specific credentials exist.

        Raises:
            SystemExit: If required credentials are missing.
        """

    @abstractmethod
    async def start(self, force_build: bool) -> None:
        """Starts the environment and optionally forces a build."""

    @abstractmethod
    async def stop(self, delete: bool):
        """Stops the environment and optionally deletes it."""

    async def prepare_logs_for_host(self) -> None:
        """Fix log file permissions so the host process can read them.

        Called before agent logs are read on the host side (e.g. for trajectory
        conversion). Mounted environments (Docker on Linux) need to chown files
        written by the in-container agent user; other environments are no-ops.
        """

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str):
        """
        Adds a local file to the environment.

        Args:
            source_path: The path to the source local file.
            target_path: The path to which to copy the file.
        """

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        """
        Adds a local directory to the environment.

        Args:
            source_dir: The path to the source local directory.
            target_dir: The path to which to copy the directory.
        """

    @abstractmethod
    async def download_file(self, source_path: str, target_path: Path | str):
        """
        Downloads a file from the environment to the local machine.

        Args:
            source_path: The path to the source file in the environment.
            target_path: The local path to which to copy the file.
        """

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str):
        """
        Downloads a directory from the environment to the local machine. This overwrites
        existing files in the target directory.

        Args:
            source_dir: The path to the source directory in the environment.
            target_dir: The local path to which to copy the directory.
        """

    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        """Download a directory through a temporary tar archive with excludes."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        exclude_flags = " ".join(
            f"--exclude={shlex.quote(pattern)}" for pattern in exclude
        )
        env_tar_filename = _TRANSFER_TAR_TEMPLATE.format(uuid=uuid.uuid4())
        env_tar_path = str(_ENV_TRANSFER_TAR_DIR / env_tar_filename)
        source_path = shlex.quote(source_dir)

        result = await self.exec(
            f"tar czf {shlex.quote(env_tar_path)} {exclude_flags} -C {source_path} .",
            timeout_sec=120,
            user="root",
        )
        if result.return_code != 0:
            output = result.stderr or result.stdout or "no output"
            raise RuntimeError(
                "Failed to create transfer archive for "
                f"{source_dir!r} with code {result.return_code}: {output}"
            )

        with tempfile.TemporaryDirectory() as host_tmp_dir:
            host_tar_path = Path(host_tmp_dir) / env_tar_filename
            await self.download_file(
                source_path=env_tar_path,
                target_path=host_tar_path,
            )

            with tarfile.open(host_tar_path, "r:gz") as tf:
                tf.extractall(path=target, filter="data")

        cleanup_result = await self.exec(
            f"rm -f {shlex.quote(env_tar_path)}",
            timeout_sec=120,
            user="root",
        )
        if cleanup_result.return_code != 0:
            output = cleanup_result.stderr or cleanup_result.stdout or "no output"
            self.logger.warning(
                "Failed to remove transfer archive "
                f"{env_tar_path!r} with code {cleanup_result.return_code}: {output}"
            )

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """
        Executes a command in the environment.

        Args:
            command: The command to execute.
            cwd: The working directory in which to execute the command.
            env: The environment  variables to set.
            timeout_sec: The timeout in seconds.
            user: Username or UID to run the command as. None falls back to
                ``self.default_user``; if that is also None the environment's
                container default (typically root) is used.
        """

    async def exec_stream(
        self,
        command: str,
        on_output: ExecOutputCallback,
        input_queue: ExecInputQueue | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute and report output incrementally when the backend supports it.

        The compatibility fallback reports captured output after completion. Backends
        override this to provide real-time delivery. Backends advertising
        ``capabilities.exec_stdin`` also forward ``input_queue`` to the process.
        """
        if input_queue is not None:
            raise NotImplementedError(
                "This environment does not support streamed stdin"
            )
        result = await self.exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
        if result.stdout:
            await on_output("stdout", result.stdout)
        if result.stderr:
            await on_output("stderr", result.stderr)
        return result

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a directory.

        Uses ``test -d`` on POSIX targets and cmd.exe's ``if exist "<path>\\"``
        idiom on Windows (the trailing backslash matches only directories).
        Subclasses may override with a native SDK call.
        """
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=True),
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a regular file.

        Uses ``test -f`` on POSIX targets. On Windows, checks that the path
        exists but is not a directory. Subclasses may override with a
        native SDK call.
        """
        result = await self.exec(
            self._path_kind_check_command(path, require_dir=False),
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    def _path_kind_check_command(self, path: str, *, require_dir: bool) -> str:
        """Build an OS-aware command that exits 0 iff *path* matches the kind.

        ``require_dir=True`` checks for a directory; ``False`` checks for a
        regular file. On Windows the trailing-backslash ``if exist`` idiom
        is used to distinguish directories from files.
        """
        if self.task_os == TaskOS.WINDOWS:
            quoted_path = quote_shell_arg(path, self.task_os)
            quoted_as_dir = quote_shell_arg(str(path) + "\\", self.task_os)
            if require_dir:
                return f"if exist {quoted_as_dir} (exit 0) else (exit 1)"
            return (
                f"if not exist {quoted_path} exit 1 & "
                f"if exist {quoted_as_dir} exit 1 & "
                f"exit 0"
            )
        flag = "d" if require_dir else "f"
        return f"test -{flag} {shlex.quote(path)}"

    async def run_healthcheck(
        self, healthcheck: HealthcheckConfig | None = None
    ) -> None:
        """Run a healthcheck, defaulting to the environment-level config.

        Mirrors Docker HEALTHCHECK semantics: during the start period,
        failures don't count toward retries. After the start period,
        consecutive failures are counted and the check fails after
        exceeding the retry limit.

        Args:
            healthcheck: Optional override. When ``None``, falls back to
                ``task_env_config.healthcheck`` (the top-level healthcheck).
                Callers pass a per-step config here to run a step-scoped
                healthcheck.
        """
        hc = (
            healthcheck if healthcheck is not None else self.task_env_config.healthcheck
        )
        if hc is None:
            return

        self.logger.debug(f"Running healthcheck: {hc.command}")

        start_time = time.monotonic()
        start_period_end = start_time + hc.start_period_sec
        consecutive_failures = 0

        while True:
            now = time.monotonic()
            in_start_period = now < start_period_end

            result = await self.exec(hc.command, timeout_sec=int(hc.timeout_sec))

            if result.return_code == 0:
                self.logger.debug("Healthcheck passed")
                return

            self.logger.debug(
                f"Healthcheck failed (rc={result.return_code}, "
                f"in_start_period={in_start_period})"
            )

            if in_start_period:
                await asyncio.sleep(hc.start_interval_sec)
            else:
                consecutive_failures += 1
                if consecutive_failures >= hc.retries:
                    raise HealthcheckError(
                        f"Healthcheck failed after {hc.retries} consecutive "
                        f"retries: {hc.command}"
                    )
                await asyncio.sleep(hc.interval_sec)

    async def attach(self) -> None:
        """Attaches to the environment using os.execvp."""
        raise NotImplementedError("This environment does not support attaching.")

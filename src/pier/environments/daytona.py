from __future__ import annotations

import asyncio
import atexit
import ipaddress
import json
import os
import shlex
import socket
import tempfile
import urllib.parse
import urllib.request
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

from pier.environments.agent_setup import dockerfile_install_commands
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from pier.environments.docker import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    RESOURCES_COMPOSE_NAME,
    write_resources_compose_file,
)
from pier.environments.docker.docker import _sanitize_docker_image_name
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.config import ResourceMode
from pier.models.trial.paths import EnvironmentPaths, TrialPaths
from pier.utils.env import resolve_env_vars
from pier.utils.logger import logger
from pier.utils.optional_import import MissingExtraError

try:
    from daytona import (
        AsyncDaytona,
        AsyncSandbox,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        DaytonaNotFoundError,
        FileDownloadRequest,
        FileUpload,
        Image,
        Resources,
        SessionExecuteRequest,
    )
    from daytona._async.snapshot import SnapshotState

    try:
        from daytona import DaytonaConfig
    except ImportError:
        DaytonaConfig = None  # type: ignore[assignment]

    _HAS_DAYTONA = True
except ImportError:
    _HAS_DAYTONA = False

if TYPE_CHECKING:
    from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams

_SandboxParams = Union["CreateSandboxFromImageParams", "CreateSandboxFromSnapshotParams"]

DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS = 10


def _daytona_preflight() -> None:
    has_api_key = bool(os.environ.get("DAYTONA_API_KEY"))
    has_jwt_auth = bool(
        os.environ.get("DAYTONA_JWT_TOKEN")
        and os.environ.get("DAYTONA_ORGANIZATION_ID")
    )
    if not (has_api_key or has_jwt_auth):
        raise SystemExit(
            "Daytona requires either DAYTONA_API_KEY, or both "
            "DAYTONA_JWT_TOKEN and DAYTONA_ORGANIZATION_ID, to be set. "
            "Please set the required environment variables and try again."
        )


def _collapse_cidrs_to_budget(cidrs: list[str], *, budget: int) -> list[str]:
    networks = [
        ipaddress.ip_network(cidr, strict=False)
        for cidr in cidrs
        if ipaddress.ip_network(cidr, strict=False).version == 4
    ]
    working = list(ipaddress.collapse_addresses(networks))
    while len(working) > budget:
        working.sort(key=lambda net: (-net.prefixlen, int(net.network_address)))
        working[0] = working[0].supernet()
        working = list(ipaddress.collapse_addresses(working))
    return sorted(str(net) for net in working)


def _cidrs_from_domain_resolution(
    domain_resolution: dict[str, list[str]],
) -> list[str]:
    cidrs: list[str] = []
    for addrs in domain_resolution.values():
        for addr in addrs:
            ip = ipaddress.ip_address(addr)
            if ip.version == 4:
                cidrs.append(f"{addr}/32")
    return _collapse_cidrs_to_budget(
        sorted(set(cidrs)),
        budget=DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS,
    )


# Address space a cloud sandbox can never reach: answers in these ranges come from a
# split-horizon (corporate/VPN) resolver's internal view of the domain, not the public
# endpoint the sandbox will connect to. Shipping them in the allowlist silently blocks
# every real connection.
_SANDBOX_UNREACHABLE_NETS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",  # "this network"
        "10.0.0.0/8",  # RFC1918
        "100.64.0.0/10",  # CGNAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local
        "172.16.0.0/12",  # RFC1918
        "192.168.0.0/16",  # RFC1918
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved
    )
)

# Public DNS-over-HTTPS resolvers used as a fallback when the local resolver returns no
# sandbox-reachable answer for an allowlist domain (split-horizon DNS, or no A record in
# the local view). Queried over plain HTTPS with the JSON API; no extra dependencies.
_DOH_RESOLVER_URLS = (
    "https://dns.google/resolve",
    "https://cloudflare-dns.com/dns-query",
)
_DOH_TIMEOUT_SECONDS = 5.0
_DNS_TYPE_A = 1


def _is_sandbox_reachable_ipv4(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.version != 4:
        return False
    return not any(ip in net for net in _SANDBOX_UNREACHABLE_NETS)


def _resolve_domain_via_public_doh(domain: str) -> list[str]:
    """Best-effort A-record lookup through public DNS-over-HTTPS resolvers.

    Used only when the local resolver produced no sandbox-reachable answer, so a
    launcher behind split-horizon DNS still allowlists the public addresses the
    sandbox will actually connect to. Returns [] when every resolver fails.
    """
    for resolver_url in _DOH_RESOLVER_URLS:
        query = urllib.parse.urlencode({"name": domain, "type": "A"})
        request = urllib.request.Request(
            f"{resolver_url}?{query}",
            headers={"accept": "application/dns-json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=_DOH_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            continue
        addrs = sorted(
            {
                answer.get("data", "")
                for answer in payload.get("Answer") or []
                if answer.get("type") == _DNS_TYPE_A
                and _is_sandbox_reachable_ipv4(answer.get("data", ""))
            }
        )
        if addrs:
            return addrs
    return []


def resolve_network_allowlist_to_daytona_cidrs(
    domains: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve exact allowlist domains into Daytona-compatible IPv4 CIDRs.

    Daytona network limits accept IPv4 CIDR blocks only. Leading-dot suffix
    domains cannot be resolved deterministically, so they are ignored here.

    Resolution runs on the LAUNCHER, but the allowlist gates the SANDBOX's
    egress, so answers only the launcher's network can reach (split-horizon
    corporate/VPN DNS returning private or CGNAT addresses) are dropped, and a
    domain with no sandbox-reachable answer falls back to public DNS-over-HTTPS.
    Without this, a launcher behind split-horizon DNS ships an allowlist of
    internal addresses and the sandbox's every connection is silently blocked.
    """
    domain_resolution: dict[str, list[str]] = {}
    for domain in sorted(set(domains)):
        if domain.startswith("."):
            continue
        try:
            local_addrs = sorted(
                {
                    info[4][0]
                    for info in socket.getaddrinfo(
                        domain,
                        443,
                        family=socket.AF_INET,
                        type=socket.SOCK_STREAM,
                    )
                }
            )
        except socket.gaierror:
            local_addrs = []
        addrs = [addr for addr in local_addrs if _is_sandbox_reachable_ipv4(addr)]
        dropped = sorted(set(local_addrs) - set(addrs))
        if dropped:
            logger.warning(
                "Ignoring local resolver answers for %s that a Daytona sandbox "
                "cannot reach (split-horizon/corporate DNS?): %s",
                domain,
                ", ".join(dropped),
            )
        if not addrs:
            addrs = _resolve_domain_via_public_doh(domain)
            if addrs:
                logger.warning(
                    "Local resolver returned no sandbox-reachable IPv4 answer "
                    "for %s; using public DNS-over-HTTPS answer instead: %s",
                    domain,
                    ", ".join(addrs),
                )
            else:
                logger.warning(
                    "Could not resolve any sandbox-reachable IPv4 address for "
                    "allowlist domain %s (local resolver and public "
                    "DNS-over-HTTPS both failed); connections to it from the "
                    "sandbox will be blocked.",
                    domain,
                )
        domain_resolution[domain] = addrs

    return domain_resolution, _cidrs_from_domain_resolution(domain_resolution)


class DaytonaClientManager:
    """Singleton manager for the AsyncDaytona client."""

    _instance: DaytonaClientManager | None = None
    _lock = asyncio.Lock()

    def __init__(self):
        if not _HAS_DAYTONA:
            raise MissingExtraError(package="daytona", extra="daytona")
        self._client: AsyncDaytona | None = None
        self._client_lock = asyncio.Lock()
        self._logger = logger.getChild(__name__)
        self._cleanup_registered = False
        self._client_config_set = False
        self._connection_pool_maxsize: int | None = None

    @classmethod
    async def get_instance(cls) -> "DaytonaClientManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def configure(self, *, connection_pool_maxsize: int | None) -> None:
        """Record Daytona client options for the process-wide client."""
        async with self._client_lock:
            if self._client_config_set:
                if self._connection_pool_maxsize != connection_pool_maxsize:
                    self._logger.warning(
                        "Daytona client already configured with "
                        "connection_pool_maxsize=%r; ignoring %r",
                        self._connection_pool_maxsize,
                        connection_pool_maxsize,
                    )
                return
            if self._client is not None:
                self._logger.warning(
                    "Daytona client was built before explicit configuration; "
                    "ignoring connection_pool_maxsize=%r",
                    connection_pool_maxsize,
                )
                return
            self._connection_pool_maxsize = connection_pool_maxsize
            self._client_config_set = True

    async def get_client(self) -> AsyncDaytona:
        async with self._client_lock:
            if self._client is None:
                self._logger.debug("Creating new AsyncDaytona client")
                if self._client_config_set and DaytonaConfig is not None:
                    self._client = AsyncDaytona(
                        DaytonaConfig(
                            connection_pool_maxsize=self._connection_pool_maxsize
                        )
                    )
                elif self._client_config_set:
                    self._logger.warning(
                        "Installed Daytona SDK does not expose DaytonaConfig; "
                        "ignoring client connection_pool_maxsize."
                    )
                    self._client = AsyncDaytona()
                else:
                    self._client = AsyncDaytona()
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_sync)
                    self._cleanup_registered = True
            return self._client

    def _cleanup_sync(self) -> None:
        try:
            asyncio.run(self._cleanup())
        except Exception as e:
            self._logger.error(f"Error during Daytona client cleanup: {e}")

    async def _cleanup(self) -> None:
        async with self._client_lock:
            if self._client is None:
                return
            try:
                await self._client.close()
            finally:
                self._client = None
            self._client_config_set = False
            self._connection_pool_maxsize = None


class _DaytonaStrategy:
    def __init__(self, env: "DaytonaEnvironment"):
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None: ...

    @abstractmethod
    async def stop(self, delete: bool) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None: ...

    @abstractmethod
    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None: ...

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None: ...

    @abstractmethod
    async def is_dir(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    @abstractmethod
    async def attach(self) -> None: ...


class _DaytonaDirect(_DaytonaStrategy):
    async def start(self, force_build: bool) -> None:
        env = self._env
        resources = env._sandbox_resources()

        env._client_manager = await DaytonaClientManager.get_instance()
        await env._configure_daytona_client()
        daytona = await env._client_manager.get_client()

        snapshot_name: str | None = None
        snapshot_exists = False
        if env._snapshot_template_name:
            snapshot_name = env._snapshot_template_name.format(name=env.environment_name)
            try:
                snapshot = await daytona.snapshot.get(snapshot_name)
                snapshot_exists = snapshot.state == SnapshotState.ACTIVE
            except Exception:
                snapshot_exists = False

        if snapshot_exists and force_build:
            env.logger.warning(
                "Snapshot template specified but force_build is True. "
                "Snapshot will be used instead of building from scratch."
            )
        if snapshot_exists and env.agent_install_spec is not None:
            env.logger.warning(
                "Daytona snapshot is being used, so Pier cannot inline the agent "
                "install into the image. Agent setup will install at runtime."
            )
            env.agent_install_spec = None

        params: _SandboxParams
        if snapshot_exists and snapshot_name:
            params = CreateSandboxFromSnapshotParams(
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                snapshot=snapshot_name,
                **env._network_params(),
            )
        elif force_build or not env.task_env_config.docker_image:
            image = env._with_agent_install(Image.from_dockerfile(env._dockerfile_path))
            kwargs = {
                "image": image,
                "auto_delete_interval": env._auto_delete_interval,
                "auto_stop_interval": env._auto_stop_interval,
                **env._network_params(),
            }
            if resources is not None:
                kwargs["resources"] = resources
            params = CreateSandboxFromImageParams(
                **kwargs,
            )
        else:
            image = env._with_agent_install(Image.base(env.task_env_config.docker_image))
            kwargs = {
                "image": image,
                "auto_delete_interval": env._auto_delete_interval,
                "auto_stop_interval": env._auto_stop_interval,
                **env._network_params(),
            }
            if resources is not None:
                kwargs["resources"] = resources
            params = CreateSandboxFromImageParams(
                **kwargs,
            )

        await env._create_sandbox(params=params)
        await env._pin_resolved_hosts()
        await env._sandbox_exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} && "
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        try:
            if env._sandbox:
                try:
                    await env._stop_sandbox()
                except Exception as e:
                    env.logger.error(f"Error stopping sandbox {env._sandbox.id}: {e}")
                finally:
                    env._sandbox = None
            else:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
        finally:
            env._client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        file_info = await self._env._sandbox.fs.get_file_info(path)
        return file_info.is_dir

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        file_info = await self._env._sandbox.fs.get_file_info(path)
        return not file_info.is_dir

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()
        os.execvp("ssh", ["ssh", f"{ssh_access.token}@ssh.app.daytona.io"])


class _DaytonaDinD(_DaytonaStrategy):
    _DOCKER_DAEMON_TIMEOUT_SEC = 60
    _COMPOSE_DIR = "/pier/compose"
    _ENVIRONMENT_DIR = "/pier/environment"
    _LOGS_DIR = "/pier/logs"
    _RESOURCES_COMPOSE_PATH = f"{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}"

    def __init__(self, env: "DaytonaEnvironment"):
        super().__init__(env)
        self._use_prebuilt = False
        self._resolved_task_env: dict[str, str] = {}
        pier_keys = set(self._infra_env_vars().keys())
        if self._env.task_env_config.env:
            self._resolved_task_env = resolve_env_vars(self._env.task_env_config.env)

        resolved_task_keys = set(self._resolved_task_env.keys()) | set(
            self._env._persistent_env.keys()
        )
        if resolved_task_keys:
            collisions = pier_keys & resolved_task_keys
            if collisions:
                self._env.logger.warning(
                    "Environment vars override Pier compose variable(s): %s",
                    ", ".join(sorted(collisions)),
                )

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._env._sandbox_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell="sh -c",
        )

    def _infra_env_vars(self) -> dict[str, str]:
        env_vars: dict[str, str] = {
            "CONTEXT_DIR": self._ENVIRONMENT_DIR,
            "MAIN_IMAGE_NAME": _sanitize_docker_image_name(
                f"pier__{self._env.environment_name}"
            ),
            "HOST_VERIFIER_LOGS_PATH": f"{self._LOGS_DIR}/verifier",
            "HOST_AGENT_LOGS_PATH": f"{self._LOGS_DIR}/agent",
            "HOST_ARTIFACTS_PATH": f"{self._LOGS_DIR}/artifacts",
            "ENV_VERIFIER_LOGS_PATH": str(EnvironmentPaths.verifier_dir),
            "ENV_AGENT_LOGS_PATH": str(EnvironmentPaths.agent_dir),
            "ENV_ARTIFACTS_PATH": str(EnvironmentPaths.artifacts_dir),
        }
        if (cpus := self._env._effective_cpus) is not None:
            env_vars["CPUS"] = str(cpus)
        if (memory_mb := self._env._effective_memory_mb) is not None:
            env_vars["MEMORY"] = f"{memory_mb}M"
        if self._use_prebuilt and self._env.task_env_config.docker_image:
            env_vars["PREBUILT_IMAGE_NAME"] = self._env.task_env_config.docker_image
        return env_vars

    def _compose_env_vars(self) -> dict[str, str]:
        env_vars = self._infra_env_vars()
        env_vars.update(self._resolved_task_env)
        env_vars.update(self._env._persistent_env)
        return env_vars

    def _compose_file_flags(self) -> list[str]:
        build_or_prebuilt = (
            "docker-compose-prebuilt.yaml"
            if self._use_prebuilt
            else "docker-compose-build.yaml"
        )
        files = [
            f"{self._COMPOSE_DIR}/docker-compose-base.yaml",
            self._RESOURCES_COMPOSE_PATH,
            f"{self._COMPOSE_DIR}/{build_or_prebuilt}",
            f"{self._ENVIRONMENT_DIR}/docker-compose.yaml",
        ]
        if self._env._compose_should_block_main_network():
            files.append(f"{self._COMPOSE_DIR}/docker-compose-no-network.yaml")

        flags: list[str] = []
        for file in files:
            flags.extend(["-f", file])
        return flags

    @property
    def _project_name(self) -> str:
        return self._env.session_id.lower().replace(".", "-")

    def _compose_cmd(self, subcommand: list[str]) -> str:
        parts = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._ENVIRONMENT_DIR,
            *self._compose_file_flags(),
            *subcommand,
        ]
        return shlex.join(parts)

    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._vm_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    async def _wait_for_docker_daemon(self) -> None:
        last_output = ""
        for _ in range(self._DOCKER_DAEMON_TIMEOUT_SEC // 2):
            result = await self._vm_exec("docker info", timeout_sec=10)
            if result.return_code == 0:
                return
            last_output = (result.stdout or "") + (result.stderr or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {self._DOCKER_DAEMON_TIMEOUT_SEC}s. "
            f"Last output: {last_output}"
        )

    async def _wait_for_main_container(self, timeout_sec: int = 60) -> None:
        for _ in range(timeout_sec // 2):
            result = await self._compose_exec(
                ["exec", "-T", "main", "true"],
                timeout_sec=10,
            )
            if result.return_code == 0:
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

    async def start(self, force_build: bool) -> None:
        env = self._env
        resources = env._sandbox_resources()

        env._client_manager = await DaytonaClientManager.get_instance()
        await env._configure_daytona_client()
        dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
        dind_snapshot: str | None = env._kwargs.get("dind_snapshot")

        if dind_snapshot:
            params: _SandboxParams = CreateSandboxFromSnapshotParams(
                snapshot=dind_snapshot,
                auto_delete_interval=env._auto_delete_interval,
                auto_stop_interval=env._auto_stop_interval,
                **env._network_params(),
            )
        else:
            kwargs = {
                "image": Image.base(dind_image),
                "auto_delete_interval": env._auto_delete_interval,
                "auto_stop_interval": env._auto_stop_interval,
                **env._network_params(),
            }
            if resources is not None:
                kwargs["resources"] = resources
            params = CreateSandboxFromImageParams(
                **kwargs,
            )

        await env._create_sandbox(params=params)
        await env._pin_resolved_hosts()

        await self._vm_exec(
            "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &",
            timeout_sec=10,
        )
        await self._wait_for_docker_daemon()

        for path in (
            COMPOSE_BASE_PATH,
            COMPOSE_BUILD_PATH,
            COMPOSE_PREBUILT_PATH,
            COMPOSE_NO_NETWORK_PATH,
        ):
            await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")
        await self._stage_resources_compose_file()

        await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)
        await self._vm_exec(
            f"mkdir -p {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts && "
            f"chmod 777 {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts"
        )

        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)

        result = await self._compose_exec(
            ["build"],
            timeout_sec=round(env.task_env_config.build_timeout_sec),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose build failed: {result.stdout} {result.stderr}"
            )

        result = await self._compose_exec(["up", "-d"], timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {result.stdout} {result.stderr}"
            )

        await self._wait_for_main_container()

    async def _stage_resources_compose_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = write_resources_compose_file(
                Path(tmp) / RESOURCES_COMPOSE_NAME,
                cpu_request=self._env._resource_request_value(
                    "cpu", auto_mode=ResourceMode.REQUEST
                ),
                cpu_limit=self._env._resource_limit_value(
                    "cpu", auto_mode=ResourceMode.REQUEST
                ),
                memory_request_mb=self._env._resource_request_value(
                    "memory", auto_mode=ResourceMode.REQUEST
                ),
                memory_limit_mb=self._env._resource_limit_value(
                    "memory", auto_mode=ResourceMode.REQUEST
                ),
            )
            await self._env._sdk_upload_file(local_path, self._RESOURCES_COMPOSE_PATH)

    async def stop(self, delete: bool) -> None:
        env = self._env
        if not delete:
            env.logger.info(
                "Daytona sandboxes are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        if env._sandbox:
            try:
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as e:
                env.logger.warning(f"docker compose down failed: {e}")

        try:
            if env._sandbox:
                try:
                    await env._stop_sandbox()
                except Exception as e:
                    env.logger.error(f"Error stopping sandbox {env._sandbox.id}: {e}")
                finally:
                    env._sandbox = None
            else:
                env.logger.warning(
                    "Sandbox not found. Please build the environment first."
                )
        finally:
            env._client_manager = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                parts.extend(["-e", f"{key}={value}"])
        if user is not None:
            parts.extend(["-u", str(user)])
        parts.extend(["main", "bash", "-c", command])
        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            result = await self._compose_exec(
                ["cp", temp, f"main:{target_path}"],
                timeout_sec=60,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp)
            result = await self._compose_exec(
                ["cp", f"{temp}/.", f"main:{target_dir}"],
                timeout_sec=120,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    def _sandbox_log_path(self, container_path: str) -> str | None:
        mappings = {
            str(EnvironmentPaths.verifier_dir): f"{self._LOGS_DIR}/verifier",
            str(EnvironmentPaths.agent_dir): f"{self._LOGS_DIR}/agent",
            str(EnvironmentPaths.artifacts_dir): f"{self._LOGS_DIR}/artifacts",
        }
        for env_prefix, sandbox_prefix in mappings.items():
            if container_path == env_prefix or container_path.startswith(
                env_prefix + "/"
            ):
                return container_path.replace(env_prefix, sandbox_prefix, 1)
        return None

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox_path = self._sandbox_log_path(source_path)
        if sandbox_path:
            await self._env._sdk_download_file(sandbox_path, target_path)
            return

        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            result = await self._compose_exec(
                ["cp", f"main:{source_path}", temp],
                timeout_sec=60,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_file(temp, target_path)
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        sandbox_path = self._sandbox_log_path(source_dir)
        if sandbox_path:
            await self._env._sdk_download_dir(sandbox_path, target_dir)
            return

        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"main:{source_dir}/.", temp],
                timeout_sec=120,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_dir(temp, target_dir)
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}",
            timeout_sec=10,
            user=user,
        )
        return result.return_code == 0

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        ssh_access = await env._sandbox.create_ssh_access()
        compose_cmd = self._compose_cmd(["exec", "-it", "main", "bash"])
        compose_env = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in self._compose_env_vars().items()
        )
        remote_cmd = f"{compose_env} {compose_cmd}"
        os.execvp(
            "ssh",
            ["ssh", "-t", f"{ssh_access.token}@ssh.app.daytona.io", remote_cmd],
        )


class DaytonaEnvironment(BaseEnvironment):
    @classmethod
    def preflight(cls) -> None:
        _daytona_preflight()

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        snapshot_template_name: str | None = None,
        network_block_all: bool | None = None,
        network_allow_list: str | list[str] | None = None,
        auto_stop_interval_mins: int = 0,
        auto_delete_interval_mins: int = 0,
        **kwargs: Any,
    ):
        if not _HAS_DAYTONA:
            raise MissingExtraError(package="daytona", extra="daytona")

        self._compose_mode = (environment_dir / "docker-compose.yaml").exists()
        self._kwargs = kwargs
        self._explicit_network_block_all = network_block_all
        self._explicit_network_allow_list = network_allow_list

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._auto_stop_interval = auto_stop_interval_mins
        self._auto_delete_interval = auto_delete_interval_mins
        self._snapshot_template_name = snapshot_template_name
        self._sandbox: AsyncSandbox | None = None
        self._client_manager: DaytonaClientManager | None = None
        self._resolved_network_allow_list: str | None = None
        self._network_resolution_debug: dict[str, Any] = {}

        self._strategy: _DaytonaStrategy = (
            _DaytonaDinD(self) if self._compose_mode else _DaytonaDirect(self)
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DAYTONA

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            filtered_egress=True,
            preinstall_agents=not self._compose_mode,
            docker_compose=True,
        )

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_request=True,
            memory_request=True,
        )

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    def _validate_definition(self) -> None:
        path = (
            self._environment_docker_compose_path
            if self._compose_mode
            else self._dockerfile_path
        )
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    def _with_agent_install(self, image: Image) -> Image:
        install = self.agent_install_spec
        if install is None:
            return image
        commands = dockerfile_install_commands(
            install,
            user=self._resolve_user(None),
        )
        return image.dockerfile_commands(commands)

    def _network_params(self) -> dict[str, Any]:
        if self._explicit_network_allow_list:
            allow_list = self._explicit_network_allow_list
            if isinstance(allow_list, list):
                allow_list = ",".join(allow_list)
            return {"network_block_all": False, "network_allow_list": allow_list}

        if self._explicit_network_block_all is not None:
            expected = not self.task_env_config.allow_internet
            if self._explicit_network_block_all != expected:
                self.logger.warning(
                    "network_block_all=%s overrides task config allow_internet=%s",
                    self._explicit_network_block_all,
                    self.task_env_config.allow_internet,
                )
            return {"network_block_all": self._explicit_network_block_all}

        if self.task_env_config.allow_internet:
            return {"network_block_all": False}

        if not self.network_allowlist.domains:
            return {"network_block_all": True}

        if self._resolved_network_allow_list is None:
            suffix_domains = [
                domain
                for domain in self.network_allowlist.domains
                if domain.startswith(".")
            ]
            if suffix_domains:
                self.logger.warning(
                    "Daytona network limits only accept IPv4 CIDRs; ignoring "
                    "suffix allowlist domains: %s",
                    ", ".join(suffix_domains),
                )

            domain_resolution, cidrs = resolve_network_allowlist_to_daytona_cidrs(
                self.network_allowlist.domains
            )
            full_cidr_count = sum(len(addrs) for addrs in domain_resolution.values())
            if full_cidr_count > DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS:
                self.logger.warning(
                    "Resolved Daytona network allowlist has %d IPv4 addresses; "
                    "collapsing to %d CIDR entries. This can allow broader CDN "
                    "ranges than the original domains.",
                    full_cidr_count,
                    DAYTONA_MAX_NETWORK_ALLOWLIST_CIDRS,
                )
            self._network_resolution_debug = {
                "domains": self.network_allowlist.domains,
                "domain_resolution": domain_resolution,
                "cidr_allowlist": cidrs,
            }
            self._resolved_network_allow_list = ",".join(cidrs) if cidrs else ""

        if not self._resolved_network_allow_list:
            self.logger.warning(
                "No Daytona-compatible CIDRs resolved from network allowlist; "
                "blocking all network access."
            )
            return {"network_block_all": True}

        return {
            "network_block_all": False,
            "network_allow_list": self._resolved_network_allow_list,
        }

    async def _configure_daytona_client(self) -> None:
        if self._client_manager is None:
            raise RuntimeError(
                "Client manager not initialized. This should never happen."
            )
        if "connection_pool_maxsize" not in self._kwargs:
            return
        await self._client_manager.configure(
            connection_pool_maxsize=self._kwargs["connection_pool_maxsize"],
        )

    def _sandbox_resources(self) -> Resources | None:
        kwargs = {}
        if (cpus := self._effective_cpus) is not None:
            kwargs["cpu"] = cpus
        if (memory_mb := self._effective_memory_mb) is not None:
            kwargs["memory"] = memory_mb // 1024
        if (storage_mb := self._effective_storage_mb) is not None:
            kwargs["disk"] = storage_mb // 1024
        return Resources(**kwargs) if kwargs else None

    async def _pin_resolved_hosts(self) -> None:
        """Pin resolved allowlist domains so restricted sandboxes do not need DNS."""
        domain_resolution = self._network_resolution_debug.get("domain_resolution")
        if not isinstance(domain_resolution, dict):
            return

        lines: list[str] = []
        for domain, addrs in sorted(domain_resolution.items()):
            if not addrs or not isinstance(domain, str):
                continue
            addr = next((value for value in addrs if isinstance(value, str)), None)
            if addr:
                lines.append(f"{addr} {domain}")

        if not lines:
            return

        hosts = "\n".join(lines) + "\n"
        command = (
            "cat >> /etc/hosts <<'EOF'\n"
            "# Pier Daytona network allowlist host pins\n"
            f"{hosts}"
            "EOF"
        )
        shell = "sh -c" if self._compose_mode else "bash -c"
        result = await self._sandbox_exec(command, shell=shell)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to pin Daytona resolved hosts: {result.stdout} {result.stderr}"
            )

    def _compose_should_block_main_network(self) -> bool:
        if self.task_env_config.allow_internet:
            return False
        if self._explicit_network_block_all is False:
            return False
        if self._explicit_network_allow_list:
            return False
        return not self._resolved_network_allow_list

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(self, params: _SandboxParams) -> None:
        if not self._client_manager:
            raise RuntimeError(
                "Client manager not initialized. This should never happen."
            )

        daytona = await self._client_manager.get_client()
        create_task = asyncio.ensure_future(
            daytona.create(
                params=params,
                timeout=round(self.task_env_config.build_timeout_sec),
            )
        )
        try:
            self._sandbox = await asyncio.shield(create_task)
        except asyncio.CancelledError:
            try:
                self._sandbox = await asyncio.wait_for(create_task, timeout=30)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                create_task.cancel()
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _stop_sandbox(self) -> None:
        if self._sandbox:
            await self._sandbox.delete()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get_session_command_with_retry(self, session_id: str, command_id: str):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        return await self._sandbox.process.get_session_command(session_id, command_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get_session_command_logs_with_retry(
        self,
        session_id: str,
        command_id: str,
    ):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        return await self._sandbox.process.get_session_command_logs(
            session_id,
            command_id,
        )

    async def _poll_response(self, session_id: str, command_id: str) -> ExecResult:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")

        response = await self._get_session_command_with_retry(session_id, command_id)
        while response.exit_code is None:
            await asyncio.sleep(1)
            response = await self._get_session_command_with_retry(
                session_id,
                response.id,
            )

        logs = await self._get_session_command_logs_with_retry(session_id, command_id)
        return ExecResult(
            stdout=logs.stdout,
            stderr=logs.stderr,
            return_code=int(response.exit_code),
        )

    async def _sandbox_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash -c",
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")

        session_id = str(uuid4())
        await self._sandbox.process.create_session(session_id)

        command = f"{shell} {shlex.quote(command)}"
        if env:
            env_args = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
            command = f"env {env_args} {command}"
        if timeout_sec:
            command = f"timeout {timeout_sec} {command}"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if user is not None:
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        response = await self._sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=command, run_async=True),
            timeout=timeout_sec,
        )
        if response.cmd_id is None:
            raise RuntimeError("Cannot find command ID.")
        return await self._poll_response(session_id, response.cmd_id)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        await self._sandbox.fs.upload_file(str(source_path), target_path)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")

        file_uploads = []
        source_dir = Path(source_dir)
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_dir).as_posix()
                file_uploads.append(
                    FileUpload(
                        source=str(file_path),
                        destination=f"{target_dir}/{relative_path}",
                    )
                )

        if file_uploads:
            await self._sandbox.fs.upload_files(files=file_uploads)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")
        await self._sandbox.fs.download_file(source_path, str(target_path))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please build the environment first.")

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        search_result = await self._sandbox.fs.search_files(source_dir, "*")

        file_downloads = []
        for file_path in search_result.files:
            try:
                file_info = await self._sandbox.fs.get_file_info(file_path)
            except DaytonaNotFoundError:
                self.logger.debug(
                    f"Skipping file not found during download_dir: {file_path}"
                )
                continue

            if not file_info.is_dir:
                path_obj = Path(file_path)
                relative_path = path_obj.relative_to(Path(source_dir))
                local_file_path = target_dir / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                file_downloads.append(
                    FileDownloadRequest(
                        source=file_path,
                        destination=str(local_file_path),
                    )
                )

        if file_downloads:
            await self._sandbox.fs.download_files(files=file_downloads)

    async def start(self, force_build: bool) -> None:
        return await self._strategy.start(force_build)

    async def stop(self, delete: bool) -> None:
        return await self._strategy.stop(delete)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)
        effective_cwd = cwd or self.task_env_config.workdir
        return await self._strategy.exec(
            command,
            cwd=effective_cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        return await self._strategy.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        return await self._strategy.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        return await self._strategy.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        return await self._strategy.download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_dir(path, user=self._resolve_user(user))

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_file(path, user=self._resolve_user(user))

    async def attach(self) -> None:
        return await self._strategy.attach()

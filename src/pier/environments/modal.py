from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import socket
from abc import abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    dockerfile_install_commands,
    new_proxy_token,
    proxy_environment,
    proxy_policy_env,
    squid_bootstrap_command,
    write_agent_dockerfile,
    write_docker_proxy_compose,
)
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
)
from pier.environments.docker.docker import _sanitize_docker_image_name
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.config import ResourceMode
from pier.models.trial.paths import EnvironmentPaths, TrialPaths
from pier.utils.env import resolve_env_vars
from pier.utils.optional_import import MissingExtraError

try:
    import modal
    from modal import App, Image, Sandbox, Secret, Volume

    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

_MODAL_DEFAULT_CPU_REQUEST_CORES = 0.125
_MODAL_DEFAULT_MEMORY_REQUEST_MB = 128


class _ModalStrategy:
    """Base class for Modal execution strategies.

    A direct strategy runs everything in a single sandbox container,
    while a compose (DinD) strategy runs Docker inside the sandbox and
    orchestrates multiple containers via docker-compose.

    Methods that simply delegate to the Modal SDK (upload, download,
    is_dir, is_file) have concrete defaults here so that only strategies
    with genuinely different behaviour need to override them.
    """

    def __init__(self, env: "ModalEnvironment"):
        self._env = env

    @abstractmethod
    async def start(self, force_build: bool) -> None:
        """Start the environment."""

    async def stop(self, delete: bool) -> None:
        """Stop the environment and optionally delete resources."""
        await self._teardown_sandbox()

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command in the environment's main container."""

    @abstractmethod
    async def attach(self) -> None:
        """Attach an interactive shell to the environment."""

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._sdk_upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._env._sdk_upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._sdk_download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._env._sdk_download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a directory (uses sandbox.ls)."""
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        try:
            await self._env._sandbox.ls.aio(path)
            return True
        except (NotADirectoryError, FileNotFoundError):
            return False

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        """Check if a remote path is a file (uses sandbox.ls)."""
        if not self._env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        try:
            await self._env._sandbox.ls.aio(path)
            return False
        except NotADirectoryError:
            return True
        except FileNotFoundError:
            return False

    async def _teardown_sandbox(self) -> None:
        """Terminate the sandbox and reset references."""
        env = self._env
        if not env._sandbox:
            env._app = None
            env._image = None
            await env._teardown_egress_proxy()
            return
        try:
            await env._terminate_sandbox()
            await env._sandbox.wait.aio(raise_on_termination=False)
        except Exception as e:
            env.logger.warning(f"Error terminating Modal sandbox: {e}")
        finally:
            env._sandbox = None
            env._app = None
            env._image = None
            await env._teardown_egress_proxy()

    async def exec_on_vm(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash",
    ) -> ExecResult:
        """Run a command directly on the sandbox VM (bypasses compose)."""
        return await self._env._sdk_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, shell=shell
        )


class _ModalDirect(_ModalStrategy):
    """Single-container sandbox — runs agent and verifier directly in the
    Modal sandbox.

    Inherits upload/download/is_dir/is_file from `_ModalStrategy` since
    the default SDK implementations are sufficient for a single container.
    """

    async def start(self, force_build: bool) -> None:
        env = self._env

        docker_image = env.task_env_config.docker_image
        if docker_image:
            registry_secret = (
                Secret.from_name(env._registry_secret) if env._registry_secret else None
            )
            if ".dkr.ecr." in docker_image:
                env._image = Image.from_aws_ecr(
                    docker_image,
                    secret=registry_secret,
                )
            else:
                env._image = Image.from_registry(
                    docker_image,
                    secret=registry_secret,
                )
        else:
            env._image = Image.from_dockerfile(
                env._environment_definition_path,
                context_dir=env.environment_dir,
            )
        env._image = env._with_agent_install(env._image)

        env._app = await App.lookup.aio(
            name=env._app_name,
            create_if_missing=True,
        )

        await env._ensure_egress_proxy()
        experimental = (
            {"vm_runtime": True} if getattr(env, "_vm_runtime", False) else None
        )
        env._sandbox = await env._create_sandbox(experimental_options=experimental)

        await env._sandbox.filesystem.make_directory.aio(
            str(EnvironmentPaths.agent_dir), create_parents=True
        )
        await env._sandbox.filesystem.make_directory.aio(
            str(EnvironmentPaths.verifier_dir), create_parents=True
        )

        # Make log directories world-writable so non-root agent/verifier
        # users can write to them.
        await self.exec(
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        return await self._env._sdk_exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec
        )

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        os.execvp(
            "modal",
            [
                "modal",
                "shell",
                env._sandbox.object_id,
            ],
        )


class _ModalDinD(_ModalStrategy):
    """Docker-in-Docker compose strategy for multi-container tasks.

    Two sandbox runtimes are supported:

    * **gVisor DinD** (default): ``experimental_options={"enable_docker": True}``
      on Modal's managed DinD image. gVisor lacks iptables/netlink for veth
      pairs, so Pier forces ``network_mode: host`` and ``bridge: none``.
    * **VM runtime** (``vm_runtime=True``): a real Linux kernel with Ubuntu +
      dockerd. Native Docker bridge networking works, so compose DNS aliases
      and inter-container networking behave like a normal Linux host.

    Topology:
        Local machine (pier CLI)
          └── Modal Sandbox (DinD / VM + dockerd)
                ├── dockerd
                └── docker compose
                      ├── main        ← agent runs here, exec/upload/download target
                      ├── sidecar     ← additional services
                      └── ...
    """

    # Max iterations when polling for Docker daemon readiness.
    # Each iteration sleeps 2s; worst-case wall clock time is ~160s.
    _DOCKER_DAEMON_POLL_LIMIT = 80
    _COMPOSE_DIR = "/pier/compose"
    _ENVIRONMENT_DIR = "/pier/environment"
    _LOGS_DIR = "/pier/logs"
    _HOST_NETWORK_COMPOSE = "docker-compose-host-network.yaml"
    _MOUNTS_COMPOSE = "docker-compose-mounts.yaml"
    _VM_NETWORK_COMPOSE = "docker-compose-vm-network.yaml"
    _EGRESS_PROXY_COMPOSE = "docker-compose-egress-proxy.json"
    _BASE_IMAGE_COMPOSE = "docker-compose-main-base.yaml"
    _AGENT_IMAGE_COMPOSE = "docker-compose-main-agent.yaml"
    _AGENT_BUILD_DIR = "/pier/compose/agent-build"
    _EGRESS_PROXY_DIR = "/pier/compose/egress-proxy"

    def __init__(self, env: "ModalEnvironment"):
        super().__init__(env)
        self._use_prebuilt = False
        self._base_image_overlay_ready = False
        self._agent_image_overlay_ready = False
        self._egress_proxy_overlay_ready = False

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

    @staticmethod
    def _build_host_network_overlay(
        environment_dir: Path, *, use_prebuilt: bool = False
    ) -> str:
        """Generate a compose overlay that sets host networking on all services.

        Parses service names from the task's docker-compose.yaml so the
        overlay covers all services regardless of naming conventions.
        Only adds ``build.network: host`` for services that have a build
        context (not pure image-based services like redis).

        Each service gets ``extra_hosts`` entries mapping every *other*
        service name to ``127.0.0.1`` so that Docker DNS hostnames
        (e.g. ``redis``, ``agent1``) resolve correctly under host networking.
        """
        import yaml

        compose_path = environment_dir / "docker-compose.yaml"
        services: dict[str, bool] = {}  # name -> has_build
        aliases: dict[str, list[str]] = {}  # name -> network aliases
        if compose_path.exists():
            doc = yaml.safe_load(compose_path.read_text())
            if doc and "services" in doc:
                for name, cfg in doc["services"].items():
                    has_build = isinstance(cfg, dict) and "build" in cfg
                    services[name] = has_build
                    # Collect network aliases (records.example.internal, …):
                    # under host networking the compose network DNS is gone, so
                    # they must ride extra_hosts alongside the service names.
                    svc_aliases: list[str] = []
                    nets = cfg.get("networks") if isinstance(cfg, dict) else None
                    if isinstance(nets, dict):
                        for net_cfg in nets.values():
                            if isinstance(net_cfg, dict):
                                svc_aliases.extend(net_cfg.get("aliases") or [])
                    if svc_aliases:
                        aliases[name] = svc_aliases

        # Fallback if parsing fails
        if not services:
            services = {"main": True, "sidecar": True, "redis": False}

        # main always needs host networking.  In build mode it also needs
        # build.network: host; in prebuilt mode only network_mode: host.
        if use_prebuilt:
            services.setdefault("main", False)
        else:
            services["main"] = True

        service_names = list(services.keys())
        lines = ["services:"]
        for svc, has_build in services.items():
            lines.append(f"  {svc}:")
            if has_build:
                lines.append("    build:")
                lines.append("      network: host")
            lines.append("    network_mode: host")
            if svc == "main":
                # The docker environment injects these via its mounts overlay;
                # DinD must add them itself or /logs/{agent,verifier,artifacts}
                # never exist inside main and every agent setup fails.
                # (staticmethod: hardcode _LOGS_DIR = /pier/logs)
                lines.append("    volumes:")
                lines.append("      - /pier/logs/agent:/logs/agent")
                lines.append("      - /pier/logs/verifier:/logs/verifier")
                lines.append("      - /pier/logs/artifacts:/logs/artifacts")
            # Task compose files may attach services to named networks, which
            # is mutually exclusive with network_mode — reset the key so the
            # host-network override wins.
            lines.append("    networks: !reset []")
            # Map all other service names to localhost so Docker DNS
            # hostnames work under host networking.
            others = [s for s in service_names if s != svc]
            other_hosts: list[str] = []
            for other in others:
                other_hosts.append(other)
                other_hosts.extend(aliases.get(other, []))
            # A service's own aliases must also resolve (self-referencing URLs)
            other_hosts.extend(aliases.get(svc, []))
            if other_hosts:
                lines.append("    extra_hosts:")
                for host in other_hosts:
                    lines.append(f'      - "{host}:127.0.0.1"')
            # NOTE: Do NOT add environment: here — it replaces (not merges)
            # the service's entire environment block from the base compose
            # file, wiping out AGENT_ID, API keys, etc.
        return "\n".join(lines)

    async def _vm_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str | None = None,
    ) -> ExecResult:
        """Run a command on the DinD sandbox VM."""
        return await self._env._sdk_exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            shell=shell or self._env._default_shell,
        )

    def _compose_referenced_env_vars(self) -> dict[str, str]:
        """Extract env vars referenced in the task's docker-compose.yaml.

        Parses ``${VAR_NAME}`` and ``${VAR_NAME:-default}`` patterns from the
        compose file and returns values from os.environ for any that are set.
        """
        compose_path = self._env.environment_dir / "docker-compose.yaml"
        if not compose_path.exists():
            return {}

        content = compose_path.read_text()
        # Match ${VAR}, ${VAR:-default}, and bare $VAR references
        matches = re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)\b",
            content,
        )
        var_names = {g1 or g2 for g1, g2 in matches}

        env_vars: dict[str, str] = {}
        for name in var_names:
            value = os.environ.get(name)
            if value is not None:
                env_vars[name] = value
        return env_vars

    def _infra_env_vars(self) -> dict[str, str]:
        """Pier infrastructure vars required by the compose templates."""
        env_vars: dict[str, str] = {
            "CONTEXT_DIR": self._ENVIRONMENT_DIR,
            "MAIN_IMAGE_NAME": _sanitize_docker_image_name(
                f"hb__{self._env.environment_name}"
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
        """All environment variables for docker compose commands."""
        env_vars: dict[str, str] = self._compose_referenced_env_vars()
        env_vars.update(self._infra_env_vars())
        if self._resolved_task_env:
            env_vars.update(self._resolved_task_env)
        if self._env._persistent_env:
            env_vars.update(self._env._persistent_env)
        return env_vars

    @property
    def _use_vm_runtime(self) -> bool:
        return bool(getattr(self._env, "_vm_runtime", False))

    def _compose_file_flags(
        self,
        *,
        include_base_image: bool = False,
        include_agent_image: bool = True,
        include_vm_networking: bool = True,
    ) -> list[str]:
        """Return -f flag pairs for all compose files as a flat list."""
        build_or_prebuilt = (
            "docker-compose-prebuilt.yaml"
            if self._use_prebuilt
            else "docker-compose-build.yaml"
        )
        files = [
            f"{self._COMPOSE_DIR}/docker-compose-base.yaml",
            f"{self._COMPOSE_DIR}/{build_or_prebuilt}",
            f"{self._ENVIRONMENT_DIR}/docker-compose.yaml",
        ]
        if self._use_vm_runtime:
            files.append(f"{self._COMPOSE_DIR}/{self._MOUNTS_COMPOSE}")
            if include_vm_networking:
                files.append(f"{self._COMPOSE_DIR}/{self._VM_NETWORK_COMPOSE}")
                if self._egress_proxy_overlay_ready:
                    files.append(f"{self._COMPOSE_DIR}/{self._EGRESS_PROXY_COMPOSE}")
            if include_base_image and self._base_image_overlay_ready:
                files.append(f"{self._COMPOSE_DIR}/{self._BASE_IMAGE_COMPOSE}")
            if include_agent_image and self._agent_image_overlay_ready:
                files.append(f"{self._COMPOSE_DIR}/{self._AGENT_IMAGE_COMPOSE}")
        else:
            if not self._env.task_env_config.allow_internet:
                files.append(f"{self._COMPOSE_DIR}/docker-compose-no-network.yaml")
            # gVisor DinD lacks netlink for veth pairs — force host networking.
            files.append(f"{self._COMPOSE_DIR}/{self._HOST_NETWORK_COMPOSE}")

        flags: list[str] = []
        for f in files:
            flags.extend(["-f", f])
        return flags

    def _build_mounts_overlay(self) -> str:
        """Bind Pier log dirs into the main service under native networking."""
        return (
            "services:\n"
            "  main:\n"
            "    volumes:\n"
            f"      - {self._LOGS_DIR}/agent:/logs/agent\n"
            f"      - {self._LOGS_DIR}/verifier:/logs/verifier\n"
            f"      - {self._LOGS_DIR}/artifacts:/logs/artifacts\n"
        )

    @staticmethod
    def _service_network_names(service: dict[str, Any]) -> list[str]:
        networks = service.get("networks") or {}
        if isinstance(networks, dict):
            return list(networks)
        if isinstance(networks, list):
            return [name for name in networks if isinstance(name, str)]
        return []

    def _build_vm_network_overlay(self, config: dict[str, Any]) -> str:
        """Apply task-level internet policy without opening sidecar networks."""
        import yaml

        services = config.get("services") or {}
        networks = config.get("networks") or {}
        main = services.get("main") or {}

        if EGRESS_PROXY_SERVICE in services:
            raise ValueError(
                f"Compose service name {EGRESS_PROXY_SERVICE!r} is reserved by Pier."
            )
        reserved_networks = {
            "pier-egress-internal",
            "pier-egress-external",
            "pier-main-internet",
        }
        collisions = reserved_networks.intersection(networks)
        if collisions:
            raise ValueError(
                "Compose network names reserved by Pier: "
                + ", ".join(sorted(collisions))
            )

        if self._env.task_env_config.allow_internet:
            if isinstance(main, dict) and main.get("network_mode"):
                return "networks: {}\n"
            main_networks = self._service_network_names(main)
            overlay = {
                "services": {
                    "main": {
                        "networks": list(
                            dict.fromkeys([*main_networks, "pier-main-internet"])
                        )
                    }
                },
                "networks": {"pier-main-internet": {}},
            }
            return yaml.safe_dump(overlay, sort_keys=False)

        bypass_network_services = {
            name: cfg.get("network_mode")
            for name, cfg in services.items()
            if isinstance(cfg, dict)
            and cfg.get("network_mode")
            and cfg.get("network_mode") != "none"
        }
        if bypass_network_services:
            details = ", ".join(
                f"{name}={mode}"
                for name, mode in sorted(bypass_network_services.items())
            )
            raise ValueError(
                "allow_internet=False cannot isolate compose services with "
                f"network_mode: {details}"
            )

        external_networks = [
            name
            for name, cfg in networks.items()
            if isinstance(cfg, dict) and cfg.get("external")
        ]
        if external_networks:
            raise ValueError(
                "allow_internet=False cannot isolate external compose networks: "
                + ", ".join(sorted(external_networks))
            )

        if (
            isinstance(main, dict)
            and main.get("network_mode") == "none"
            and self._env.network_allowlist.domains
        ):
            raise ValueError(
                "Filtered inference egress is incompatible with main "
                "network_mode=none. Use an internal compose network instead."
            )

        isolated_networks = {name: {"internal": True} for name in networks.keys()}
        isolated_networks.setdefault("default", {"internal": True})
        overlay = {"networks": isolated_networks}
        return yaml.safe_dump(overlay, sort_keys=False)

    def _validate_vm_network_isolation(self, config: dict[str, Any]) -> None:
        """Fail before startup if any task service retains unrestricted egress."""
        if self._env.task_env_config.allow_internet:
            return

        services = config.get("services") or {}
        networks = config.get("networks") or {}
        violations: list[str] = []

        for service_name, service in services.items():
            if service_name == EGRESS_PROXY_SERVICE or not isinstance(service, dict):
                continue
            network_mode = service.get("network_mode")
            if network_mode == "none":
                continue
            if network_mode:
                violations.append(f"{service_name} uses network_mode={network_mode}")
                continue
            for network_name in self._service_network_names(service):
                network = networks.get(network_name)
                if not isinstance(network, dict) or not network.get("internal"):
                    violations.append(
                        f"{service_name} is attached to non-internal network "
                        f"{network_name}"
                    )

        if violations:
            raise RuntimeError(
                "allow_internet=False network isolation failed: "
                + "; ".join(sorted(violations))
            )

    def _main_image_name(self, suffix: str) -> str:
        install = self._env.agent_install_spec
        fingerprint = install.fingerprint() if install else "none"
        return _sanitize_docker_image_name(
            f"hb__{self._env.environment_name}__{suffix}-{fingerprint}"
        )

    async def _upload_text(self, filename: str, content: str) -> None:
        path = self._env.trial_paths.trial_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        await self._env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{filename}")

    async def _prepare_vm_networking(self) -> None:
        source_config = await self._resolve_compose_config(include_vm_networking=False)
        network_overlay = self._build_vm_network_overlay(source_config)
        await self._upload_text(self._VM_NETWORK_COMPOSE, network_overlay)

        allowlist = self._env.network_allowlist
        if not self._env.task_env_config.allow_internet and allowlist.domains:
            token = new_proxy_token()
            self._env._egress_proxy_env = proxy_environment(
                token,
                EGRESS_PROXY_SERVICE,
                EGRESS_PROXY_PORT,
                no_proxy_hosts=self._compose_no_proxy_hosts(source_config),
            )
            local_proxy_dir = self._env.trial_paths.trial_dir / "modal-egress-proxy"
            local_compose_path = (
                self._env.trial_paths.trial_dir / self._EGRESS_PROXY_COMPOSE
            )
            main = source_config.get("services", {}).get("main") or {}
            write_docker_proxy_compose(
                path=local_compose_path,
                proxy_dir=local_proxy_dir,
                compose_proxy_dir=self._EGRESS_PROXY_DIR,
                allowlist=allowlist,
                token=token,
                main_networks=self._service_network_names(main),
            )
            await self._env._sdk_upload_dir(local_proxy_dir, self._EGRESS_PROXY_DIR)
            await self._env._sdk_upload_file(
                local_compose_path,
                f"{self._COMPOSE_DIR}/{self._EGRESS_PROXY_COMPOSE}",
            )
            self._egress_proxy_overlay_ready = True

        resolved_config = await self._resolve_compose_config()
        self._validate_vm_network_isolation(resolved_config)

    def _compose_no_proxy_hosts(self, config: dict[str, Any]) -> list[str]:
        """Names that must keep using task-internal compose networking."""
        services = config.get("services") or {}
        hosts: set[str] = set(services.keys())

        for config in services.values():
            if not isinstance(config, dict):
                continue
            for key in ("hostname", "container_name"):
                value = config.get(key)
                if isinstance(value, str) and value:
                    hosts.add(value)

            networks = config.get("networks") or {}
            if isinstance(networks, dict):
                for network in networks.values():
                    if isinstance(network, dict):
                        hosts.update(network.get("aliases") or [])

            extra_hosts = config.get("extra_hosts") or {}
            if isinstance(extra_hosts, dict):
                hosts.update(extra_hosts.keys())
            elif isinstance(extra_hosts, list):
                for entry in extra_hosts:
                    if isinstance(entry, str):
                        hosts.add(entry.split("=", 1)[0].split(":", 1)[0])

        for server in self._env.task_env_config.mcp_servers:
            if server.url and (hostname := urlparse(server.url).hostname):
                hosts.add(hostname)

        return sorted(host for host in hosts if isinstance(host, str) and host)

    async def _resolve_compose_config(
        self, *, include_vm_networking: bool = True
    ) -> dict[str, Any]:
        result = await self._compose_exec(
            ["config", "--format", "json"],
            include_agent_image=False,
            include_vm_networking=include_vm_networking,
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose config failed: {result.stdout} {result.stderr}"
            )
        try:
            config = json.loads(result.stdout or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("docker compose config returned invalid JSON") from exc
        if not isinstance(config, dict):
            raise RuntimeError("docker compose config did not return a mapping")
        return config

    async def _resolve_main_compose_config(self) -> dict[str, Any]:
        config = await self._resolve_compose_config()
        try:
            main = config["services"]["main"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "docker compose config did not define service 'main'"
            ) from exc
        if not isinstance(main, dict):
            raise RuntimeError("docker compose service 'main' is not a mapping")
        return main

    async def _prepare_agent_base_image(self) -> str | None:
        install = self._env.agent_install_spec
        if install is None:
            return None

        main = await self._resolve_main_compose_config()
        if main.get("build"):
            base_image = self._main_image_name("base")
            await self._upload_text(
                self._BASE_IMAGE_COMPOSE,
                f"services:\n  main:\n    image: {base_image}\n",
            )
            self._base_image_overlay_ready = True
            return base_image

        image = main.get("image")
        if not isinstance(image, str) or not image:
            raise RuntimeError(
                "Cannot preinstall the agent: compose service 'main' has "
                "neither a build definition nor an image."
            )
        return image

    async def _build_agent_image(
        self, base_image: str | None, *, force_build: bool
    ) -> None:
        install = self._env.agent_install_spec
        if install is None:
            return
        if not base_image:
            raise RuntimeError("Missing base image for agent preinstallation")

        local_build_dir = self._env.trial_paths.trial_dir / "modal-agent-build"
        write_agent_dockerfile(
            build_dir=local_build_dir,
            source_environment_dir=local_build_dir,
            prebuilt_image_name=base_image,
            install=install,
            user=self._env._resolve_user(None),
        )
        await self._env._sdk_upload_dir(local_build_dir, self._AGENT_BUILD_DIR)

        agent_image = self._main_image_name("agent")
        command = ["docker", "build", "-t", agent_image]
        if force_build:
            command.append("--no-cache")
        command.append(self._AGENT_BUILD_DIR)
        result = await self._vm_exec(
            shlex.join(command),
            timeout_sec=round(self._env.task_env_config.build_timeout_sec),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"agent image build failed: {result.stdout} {result.stderr}"
            )

        await self._upload_text(
            self._AGENT_IMAGE_COMPOSE,
            "services:\n"
            "  main:\n"
            f"    image: {agent_image}\n"
            "    build: !reset null\n"
            "    pull_policy: never\n",
        )
        self._agent_image_overlay_ready = True

    @property
    def _project_name(self) -> str:
        return self._env.session_id.lower().replace(".", "-")

    def _compose_cmd(
        self,
        subcommand: list[str],
        *,
        include_base_image: bool = False,
        include_agent_image: bool = True,
        include_vm_networking: bool = True,
    ) -> str:
        """Build a fully shell-escaped docker compose command string."""
        parts = [
            "docker",
            "compose",
            "-p",
            self._project_name,
            "--project-directory",
            self._ENVIRONMENT_DIR,
            *self._compose_file_flags(
                include_base_image=include_base_image,
                include_agent_image=include_agent_image,
                include_vm_networking=include_vm_networking,
            ),
            *subcommand,
        ]
        return shlex.join(parts)

    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
        *,
        include_base_image: bool = False,
        include_agent_image: bool = True,
        include_vm_networking: bool = True,
    ) -> ExecResult:
        """Run a docker compose subcommand on the sandbox."""
        return await self._vm_exec(
            self._compose_cmd(
                subcommand,
                include_base_image=include_base_image,
                include_agent_image=include_agent_image,
                include_vm_networking=include_vm_networking,
            ),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    async def _wait_for_docker_daemon(self) -> None:
        """Poll until the Docker daemon inside the sandbox is responsive."""
        self._env.logger.debug("Waiting for Docker daemon inside DinD sandbox...")
        last_output = ""
        for _ in range(self._DOCKER_DAEMON_POLL_LIMIT):
            result = await self._vm_exec("docker info", timeout_sec=10)
            if result.return_code == 0:
                self._env.logger.debug("Docker daemon is ready")
                return
            last_output = (result.stdout or "") + (result.stderr or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {self._DOCKER_DAEMON_POLL_LIMIT} "
            f"poll attempts. Last output: {last_output}"
        )

    async def _wait_for_main_container(self, timeout_sec: int = 60) -> None:
        """Poll until the 'main' compose service is running."""
        self._env.logger.debug("Waiting for main container to be running...")
        for _ in range(timeout_sec // 2):
            result = await self._compose_exec(
                ["exec", "-T", "main", "true"], timeout_sec=10
            )
            if result.return_code == 0:
                self._env.logger.debug("Main container is running")
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"Main container not running after {timeout_sec}s")

    async def start(self, force_build: bool) -> None:
        env = self._env
        use_vm = self._use_vm_runtime

        if use_vm:
            # Modal's recommended Docker-in-Sandbox path: real Linux kernel +
            # stock dockerd (not gVisor enable_docker). Native bridge/iptables
            # work, so we can keep task compose networks instead of host-net.
            # https://modal.com/docs/guide/vm-sandboxes
            env._image = (
                Image.from_registry("ubuntu:24.04")
                .env({"DEBIAN_FRONTEND": "noninteractive"})
                .apt_install(
                    [
                        "ca-certificates",
                        "curl",
                        "docker-buildx",
                        "docker-compose-v2",
                        "docker.io",
                    ]
                )
                .run_commands(
                    "mkdir -p /pier/compose /pier/environment /pier/logs /etc/docker",
                )
            )
            experimental_options = {"vm_runtime": True}
            sandbox_command: tuple[str, ...] | None = ("/usr/bin/dockerd", "-D")
        else:
            dind_image: str = env._kwargs.get("dind_image", "docker:28.3.3-dind")
            # Pre-configure dockerd for gVisor sandboxes which lack iptables
            # kernel modules and netlink permissions for creating veth pairs.
            # Disabling iptables and the default bridge avoids both issues.
            # All compose services must use network_mode: host.
            env._image = Image.from_registry(dind_image).dockerfile_commands(
                "RUN mkdir -p /etc/docker "
                '&& echo \'{"iptables": false, "bridge": "none"}\' '
                "> /etc/docker/daemon.json"
            )
            experimental_options = {"enable_docker": True}
            sandbox_command = None

        env._app = await App.lookup.aio(
            name=env._app_name,
            create_if_missing=True,
        )

        # DinD / VM sandbox needs network for Docker daemon and image pulls
        env._sandbox = await env._create_sandbox(
            block_network=False,
            experimental_options=experimental_options,
            sandbox_command=sandbox_command,
        )

        # Wait for Docker daemon to be ready inside the sandbox
        await self._wait_for_docker_daemon()

        if use_vm:
            env.logger.info(
                "Modal VM runtime: using native Docker bridge networking "
                "(no host-network overlay)."
            )
        else:
            env.logger.debug(
                "DinD mode uses host networking: no port isolation between "
                "services, no Docker DNS service discovery (extra_hosts entries "
                "map service names to 127.0.0.1 instead), and no network "
                "namespace isolation."
            )

        # Upload Pier compose files to the sandbox
        for path in (
            COMPOSE_BASE_PATH,
            COMPOSE_BUILD_PATH,
            COMPOSE_PREBUILT_PATH,
            COMPOSE_NO_NETWORK_PATH,
        ):
            await env._sdk_upload_file(path, f"{self._COMPOSE_DIR}/{path.name}")

        # Upload task environment directory (Dockerfiles, compose file, etc.)
        await env._sdk_upload_dir(env.environment_dir, self._ENVIRONMENT_DIR)

        # Create log directories on sandbox (volume-mounted into main container)
        # chmod 777 so non-root agent/verifier users can write to them.
        await self._vm_exec(
            f"mkdir -p {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts && "
            f"chmod 777 {self._LOGS_DIR}/verifier {self._LOGS_DIR}/agent "
            f"{self._LOGS_DIR}/artifacts"
        )

        # Build and start compose services
        self._use_prebuilt = not force_build and bool(env.task_env_config.docker_image)

        if use_vm:
            mounts = self._build_mounts_overlay()
            await self._upload_text(self._MOUNTS_COMPOSE, mounts)
            await self._prepare_vm_networking()
        else:
            # Under host networking every service shares the sandbox netns,
            # where non-root users can't bind ports <1024 (unlike a private
            # container netns). Hardened services that expose 443 need this.
            sysctl = await self._vm_exec(
                "sysctl -w net.ipv4.ip_unprivileged_port_start=0 || true",
                timeout_sec=10,
            )
            if "= 0" not in (sysctl.stdout or ""):
                env.logger.warning(
                    "Could not lower ip_unprivileged_port_start; services binding "
                    f"privileged ports as non-root will fail: {sysctl.stderr}"
                )

            overlay = self._build_host_network_overlay(
                env.environment_dir, use_prebuilt=self._use_prebuilt
            )
            await self._vm_exec(
                f"cat > {self._COMPOSE_DIR}/{self._HOST_NETWORK_COMPOSE} << 'YAML'\n"
                f"{overlay}\n"
                f"YAML",
                timeout_sec=10,
            )

        agent_base_image = await self._prepare_agent_base_image() if use_vm else None

        env.logger.debug("Building compose services inside DinD sandbox...")
        build_command = ["build"]
        if force_build:
            build_command.append("--no-cache")
        result = await self._compose_exec(
            build_command,
            timeout_sec=round(env.task_env_config.build_timeout_sec),
            include_base_image=use_vm,
            include_agent_image=False,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose build failed: {result.stdout} {result.stderr}"
            )

        if use_vm:
            await self._build_agent_image(agent_base_image, force_build=force_build)

        env.logger.debug("Starting compose services inside DinD sandbox...")
        result = await self._compose_exec(["up", "-d"], timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"docker compose up failed: {result.stdout} {result.stderr}"
            )

        await self._wait_for_main_container()

    async def stop(self, delete: bool) -> None:
        if self._env._sandbox:
            try:
                await self._compose_exec(["down", "--remove-orphans"], timeout_sec=30)
            except Exception as e:
                self._env.logger.warning(f"docker compose down failed: {e}")

        await self._teardown_sandbox()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute command inside the main compose container."""
        parts: list[str] = ["exec", "-T"]
        if cwd:
            parts.extend(["-w", cwd])
        if env:
            for k, v in env.items():
                parts.extend(["-e", f"{k}={v}"])
        if user is not None:
            parts.extend(["-u", str(user)])
        parts.extend(["main", "bash", "-c", command])

        return await self._compose_exec(parts, timeout_sec=timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Two-hop upload: SDK → sandbox temp, docker compose cp → main."""
        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._env._sdk_upload_file(source_path, temp)
            result = await self._compose_exec(
                ["cp", temp, f"main:{target_path}"], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Two-hop upload: SDK → sandbox temp dir, docker compose cp → main."""
        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._env._sdk_upload_dir(source_dir, temp)
            result = await self._compose_exec(
                ["cp", f"{temp}/.", f"main:{target_dir}"], timeout_sec=120
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    def _sandbox_log_path(self, container_path: str) -> str | None:
        """Map a container log path to its sandbox volume-mount location.

        Paths under /logs/{verifier,agent,artifacts} inside the main container
        are volume-mounted to /pier/logs/... on the sandbox, so they can be
        accessed directly without docker compose cp.
        """
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
        """Download a file from the main container.

        Fast path: if the file is under a volume-mounted log dir, download
        directly from the sandbox. Slow path: docker compose cp to sandbox
        temp, then SDK download.
        """
        sandbox_path = self._sandbox_log_path(source_path)
        if sandbox_path:
            await self._env._sdk_download_file(sandbox_path, target_path)
            return

        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            result = await self._compose_exec(
                ["cp", f"main:{source_path}", temp], timeout_sec=60
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"docker compose cp failed: {result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_file(temp, target_path)
        finally:
            await self._vm_exec(f"rm -f {shlex.quote(temp)}", timeout_sec=10)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the main container.

        Fast path: if under a volume-mounted log dir, download directly from
        the sandbox. Slow path: docker compose cp to sandbox temp, then SDK
        download.
        """
        sandbox_path = self._sandbox_log_path(source_dir)
        if sandbox_path:
            await self._env._sdk_download_dir(sandbox_path, target_dir)
            return

        temp = f"/tmp/pier_{uuid4().hex}"
        try:
            await self._vm_exec(f"mkdir -p {shlex.quote(temp)}", timeout_sec=10)
            result = await self._compose_exec(
                ["cp", f"main:{source_dir}/.", temp], timeout_sec=120
            )
            if result.return_code != 0:
                self._env.logger.error(
                    f"download_dir: docker compose cp failed: "
                    f"{result.stdout} {result.stderr}"
                )
                raise RuntimeError(
                    f"download_dir: docker compose cp failed: "
                    f"{result.stdout} {result.stderr}"
                )
            await self._env._sdk_download_dir(temp, target_dir)
        finally:
            await self._vm_exec(f"rm -rf {shlex.quote(temp)}", timeout_sec=10)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -d {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(
            f"test -f {shlex.quote(path)}", timeout_sec=10, user=user
        )
        return result.return_code == 0

    async def attach(self) -> None:
        env = self._env
        if not env._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Drop into the main compose container, not the DinD sandbox VM
        compose_exec_cmd = self._compose_cmd(["exec", "main", "bash"])
        os.execvp(
            "modal",
            ["modal", "shell", env._sandbox.object_id, "--cmd", compose_exec_cmd],
        )


class ModalEnvironment(BaseEnvironment):
    environment_dir: Path
    environment_name: str
    session_id: str
    trial_paths: TrialPaths
    config: EnvironmentConfig

    @classmethod
    def preflight(cls) -> None:
        import os
        from pathlib import Path

        modal_config = Path.home() / ".modal.toml"
        has_env_token = os.environ.get("MODAL_TOKEN_ID") and os.environ.get(
            "MODAL_TOKEN_SECRET"
        )
        if not modal_config.exists() and not has_env_token:
            raise SystemExit(
                "Modal requires authentication. Run 'modal token new' to set up "
                "credentials, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "environment variables."
            )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.MODAL

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            cpu_request=True,
            memory_limit=True,
            memory_request=True,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return self._capabilities

    @property
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        if self.task_env_config.docker_image:
            return
        if self._compose_mode:
            path = self.environment_dir / "docker-compose.yaml"
        else:
            path = self._environment_definition_path
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Please ensure the file exists.")

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        secrets: list[str] | None = None,
        registry_secret: str | None = None,
        volumes: dict[str, str] | None = None,
        app_name: str = "__pier__",
        sandbox_timeout_secs: int = 60 * 60 * 24,
        sandbox_idle_timeout_secs: int | None = None,
        vm_runtime: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a ModalEnvironment instance.

        Args:
            environment_dir: The directory containing the environment definition files.
            environment_name: The name identifier for this environment instance.
            session_id: Unique session identifier for this environment instance.
            trial_paths: Path configuration object containing trial-related directory
                paths.
            task_env_config: Environment configuration specifying resources (CPU,
                memory), GPU types, and network access.
            secrets: Optional list of Modal secret names to mount into the sandbox.
            registry_secret: Optional Modal secret name for authenticating with
                a private container registry (e.g. AWS ECR). When set, the
                Dockerfile's FROM image is pulled using Image.from_aws_ecr()
                instead of Image.from_dockerfile().
            volumes: Optional mapping of mount paths to Modal volume names.
            app_name: Name of the Modal App to use. All sandboxes created
                with the same app name share a single Modal App. Default
                is "__pier__".
            sandbox_timeout_secs: Maximum lifetime of the sandbox in seconds.
                The sandbox will be terminated after this duration regardless of
                activity. Default is 86400 (24 hours). See Modal sandbox docs:
                https://modal.com/docs/reference/modal.Sandbox#create
            sandbox_idle_timeout_secs: Seconds of inactivity after which the
                sandbox will be automatically terminated. None means no idle
                timeout (default). See Modal sandbox docs:
                https://modal.com/docs/reference/modal.Sandbox#create
            vm_runtime: If True, create Modal VM Sandboxes
                (``experimental_options={{"vm_runtime": True}}``) with a real
                Linux kernel. For compose tasks this replaces gVisor
                ``enable_docker`` DinD and enables native Docker networking.
                See https://modal.com/docs/guide/vm-sandboxes. CPU-only.
        """
        # Detect compose mode *before* super().__init__ which calls
        # _validate_definition
        self._compose_mode = (environment_dir / "docker-compose.yaml").exists()
        self._vm_runtime = bool(vm_runtime)
        isolated_runtime = not self._compose_mode or self._vm_runtime
        self._capabilities = EnvironmentCapabilities(
            gpus=not self._vm_runtime,  # Modal VM sandboxes are CPU-only today
            disable_internet=isolated_runtime,
            filtered_egress=isolated_runtime,
            preinstall_agents=isolated_runtime,
            docker_compose=True,
        )
        self._kwargs = kwargs
        if not _HAS_MODAL:
            raise MissingExtraError(package="modal", extra="modal")

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )
        self._image: Image | None = None
        self._app: App | None = None
        self._sandbox: Sandbox | None = None
        self._egress_proxy_sandbox: Sandbox | None = None
        self._egress_proxy_env: dict[str, str] = {}
        self._egress_cidr_allowlist: list[str] | None = None
        self._secrets = secrets or []
        self._registry_secret = registry_secret
        self._volumes = volumes or {}
        self._app_name = app_name
        self._sandbox_timeout = sandbox_timeout_secs
        self._sandbox_idle_timeout = sandbox_idle_timeout_secs

        # Select strategy based on compose mode
        self._strategy: _ModalStrategy = (
            _ModalDinD(self) if self._compose_mode else _ModalDirect(self)
        )
        self.logger.debug(
            "Selected strategy: %s (vm_runtime=%s)",
            self._strategy.__class__.__name__,
            self._vm_runtime,
        )

    @property
    def _default_shell(self) -> str:
        """Shell available on the sandbox VM.

        Alpine-based gVisor DinD images only have ``sh``; Ubuntu VM images and
        standard single-container images have ``bash``.
        """
        if self._compose_mode and not self._vm_runtime:
            return "sh"
        return "bash"

    def _cpu_config(self) -> int | float | tuple[int | float, int] | None:
        """Resolve CPU configuration for sandbox creation.

        Modal's scalar form is request-only. Tuple form carries request and
        limit, which Pier uses for stricter resource modes.
        """
        cpus = self._effective_cpus
        if cpus is None:
            return None
        if self._cpu_resource_mode == ResourceMode.REQUEST:
            return cpus
        if self._cpu_resource_mode == ResourceMode.LIMIT:
            return (min(_MODAL_DEFAULT_CPU_REQUEST_CORES, cpus), cpus)
        return (cpus, cpus)

    def _memory_config(self) -> int | tuple[int, int] | None:
        memory_mb = self._effective_memory_mb
        if memory_mb is None:
            return None
        if self._memory_resource_mode in (ResourceMode.AUTO, ResourceMode.REQUEST):
            return memory_mb
        if self._memory_resource_mode == ResourceMode.LIMIT:
            return (min(_MODAL_DEFAULT_MEMORY_REQUEST_MB, memory_mb), memory_mb)
        return (memory_mb, memory_mb)

    def _gpu_config(self) -> str | None:
        """Resolve GPU configuration string for sandbox creation.

        When a specific GPU type is requested, append Modal's ``!`` suffix
        to pin to that exact type. Without ``!``, Modal may silently
        upgrade requests (e.g. ``H100`` → ``H200``, ``A100-40GB`` →
        ``A100-80GB``) "at no extra cost" — convenient for general
        workloads, but breaks benchmark reproducibility because the same
        ``task.toml`` can land on different hardware across trials.

        No ``!`` is appended when ``gpu_types`` is unset (the default
        ``any``), since "any" already means "accept whatever is available".
        """
        if self._effective_gpus <= 0:
            return None
        gpu_type = "any"
        if self.task_env_config.gpu_types:
            if len(self.task_env_config.gpu_types) > 1:
                self.logger.debug(
                    "Multiple GPU types specified but Modal only supports one "
                    "GPU type. Using the first GPU type."
                )
            # Pin to exact type to prevent silent Modal upgrades (e.g. H100 → H200).
            gpu_type = f"{self.task_env_config.gpu_types[0]}!"
        return f"{gpu_type}:{self._effective_gpus}"

    def _secrets_config(self) -> list:
        secrets = [Secret.from_name(secret) for secret in self._secrets]
        # Inject resolved [environment.env] from task.toml into the sandbox
        if self._persistent_env:
            secrets.append(
                Secret.from_dict(dict[str, str | None](self._persistent_env))
            )
        return secrets

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self._egress_proxy_env:
            return env
        merged = dict(self._egress_proxy_env)
        if env:
            merged.update(env)
        return merged or None

    def _volumes_config(self) -> dict[str, Volume]:
        return {
            mount_path: Volume.from_name(volume_name)
            for mount_path, volume_name in self._volumes.items()
        }

    def _with_agent_install(self, image: Image) -> Image:
        install = self.agent_install_spec
        if install is None:
            return image
        commands = dockerfile_install_commands(
            install,
            user=self._resolve_user(None),
        )
        return image.dockerfile_commands(*commands)

    @staticmethod
    def _resolve_ipv4(host: str) -> str:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
        return infos[0][4][0]

    async def _ensure_egress_proxy(self) -> None:
        allowlist = self.network_allowlist
        if self.task_env_config.allow_internet or not allowlist.domains:
            return
        if self._compose_mode:
            raise ValueError(
                "Filtered inference egress is not supported for Modal docker-compose tasks."
            )
        if self._egress_proxy_sandbox is not None:
            return

        token = new_proxy_token()
        proxy_image = Image.debian_slim(python_version="3.12").apt_install(
            "apache2-utils",
            "ca-certificates",
            "squid",
        )
        self._egress_proxy_sandbox = await Sandbox.create.aio(
            "sh",
            "-c",
            squid_bootstrap_command(),
            app=self._app,
            image=proxy_image,
            env=proxy_policy_env(allowlist, token),
            unencrypted_ports=[EGRESS_PROXY_PORT],
            readiness_probe=modal.sandbox.Probe.with_tcp(EGRESS_PROXY_PORT),
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=f"{self.session_id}-egress-proxy",
        )
        tunnel = (await self._egress_proxy_sandbox.tunnels.aio(timeout=60))[
            EGRESS_PROXY_PORT
        ]
        if tunnel.unencrypted_host and tunnel.unencrypted_port:
            proxy_host = tunnel.unencrypted_host
            proxy_port = tunnel.unencrypted_port
        else:
            proxy_host = tunnel.host
            proxy_port = tunnel.port
        proxy_ip = self._resolve_ipv4(proxy_host)
        self._egress_cidr_allowlist = [f"{proxy_ip}/32"]
        self._egress_proxy_env = proxy_environment(token, proxy_host, proxy_port)

    async def _teardown_egress_proxy(self) -> None:
        if self._egress_proxy_sandbox is None:
            self._egress_proxy_env = {}
            self._egress_cidr_allowlist = None
            return
        try:
            await self._egress_proxy_sandbox.terminate.aio()
            await self._egress_proxy_sandbox.wait.aio(raise_on_termination=False)
        except Exception as e:
            self.logger.warning(f"Error terminating Modal egress proxy sandbox: {e}")
        finally:
            self._egress_proxy_sandbox = None
            self._egress_proxy_env = {}
            self._egress_cidr_allowlist = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(
        self,
        *,
        block_network: bool | None = None,
        cidr_allowlist: list[str] | None = None,
        experimental_options: dict[str, Any] | None = None,
        sandbox_command: tuple[str, ...] | None = ("sleep", "infinity"),
    ) -> Sandbox:
        """Create a sandbox with retry logic for transient failures.

        Modal's Sandbox inherits the container CMD from `*args`; if omitted, the image
        must define CMD/ENTRYPOINT. Many benchmark images omit both — pass the default
        long-running stub (mirror of docker-compose's ``sleep infinity``).

        DinD uses images that already define dockerd — pass ``sandbox_command=None`` to
        keep the upstream entrypoint/command.
        """
        if block_network is None:
            block_network = (
                not self.task_env_config.allow_internet
                and not self._egress_cidr_allowlist
            )

        kwargs: dict[str, Any] = {}
        if experimental_options:
            kwargs["experimental_options"] = experimental_options

        sandbox_args = sandbox_command if sandbox_command is not None else ()

        if (cpu := self._cpu_config()) is not None:
            kwargs["cpu"] = cpu
        if (memory := self._memory_config()) is not None:
            kwargs["memory"] = memory
        if (gpu := self._gpu_config()) is not None:
            kwargs["gpu"] = gpu

        sandbox = await Sandbox.create.aio(
            *sandbox_args,
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=self.session_id,
            block_network=block_network,
            cidr_allowlist=cidr_allowlist or self._egress_cidr_allowlist,
            secrets=self._secrets_config(),
            volumes=self._volumes_config(),  # type: ignore[arg-type]
            **kwargs,
        )
        self.logger.info("Modal sandbox created: %s", sandbox.object_id)
        return sandbox

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _terminate_sandbox(self) -> None:
        """Terminate sandbox with retry logic."""
        if self._sandbox:
            await self._sandbox.terminate.aio()

    async def _sdk_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        shell: str = "bash",
        login: bool = False,
    ) -> ExecResult:
        """Execute a command directly on the Modal sandbox VM.

        This is the low-level exec that talks to the Modal SDK.  Strategies
        should use this (or the public ``exec``) rather than calling
        ``sandbox.exec.aio`` directly.

        Args:
            shell: Shell to use (``"bash"`` for standard images,
                ``"sh"`` for Alpine-based images like docker:dind).
            login: If True, use a login shell (``-lc``) so that
                ``.bashrc``, ``.profile``, etc. are sourced.
        """
        # Merge persistent env vars (--ae flags) with per-exec env vars
        env = self._merge_env(env)

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        process = await self._sandbox.exec.aio(
            shell,
            "-lc" if login else "-c",
            command,
            workdir=cwd,
            secrets=[Secret.from_dict(env)] if env else [],  # type: ignore
            timeout=timeout_sec,
        )

        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()
        return_code = await process.wait.aio()

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_upload_file(self, source_path: Path | str, target_path: str) -> None:
        """
        Adds a local file to the environment.

        Args:
            source_path: The path to the source local file.
            target_path: The path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Creates parent directories on the remote side if needed.
        await self._sandbox.filesystem.copy_from_local.aio(source_path, target_path)

    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """
        Adds a local directory to the environment.

        Args:
            source_dir: The path to the source local directory.
            target_dir: The path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        shell = self._default_shell
        await self._sdk_exec(f"mkdir -p {shlex.quote(target_dir)}", shell=shell)
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_path).as_posix()
                target_file_path = str(PurePosixPath(target_dir) / relative_path)

                target_file_parent = str(PurePosixPath(target_file_path).parent)
                if target_file_parent != target_dir:
                    await self._sdk_exec(
                        f"mkdir -p {shlex.quote(target_file_parent)}", shell=shell
                    )

                await self._sdk_upload_file(file_path, target_file_path)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _sdk_download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        """
        Downloads a file from the environment to the local machine.

        Args:
            source_path: The path to the source file in the environment.
            target_path: The local path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        # Creates parent directories locally if needed.
        await self._sandbox.filesystem.copy_to_local.aio(source_path, target_path)

    async def _sdk_download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """
        Downloads a directory from the environment to the local machine. This overwrites
        existing files in the target directory.

        Args:
            source_dir: The path to the source directory in the environment.
            target_dir: The local path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Run find on the sandbox VM directly via exec_on_vm, not through
        # the strategy's exec (which in DinD mode routes to the compose
        # container, not the sandbox filesystem).
        result = await self._strategy.exec_on_vm(
            f"find {shlex.quote(source_dir)} -type f", shell="sh"
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to list files in {source_dir}: {result.stderr}")

        if not result.stdout or not result.stdout.strip():
            return

        file_paths = [p for p in result.stdout.strip().split("\n") if p.strip()]

        sem = asyncio.Semaphore(5)

        async def _download_one(remote_path: str) -> None:
            async with sem:
                rel = Path(remote_path).relative_to(Path(source_dir))
                local_path = target_dir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self._sdk_download_file(remote_path, local_path)

        async with asyncio.TaskGroup() as tg:
            for p in file_paths:
                tg.create_task(_download_one(p))

    async def start(self, force_build: bool) -> None:
        try:
            await self._strategy.start(force_build)
        except BaseException:
            cleanup = asyncio.create_task(self._strategy._teardown_sandbox())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise

    async def stop(self, delete: bool):
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

        if isinstance(self._strategy, _ModalDinD):
            # docker compose exec supports -u natively. Never su-wrap here:
            # compose exec runs as the image's (possibly non-root) default
            # user, and `su` upward to root prompts for a password and hangs.
            return await self._strategy.exec(
                command, cwd=effective_cwd, env=env, timeout_sec=timeout_sec, user=user
            )

        if user is not None:
            # Modal doesn't support user= on exec; wrap with su.
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(user))
            command = f"su -m {user_arg} -s /bin/bash -c {shlex.quote(command)}"

        return await self._strategy.exec(
            command, cwd=effective_cwd, env=env, timeout_sec=timeout_sec
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        return await self._strategy.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        return await self._strategy.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str):
        return await self._strategy.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        return await self._strategy.download_dir(source_dir, target_dir)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_dir(path, user=self._resolve_user(user))

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return await self._strategy.is_file(path, user=self._resolve_user(user))

    async def attach(self) -> None:
        return await self._strategy.attach()

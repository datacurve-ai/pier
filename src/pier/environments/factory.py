from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import NamedTuple

from pier.constants import PYPI_PACKAGE_NAME
from pier.environments.base import BaseEnvironment
from pier.environments.capabilities import EnvironmentResourceCapabilities
from pier.environments.resource_policies import validate_resource_capabilities
from pier.models.agent.install import AgentInstallSpec
from pier.models.agent.network import NetworkAllowlist
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.config import (
    EnvironmentConfig as TrialEnvironmentConfig,
)
from pier.models.trial.config import ResourceMode
from pier.models.trial.paths import TrialPaths


class _EnvEntry(NamedTuple):
    module: str
    class_name: str
    pip_extra: str | None


# Registry of built-in environment types. Modules are imported lazily so optional
# vendor SDKs are only loaded when that environment is requested. Pier only ships
# Docker and Modal; other Pier environment implementations are not included.
_ENVIRONMENT_REGISTRY: dict[EnvironmentType, _EnvEntry] = {
    EnvironmentType.DOCKER: _EnvEntry(
        "pier.environments.docker.docker",
        "DockerEnvironment",
        None,
    ),
    EnvironmentType.MODAL: _EnvEntry(
        "pier.environments.modal",
        "ModalEnvironment",
        "modal",
    ),
    EnvironmentType.DAYTONA: _EnvEntry(
        "pier.environments.daytona",
        "DaytonaEnvironment",
        "daytona",
    ),
}


def _load_environment_class(env_type: EnvironmentType) -> type[BaseEnvironment]:
    """Lazily import and return the environment class for *env_type*."""
    entry = _ENVIRONMENT_REGISTRY.get(env_type)
    if entry is None:
        raise ValueError(
            f"Unsupported environment type: {env_type}. This could be because the "
            "environment is not registered in the EnvironmentFactory or because "
            "the environment type is invalid."
        )

    try:
        module = importlib.import_module(entry.module)
    except ImportError as exc:
        if entry.pip_extra is not None:
            raise ImportError(
                f"The '{env_type.value}' environment requires extra dependencies. "
                f"Install them with:\n"
                f"  pip install {PYPI_PACKAGE_NAME}\n"
                f"  uv tool install {PYPI_PACKAGE_NAME}"
            ) from exc
        raise

    cls: type[BaseEnvironment] = getattr(module, entry.class_name)
    return cls


class EnvironmentFactory:
    @classmethod
    def supports_exec_stdin(
        cls,
        type: EnvironmentType | None,
        import_path: str | None = None,
    ) -> bool:
        """Return whether the configured environment supports streamed stdin."""
        environment_class: type[BaseEnvironment] | None = None
        if import_path is not None and ":" in import_path:
            module_path, class_name = import_path.split(":", 1)
            try:
                environment_class = getattr(
                    importlib.import_module(module_path), class_name
                )
            except (ImportError, AttributeError):
                return False
        elif type is not None and type in _ENVIRONMENT_REGISTRY:
            environment_class = _load_environment_class(type)
        return bool(
            environment_class is not None and environment_class.supports_exec_stdin()
        )

    @classmethod
    def create_environment(
        cls,
        type: EnvironmentType,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        agent_install_spec: AgentInstallSpec | None = None,
        network_allowlist: NetworkAllowlist | None = None,
        default_user: str | int | None = None,
        **kwargs,
    ) -> BaseEnvironment:
        environment_class = _load_environment_class(type)

        return environment_class(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            agent_install_spec=agent_install_spec,
            network_allowlist=network_allowlist,
            default_user=default_user,
            **kwargs,
        )

    @classmethod
    def run_preflight(
        cls,
        type: EnvironmentType | None,
        import_path: str | None = None,
    ) -> None:
        """Run credential preflight checks for the given environment type."""
        if import_path is not None:
            if ":" not in import_path:
                return
            module_path, class_name = import_path.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                env_class = getattr(module, class_name)
                if hasattr(env_class, "preflight"):
                    env_class.preflight()
            except (ImportError, AttributeError):
                pass
            return

        if type is None or type not in _ENVIRONMENT_REGISTRY:
            return

        env_class = _load_environment_class(type)
        env_class.preflight()

    @classmethod
    def resource_capabilities(
        cls,
        type: EnvironmentType | None,
        import_path: str | None = None,
    ) -> EnvironmentResourceCapabilities | None:
        if import_path is not None:
            if ":" not in import_path:
                return None
            module_path, class_name = import_path.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                env_class = getattr(module, class_name)
            except (ImportError, AttributeError):
                return None
            resource_capabilities = getattr(env_class, "resource_capabilities", None)
            if callable(resource_capabilities):
                return resource_capabilities()
            return None

        if type is None or type not in _ENVIRONMENT_REGISTRY:
            return None

        env_class = _load_environment_class(type)
        return env_class.resource_capabilities()

    @classmethod
    def validate_resource_policies(cls, config: TrialEnvironmentConfig) -> None:
        resource_capabilities = cls.resource_capabilities(
            config.type, config.import_path
        )
        if resource_capabilities is None:
            return
        environment_label = (
            config.import_path
            if config.import_path is not None
            else config.type.value
            if config.type is not None
            else "environment"
        )
        validate_resource_capabilities(
            environment_label=environment_label,
            resource_capabilities=resource_capabilities,
            cpu_enforcement_policy=config.cpu_enforcement_policy,
            memory_enforcement_policy=config.memory_enforcement_policy,
        )

    @classmethod
    def create_environment_from_import_path(
        cls,
        import_path: str,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        agent_install_spec: AgentInstallSpec | None = None,
        network_allowlist: NetworkAllowlist | None = None,
        default_user: str | int | None = None,
        **kwargs,
    ) -> BaseEnvironment:
        """
        Create an environment from an import path.

        Args:
            import_path (str): The import path of the environment. In the format
                'module.path:ClassName'.

        Returns:
            BaseEnvironment: The created environment.

        Raises:
            ValueError: If the import path is invalid.
        """
        if ":" not in import_path:
            raise ValueError("Import path must be in format 'module.path:ClassName'")

        module_path, class_name = import_path.split(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"Failed to import module '{module_path}': {e}") from e

        try:
            Environment = getattr(module, class_name)
        except AttributeError as e:
            raise ValueError(
                f"Module '{module_path}' has no class '{class_name}'"
            ) from e

        return Environment(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            agent_install_spec=agent_install_spec,
            network_allowlist=network_allowlist,
            default_user=default_user,
            **kwargs,
        )

    @classmethod
    def create_environment_from_config(
        cls,
        config: TrialEnvironmentConfig,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        agent_install_spec: AgentInstallSpec | None = None,
        network_allowlist: NetworkAllowlist | None = None,
        default_user: str | int | None = None,
        **kwargs,
    ) -> BaseEnvironment:
        """
        Create an environment from an environment configuration.

        Args:
            config (TrialEnvironmentConfig): The configuration of the environment.

        Returns:
            BaseEnvironment: The created environment.

        Raises:
            ValueError: If the configuration is invalid.
        """
        env_constructor_kwargs = {
            "override_cpus": config.override_cpus,
            "override_memory_mb": config.override_memory_mb,
            "override_storage_mb": config.override_storage_mb,
            "override_gpus": config.override_gpus,
            "cpu_enforcement_policy": config.cpu_enforcement_policy,
            "memory_enforcement_policy": config.memory_enforcement_policy,
            "suppress_override_warnings": config.suppress_override_warnings,
            "mounts_json": config.mounts,
            "persistent_env": config.env,
            **config.kwargs,
            **kwargs,
        }
        if config.cpu_enforcement_policy == ResourceMode.AUTO:
            env_constructor_kwargs.pop("cpu_enforcement_policy")
        if config.memory_enforcement_policy == ResourceMode.AUTO:
            env_constructor_kwargs.pop("memory_enforcement_policy")

        if config.import_path is not None:
            return cls.create_environment_from_import_path(
                config.import_path,
                environment_dir=environment_dir,
                environment_name=environment_name,
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task_env_config,
                logger=logger,
                agent_install_spec=agent_install_spec,
                network_allowlist=network_allowlist,
                default_user=default_user,
                **env_constructor_kwargs,
            )
        elif config.type is not None:
            return cls.create_environment(
                type=config.type,
                environment_dir=environment_dir,
                environment_name=environment_name,
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task_env_config,
                logger=logger,
                agent_install_spec=agent_install_spec,
                network_allowlist=network_allowlist,
                default_user=default_user,
                **env_constructor_kwargs,
            )
        else:
            raise ValueError(
                "At least one of environment type or import_path must be set."
            )

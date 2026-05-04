from __future__ import annotations

import asyncio
from dataclasses import dataclass
from logging import Logger
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pier.agents.factory import AgentFactory
from pier.environments.base import BaseEnvironment
from pier.environments.factory import EnvironmentFactory
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName
from pier.models.task.config import TaskOS
from pier.models.task.task import Task
from pier.models.trial.config import AgentConfig, EnvironmentConfig
from pier.models.trial.paths import TrialPaths


class AgentSetupTimeoutError(asyncio.TimeoutError):
    pass


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class EnvironmentStartTimeoutError(asyncio.TimeoutError):
    pass


@dataclass
class TrialExecution:
    task: Task
    agent: Any
    environment: BaseEnvironment
    agent_timeout_sec: float | None
    agent_setup_timeout_sec: float
    environment_build_timeout_sec: float

    @classmethod
    def create(
        cls,
        *,
        task: Task,
        agent_config: AgentConfig,
        environment_config: EnvironmentConfig,
        trial_paths: TrialPaths,
        session_id: str,
        logger: Logger,
        timeout_multiplier: float,
        agent_timeout_multiplier: float | None,
        agent_setup_timeout_multiplier: float | None,
        environment_build_timeout_multiplier: float | None,
        default_agent_setup_timeout_sec: float,
    ) -> "TrialExecution":
        agent_timeout_sec = cls._resolve_agent_timeout(
            agent_config=agent_config,
            task_agent_timeout_sec=task.config.agent.timeout_sec,
            timeout_multiplier=timeout_multiplier,
            agent_timeout_multiplier=agent_timeout_multiplier,
        )
        agent = cls._create_agent(
            agent_config=agent_config,
            task=task,
            trial_paths=trial_paths,
            logger=logger,
            agent_timeout_sec=agent_timeout_sec,
        )
        environment = cls._create_environment(
            environment_config=environment_config,
            task=task,
            session_id=session_id,
            trial_paths=trial_paths,
            logger=logger,
            agent=agent,
        )
        return cls(
            task=task,
            agent=agent,
            environment=environment,
            agent_timeout_sec=agent_timeout_sec,
            agent_setup_timeout_sec=cls._resolve_agent_setup_timeout(
                agent_config=agent_config,
                default_timeout_sec=default_agent_setup_timeout_sec,
                timeout_multiplier=timeout_multiplier,
                agent_setup_timeout_multiplier=agent_setup_timeout_multiplier,
            ),
            environment_build_timeout_sec=cls._resolve_environment_build_timeout(
                task_environment_build_timeout_sec=task.config.environment.build_timeout_sec,
                timeout_multiplier=timeout_multiplier,
                environment_build_timeout_multiplier=environment_build_timeout_multiplier,
            ),
        )

    @staticmethod
    def _resolve_agent_timeout(
        *,
        agent_config: AgentConfig,
        task_agent_timeout_sec: float | None,
        timeout_multiplier: float,
        agent_timeout_multiplier: float | None,
    ) -> float | None:
        base_timeout = agent_config.override_timeout_sec or task_agent_timeout_sec
        if base_timeout is None:
            return None

        cap = agent_config.max_timeout_sec or float("inf")
        multiplier = (
            agent_timeout_multiplier
            if agent_timeout_multiplier is not None
            else timeout_multiplier
        )
        return min(base_timeout, cap) * multiplier

    @staticmethod
    def _resolve_agent_setup_timeout(
        *,
        agent_config: AgentConfig,
        default_timeout_sec: float,
        timeout_multiplier: float,
        agent_setup_timeout_multiplier: float | None,
    ) -> float:
        multiplier = (
            agent_setup_timeout_multiplier
            if agent_setup_timeout_multiplier is not None
            else timeout_multiplier
        )
        return (
            agent_config.override_setup_timeout_sec
            if agent_config.override_setup_timeout_sec is not None
            else default_timeout_sec
        ) * multiplier

    @staticmethod
    def _resolve_environment_build_timeout(
        *,
        task_environment_build_timeout_sec: float,
        timeout_multiplier: float,
        environment_build_timeout_multiplier: float | None,
    ) -> float:
        multiplier = (
            environment_build_timeout_multiplier
            if environment_build_timeout_multiplier is not None
            else timeout_multiplier
        )
        return task_environment_build_timeout_sec * multiplier

    @staticmethod
    def _create_agent(
        *,
        agent_config: AgentConfig,
        task: Task,
        trial_paths: TrialPaths,
        logger: Logger,
        agent_timeout_sec: float | None,
    ):
        extra_kwargs: dict[str, Any] = {}
        if agent_config.name == AgentName.ORACLE.value:
            extra_kwargs = {
                "task_dir": task.task_dir,
                "trial_paths": trial_paths,
                "agent_timeout_sec": agent_timeout_sec,
            }
        if task.config.environment.mcp_servers:
            extra_kwargs["mcp_servers"] = task.config.environment.mcp_servers
        if task.config.environment.skills_dir:
            extra_kwargs["skills_dir"] = task.config.environment.skills_dir

        return AgentFactory.create_agent_from_config(
            agent_config,
            logs_dir=trial_paths.agent_dir,
            logger=logger,
            **extra_kwargs,
        )

    @staticmethod
    def _create_environment(
        *,
        environment_config: EnvironmentConfig,
        task: Task,
        session_id: str,
        trial_paths: TrialPaths,
        logger: Logger,
        agent,
    ) -> BaseEnvironment:
        return EnvironmentFactory.create_environment_from_config(
            config=environment_config,
            environment_dir=task.paths.environment_dir,
            environment_name=task.name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            logger=logger,
            agent_install_spec=agent.install_spec(),
            network_allowlist=agent.network_allowlist(),
            default_user=task.config.agent.user,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EnvironmentStartTimeoutError),
    )
    async def start_environment(self, *, force_build: bool) -> None:
        try:
            await asyncio.wait_for(
                self.environment.start(force_build=force_build),
                timeout=self.environment_build_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise EnvironmentStartTimeoutError(
                f"Environment start timed out after {
                    self.environment_build_timeout_sec
                } seconds"
            ) from e

    def ensure_agent_supports_environment(self) -> None:
        if self.environment.task_os != TaskOS.WINDOWS or self.agent.SUPPORTS_WINDOWS:
            return

        raise RuntimeError(
            f"Agent '{self.agent.name()}' does not support Windows containers. "
            "Only agents with SUPPORTS_WINDOWS = True can run Windows tasks "
            "(currently: oracle, nop)."
        )

    async def setup_agent(self) -> None:
        self.ensure_agent_supports_environment()
        try:
            await asyncio.wait_for(
                self.agent.setup(environment=self.environment),
                timeout=self.agent_setup_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise AgentSetupTimeoutError(
                f"Agent setup timed out after {self.agent_setup_timeout_sec} seconds"
            ) from e

    async def run_agent(
        self,
        *,
        instruction: str,
        context: AgentContext,
    ) -> None:
        try:
            await asyncio.wait_for(
                self.agent.run(
                    instruction=instruction,
                    environment=self.environment,
                    context=context,
                ),
                timeout=self.agent_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise AgentTimeoutError(
                f"Agent execution timed out after {self.agent_timeout_sec} seconds"
            ) from e

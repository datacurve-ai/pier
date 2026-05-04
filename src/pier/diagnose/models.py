from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from pier.models.agent.context import AgentContext
from pier.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from pier.models.trial.config import AgentConfig, EnvironmentConfig
from pier.models.trial.result import AgentInfo, ExceptionInfo, TimingInfo


class DiagnosticConfig(BaseModel):
    """Configuration for one diagnostic run over an existing Pier job."""

    run_name: str
    source_job_dir: Path
    prompt_path: Path
    agent: AgentConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    n_concurrent: int = 1
    overwrite: bool = False
    trial_names: list[str] | None = None
    limit: int | None = None
    filter_passing: bool | None = None


class DiagnosticItem(BaseModel):
    source_trial_dir: Path
    source_trial_name: str
    task_dir: Path


class DiagnosticItemResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_trial_name: str
    diagnostic_trial_name: str
    task_name: str
    task_id: LocalTaskId | GitTaskId | PackageTaskId
    task_checksum: str
    source_trial_uri: str
    diagnostic_trial_uri: str
    agent_info: AgentInfo | None = None
    agent_result: AgentContext | None = None
    diagnostic_result: dict[str, Any] | None = None
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None


class DiagnosticJobResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_name: str
    source_job_dir: Path
    diagnostic_dir: Path
    config: DiagnosticConfig
    started_at: datetime | None = None
    finished_at: datetime | None = None
    item_results: list[DiagnosticItemResult] = Field(default_factory=list)
    failed_items: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def n_items(self) -> int:
        return len(self.item_results) + len(self.failed_items)

    @computed_field
    @property
    def n_failed(self) -> int:
        return len(
            [r for r in self.item_results if r.exception_info is not None]
        ) + len(self.failed_items)

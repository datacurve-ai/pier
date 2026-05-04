from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shlex
import shutil
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from shortuuid import ShortUUID

from pier.agents.installed.base import BaseInstalledAgent
from pier.diagnose.models import (
    DiagnosticConfig,
    DiagnosticItem,
    DiagnosticItemResult,
    DiagnosticJobResult,
)
from pier.models.agent.context import AgentContext
from pier.models.task.task import Task
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import ExceptionInfo, TimingInfo, TrialResult
from pier.trial.execution import TrialExecution
from pier.utils.logger import logger as global_logger

DIAGNOSTIC_DIRNAME = "diagnostic"
DIAGNOSTIC_INPUT_DIRNAME = "input"
DIAGNOSTIC_TASK_DIRNAME = "task"
DIAGNOSTIC_TRIAL_DIRNAME = "trial"
DIAGNOSTIC_RESULT_FILENAME = "diagnostic-result.json"
DIAGNOSTIC_MARKDOWN_FILENAME = "diagnostic-result.md"
DIAGNOSTIC_PROMPT_FILENAME = "prompt.md"


@dataclass(frozen=True)
class DiagnosticPaths:
    diagnostic_dir: Path

    @property
    def mount_paths(self) -> TrialPaths:
        return TrialPaths(self.diagnostic_dir)

    def mkdir(self) -> None:
        self.mount_paths.mkdir()

    @property
    def config_path(self) -> Path:
        return self.diagnostic_dir / "diagnostic-config.json"

    @property
    def result_path(self) -> Path:
        return self.diagnostic_dir / "diagnostic-metadata.json"

    @property
    def log_path(self) -> Path:
        return self.diagnostic_dir / "diagnostic.log"

    @property
    def agent_dir(self) -> Path:
        return self.mount_paths.agent_dir

    @property
    def artifacts_dir(self) -> Path:
        return self.mount_paths.artifacts_dir

    @property
    def artifacts_manifest_path(self) -> Path:
        return self.mount_paths.artifacts_manifest_path

    @property
    def exception_message_path(self) -> Path:
        return self.diagnostic_dir / "exception.txt"


def is_job_dir(path: Path) -> bool:
    return (path / "job.log").exists()


def is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() and (path / "result.json").exists()


def diagnostic_run_dir(source_job_dir: Path, run_name: str) -> Path:
    return source_job_dir / "diagnostics" / run_name


def _replace_template_vars(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _is_passing_trial(result: TrialResult) -> bool:
    has_reward_one = (
        result.verifier_result is not None
        and result.verifier_result.rewards is not None
        and result.verifier_result.rewards.get("reward", 0) == 1.0
    )
    return has_reward_one and result.exception_info is None


class DiagnosticRunner:
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.source_job_dir = config.source_job_dir.resolve()
        self.diagnostic_dir = diagnostic_run_dir(self.source_job_dir, config.run_name)

    def collect_items(self) -> list[DiagnosticItem]:
        if not is_job_dir(self.source_job_dir):
            raise ValueError(f"Not a Pier job directory: {self.source_job_dir}")

        trial_dirs = sorted(
            d for d in self.source_job_dir.iterdir() if d.is_dir() and is_trial_dir(d)
        )

        if self.config.trial_names:
            selected = set(self.config.trial_names)
            trial_dirs = [d for d in trial_dirs if d.name in selected]
            missing = selected - {d.name for d in trial_dirs}
            if missing:
                raise ValueError(
                    f"Trial(s) not found in {self.source_job_dir}: "
                    f"{', '.join(sorted(missing))}"
                )

        items: list[DiagnosticItem] = []
        for trial_dir in trial_dirs:
            source_result = TrialResult.model_validate_json(
                (trial_dir / "result.json").read_text(encoding="utf-8")
            )
            if self.config.filter_passing is not None:
                is_passing = _is_passing_trial(source_result)
                if self.config.filter_passing != is_passing:
                    continue

            task_dir = source_result.config.task.get_local_path()
            if not task_dir.exists():
                raise FileNotFoundError(
                    f"Task directory for trial {trial_dir.name} does not exist: "
                    f"{task_dir}"
                )
            items.append(
                DiagnosticItem(
                    source_trial_dir=trial_dir,
                    source_trial_name=trial_dir.name,
                    task_dir=task_dir,
                )
            )

        if self.config.limit is not None:
            items = items[: self.config.limit]

        if not items:
            raise ValueError(f"No trial directories selected in {self.source_job_dir}")

        return items

    async def run_job(
        self,
        on_total: Callable[[int], None] | None = None,
        on_item_complete: Callable[[], None] | None = None,
    ) -> DiagnosticJobResult:
        self._prepare_diagnostic_dir()
        items = self.collect_items()
        if on_total is not None:
            on_total(len(items))

        (self.diagnostic_dir / "config.json").write_text(
            self.config.model_dump_json(indent=4), encoding="utf-8"
        )

        job_result = DiagnosticJobResult(
            run_name=self.config.run_name,
            source_job_dir=self.source_job_dir,
            diagnostic_dir=self.diagnostic_dir,
            config=self.config,
            started_at=datetime.now(timezone.utc),
        )

        semaphore = asyncio.Semaphore(self.config.n_concurrent)

        async def _run_one(item: DiagnosticItem) -> None:
            try:
                async with semaphore:
                    result = await self.run_item(item)
                job_result.item_results.append(result)
            except Exception as e:
                job_result.failed_items.append(f"{item.source_trial_name}: {e}")
            finally:
                if on_item_complete is not None:
                    on_item_complete()
                self._write_job_result(job_result)

        async with asyncio.TaskGroup() as tg:
            for item in items:
                tg.create_task(_run_one(item))

        job_result.item_results.sort(key=lambda r: r.source_trial_name)
        job_result.failed_items.sort()
        job_result.finished_at = datetime.now(timezone.utc)
        self._write_job_result(job_result)
        return job_result

    def _prepare_diagnostic_dir(self) -> None:
        config_path = self.diagnostic_dir / "config.json"

        if self.config.overwrite and self.diagnostic_dir.exists():
            shutil.rmtree(self.diagnostic_dir)

        if config_path.exists():
            existing_config = DiagnosticConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
            if self._resume_key(existing_config) != self._resume_key(self.config):
                raise FileExistsError(
                    f"Diagnostic run {self.diagnostic_dir} already exists and cannot "
                    "be resumed with a different config. Use --overwrite to replace it."
                )
            self.config = self.config.model_copy(
                update={
                    "prompt_path": existing_config.prompt_path,
                }
            )
            return

        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.config.model_copy(
            update={
                "prompt_path": self._copy_input_file(
                    self.config.prompt_path, DIAGNOSTIC_PROMPT_FILENAME
                ),
            }
        )

    def _copy_input_file(self, source: Path, filename: str) -> Path:
        target = self.diagnostic_dir / filename
        shutil.copyfile(source, target)
        return target

    @staticmethod
    def _resume_key(config: DiagnosticConfig) -> dict[str, Any]:
        return config.model_dump(exclude={"overwrite", "prompt_path"})

    async def run_item(self, item: DiagnosticItem) -> DiagnosticItemResult:
        paths = DiagnosticPaths(self.diagnostic_dir / item.source_trial_name)
        cached = paths.result_path
        if not self.config.overwrite and cached.exists():
            return DiagnosticItemResult.model_validate_json(
                cached.read_text(encoding="utf-8")
            )
        if self.config.overwrite and paths.diagnostic_dir.exists():
            shutil.rmtree(paths.diagnostic_dir)

        trial = DiagnosticTrial(self.config, item, self.diagnostic_dir)
        return await trial.run()

    def _write_job_result(self, result: DiagnosticJobResult) -> None:
        (self.diagnostic_dir / "result.json").write_text(
            result.model_dump_json(indent=4), encoding="utf-8"
        )


class DiagnosticTrial:
    _AGENT_SETUP_TIMEOUT_SEC = 360.0

    def __init__(
        self,
        config: DiagnosticConfig,
        item: DiagnosticItem,
        diagnostic_dir: Path,
    ):
        self.config = config
        self.item = item
        self.diagnostic_dir = diagnostic_dir
        self.source_result = TrialResult.model_validate_json(
            (item.source_trial_dir / "result.json").read_text(encoding="utf-8")
        )
        self._task = Task(item.task_dir)
        self._diagnostic_paths = DiagnosticPaths(
            diagnostic_dir / item.source_trial_name
        )
        self._trial_paths = self._diagnostic_paths.mount_paths
        self._diagnostic_paths.mkdir()
        self._are_agent_logs_downloaded = False
        self._log_handler: logging.Handler | None = None
        self._init_logger()

        self._execution = TrialExecution.create(
            task=self._task,
            agent_config=config.agent,
            environment_config=config.environment,
            trial_paths=self._trial_paths,
            session_id=self._session_id(),
            logger=self._logger,
            timeout_multiplier=config.timeout_multiplier,
            agent_timeout_multiplier=config.agent_timeout_multiplier,
            agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
            default_agent_setup_timeout_sec=self._AGENT_SETUP_TIMEOUT_SEC,
        )
        self._agent = self._execution.agent
        self._environment = self._execution.environment

        self._result = DiagnosticItemResult(
            source_trial_name=item.source_trial_name,
            diagnostic_trial_name=item.source_trial_name,
            task_name=self._task.name,
            task_id=self.source_result.task_id,
            task_checksum=self._task.checksum,
            source_trial_uri=item.source_trial_dir.expanduser().resolve().as_uri(),
            diagnostic_trial_uri=self._diagnostic_paths.diagnostic_dir.expanduser()
            .resolve()
            .as_uri(),
            agent_info=self._agent.to_agent_info(),
        )

    @property
    def result(self) -> DiagnosticItemResult:
        return self._result

    def _session_id(self) -> str:
        raw = f"diag__{self.config.run_name}__{self.item.source_trial_name}"
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        suffix = ShortUUID().random(length=5)
        prefix = cleaned[:20].rstrip("_.-") or "diag"
        return f"{prefix}__{digest}_{suffix}"

    def _init_logger(self) -> None:
        self._logger = global_logger.getChild(
            f"{__name__}.{self.config.run_name}.{self.item.source_trial_name}"
        )
        file_handler = logging.FileHandler(self._diagnostic_paths.log_path)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)
        self._log_handler = file_handler

    def _close_logger_handler(self) -> None:
        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    async def run(self) -> DiagnosticItemResult:
        self._diagnostic_paths.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostic_paths.config_path.write_text(
            self.config.model_dump_json(indent=4), encoding="utf-8"
        )
        self.result.started_at = datetime.now(timezone.utc)

        try:
            await self._setup_environment()
            await self._environment.run_healthcheck()
            self._environment.default_user = self._task.config.agent.user
            try:
                await self._setup_agent()
                self.result.agent_info = self._agent.to_agent_info()
                await self._upload_diagnostic_inputs()
                await self._execute_agent()
            finally:
                self._environment.default_user = None

            await self._maybe_download_logs(
                source_dir=self._environment.env_paths.agent_dir.as_posix(),
                target_dir=self._diagnostic_paths.agent_dir,
            )
            self._maybe_populate_agent_context(self.result.agent_result)
            await self._download_artifacts()
            self._parse_diagnostic_result()

        except asyncio.CancelledError as e:
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._diagnostic_paths.exception_message_path.write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            await self._best_effort_download_outputs()
            raise

        except Exception as e:
            self._logger.debug(f"Diagnostic trial failed: {e}")
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._diagnostic_paths.exception_message_path.write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            await self._best_effort_download_outputs()

        finally:
            await self._cleanup_and_finalize()
            self._close_logger_handler()

        return self.result

    async def _setup_environment(self) -> None:
        self.result.environment_setup = TimingInfo(
            started_at=datetime.now(timezone.utc)
        )
        try:
            await self._execution.start_environment(
                force_build=self.config.environment.force_build
            )
        finally:
            self.result.environment_setup.finished_at = datetime.now(timezone.utc)

    async def _setup_agent(self) -> None:
        self.result.agent_setup = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            await self._execution.setup_agent()
        finally:
            self.result.agent_setup.finished_at = datetime.now(timezone.utc)

    async def _upload_diagnostic_inputs(self) -> None:
        task_dir = self._diagnostic_task_dir()
        trial_dir = self._diagnostic_trial_dir()
        await self._environment.exec(
            "mkdir -p "
            f"{shlex.quote(task_dir.as_posix())} "
            f"{shlex.quote(trial_dir.as_posix())} "
            f"{shlex.quote(self._environment.env_paths.artifacts_dir.as_posix())}",
            user="root",
            timeout_sec=30,
        )
        await self._environment.upload_dir(self.item.task_dir, task_dir.as_posix())
        await self._environment.upload_dir(
            self.item.source_trial_dir, trial_dir.as_posix()
        )

    async def _execute_agent(self) -> None:
        self.result.agent_execution = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            self.result.agent_result = AgentContext()
            await self._execution.run_agent(
                instruction=self._render_instruction(),
                context=self.result.agent_result,
            )
        finally:
            self.result.agent_execution.finished_at = datetime.now(timezone.utc)

    def _render_instruction(self) -> str:
        diagnostic_task_dir = self._diagnostic_task_dir()
        diagnostic_trial_dir = self._diagnostic_trial_dir()
        diagnostic_result_path = self._diagnostic_result_path()
        diagnostic_markdown_path = self._diagnostic_markdown_path()
        values = {
            "task_dir": diagnostic_task_dir.as_posix(),
            "trial_dir": diagnostic_trial_dir.as_posix(),
            "source_trial_name": self.item.source_trial_name,
            "task_name": self._task.name,
            "diagnostic_result_path": diagnostic_result_path.as_posix(),
            "diagnostic_markdown_path": diagnostic_markdown_path.as_posix(),
        }
        prompt = _replace_template_vars(
            self.config.prompt_path.read_text(encoding="utf-8"), values
        )

        sections = [
            prompt.rstrip(),
            "Diagnostic runtime contract:",
            f"- Task files are available at `{diagnostic_task_dir.as_posix()}`.",
            f"- Source trial files are available at `{diagnostic_trial_dir.as_posix()}`.",
            f"- Write a valid JSON object to `{diagnostic_result_path.as_posix()}`.",
            f"- You may also write Markdown notes to `{diagnostic_markdown_path.as_posix()}`.",
        ]
        return "\n\n".join(sections) + "\n"

    def _diagnostic_input_dir(self) -> PurePosixPath:
        return (
            self._environment.env_paths.logs_dir.parent
            / DIAGNOSTIC_DIRNAME
            / DIAGNOSTIC_INPUT_DIRNAME
        )

    def _diagnostic_task_dir(self) -> PurePosixPath:
        return self._diagnostic_input_dir() / DIAGNOSTIC_TASK_DIRNAME

    def _diagnostic_trial_dir(self) -> PurePosixPath:
        return self._diagnostic_input_dir() / DIAGNOSTIC_TRIAL_DIRNAME

    def _diagnostic_result_path(self) -> PurePosixPath:
        return self._environment.env_paths.artifacts_dir / DIAGNOSTIC_RESULT_FILENAME

    def _diagnostic_markdown_path(self) -> PurePosixPath:
        return self._environment.env_paths.artifacts_dir / DIAGNOSTIC_MARKDOWN_FILENAME

    async def _maybe_download_logs(self, source_dir: str, target_dir: Path) -> None:
        if self._are_agent_logs_downloaded:
            return
        if self._environment.capabilities.mounted:
            await self._environment.prepare_logs_for_host()
            self._are_agent_logs_downloaded = True
            return
        try:
            await self._environment.download_dir(
                source_dir=source_dir, target_dir=target_dir
            )
        except Exception:
            self._logger.error(f"Failed to download logs to {target_dir}")
        self._are_agent_logs_downloaded = True

    def _maybe_populate_agent_context(self, agent_result: AgentContext | None) -> None:
        if (
            agent_result is None
            or not agent_result.is_empty()
            or not isinstance(self._agent, BaseInstalledAgent)
        ):
            return
        self._agent.populate_context_post_run(agent_result)

    async def _download_artifacts(self) -> None:
        self._diagnostic_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self._environment.capabilities.mounted:
            try:
                await self._environment.download_dir(
                    source_dir=self._environment.env_paths.artifacts_dir.as_posix(),
                    target_dir=self._diagnostic_paths.artifacts_dir,
                )
            except Exception:
                self._logger.warning("Failed to download diagnostic artifacts")
        self._write_artifacts_manifest()

    def _write_artifacts_manifest(self) -> None:
        entries = []
        for name, source in (
            (
                DIAGNOSTIC_RESULT_FILENAME,
                self._diagnostic_result_path().as_posix(),
            ),
            (
                DIAGNOSTIC_MARKDOWN_FILENAME,
                self._diagnostic_markdown_path().as_posix(),
            ),
        ):
            path = self._diagnostic_paths.artifacts_dir / name
            entries.append(
                {
                    "source": source,
                    "destination": f"artifacts/{name}",
                    "type": "file",
                    "status": "ok" if path.exists() else "missing",
                }
            )
        self._diagnostic_paths.artifacts_manifest_path.write_text(
            json.dumps(entries, indent=2), encoding="utf-8"
        )

    def _parse_diagnostic_result(self) -> None:
        result_path = self._diagnostic_paths.artifacts_dir / DIAGNOSTIC_RESULT_FILENAME
        if not result_path.exists():
            raise FileNotFoundError(
                "Diagnostic result was not written to "
                f"{self._diagnostic_result_path().as_posix()}"
            )
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("diagnostic-result.json must contain a JSON object")
        self.result.diagnostic_result = parsed

    async def _best_effort_download_outputs(self) -> None:
        try:
            await self._maybe_download_logs(
                source_dir=self._environment.env_paths.agent_dir.as_posix(),
                target_dir=self._diagnostic_paths.agent_dir,
            )
            self._maybe_populate_agent_context(self.result.agent_result)
        except Exception:
            pass
        try:
            await self._download_artifacts()
            if self.result.diagnostic_result is None:
                self._parse_diagnostic_result()
        except Exception:
            pass

    async def _cleanup_and_finalize(self) -> None:
        try:
            await asyncio.shield(
                self._environment.stop(delete=self.config.environment.delete)
            )
        except asyncio.CancelledError:
            self._logger.warning(
                "Cleanup interrupted, but environment stop is shielded"
            )
        except Exception as e:
            self._logger.warning(f"Diagnostic environment cleanup failed: {e}")
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)

        self.result.finished_at = datetime.now(timezone.utc)
        self._diagnostic_paths.result_path.write_text(
            self.result.model_dump_json(indent=4), encoding="utf-8"
        )

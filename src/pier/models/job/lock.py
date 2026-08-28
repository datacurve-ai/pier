from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, Field, field_validator

from pier.constants import PYPI_PACKAGE_NAME
from pier.models.job.config import JobConfig, RetryConfig
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from pier.utils.env import sanitize_env_assignment

LOCK_FILENAME = "lock.json"
_DIGEST_PREFIX = "sha256:"
_ENV_ARG_FLAGS = {"--ae", "--agent-env", "--ve", "--verifier-env"}


def _validate_digest(value: str) -> str:
    if not value.startswith(_DIGEST_PREFIX):
        raise ValueError(f"Digest must start with '{_DIGEST_PREFIX}'. Got: {value}")
    hex_digest = value.removeprefix(_DIGEST_PREFIX)
    if len(hex_digest) != 64 or any(c not in "0123456789abcdef" for c in hex_digest):
        raise ValueError(
            f"Digest must be in 'sha256:<64 hex chars>' format. Got: {value}"
        )
    return value


def _prefixed_digest(value: str) -> str:
    return value if value.startswith(_DIGEST_PREFIX) else f"{_DIGEST_PREFIX}{value}"


def sanitize_cli_invocation(argv: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False

    for arg in argv:
        if redact_next:
            sanitized.append(sanitize_env_assignment(arg))
            redact_next = False
            continue

        if arg in _ENV_ARG_FLAGS:
            sanitized.append(arg)
            redact_next = True
            continue

        flag, sep, value = arg.partition("=")
        if sep and flag in _ENV_ARG_FLAGS:
            sanitized.append(f"{flag}={sanitize_env_assignment(value)}")
        else:
            sanitized.append(arg)

    return sanitized


class PierLockInfo(BaseModel):
    version: str | None = None
    git_commit_hash: str | None = None
    is_editable: bool | None = None


class TaskLock(BaseModel):
    name: str
    type: Literal["local", "git", "package"]
    digest: str
    source: str | None = None
    path: Path | None = None
    git_url: str | None = None
    git_commit_id: str | None = None

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)


class TrialLock(BaseModel):
    task: TaskLock
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    verifier_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    agent: AgentConfig
    environment: EnvironmentConfig
    verifier: VerifierConfig


class JobLock(BaseModel):
    schema_version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pier: PierLockInfo = Field(default_factory=PierLockInfo)
    invocation: list[str] | None = None
    n_concurrent_trials: int
    optimize_concurrency: bool = False
    retry: RetryConfig
    trials: list[TrialLock] = Field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JobLock):
            return NotImplemented
        return self._canonical_payload() == other._canonical_payload()

    def _canonical_payload(self) -> dict:
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "pier", "invocation"},
        )
        retry = payload.get("retry")
        if isinstance(retry, dict):
            for key in ("include_exceptions", "exclude_exceptions"):
                value = retry.get(key)
                if isinstance(value, list):
                    retry[key] = sorted(value)

        trials = payload.get("trials")
        if isinstance(trials, list):
            payload["trials"] = sorted(
                trials,
                key=lambda trial: json.dumps(
                    trial, sort_keys=True, separators=(",", ":")
                ),
            )
        return payload


def build_job_lock(
    *,
    config: JobConfig,
    trial_configs: Sequence[TrialConfig],
    invocation: list[str] | None = None,
) -> JobLock:
    return JobLock(
        pier=_get_pier_info(),
        invocation=sanitize_cli_invocation(
            invocation if invocation is not None else list(sys.argv)
        ),
        n_concurrent_trials=config.n_concurrent_trials,
        optimize_concurrency=config.optimize_concurrency,
        retry=config.retry,
        trials=[_build_lock_trial(trial_config) for trial_config in trial_configs],
    )


def _build_lock_trial(trial_config: TrialConfig) -> TrialLock:
    return TrialLock(
        task=_build_lock_trial_task(trial_config.task),
        timeout_multiplier=trial_config.timeout_multiplier,
        agent_timeout_multiplier=trial_config.agent_timeout_multiplier,
        verifier_timeout_multiplier=trial_config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=trial_config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=(
            trial_config.environment_build_timeout_multiplier
        ),
        agent=trial_config.agent,
        environment=trial_config.environment,
        verifier=trial_config.verifier,
    )


def _build_lock_trial_task(task_config: TaskConfig) -> TaskLock:
    task_id = task_config.get_task_id()
    name = task_id.get_name()

    if task_config.is_package_task():
        task_type: Literal["local", "git", "package"] = "package"
        if task_config.ref is None or not task_config.ref.startswith(_DIGEST_PREFIX):
            raise ValueError(
                f"Package task {name!r} was not resolved to a sha256 digest."
            )
        digest = task_config.ref
    else:
        task_type = "git" if task_config.is_git_task() else "local"
        digest = _prefixed_digest(_compute_content_hash(task_config.get_local_path()))

    return TaskLock(
        name=name,
        type=task_type,
        digest=digest,
        source=task_config.source,
        path=task_config.path,
        git_url=task_config.git_url,
        git_commit_id=task_config.git_commit_id,
    )


def _compute_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        return digest.hexdigest()

    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        if ".git" in file_path.relative_to(path).parts:
            continue
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _get_pier_info() -> PierLockInfo:
    return PierLockInfo(
        version=_get_pier_version(),
        git_commit_hash=_get_pier_git_commit_hash(),
        is_editable=_get_pier_is_editable_install(),
    )


def _get_pier_version() -> str | None:
    try:
        return version(PYPI_PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def _get_pier_git_commit_hash() -> str | None:
    direct_url_data = _get_pier_direct_url_data()
    if direct_url_data is None:
        return None

    vcs_info = direct_url_data.get("vcs_info")
    if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
        commit_id = vcs_info.get("commit_id")
        if isinstance(commit_id, str) and commit_id:
            return commit_id

    if not _is_pier_editable_install(direct_url_data):
        return None

    repo_path = _get_file_path_from_direct_url(direct_url_data.get("url"))
    if repo_path is None:
        return None

    return _get_git_commit_hash(repo_path)


def _get_git_commit_hash(repo_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _get_pier_is_editable_install() -> bool | None:
    direct_url_data = _get_pier_direct_url_data()
    if direct_url_data is None:
        return None
    return _is_pier_editable_install(direct_url_data)


def _get_pier_direct_url_data() -> dict | None:
    try:
        dist = distribution("pier")
    except PackageNotFoundError:
        return None

    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        return {}

    try:
        direct_url_data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    return direct_url_data if isinstance(direct_url_data, dict) else None


def _is_pier_editable_install(direct_url_data: dict) -> bool:
    dir_info = direct_url_data.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return bool(dir_info.get("editable", False))


def _get_file_path_from_direct_url(url: object) -> Path | None:
    if not isinstance(url, str):
        return None

    parsed_url = urlparse(url)
    if parsed_url.scheme != "file" or parsed_url.netloc not in ("", "localhost"):
        return None

    return Path(url2pathname(parsed_url.path))

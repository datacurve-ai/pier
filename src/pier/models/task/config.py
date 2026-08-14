# NOTE: When updating this file, also update the corresponding docs page:
# docs/content/docs/tasks/index.mdx

import re
import tomllib
import warnings
from enum import Enum
from typing import Any

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

from pier.constants import ORG_NAME_PATTERN


class TaskOS(str, Enum):
    """Target operating system for a task's container."""

    LINUX = "linux"
    WINDOWS = "windows"


class VerifierEnvironmentMode(str, Enum):
    """Whether the verifier runs in the agent's environment or its own."""

    SHARED = "shared"
    SEPARATE = "separate"


class NetworkMode(str, Enum):
    """Network access policy for agent and verifier execution.

    Mirrors Harbor's ``NetworkMode`` (harbor.models.task.config): tasks authored
    against Harbor >= 0.18 declare network policy via ``network_mode`` (at
    [environment] scope, with optional [agent]/[verifier] phase overrides)
    instead of the legacy ``allow_internet`` boolean. Values must stay in sync
    with Harbor so the same task.toml parses identically in both harnesses.
    """

    NO_NETWORK = "no-network"
    PUBLIC = "public"
    ALLOWLIST = "allowlist"


class NetworkPolicyFieldsMixin(BaseModel):
    """``network_mode`` field shared by the [environment] scope and the
    [agent]/[verifier] phase overrides (Harbor's PhaseNetworkPolicyConfig
    equivalent). Pier currently enforces 'no-network' and 'public' only;
    'allowlist' (and its 'allowed_hosts' companion) is rejected at parse time
    rather than silently mis-enforced."""

    network_mode: NetworkMode | None = Field(
        default=None,
        description="Network access policy ('no-network' or 'public'). On "
        "[agent]/[verifier] this is an explicit phase override; on "
        "[environment] it applies to both phases. When unset everywhere, the "
        "legacy 'allow_internet' boolean applies.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Unsupported (Harbor allowlist-mode companion field); "
        "declared only so its presence fails loudly instead of being ignored.",
    )

    @model_validator(mode="after")
    def _validate_network_policy(self) -> "NetworkPolicyFieldsMixin":
        if self.network_mode == NetworkMode.ALLOWLIST:
            raise ValueError(
                "network_mode='allowlist' is not supported by pier yet; "
                "use 'no-network' or 'public'."
            )
        if self.allowed_hosts is not None:
            raise ValueError("allowed_hosts is not supported by pier yet.")
        return self


class Author(BaseModel):
    """Author information for a package or dataset."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(default=None, description="Author email address")


class PackageInfo(BaseModel):
    """Package metadata for the [task] section of task.toml.

    This section identifies the package in the registry with a unique name.
    """

    name: str = Field(
        ...,
        description="Package name in org/name format (e.g., 'pier/hello-world')",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the task",
    )
    authors: list[Author] = Field(
        default_factory=list,
        description="List of package authors",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for search and categorization",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate that name follows org/name format."""
        if not re.match(ORG_NAME_PATTERN, v) or ".." in v:
            raise ValueError(
                f"Package name must be in 'org/name' format with alphanumeric characters, "
                f"hyphens, underscores, and dots. Cannot start with a dot or contain '..'. Got: {v}"
            )
        return v

    @property
    def org(self) -> str:
        """Extract organization from package name."""
        return self.name.split("/")[0]

    @property
    def short_name(self) -> str:
        """Extract short name (without org) from package name."""
        return self.name.split("/")[1]


MAIN_SERVICE_NAME = "main"

_COMPOSE_SERVICE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _validate_compose_service_name(value: str | None) -> str | None:
    if value is None:
        return value
    value = value.strip()
    if not _COMPOSE_SERVICE_NAME_PATTERN.match(value):
        raise ValueError(
            f"Invalid Docker Compose service name: {value!r}. Service names "
            "must start with an alphanumeric character and contain only "
            "alphanumeric characters, hyphens, underscores, and dots."
        )
    return value


class VerifierCollectConfig(BaseModel):
    """A command run inside a compose service after the agent phase ends.

    Collect hooks let tasks snapshot runtime state into files before the
    environment is torn down, so the files can be declared as artifacts and
    read by a separate verifier (e.g. capture the agent's change set as
    ``/logs/artifacts/model.patch``). Mirrors Harbor's ``[[verifier.collect]]``
    blocks. Pier only runs hooks targeting the main service; hooks targeting
    compose sidecar services are skipped with a warning.
    """

    command: str = Field(..., description="Shell command to run in the service.")
    service: str = Field(
        default=MAIN_SERVICE_NAME,
        description="Compose service to run the command in. Defaults to main. "
        "Pier only runs hooks targeting main (the agent's container).",
    )
    timeout_sec: float = Field(
        default=60.0,
        description="Timeout in seconds for the collect command.",
    )
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the command as. None uses the "
        "service container's default user.",
    )

    @field_validator("service")
    @classmethod
    def _validate_service(cls, value: str) -> str:
        validated = _validate_compose_service_name(value)
        if validated is None:
            raise ValueError("Collect hook service must not be empty.")
        return validated


class VerifierConfig(NetworkPolicyFieldsMixin):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the verifier as. None uses the environment's default USER (e.g., root).",
    )
    environment_mode: VerifierEnvironmentMode | None = Field(
        default=None,
        description=(
            "Whether the verifier runs in the agent's environment ('shared') "
            "or in a dedicated container ('separate'). When omitted: defaults "
            "to 'separate' if a verifier 'environment' is set, otherwise "
            "'shared'."
        ),
    )
    environment: "EnvironmentConfig | None" = Field(
        default=None,
        description=(
            "Environment definition for the separate verifier container. "
            "Same schema as the top-level [environment] section. When set "
            "without an explicit environment_mode, implies "
            "environment_mode='separate'. When unset with "
            "environment_mode='separate', a fresh copy of the top-level "
            "[environment] is used. Conflicts with environment_mode='shared'."
        ),
    )
    collect: list[VerifierCollectConfig] = Field(
        default_factory=list,
        description=(
            "Commands run in the agent environment after the agent phase ends "
            "and before artifact collection ([[verifier.collect]] blocks in "
            "task.toml). Use these to snapshot runtime state into files that "
            "artifact entries can then collect."
        ),
    )

    @model_validator(mode="after")
    def _validate_mode_env_consistency(self) -> "VerifierConfig":
        if (
            self.environment_mode == VerifierEnvironmentMode.SHARED
            and self.environment is not None
        ):
            raise ValueError(
                "[verifier].environment_mode='shared' is incompatible with "
                "[verifier.environment]; either omit the environment or set "
                "environment_mode='separate'."
            )
        return self


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(NetworkPolicyFieldsMixin):
    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the agent as. None uses the environment's default USER (e.g., root).",
    )


class HealthcheckConfig(BaseModel):
    """Healthcheck configuration mirroring Docker HEALTHCHECK options.

    Runs a command repeatedly after environment start to verify readiness.
    All retries must pass before agent setup begins.
    """

    command: str = Field(..., description="Shell command to run. Exit 0 means healthy.")
    interval_sec: float = Field(
        default=5.0,
        description="Time in seconds between healthcheck attempts.",
    )
    timeout_sec: float = Field(
        default=30.0,
        description="Maximum time in seconds for a single healthcheck command to run.",
    )
    start_period_sec: float = Field(
        default=0.0,
        description="Grace period in seconds after environment start during which "
        "failures do not count toward retries.",
    )
    start_interval_sec: float = Field(
        default=5.0,
        description="Interval in seconds between checks during the start period.",
    )
    retries: int = Field(
        default=3,
        description="Number of consecutive failures before the healthcheck is considered failed.",
    )


class EnvironmentConfig(NetworkPolicyFieldsMixin):
    build_timeout_sec: float = 600.0  # 10 minutes default
    docker_image: str | None = None
    os: TaskOS = Field(
        default=TaskOS.LINUX,
        description="Target operating system for the task's container. "
        "Defaults to 'linux' for back-compat. Set to 'windows' to target "
        "Windows containers (requires Docker Desktop in Windows container "
        "mode on a Windows host).",
    )
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = Field(
        default=None,
        description="List of acceptable GPU types (e.g., ['H100', 'A100', 'T4']). None "
        "means any GPU type is acceptable.",
    )
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access in the environment. "
        "Legacy boolean: when 'network_mode' is set (here or as an "
        "[agent]/[verifier] phase override), the resolved mode wins and this "
        "field is overwritten at TaskConfig validation time.",
    )
    mcp_servers: list["MCPServerConfig"] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables required for the task and resolved from the host at runtime. "
        "Supports ${VAR} and ${VAR:-default} template syntax.",
    )
    skills_dir: str | None = Field(
        default=None,
        description="Path to skills directory in the environment. "
        "Contents are copied to the agent's skills config directory.",
    )
    healthcheck: HealthcheckConfig | None = Field(
        default=None,
        description="Healthcheck to run after environment start to verify readiness. "
        "Mirrors Docker HEALTHCHECK semantics.",
    )
    workdir: str | None = Field(
        default=None,
        description="Default working directory for command execution. "
        "Overrides the container's WORKDIR when set.",
    )

    # Deprecated fields - marked as excluded so they don't appear in serialization by default
    memory: str | None = Field(
        default=None,
        deprecated="Use 'memory_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )
    storage: str | None = Field(
        default=None,
        deprecated="Use 'storage_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )

    @field_validator("os", mode="before")
    @classmethod
    def normalize_os(cls, v: Any) -> Any:
        """Accept case-insensitive string values for the os field."""
        if isinstance(v, str):
            return v.lower()
        return v

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> int:
        size_str = size_str.strip().upper()

        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024)
        elif size_str.endswith("M"):
            return int(float(size_str[:-1]))
        elif size_str.endswith("K"):
            return int(float(size_str[:-1]) / 1024)
        else:
            raise ValueError(
                f"Invalid size format: {size_str}. Expected format like '1G', "
                "'512M', etc."
            )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_resource_fields(cls, data: Any) -> Any:
        """Map deprecated memory/storage fields to memory_mb/storage_mb."""
        if not isinstance(data, dict):
            return data

        if "memory" in data:
            warnings.warn(
                "The 'memory' field is deprecated. Use 'memory_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            memory = data.pop("memory")
            if isinstance(memory, str):
                memory_mb = cls._parse_size_to_mb(memory)
                if "memory_mb" in data and data["memory_mb"] != memory_mb:
                    raise ValueError(
                        "Conflicting 'memory' and 'memory_mb' values: "
                        f"memory={memory!r} ({memory_mb} MB) != "
                        f"memory_mb={data['memory_mb']!r}."
                    )
                data.setdefault("memory_mb", memory_mb)

        if "storage" in data:
            warnings.warn(
                "The 'storage' field is deprecated. Use 'storage_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            storage = data.pop("storage")
            if isinstance(storage, str):
                storage_mb = cls._parse_size_to_mb(storage)
                if "storage_mb" in data and data["storage_mb"] != storage_mb:
                    raise ValueError(
                        "Conflicting 'storage' and 'storage_mb' values: "
                        f"storage={storage!r} ({storage_mb} MB) != "
                        f"storage_mb={data['storage_mb']!r}."
                    )
                data.setdefault("storage_mb", storage_mb)

        return data


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server available to the agent."""

    name: str
    transport: str = "sse"  # "sse" | "streamable-http" | "stdio"
    url: str | None = None  # required for sse/streamable-http
    command: str | None = None  # for stdio
    args: list[str] = Field(default_factory=list)  # for stdio

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required for transport '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for transport 'stdio'")
        return self


class ArtifactConfig(BaseModel):
    source: str
    destination: str | None = None
    exclude: list[str] = Field(
        default_factory=list,
        description="Patterns to exclude when downloading a directory artifact "
        "(passed as tar --exclude flags).",
    )


class StepConfig(BaseModel):
    name: str
    agent: AgentConfig = Field(default_factory=AgentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    min_reward: float | dict[str, float] | None = Field(
        default=None,
        description="If set, abort remaining steps when this step's rewards do "
        "not meet the threshold(s). A float gates on the 'reward' key (1D "
        "convention); a dict gates on each declared key (aborts if any key is "
        "below its threshold or missing from the rewards dict). A missing "
        "verifier_result (verifier crash) or missing gated key is treated as "
        "-inf. Ignored when verification is globally disabled.",
    )
    healthcheck: HealthcheckConfig | None = Field(
        default=None,
        description="Optional per-step healthcheck run after this step's setup "
        "completes and before the agent runs. Mirrors the semantics of the "
        "top-level environment healthcheck; start_period_sec applies as a grace "
        "period after setup. Supplements rather than replaces the top-level "
        "healthcheck.",
    )
    artifacts: list[str | ArtifactConfig] = Field(
        default_factory=list,
        description="Artifacts to collect after this step's verification into "
        "steps/{name}/artifacts/. Appended to task-level and trial-level "
        "artifacts during this step's collection pass.",
    )


class MultiStepRewardStrategy(str, Enum):
    """Strategy for deriving a trial-level reward from per-step verifier results."""

    MEAN = "mean"
    FINAL = "final"


class TaskConfig(BaseModel):
    schema_version: str = "1.2"
    task: PackageInfo | None = Field(
        default=None,
        description="Package information for the task, parsed from the [task] section of task.toml.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None
    multi_step_reward_strategy: MultiStepRewardStrategy | None = Field(
        default=None,
        description=(
            "How to derive the trial-level reward from per-step verifier "
            "results in a multi-step task. 'mean' computes per-key means "
            "across steps (missing keys treated as 0; steps without a "
            "verifier_result excluded). 'final' uses the last step's "
            "verifier_result verbatim. Only applies to multi-step tasks; "
            "leave unset for single-step tasks. Defaults to 'mean' when "
            "unset on a multi-step task."
        ),
    )
    steps: list[StepConfig] | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        return data

    @model_validator(mode="after")
    def resolve_network_modes(self) -> "TaskConfig":
        """Resolve Harbor-style ``network_mode`` declarations onto the legacy
        ``allow_internet`` booleans that pier's environments enforce.

        Precedence per phase (mirrors Harbor): explicit [agent]/[verifier]
        override > [environment] scope > legacy ``allow_internet``. 'public'
        maps to allow_internet=True; 'no-network' to allow_internet=False.

        Before this resolution existed, pier silently IGNORED network_mode and
        fell back to allow_internet's default of True.
        """
        task_mode = self.environment.network_mode

        agent_mode = self.agent.network_mode or task_mode
        if agent_mode is not None:
            self.environment.network_mode = agent_mode
            self.environment.allow_internet = agent_mode == NetworkMode.PUBLIC

        def _resolve_verifier(verifier: VerifierConfig) -> None:
            mode = verifier.network_mode or task_mode
            if mode is None:
                return
            if (
                verifier.environment is None
                and verifier.network_mode is not None
                and verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
            ):
                # An explicit [verifier] override with environment_mode =
                # 'separate' but no spelled-out [verifier.environment] would be
                # lost when the runtime materializes the fresh copy — do it now.
                verifier.environment = self.environment.model_copy(deep=True)
            if verifier.environment is not None:
                verifier.environment.network_mode = mode
                verifier.environment.allow_internet = mode == NetworkMode.PUBLIC
            elif (
                verifier.network_mode is not None
                and agent_mode is not None
                and verifier.network_mode != agent_mode
            ):
                # Shared-environment verifier: it runs in the agent's container,
                # so a conflicting explicit override cannot be enforced.
                raise ValueError(
                    "[verifier].network_mode conflicts with the agent "
                    "environment's resolved network policy but the verifier "
                    "shares that environment; use environment_mode='separate' "
                    "to give the verifier its own network policy."
                )

        _resolve_verifier(self.verifier)
        for step in self.steps or []:
            _resolve_verifier(step.verifier)
        return self

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> "TaskConfig":
        toml_dict = tomllib.loads(toml_data)
        return cls.model_validate(toml_dict)

    def model_dump_toml(self) -> str:
        data = self._without_none(self.model_dump(mode="json"))

        parts: list[str] = []
        emitted: set[str] = set()
        root_fields = [
            "schema_version",
            "source",
            "multi_step_reward_strategy",
            "artifacts",
        ]
        known_sections = (
            "task",
            "steps",
            "metadata",
            "verifier",
            "agent",
            "environment",
            "solution",
        )
        root_data: dict[str, Any] = {}
        for field in root_fields:
            if field in data and not isinstance(data[field], dict):
                root_data[field] = data[field]
        for field, value in data.items():
            if field in root_fields or field in known_sections:
                continue
            if not self._is_toml_table_like(value):
                root_data[field] = value
        if root_data:
            parts.append(toml.dumps(root_data))
            emitted.update(root_data)

        if "task" in data:
            parts.append(toml.dumps({"task": data["task"]}))
            emitted.add("task")

        if "steps" in data:
            parts.append(toml.dumps({"steps": data["steps"]}))
            emitted.add("steps")

        for section in ("metadata", "verifier", "agent", "environment", "solution"):
            if section in data:
                parts.append(toml.dumps({section: data[section]}))
                emitted.add(section)

        for field, value in data.items():
            if field not in emitted:
                parts.append(toml.dumps({field: value}))
                emitted.add(field)

        return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"

    @staticmethod
    def _is_toml_table_like(value: Any) -> bool:
        return isinstance(value, dict) or (
            isinstance(value, list) and any(isinstance(item, dict) for item in value)
        )

    @classmethod
    def _without_none(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._without_none(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [cls._without_none(item) for item in value]
        return value

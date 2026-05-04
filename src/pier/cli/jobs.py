import json
import shutil
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from pier.cli.host_env import confirm_host_env_access
from pier.cli.utils import parse_env_vars, parse_kwargs, run_async
from pier.models.agent.name import AgentName
from pier.models.environment_type import EnvironmentType
from pier.models.job.config import (
    DatasetConfig,
    JobConfig,
)
from pier.models.job.result import JobStats
from pier.models.task.paths import TaskPaths
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
)
from pier.models.trial.paths import TrialPaths
from pier.models.trial.result import TrialResult

jobs_app = Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)
console = Console()


def _format_duration(started_at: datetime | None, finished_at: datetime | None) -> str:
    if started_at is None or finished_at is None:
        return "unknown"

    total_seconds = max(0, int((finished_at - started_at).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_group_title(evals_key: str, job_result) -> str:
    parts = evals_key.split("__")
    if len(parts) == 3:
        agent_name, model_name, dataset_name = parts
    else:
        agent_name, dataset_name = parts
        model_name = None

    for trial_result in job_result.trial_results:
        trial_evals_key = JobStats.format_agent_evals_key(
            trial_result.agent_info.name,
            (
                trial_result.agent_info.model_info.name
                if trial_result.agent_info.model_info
                else None
            ),
            trial_result.source or "adhoc",
        )
        if trial_evals_key != evals_key:
            continue

        agent_name = trial_result.agent_info.name
        model_name = (
            trial_result.agent_info.model_info.name
            if trial_result.agent_info.model_info
            else model_name
        )
        dataset_name = trial_result.source or dataset_name
        break

    title_parts = [dataset_name, agent_name]
    if model_name:
        title_parts.append(model_name)
    return " • ".join(title_parts)


def print_job_results_tables(job_result) -> None:
    for evals_key, dataset_stats in job_result.stats.evals.items():
        console.print(f"[bold]{_format_group_title(evals_key, job_result)}[/bold]")

        summary_table = Table(show_header=True)
        summary_table.add_column("Trials", justify="right")
        summary_table.add_column("Exceptions", justify="right")

        summary_row = [str(dataset_stats.n_trials), str(dataset_stats.n_errors)]
        if dataset_stats.metrics:
            for i, metric in enumerate(dataset_stats.metrics):
                for key, value in metric.items():
                    metric_label = (
                        f"Metric {i + 1}: {key}"
                        if len(dataset_stats.metrics) > 1
                        else key
                    )
                    summary_table.add_column(metric_label.title(), justify="right")
                    if isinstance(value, float):
                        summary_row.append(f"{value:.3f}")
                    else:
                        summary_row.append(str(value))

        if dataset_stats.pass_at_k:
            for k, value in sorted(dataset_stats.pass_at_k.items()):
                summary_table.add_column(f"Pass@{k}", justify="right")
                summary_row.append(f"{value:.3f}")

        summary_table.add_row(*summary_row)
        console.print(summary_table)

        if dataset_stats.reward_stats:
            reward_table = Table(show_header=True)
            reward_table.add_column("Reward")
            reward_table.add_column("Count", justify="right")
            for reward_key, reward_values in dataset_stats.reward_stats.items():
                for reward_value, trial_names in reward_values.items():
                    count = len(trial_names)
                    reward_table.add_row(str(reward_value), str(count))
            console.print()
            console.print(reward_table)

        if dataset_stats.exception_stats:
            exception_table = Table(show_header=True)
            exception_table.add_column("Exception")
            exception_table.add_column("Count", justify="right")
            for (
                exception_type,
                trial_names,
            ) in dataset_stats.exception_stats.items():
                count = len(trial_names)
                exception_table.add_row(exception_type, str(count))
            console.print()
            console.print(exception_table)

        console.print()


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def start(
    config_path: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="A job configuration path in yaml or json format. "
            "Should implement the schema of pier.models.job.config:JobConfig. "
            "Allows for more granular control over the job configuration.",
            rich_help_panel="Config",
            show_default=False,
        ),
    ] = None,
    job_name: Annotated[
        str | None,
        Option(
            "--job-name",
            help="Name of the job (default: timestamp)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    jobs_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--jobs-dir",
            help=f"Directory to store job results (default: {
                JobConfig.model_fields['jobs_dir'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_attempts: Annotated[
        int | None,
        Option(
            "-k",
            "--n-attempts",
            help=f"Number of attempts per trial (default: {
                JobConfig.model_fields['n_attempts'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        Option(
            "--timeout-multiplier",
            help=f"Multiplier for task timeouts (default: {
                JobConfig.model_fields['timeout_multiplier'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-timeout-multiplier",
            help="Multiplier for agent execution timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--verifier-timeout-multiplier",
            help="Multiplier for verifier timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_setup_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-setup-timeout-multiplier",
            help="Multiplier for agent setup timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    environment_build_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--environment-build-timeout-multiplier",
            help="Multiplier for environment build timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[
        bool,
        Option(
            "-q",
            "--quiet",
            "--silent",
            help="Suppress individual trial progress displays",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    debug: Annotated[
        bool,
        Option(
            "--debug",
            help="Enable debug logging",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    n_concurrent_trials: Annotated[
        int | None,
        Option(
            "-n",
            "--n-concurrent",
            help=f"Number of concurrent trials to run (default: {
                JobConfig.model_fields['n_concurrent_trials'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        Option(
            "-r",
            "--max-retries",
            help="Maximum number of retry attempts (default: 0)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_include_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-include",
            help="Exception types to retry on. If not specified, all exceptions except "
            "those in --retry-exclude are retried (can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_exclude_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-exclude",
            help="Exception types to NOT retry on (can be used multiple times)",
            rich_help_panel="Job Settings",
        ),
    ] = [
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    ],
    agent_name: Annotated[
        AgentName | None,
        Option(
            "-a",
            "--agent",
            help=f"Agent name (default: {AgentName.ORACLE.value})",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_import_path: Annotated[
        str | None,
        Option(
            "--agent-import-path",
            help="Import path for custom agent",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    model_names: Annotated[
        list[str] | None,
        Option(
            "-m",
            "--model",
            help="Model name for the agent (can be used multiple times)",
            rich_help_panel="Agent",
            show_default=True,
        ),
    ] = None,
    agent_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ak",
            "--agent-kwarg",
            help="Additional agent kwarg in the format 'key=value'. You can view "
            "available kwargs by looking at the agent's `__init__` method. "
            "Can be set multiple times to set multiple kwargs. Common kwargs "
            "include: version, prompt_template, etc.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_env: Annotated[
        list[str] | None,
        Option(
            "--ae",
            "--agent-env",
            help="Environment variable to pass to the agent in KEY=VALUE format. "
            "Can be used multiple times. Example: --ae AWS_REGION=us-east-1",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    environment_type: Annotated[
        EnvironmentType | None,
        Option(
            "-e",
            "--env",
            help=f"Environment type (default: {EnvironmentType.DOCKER.value})",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_import_path: Annotated[
        str | None,
        Option(
            "--environment-import-path",
            help="Import path for custom environment (module.path:ClassName).",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_force_build: Annotated[
        bool | None,
        Option(
            "--force-build/--no-force-build",
            help=f"Whether to force rebuild the environment (default: {
                '--force-build'
                if EnvironmentConfig.model_fields['force_build'].default
                else '--no-force-build'
            })",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_delete: Annotated[
        bool | None,
        Option(
            "--delete/--no-delete",
            help=f"Whether to delete the environment after completion (default: {
                '--delete'
                if EnvironmentConfig.model_fields['delete'].default
                else '--no-delete'
            })",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_cpus: Annotated[
        int | None,
        Option(
            "--override-cpus",
            help="Override the number of CPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_memory_mb: Annotated[
        int | None,
        Option(
            "--override-memory-mb",
            help="Override the memory (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_storage_mb: Annotated[
        int | None,
        Option(
            "--override-storage-mb",
            help="Override the storage (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_gpus: Annotated[
        int | None,
        Option(
            "--override-gpus",
            help="Override the number of GPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    mounts_json: Annotated[
        str | None,
        Option(
            "--mounts-json",
            help="JSON array of volume mounts for the environment container "
            "(Docker Compose service volume format)",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ek",
            "--environment-kwarg",
            help="Environment kwarg in key=value format (can be used multiple times)",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Auto-confirm when tasks declare environment variables that read from the host.",
            rich_help_panel="Job Settings",
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        Option(
            "--env-file",
            help="Path to a .env file to load into environment.",
            rich_help_panel="Job Settings",
        ),
    ] = None,
    path: Annotated[
        Path | None,
        Option(
            "-p",
            "--path",
            help="Path to a local task or dataset directory",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_task_names: Annotated[
        list[str] | None,
        Option(
            "-i",
            "--include-task-name",
            help="Task name to include from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_exclude_task_names: Annotated[
        list[str] | None,
        Option(
            "-x",
            "--exclude-task-name",
            help="Task name to exclude from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option(
            "-l",
            "--n-tasks",
            help="Maximum number of tasks to run (applied after other filters)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    sample_seed: Annotated[
        int | None,
        Option(
            "--sample-seed",
            help=(
                "Seed for deterministic random task sampling/order. "
                "Applied after include/exclude filters and before --n-tasks."
            ),
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    artifact_paths: Annotated[
        list[str] | None,
        Option(
            "--artifact",
            help="Environment path to download as an artifact after the trial "
            "(can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_env: Annotated[
        list[str] | None,
        Option(
            "--ve",
            "--verifier-env",
            help="Environment variable to pass to the verifier in KEY=VALUE format. "
            "Can be used multiple times. Example: --ve OPENAI_BASE_URL=http://localhost:8000/v1",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    disable_verification: Annotated[
        bool,
        Option(
            "--disable-verification/--enable-verification",
            help="Disable task verification (skip running tests)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
):
    from pier.job import Job

    if env_file is not None:
        if not env_file.exists():
            console.print(f"[red]❌ Env file not found: {env_file}[/red]")
            raise SystemExit(1)
        load_dotenv(env_file, override=True)

    base_config = None
    if config_path is not None:
        if config_path.suffix == ".yaml":
            base_config = JobConfig.model_validate(
                yaml.safe_load(config_path.read_text())
            )
        elif config_path.suffix == ".json":
            base_config = JobConfig.model_validate_json(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    config = base_config if base_config is not None else JobConfig()

    if job_name is not None:
        config.job_name = job_name
    if jobs_dir is not None:
        config.jobs_dir = jobs_dir
    if n_attempts is not None:
        config.n_attempts = n_attempts
    if timeout_multiplier is not None:
        config.timeout_multiplier = timeout_multiplier
    if agent_timeout_multiplier is not None:
        config.agent_timeout_multiplier = agent_timeout_multiplier
    if verifier_timeout_multiplier is not None:
        config.verifier_timeout_multiplier = verifier_timeout_multiplier
    if agent_setup_timeout_multiplier is not None:
        config.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
    if environment_build_timeout_multiplier is not None:
        config.environment_build_timeout_multiplier = (
            environment_build_timeout_multiplier
        )
    if debug:
        config.debug = debug

    if n_concurrent_trials is not None:
        config.n_concurrent_trials = n_concurrent_trials
    if quiet:
        config.quiet = quiet
    if max_retries is not None:
        config.retry.max_retries = max_retries
    if retry_include_exceptions is not None:
        config.retry.include_exceptions = set(retry_include_exceptions)
    if retry_exclude_exceptions is not None:
        config.retry.exclude_exceptions = set(retry_exclude_exceptions)

    if agent_name is not None or agent_import_path is not None:
        config.agents = []
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)

        if model_names is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                )
            ]
    else:
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        if parsed_kwargs or parsed_env:
            for agent in config.agents:
                if parsed_kwargs:
                    agent.kwargs.update(parsed_kwargs)
                if parsed_env:
                    agent.env.update(parsed_env)

    if environment_type is not None:
        config.environment.type = environment_type
    if environment_import_path is not None:
        config.environment.import_path = environment_import_path
        config.environment.type = None  # Clear type so import_path takes precedence
    if environment_force_build is not None:
        config.environment.force_build = environment_force_build
    if environment_delete is not None:
        config.environment.delete = environment_delete
    if override_cpus is not None:
        config.environment.override_cpus = override_cpus
    if override_memory_mb is not None:
        config.environment.override_memory_mb = override_memory_mb
    if override_storage_mb is not None:
        config.environment.override_storage_mb = override_storage_mb
    if override_gpus is not None:
        config.environment.override_gpus = override_gpus
    if mounts_json is not None:
        config.environment.mounts_json = json.loads(mounts_json)
    if environment_kwargs is not None:
        config.environment.kwargs.update(parse_kwargs(environment_kwargs))

    if verifier_env is not None:
        config.verifier.env.update(parse_env_vars(verifier_env))
    if disable_verification:
        config.verifier.disable = disable_verification

    if artifact_paths is not None:
        config.artifacts = list(artifact_paths)

    if path is not None:
        task_paths = TaskPaths(path)
        is_task = task_paths.is_valid(disable_verification=disable_verification)

        if is_task:
            config.tasks = [TaskConfig(path=path)]
            config.datasets = []
        else:
            config.tasks = []
            config.datasets = [
                DatasetConfig(
                    path=path,
                    task_names=dataset_task_names,
                    exclude_task_names=dataset_exclude_task_names,
                    n_tasks=n_tasks,
                    sample_seed=sample_seed,
                )
            ]

    elif (
        dataset_task_names is not None
        or dataset_exclude_task_names is not None
        or n_tasks is not None
        or sample_seed is not None
    ):
        raise ValueError("Cannot specify local dataset filters without --path.")

    async def _run_job():
        job = await Job.create(config)
        confirm_host_env_access(
            task_configs=job._task_configs,
            agents=job.config.agents,
            environment=job.config.environment,
            verifier=job.config.verifier,
            console=console,
            explicit_env_file_keys=explicit_env_file_keys,
            skip_confirm=yes,
        )
        job_result = await job.run()

        # Print the run summary before returning so users see results immediately.
        console.print()
        print_job_results_tables(job_result)
        console.print("[bold]Job Info[/bold]")
        console.print(
            f"Total runtime: {_format_duration(job_result.started_at, job_result.finished_at)}"
        )
        console.print(f"Results written to {job._job_result_path}")
        console.print(f"Inspect results by running `pier view {job.job_dir.parent}`")

        console.print()

        return job, job_result

    from pier.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    explicit_env_file_keys: set[str] = set()
    if env_file is not None:
        explicit_env_file_keys = {
            key for key in dotenv_values(env_file).keys() if key is not None
        }

    signal.signal(signal.SIGTERM, _handle_sigterm)

    job, job_result = run_async(_run_job())


@jobs_app.command()
def resume(
    job_path: Annotated[
        Path,
        Option(
            "-p",
            "--job-path",
            help="Path to the job directory containing the config.json file",
        ),
    ],
    filter_error_types: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter-error-type",
            help="Remove trials with these error types before resuming (can be used "
            "multiple times)",
            show_default=False,
        ),
    ] = ["CancelledError"],
):
    """Resume an existing job from its job directory."""
    from pier.job import Job

    job_dir = Path(job_path)
    config_path = job_dir / "config.json"

    if not job_dir.exists():
        raise ValueError(f"Job directory does not exist: {job_dir}")

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    if filter_error_types:
        filter_error_types_set = set(filter_error_types)
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            trial_paths = TrialPaths(trial_dir)

            if not trial_paths.result_path.exists():
                continue

            trial_result = TrialResult.model_validate_json(
                trial_paths.result_path.read_text()
            )
            if (
                trial_result.exception_info is not None
                and trial_result.exception_info.exception_type in filter_error_types_set
            ):
                console.print(
                    f"Removing trial directory with {
                        trial_result.exception_info.exception_type
                    }: {trial_dir.name}"
                )
                shutil.rmtree(trial_dir)

    config = JobConfig.model_validate_json(config_path.read_text())

    from pier.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    async def _run_job():
        job = await Job.create(config)
        job_result = await job.run()
        return job_result

    job_result = run_async(_run_job())

    # Print results tables
    print_job_results_tables(job_result)


jobs_app.command()(start)

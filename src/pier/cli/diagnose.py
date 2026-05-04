from __future__ import annotations

import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typer import Option, Typer

from pier.cli.host_env import confirm_host_env_access
from pier.cli.utils import parse_env_vars, parse_kwargs, run_async
from pier.diagnose.models import DiagnosticConfig
from pier.diagnose.runner import DiagnosticRunner
from pier.models.agent.name import AgentName
from pier.models.environment_type import EnvironmentType
from pier.models.job.config import JobConfig
from pier.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

diagnose_app = Typer(
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


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


@diagnose_app.command(name="run")
def run_command(
    job_dir: Annotated[Path, typer.Argument(help="Path to an existing Pier job dir")],
    prompt: Annotated[
        Path,
        Option(
            "-p",
            "--prompt",
            help="External prompt file for the diagnostic agent.",
        ),
    ],
    run_name: Annotated[
        str | None,
        Option("--name", help="Diagnostic run name. Defaults to a timestamp."),
    ] = None,
    config_path: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="A job configuration path in yaml or json format. "
            "Should implement the schema of pier.models.job.config:JobConfig. "
            "Allows for more granular control over the diagnostic configuration.",
            rich_help_panel="Config",
            show_default=False,
        ),
    ] = None,
    agent_index: Annotated[
        int,
        Option(
            "--agent-index",
            help="Agent index to use from --config when it contains multiple agents.",
        ),
    ] = 0,
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
    model_name: Annotated[
        str | None,
        Option(
            "-m",
            "--model",
            help="Model name for the agent",
            rich_help_panel="Agent",
            show_default=False,
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
            help="Override the number of CPUs for the environment.",
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
            help="Override the number of GPUs for the environment.",
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
    n_concurrent: Annotated[
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
    trial_names: Annotated[
        list[str] | None,
        Option(
            "-t",
            "--trial-name",
            help="Specific source trial name to diagnose. Can be used multiple times.",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Option("--limit", help="Maximum number of selected trials to diagnose."),
    ] = None,
    passing: Annotated[
        bool,
        Option("--passing", help="Only diagnose passing source trials."),
    ] = False,
    failing: Annotated[
        bool,
        Option("--failing", help="Only diagnose failing source trials."),
    ] = False,
    overwrite: Annotated[
        bool,
        Option(
            "--overwrite",
            help="Rerun diagnostics even if diagnostic metadata already exists.",
        ),
    ] = False,
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
):
    """Run sandboxed diagnostics over completed trials in a Pier job."""
    if passing and failing:
        raise ValueError("Cannot use both --passing and --failing.")
    if not prompt.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt}")

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

    if base_config is None and agent_name is None and agent_import_path is None:
        raise ValueError("Pass --agent/--agent-import-path or --config.")

    config = base_config if base_config is not None else JobConfig()

    if timeout_multiplier is not None:
        config.timeout_multiplier = timeout_multiplier
    if agent_timeout_multiplier is not None:
        config.agent_timeout_multiplier = agent_timeout_multiplier
    if agent_setup_timeout_multiplier is not None:
        config.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
    if environment_build_timeout_multiplier is not None:
        config.environment_build_timeout_multiplier = (
            environment_build_timeout_multiplier
        )

    if n_concurrent is not None:
        config.n_concurrent_trials = n_concurrent

    if agent_name is not None or agent_import_path is not None:
        config.agents = [
            AgentConfig(
                name=agent_name,
                import_path=agent_import_path,
                model_name=model_name,
                kwargs=parse_kwargs(agent_kwargs),
                env=parse_env_vars(agent_env),
            )
        ]
    else:
        if agent_index < 0 or agent_index >= len(config.agents):
            raise ValueError(
                f"--agent-index {agent_index} is out of range for "
                f"{len(config.agents)} configured agent(s)."
            )
        agent = config.agents[agent_index].model_copy(deep=True)
        if model_name is not None:
            agent.model_name = model_name
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        if parsed_kwargs:
            agent.kwargs.update(parsed_kwargs)
        if parsed_env:
            agent.env.update(parsed_env)
        config.agents = [agent]

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

    cfg = DiagnosticConfig(
        run_name=run_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S"),
        source_job_dir=job_dir,
        prompt_path=prompt,
        agent=config.agents[0],
        environment=config.environment,
        timeout_multiplier=config.timeout_multiplier,
        agent_timeout_multiplier=config.agent_timeout_multiplier,
        agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
        n_concurrent=config.n_concurrent_trials,
        overwrite=overwrite,
        trial_names=trial_names,
        limit=limit,
        filter_passing=True if passing else (False if failing else None),
    )

    async def _run_job():
        runner = DiagnosticRunner(cfg)
        selected_items = runner.collect_items()
        confirm_host_env_access(
            task_configs=[TaskConfig(path=item.task_dir) for item in selected_items],
            agents=[cfg.agent],
            environment=cfg.environment,
            verifier=None,
            console=console,
            explicit_env_file_keys=explicit_env_file_keys,
            skip_confirm=yes,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task("Running diagnostics...", total=None)

            def _set_total(total: int) -> None:
                progress.update(task_id, total=total)

            def _advance() -> None:
                progress.advance(task_id)

            result = await runner.run_job(
                on_total=_set_total, on_item_complete=_advance
            )

        console.print()
        console.print("[bold]Diagnostic Run[/bold]")
        console.print(
            f"Total runtime: {_format_duration(result.started_at, result.finished_at)}"
        )
        console.print(f"Results written to {result.diagnostic_dir / 'result.json'}")
        console.print(f"Items: {len(result.item_results)}")
        console.print(f"Failures: {result.n_failed}")
        if result.failed_items:
            for failed in result.failed_items:
                console.print(f"[yellow]{failed}[/yellow]")

        console.print()

        return result

    from pier.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=cfg.environment.type,
        import_path=cfg.environment.import_path,
    )

    explicit_env_file_keys: set[str] = set()
    if env_file is not None:
        explicit_env_file_keys = {
            key for key in dotenv_values(env_file).keys() if key is not None
        }

    signal.signal(signal.SIGTERM, _handle_sigterm)

    run_async(_run_job())

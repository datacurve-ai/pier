from __future__ import annotations

from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.table import Table

from pier.models.agent.name import AgentName
from pier.models.task.task import Task
from pier.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    VerifierConfig,
)
from pier.utils.env import get_required_host_vars


def confirm_host_env_access(
    *,
    task_configs: Iterable[TaskConfig],
    agents: Sequence[AgentConfig],
    environment: EnvironmentConfig,
    verifier: VerifierConfig | None,
    console: Console,
    explicit_env_file_keys: set[str] | None = None,
    skip_confirm: bool = False,
) -> None:
    import os

    is_oracle = any(a.name == AgentName.ORACLE.value for a in agents)
    explicit_env_file_keys = explicit_env_file_keys or set()
    explicit_environment_keys = set(environment.env)
    explicit_verifier_keys = set(verifier.env) if verifier is not None else set()
    sections: dict[str, list[tuple[str, str | None]]] = {}

    for task_config in task_configs:
        try:
            local_path = task_config.get_local_path()
        except ValueError:
            continue
        if not local_path.exists():
            continue
        try:
            task = Task(local_path)
        except Exception:
            continue

        env_sections = [("environment", task.config.environment.env)]
        if verifier is not None:
            env_sections.append(("verifier", task.config.verifier.env))
        if is_oracle:
            env_sections.append(("solution", task.config.solution.env))

        for section_name, env_dict in env_sections:
            filtered_env_dict = env_dict
            if section_name == "environment" and explicit_environment_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_environment_keys
                }
            elif section_name == "verifier" and explicit_verifier_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_verifier_keys
                }

            required = [
                item
                for item in get_required_host_vars(filtered_env_dict)
                if item[0] not in explicit_env_file_keys
            ]
            if not required:
                continue

            key = f"[{section_name}.env]"
            existing = list(sections.get(key, []))
            for item in required:
                if item not in existing:
                    existing.append(item)
            if existing:
                sections[key] = existing

    if not sections:
        return

    missing = []
    for section, vars_list in sections.items():
        for var_name, default in vars_list:
            if default is None and var_name not in os.environ:
                missing.append((section, var_name))

    if missing:
        table = Table(
            title="Missing Environment Variables",
            title_style="bold red",
            show_header=True,
            header_style="bold",
            padding=(0, 2),
            show_edge=False,
            show_lines=False,
        )
        table.add_column("Variable", style="cyan")
        table.add_column("Phase", style="dim")

        for section, var_name in missing:
            escaped = section.replace("[", "\\[")
            table.add_row(var_name, escaped)

        console.print()
        console.print(table)
        console.print(
            "\n[yellow]Export them in your shell or pass --env-file.[/yellow]"
        )
        raise SystemExit(1)

    if skip_confirm:
        return

    table = Table(
        title="Environment Variables",
        title_style="bold",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        show_edge=False,
        show_lines=False,
    )
    table.add_column("Variable", style="cyan")
    table.add_column("Phase", style="dim")

    for section, vars_list in sections.items():
        escaped = section.replace("[", "\\[")
        for var_name, _default in vars_list:
            table.add_row(var_name, escaped)

    console.print()
    console.print(table)
    console.print()

    response = console.input(
        "Tasks in this run will load these from your environment. "
        "[yellow]Proceed? (Y/n):[/yellow] "
    )
    if response.strip().lower() in ("n", "no"):
        raise SystemExit(0)

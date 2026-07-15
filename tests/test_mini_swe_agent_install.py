import shlex
from pathlib import Path

from pier.agents.installed.mini_swe_agent import MiniSweAgent


def _tool_install_command(agent: MiniSweAgent) -> str:
    run_script = agent.install_spec().steps[-1].run
    commands = [
        line for line in run_script.splitlines() if line.startswith("uv tool install ")
    ]

    assert len(commands) == 1
    return commands[0]


def test_install_spec_provisions_litellm_proxy_dependencies(tmp_path: Path):
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openai/gpt-5.5")

    command = _tool_install_command(agent)

    assert command == "uv tool install mini-swe-agent --with 'litellm[proxy]'"


def test_install_spec_preserves_pinned_agent_version(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        version="2.2.8",
    )

    command = _tool_install_command(agent)

    assert command == "uv tool install mini-swe-agent==2.2.8 --with 'litellm[proxy]'"


def test_install_spec_without_extra_packages_is_valid_shell_command(tmp_path: Path):
    agent = MiniSweAgent(
        logs_dir=tmp_path,
        model_name="openai/gpt-5.5",
        extra_python_packages=[],
    )

    command = _tool_install_command(agent)

    assert shlex.split(command) == [
        "uv",
        "tool",
        "install",
        "mini-swe-agent",
        "--with",
        "litellm[proxy]",
    ]

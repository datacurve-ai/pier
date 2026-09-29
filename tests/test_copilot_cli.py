import asyncio
import functools
import json
import re
import shlex
from pathlib import Path
from uuid import UUID

import pytest

from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities
from pier.agents.factory import AgentFactory
from pier.agents.installed.base import CliFlag
from pier.agents.installed.copilot_cli import CopilotCli
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig, MCPServerConfig, TaskOS
from pier.models.trial.paths import TrialPaths


_CLI_ENV_VARS = (
    "COPILOT_CLI_EFFORT",
    "COPILOT_CLI_MODE",
    "COPILOT_CLI_CONTEXT_TIER",
    "COPILOT_CLI_AGENT",
)
_TOKEN_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class _RecordingEnvironment(BaseEnvironment):
    def __init__(
        self,
        tmp_path: Path,
        exec_result: ExecResult | None = None,
    ) -> None:
        self.exec_result = exec_result or ExecResult(
            return_code=0,
            stdout="",
            stderr="",
        )
        self.exec_calls: list[dict[str, object]] = []
        self.agent_process_env_inputs: list[dict[str, str] | None] = []
        trial_paths = TrialPaths(tmp_path / "trial")
        trial_paths.mkdir()
        super().__init__(
            environment_dir=tmp_path,
            environment_name="test",
            session_id="session",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(os=TaskOS.LINUX),
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool):
        pass

    async def upload_file(self, source_path, target_path):
        pass

    async def upload_dir(self, source_dir, target_dir):
        pass

    async def download_file(self, source_path, target_path):
        pass

    async def download_dir(self, source_dir, target_dir):
        pass

    def agent_process_env(
        self,
        env: dict[str, str] | None,
    ) -> dict[str, str] | None:
        self.agent_process_env_inputs.append(dict(env) if env is not None else None)
        return env

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": user,
            }
        )
        return self.exec_result


def _clear_env(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _split_run_command(command: str) -> tuple[str, str]:
    assert command.startswith("set -o pipefail; ")
    command = command.removeprefix("set -o pipefail; ")
    setup, quoted_script = command.rsplit(" && bash -lc ", 1)
    run_script = shlex.split(f"bash -lc {quoted_script}")[-1]
    return setup, run_script


async def _run_agent(
    agent: CopilotCli,
    tmp_path: Path,
    instruction: str = "fix it",
) -> tuple[_RecordingEnvironment, str, str]:
    environment = _RecordingEnvironment(tmp_path)

    await agent.run(instruction, environment, AgentContext())

    assert len(environment.exec_calls) == 1
    call = environment.exec_calls[0]
    setup, run_script = _split_run_command(str(call["command"]))
    return environment, setup, run_script


def test_copilot_cli_is_registered_with_current_flags(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.COPILOT_CLI,
        logs_dir=tmp_path,
        model_name="gpt-5.4",
        reasoning_effort="minimal",
        context_tier="long_context",
    )

    assert isinstance(agent, CopilotCli)
    assert agent.name() == "copilot-cli"
    assert "--effort minimal" in agent.build_cli_flags()
    assert "--context long_context" in agent.build_cli_flags()
    assert "--allow-all-tools" in agent.build_cli_flags()
    assert "--no-ask-user" in agent.build_cli_flags()
    assert "--context-tier" not in agent.build_cli_flags()


def test_copilot_cli_factory_passes_installed_agent_kwargs(tmp_path: Path):
    agent = AgentFactory.create_agent_from_name(
        AgentName.COPILOT_CLI,
        logs_dir=tmp_path,
        model_name="github/gpt-5.4",
        command_model_name="gpt-5.5",
        extra_args=["--flag", "value with spaces"],
        version="1.0.76",
        extra_env={"GH_TOKEN": "gh", "CUSTOM_ENV": "value"},
    )

    assert isinstance(agent, CopilotCli)
    assert agent.model_name == "github/gpt-5.4"
    assert agent._command_model_name == "gpt-5.5"
    assert shlex.split(agent._extra_args_string()) == ["--flag", "value with spaces"]
    assert agent.version() == "1.0.76"
    assert agent._extra_env["GH_TOKEN"] == "gh"
    assert agent._extra_env["CUSTOM_ENV"] == "value"


@run_async
async def test_copilot_cli_run_builds_expected_command_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    prompt_template = tmp_path / "prompt.j2"
    prompt_template.write_text(
        "Template start\n{{ instruction }}\nTemplate end",
        encoding="utf-8",
    )
    instruction = 'Fix "quotes"\nDo not expand $HOME or `cmd`; it\'s literal'
    rendered_instruction = f"Template start\n{instruction}\nTemplate end"
    agent = CopilotCli(
        logs_dir=tmp_path,
        model_name="github/gpt-5.4",
        prompt_template_path=prompt_template,
        extra_env={
            "COPILOT_GITHUB_TOKEN": "token",
            "COPILOT_HOME": "/custom",
            "CUSTOM_ENV": "value",
        },
    )

    environment, setup, run_script = await _run_agent(
        agent,
        tmp_path,
        instruction=instruction,
    )

    expected_setup = (
        "mkdir -p /logs/agent /logs/agent/command-0 "
        '/logs/agent/copilot-home /logs/agent/copilot-logs && '
        'export PATH="$HOME/.local/bin:$PATH"'
    )
    assert setup == expected_setup
    session_match = re.search(r"--session-id ([0-9a-f-]{36})", run_script)
    assert session_match is not None
    UUID(session_match.group(1))
    normalized_script = run_script.replace(session_match.group(1), "<session-id>")
    expected_script = (
        'export PATH="$HOME/.local/bin:$PATH"; '
        "set -o pipefail; "
        f"copilot -p {shlex.quote(rendered_instruction)} "
        "--output-format json --no-color "
        "--allow-all-tools --no-ask-user --no-auto-update "
        "--model gpt-5.4 --session-id <session-id> "
        "--log-dir /logs/agent/copilot-logs "
        "2>&1 | tee /logs/agent/copilot-cli.jsonl | "
        "tee /logs/agent/copilot-cli.txt | "
        "tee /logs/agent/command-0/stdout.txt; "
        "exit ${PIPESTATUS[0]}"
    )
    assert normalized_script == expected_script
    assert environment.agent_process_env_inputs == [
        {
            "COPILOT_GITHUB_TOKEN": "token",
            "CUSTOM_ENV": "value",
            "COPILOT_HOME": "/logs/agent/copilot-home",
        }
    ]
    assert environment.exec_calls[0]["env"] == {
        "COPILOT_GITHUB_TOKEN": "token",
        "CUSTOM_ENV": "value",
        "COPILOT_HOME": "/logs/agent/copilot-home",
    }
    assert environment.exec_calls[0]["user"] is None
    assert environment.exec_calls[0]["cwd"] is None
    assert environment.exec_calls[0]["timeout_sec"] is None


@run_async
async def test_copilot_cli_run_command_model_name_overrides_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        model_name="github/gpt-5.4",
        command_model_name="gpt-5.5",
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    assert "--model gpt-5.5" in run_script
    assert "--model gpt-5.4" not in run_script


@run_async
async def test_copilot_cli_run_strips_single_provider_prefix_from_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        model_name="github/gpt-5.4",
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    assert "--model gpt-5.4" in run_script
    assert "--model github/gpt-5.4" not in run_script


@run_async
async def test_copilot_cli_run_omits_model_flag_without_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    assert " --model " not in run_script


def test_copilot_cli_cli_flags_map_all_supported_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        reasoning_effort="HIGH",
        mode="PLAN",
        context_tier="LONG_CONTEXT",
        agent="copilot",
        allow_all_tools=True,
        no_ask_user=True,
        no_auto_update=True,
    )

    assert agent.build_cli_flags() == (
        "--effort high --mode plan --context long_context --agent copilot "
        "--allow-all-tools --no-ask-user --no-auto-update"
    )


def test_copilot_cli_cli_flags_omit_explicitly_disabled_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        allow_all_tools=False,
        no_ask_user=False,
        no_auto_update=False,
    )

    assert agent.build_cli_flags() == ""


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("reasoning_effort", "ultra"),
        ("mode", "yolo"),
        ("context_tier", "huge"),
    ],
)
def test_copilot_cli_cli_flag_invalid_enums_raise_helpful_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwarg: str,
    value: str,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)

    with pytest.raises(ValueError, match=rf"Invalid value for '{kwarg}'.*{value}"):
        CopilotCli(logs_dir=tmp_path, **{kwarg: value})


def test_copilot_cli_cli_flags_resolve_env_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COPILOT_CLI_EFFORT", "MAX")
    monkeypatch.setenv("COPILOT_CLI_MODE", "AUTOPILOT")
    monkeypatch.setenv("COPILOT_CLI_CONTEXT_TIER", "LONG_CONTEXT")
    monkeypatch.setenv("COPILOT_CLI_AGENT", "env-agent")

    agent = CopilotCli(logs_dir=tmp_path)

    assert agent.build_cli_flags() == (
        "--effort max --mode autopilot --context long_context --agent env-agent "
        "--allow-all-tools --no-ask-user --no-auto-update"
    )


def test_copilot_cli_cli_flags_explicit_kwargs_win_over_env_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COPILOT_CLI_EFFORT", "MAX")
    monkeypatch.setenv("COPILOT_CLI_MODE", "AUTOPILOT")
    monkeypatch.setenv("COPILOT_CLI_CONTEXT_TIER", "LONG_CONTEXT")
    monkeypatch.setenv("COPILOT_CLI_AGENT", "env-agent")

    agent = CopilotCli(
        logs_dir=tmp_path,
        reasoning_effort="LOW",
        mode="PLAN",
        context_tier="DEFAULT",
        agent="kwarg-agent",
    )

    assert agent.build_cli_flags() == (
        "--effort low --mode plan --context default --agent kwarg-agent "
        "--allow-all-tools --no-ask-user --no-auto-update"
    )


def test_copilot_cli_install_spec_and_allowlist(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, version="1.0.71-2")

    spec = agent.install_spec()
    domains = set(agent.network_allowlist().domains)

    assert spec.agent_name == "copilot-cli"
    assert spec.version == "1.0.71-2"
    assert any("https://gh.io/copilot-install" in step.run for step in spec.steps)
    assert any("musl-based images" in step.run for step in spec.steps)
    assert spec.verification_command is not None
    assert "copilot --version" in spec.verification_command
    assert {".github.com", ".githubcopilot.com", ".githubusercontent.com", "gh.io"} <= domains
    bare = {domain for domain in domains if not domain.startswith(".")}
    assert not any(f".{domain}" in domains for domain in bare)


def test_copilot_cli_install_spec_root_step_covers_supported_platforms(
    tmp_path: Path,
) -> None:
    root_step = CopilotCli(logs_dir=tmp_path).install_spec().steps[0]

    assert root_step.user == "root"
    assert root_step.env == {"DEBIAN_FRONTEND": "noninteractive"}
    assert "ldd --version" in root_step.run
    assert "[ -f /etc/alpine-release ]" in root_step.run
    assert "musl-based images" in root_step.run
    assert "apt-get update" in root_step.run
    assert "apt-get install -y bash ca-certificates curl git" in root_step.run
    assert "yum install -y bash ca-certificates curl git" in root_step.run
    assert "No supported package manager found" in root_step.run


def test_copilot_cli_install_spec_quotes_configured_version_only(
    tmp_path: Path,
) -> None:
    without_version = CopilotCli(logs_dir=tmp_path).install_spec().steps[1].run
    version = "1.0.71-2 beta's"
    with_version = CopilotCli(logs_dir=tmp_path, version=version).install_spec()

    assert with_version.steps[1].user == "agent"
    assert " VERSION=" not in without_version
    assert "curl -fsSL https://gh.io/copilot-install | bash" in without_version
    assert (
        "curl -fsSL https://gh.io/copilot-install | "
        f"VERSION={shlex.quote(version)} bash"
    ) in with_version.steps[1].run
    assert with_version.verification_command == (
        'export PATH="$HOME/.local/bin:$PATH"; copilot --version'
    )


def test_copilot_cli_version_parser(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path)

    assert agent.parse_version("GitHub Copilot CLI 1.0.76.\n") == "1.0.76"
    assert agent.parse_version("GitHub Copilot CLI 1.0.71-2.\n") == "1.0.71-2"
    assert agent.parse_version("GitHub Copilot CLI 1.2.\n") == "1.2"
    assert agent.parse_version("") == ""
    assert (
        agent.parse_version("first line\nGitHub Copilot CLI 2.3.4.\nlast line")
        == "2.3.4"
    )
    assert agent.parse_version("dev-build") == "dev-build"


def test_copilot_cli_auth_uses_official_precedence(tmp_path: Path):
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "GITHUB_TOKEN": "github",
            "GH_TOKEN": "gh",
            "COPILOT_GITHUB_TOKEN": "copilot",
        },
    )

    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "copilot"}


def test_copilot_cli_auth_uses_process_env_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "copilot")
    monkeypatch.setenv("GH_TOKEN", "gh")
    monkeypatch.setenv("GITHUB_TOKEN", "github")
    agent = CopilotCli(logs_dir=tmp_path)

    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "copilot"}


def test_copilot_cli_auth_falls_back_through_process_env_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    monkeypatch.setenv("GH_TOKEN", "gh")
    monkeypatch.setenv("GITHUB_TOKEN", "github")
    gh_agent = CopilotCli(logs_dir=tmp_path)
    gh_auth = gh_agent._copilot_auth_env()
    monkeypatch.delenv("GH_TOKEN")
    github_agent = CopilotCli(logs_dir=tmp_path)

    assert gh_auth == {"COPILOT_GITHUB_TOKEN": "gh"}
    assert github_agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "github"}


def test_copilot_cli_auth_extra_env_wins_over_process_env_for_same_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "process-copilot")
    monkeypatch.setenv("GH_TOKEN", "process-gh")
    copilot_agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={"COPILOT_GITHUB_TOKEN": "extra-copilot"},
    )
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    gh_agent = CopilotCli(logs_dir=tmp_path, extra_env={"GH_TOKEN": "extra-gh"})

    assert copilot_agent._copilot_auth_env() == {
        "COPILOT_GITHUB_TOKEN": "extra-copilot"
    }
    assert gh_agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "extra-gh"}


def test_copilot_cli_auth_treats_empty_strings_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "")
    monkeypatch.setenv("GH_TOKEN", "")
    monkeypatch.setenv("GITHUB_TOKEN", "process-github")
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "COPILOT_GITHUB_TOKEN": "",
            "GH_TOKEN": "",
            "GITHUB_TOKEN": "extra-github",
        },
    )

    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "extra-github"}


def test_copilot_cli_auth_raises_clear_error_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    agent = CopilotCli(logs_dir=tmp_path)

    with pytest.raises(ValueError, match="COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN"):
        agent._copilot_auth_env()


def test_copilot_cli_mcp_config_flag_is_empty_without_servers(tmp_path: Path) -> None:
    agent = CopilotCli(logs_dir=tmp_path)

    assert agent._build_mcp_config_flag() == ""


def test_copilot_cli_mcp_config_flag_round_trips_supported_transports(
    tmp_path: Path,
) -> None:
    servers = [
        MCPServerConfig(
            name="stdio-server",
            transport="stdio",
            command="python",
            args=["-m", "server", "--name", "quoted value"],
        ),
        MCPServerConfig(
            name="http server",
            transport="streamable-http",
            url="https://example.test/mcp?q=a b&x=';$(bad)",
        ),
        MCPServerConfig(
            name="sse-server",
            transport="sse",
            url="https://example.test/sse",
        ),
    ]
    agent = CopilotCli(logs_dir=tmp_path, mcp_servers=servers)
    expected = {
        "mcpServers": {
            "stdio-server": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "server", "--name", "quoted value"],
            },
            "http server": {
                "type": "http",
                "url": "https://example.test/mcp?q=a b&x=';$(bad)",
            },
            "sse-server": {
                "type": "sse",
                "url": "https://example.test/sse",
            },
        }
    }

    flag = agent._build_mcp_config_flag()
    parts = shlex.split(flag)

    assert parts == [
        "--additional-mcp-config",
        json.dumps(expected, separators=(",", ":")),
    ]
    assert flag == (
        "--additional-mcp-config "
        f"{shlex.quote(json.dumps(expected, separators=(',', ':')))}"
    )
    assert json.loads(parts[1]) == expected


def test_copilot_cli_register_skills_command_handles_absent_dir(tmp_path: Path) -> None:
    agent = CopilotCli(logs_dir=tmp_path)

    assert agent._build_register_skills_command() == ""


def test_copilot_cli_register_skills_command_quotes_shell_sensitive_dir(
    tmp_path: Path,
) -> None:
    skills_dir = "/skills dir/with;$(bad)'quote"
    quoted = shlex.quote(skills_dir)
    agent = CopilotCli(logs_dir=tmp_path, skills_dir=skills_dir)

    command = agent._build_register_skills_command()

    assert command == (
        f"if [ -d {quoted} ]; then "
        'mkdir -p "$COPILOT_HOME/skills" && '
        f'cp -r {quoted}/* "$COPILOT_HOME/skills/" 2>/dev/null || true; '
        "fi"
    )


def test_copilot_cli_run_command_preserves_pipeline_status(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path)

    command = agent._build_run_command(
        setup="mkdir -p /logs/agent",
        instruction="fix it",
        flag_text="--session-id 1234 --allow-all-tools",
        jsonl_path="/logs/agent/copilot-cli.jsonl",
        output_path="/logs/agent/copilot-cli.txt",
    )

    assert "bash -lc" in command
    assert "--output-format json" in command
    assert "tee /logs/agent/command-0/stdout.txt" in command
    assert "exit ${PIPESTATUS[0]}" in command
    assert "cp -a" not in command


def test_copilot_cli_extra_args_string_handles_none_and_empty_values(
    tmp_path: Path,
) -> None:
    assert CopilotCli(logs_dir=tmp_path)._extra_args_string() == ""
    assert CopilotCli(logs_dir=tmp_path, extra_args="")._extra_args_string() == ""
    assert CopilotCli(logs_dir=tmp_path, extra_args=[])._extra_args_string() == ""


def test_copilot_cli_extra_args_string_round_trips_quoted_values(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, extra_args='--foo "a b"')

    extra_args = agent._extra_args_string()

    assert extra_args == "--foo 'a b'"
    assert shlex.split(extra_args) == ["--foo", "a b"]


def test_copilot_cli_extra_args_string_quotes_shell_metacharacters(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, extra_args="; touch pwned")
    extra_args = agent._extra_args_string()
    command = agent._build_run_command(
        setup="mkdir -p /logs/agent",
        instruction="fix it",
        flag_text=extra_args,
        jsonl_path="/logs/agent/copilot-cli.jsonl",
        output_path="/logs/agent/copilot-cli.txt",
    )

    run_script = shlex.split(command)[-1]

    assert shlex.split(extra_args) == [";", "touch", "pwned"]
    assert " ; touch pwned" not in run_script
    assert "';' touch pwned" in run_script


def test_copilot_cli_extra_args_list_coerces_and_quotes_values(
    tmp_path: Path,
) -> None:
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_args=["--foo", "a b", "quote'value", ";", "$(bad)", 7],
    )

    extra_args = agent._extra_args_string()
    command = agent._build_run_command(
        setup="mkdir -p /logs/agent",
        instruction="fix it",
        flag_text=extra_args,
        jsonl_path="/logs/agent/copilot-cli.jsonl",
        output_path="/logs/agent/copilot-cli.txt",
    )
    run_script = shlex.split(command)[-1]

    assert shlex.split(extra_args) == [
        "--foo",
        "a b",
        "quote'value",
        ";",
        "$(bad)",
        "7",
    ]
    assert " ; " not in run_script
    assert "';'" in run_script
    assert "'$(bad)'" in run_script


def test_copilot_cli_extra_args_string_rejects_malformed_shell_syntax(
    tmp_path: Path,
):
    agent = CopilotCli(logs_dir=tmp_path, extra_args='--foo "unterminated')

    with pytest.raises(ValueError, match="Invalid extra_args string"):
        agent._extra_args_string()


def test_copilot_cli_native_session_root_is_mounted():
    assert CopilotCli._COPILOT_HOME.as_posix() == "/logs/agent/copilot-home"


def test_copilot_cli_registers_skills_under_copilot_home(tmp_path: Path):
    agent = CopilotCli(logs_dir=tmp_path, skills_dir="/skills")

    command = agent._build_register_skills_command()

    assert '"$COPILOT_HOME/skills"' in command
    assert "~/.copilot/skills" not in command


def test_copilot_cli_extra_env_reserved_keys_are_normalized(tmp_path: Path):
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={
            "COPILOT_HOME": "/custom/home",
            "COPILOT_GITHUB_TOKEN": "",
            "SOME_OTHER_VAR": "value",
        },
    )

    assert "COPILOT_HOME" not in agent._extra_env
    assert "COPILOT_GITHUB_TOKEN" not in agent._extra_env
    assert agent._extra_env.get("SOME_OTHER_VAR") == "value"


@run_async
async def test_copilot_cli_run_resolves_bare_model_id_from_nested_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        model_name="litellm/github_copilot/gpt-5.4",
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    # Copilot CLI only accepts bare model ids, so every routing prefix goes.
    assert "--model gpt-5.4" in run_script
    assert "github_copilot" not in run_script


@run_async
async def test_copilot_cli_run_records_the_generated_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    assert agent._session_id is not None
    UUID(agent._session_id)
    assert f"--session-id {agent._session_id}" in run_script


def test_copilot_cli_auth_ignores_empty_agent_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _TOKEN_ENV_VARS)
    monkeypatch.setenv("GH_TOKEN", "process-token")
    agent = CopilotCli(logs_dir=tmp_path, extra_env={"GH_TOKEN": ""})

    # An empty agent.env override must not mask a usable process token, since
    # _exec() re-applies extra_env on top of the environment it builds.
    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "process-token"}


def test_copilot_cli_empty_copilot_token_override_is_dropped(tmp_path: Path) -> None:
    agent = CopilotCli(
        logs_dir=tmp_path,
        extra_env={"COPILOT_GITHUB_TOKEN": "", "GITHUB_TOKEN": "github"},
    )

    assert "COPILOT_GITHUB_TOKEN" not in agent._extra_env
    assert agent._copilot_auth_env() == {"COPILOT_GITHUB_TOKEN": "github"}


@run_async
async def test_copilot_cli_run_quotes_the_agent_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch, _CLI_ENV_VARS)
    agent = CopilotCli(
        logs_dir=tmp_path,
        agent="my agent; touch /tmp/pwned",
        extra_env={"COPILOT_GITHUB_TOKEN": "token"},
    )

    _, _, run_script = await _run_agent(agent, tmp_path)

    assert "--agent 'my agent; touch /tmp/pwned'" in run_script
    assert "; touch /tmp/pwned" not in run_script.replace(
        "'my agent; touch /tmp/pwned'", ""
    )


def test_cli_flag_quoting_applies_inside_a_format_template(tmp_path: Path) -> None:
    """`quote` must survive `format`, or a templated flag loses its escaping."""
    agent = CopilotCli(logs_dir=tmp_path)
    agent.CLI_FLAGS = [
        CliFlag(
            kwarg="templated",
            cli="--templated",
            format="--templated={value}",
            quote=True,
        ),
        CliFlag(kwarg="plain", cli="--plain", format="--plain={value}"),
    ]
    agent._resolved_flags = {"templated": "a b; rm -rf /", "plain": "safe"}

    flags = agent.build_cli_flags()

    assert "--templated='a b; rm -rf /'" in flags
    assert "--plain=safe" in flags

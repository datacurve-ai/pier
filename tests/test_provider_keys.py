from pathlib import Path

from pier.agents import utils as agent_utils
from pier.agents.utils import get_api_key_var_names_from_model_name


def test_provider_keys_contains_minimax():
    """MiniMax Coding Plan must be registered so `minimax/MiniMax-M3` resolves."""
    assert agent_utils.PROVIDER_KEYS["minimax"] == "MINIMAX_API_KEY"


def test_provider_model_names_includes_minimax():
    """Mirror moonshot/zai: the litellm model set must list MiniMax-M3 as well."""
    assert any(
        env == "MINIMAX_API_KEY" and "minimax/MiniMax-M3" in models
        for models, env in agent_utils.PROVIDER_MODEL_NAMES
    )


def test_get_api_key_var_names_for_minimax_model(monkeypatch):
    """End-to-end check: --model minimax/MiniMax-M3 resolves to MINIMAX_API_KEY."""
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)

    result = get_api_key_var_names_from_model_name("minimax/MiniMax-M3")

    assert result == ["MINIMAX_API_KEY"]


def test_opencode_minimax_provider_stub_sets_auth(tmp_path: Path):
    """opencode provider stub for MiniMax must set baseURL + apiKey so the
    Anthropic SDK reads MINIMAX_API_KEY (not ANTHROPIC_API_KEY) at runtime."""
    from pier.agents.installed.opencode import OpenCode

    agent = OpenCode(logs_dir=tmp_path, model_name="minimax/MiniMax-M3")
    config = agent._build_runtime_config(include_mcp=False)

    provider = config["provider"]["minimax"]
    assert provider["npm"] == "@ai-sdk/anthropic"
    assert provider["options"]["baseURL"] == "https://api.minimax.io/anthropic/v1"
    assert provider["options"]["apiKey"] == "{env:MINIMAX_API_KEY}"


def test_opencode_minimax_provider_stub_honors_minimax_base_url(
    tmp_path: Path, monkeypatch
):
    """MINIMAX_BASE_URL env overrides the default Anthropic-compatible endpoint."""
    from pier.agents.installed.opencode import OpenCode

    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic/v1")
    agent = OpenCode(logs_dir=tmp_path, model_name="minimax/MiniMax-M3")

    config = agent._build_runtime_config(include_mcp=False)
    assert (
        config["provider"]["minimax"]["options"]["baseURL"]
        == "https://api.minimaxi.com/anthropic/v1"
    )

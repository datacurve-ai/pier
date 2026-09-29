"""Unit tests for installed-agent subscription / OAuth auth support.

Covers the changes that let codex and opencode authenticate via a host
auth.json (ChatGPT-subscription OAuth) under pier:
  - the openai egress allowlist extended to the ChatGPT OAuth backend
    (chatgpt.com) + token-refresh host (auth.openai.com), for both agents, and
  - opencode's auth.json injection resolution (OPENCODE_FORCE_AUTH_JSON /
    OPENCODE_AUTH_JSON_PATH); docker/default behavior (None) when unset.

No container engine required: agents are built with __new__ and only the
attributes the methods touch are set.
"""


def test_codex_openai_allowlist_includes_chatgpt(monkeypatch):
    from pier.agents.installed.codex import Codex

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    c = Codex.__new__(Codex)
    c._extra_env = {}
    c._config_toml = None
    domains = c.network_allowlist().domains
    assert "api.openai.com" in domains  # API-key auth preserved
    assert "chatgpt.com" in domains  # ChatGPT-subscription OAuth backend
    assert "auth.openai.com" in domains  # OAuth token refresh


def _opencode_agent():
    from pier.agents.installed.opencode import OpenCode

    oc = OpenCode.__new__(OpenCode)
    oc._extra_env = {}
    return oc


def test_opencode_openai_allowlist_includes_chatgpt():
    from pier.agents.installed.opencode import OpenCode

    dom = OpenCode._DEFAULT_PROVIDER_DOMAINS["openai"]
    assert "api.openai.com" in dom  # API-key auth preserved
    assert "chatgpt.com" in dom  # ChatGPT-subscription OAuth backend
    assert "auth.openai.com" in dom  # OAuth token refresh


def test_opencode_resolve_auth_json_none_when_unset(monkeypatch):
    monkeypatch.delenv("OPENCODE_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("OPENCODE_FORCE_AUTH_JSON", raising=False)
    assert _opencode_agent()._resolve_auth_json_path() is None


def test_opencode_resolve_auth_json_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_FORCE_AUTH_JSON", raising=False)
    f = tmp_path / "auth.json"
    f.write_text("{}")
    monkeypatch.setenv("OPENCODE_AUTH_JSON_PATH", str(f))
    assert _opencode_agent()._resolve_auth_json_path() == f


def test_opencode_resolve_auth_json_force_uses_default_home(tmp_path, monkeypatch):
    from pier.agents.installed import opencode as opencode_mod

    monkeypatch.delenv("OPENCODE_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("OPENCODE_FORCE_AUTH_JSON", "1")
    auth = tmp_path / ".local" / "share" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}")
    monkeypatch.setattr(opencode_mod.Path, "home", lambda: tmp_path)
    assert _opencode_agent()._resolve_auth_json_path() == auth

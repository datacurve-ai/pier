"""Integration tests that exercise the real adapter cost methods end-to-end.

These drive the actual ``_compute_cost_from_pricing`` / ``_resolve_pricing_rates``
methods on constructed adapters (not ``pier.utils.pricing`` directly), so the
adapter -> shared-helper wiring is executed, not just imported.
"""

import json
from pathlib import Path

import pytest

from pier.agents.installed.codex import Codex
from pier.agents.installed.cursor_cli import CursorCli
from pier.agents.installed.gemini_cli import GeminiCli
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.utils.pricing import (
    PRICING_FALLBACK_ENV,
    PRICING_FILE_ENV,
    _OVERRIDES_CACHE,
)

# DeepSeek-style steep cache discount (cache reads ~0.8% of the input rate).
RATES = {
    "input_cost_per_token": 4.35e-7,
    "output_cost_per_token": 8.7e-7,
    "cache_read_input_token_cost": 3.625e-9,
}
RATES_NO_CACHE = {
    "input_cost_per_token": 4.35e-7,
    "output_cost_per_token": 8.7e-7,
}

# The issue's representative trial: 8.99M prompt tokens, 7.08M of them cache hits.
PROMPT, CACHED, OUTPUT = 8_986_237, 7_078_144, 43_110
# Correct cache-aware cost ~= $0.89; billing cache at the input rate gives ~$4.36.
EXPECTED_DISCOUNTED = 0.8939


@pytest.fixture(autouse=True)
def _clean_pricing_env(monkeypatch):
    _OVERRIDES_CACHE.clear()
    monkeypatch.delenv(PRICING_FILE_ENV, raising=False)
    monkeypatch.delenv(PRICING_FALLBACK_ENV, raising=False)
    yield
    _OVERRIDES_CACHE.clear()


def _overrides(tmp_path: Path, mapping: dict, monkeypatch) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))


@pytest.mark.parametrize("cls", [Codex, GeminiCli])
def test_adapter_cost_discounts_cache_reads(cls, tmp_path, monkeypatch):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES}, monkeypatch)
    agent = cls(logs_dir=tmp_path, model_name="deepseek-v4-pro")

    cost = agent._compute_cost_from_pricing(PROMPT, OUTPUT, CACHED)

    assert cost == pytest.approx(EXPECTED_DISCOUNTED, abs=1e-3)
    # Sanity: the discounted figure is far below the miss-rate bill (~$4.36).
    assert cost < 1.5


@pytest.mark.parametrize("cls", [Codex, GeminiCli])
def test_adapter_fails_loud_on_missing_cache_rate(cls, tmp_path, monkeypatch, caplog):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES_NO_CACHE}, monkeypatch)
    agent = cls(logs_dir=tmp_path, model_name="deepseek-v4-pro")

    with caplog.at_level("ERROR"):
        cost = agent._compute_cost_from_pricing(PROMPT, OUTPUT, CACHED)

    assert cost is None  # left unpriced rather than silently inflated
    assert any("cache-read price" in r.message for r in caplog.records)


@pytest.mark.parametrize("cls", [Codex, GeminiCli])
def test_adapter_fallback_env_restores_input_rate(cls, tmp_path, monkeypatch):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES_NO_CACHE}, monkeypatch)
    monkeypatch.setenv(PRICING_FALLBACK_ENV, "input")
    agent = cls(logs_dir=tmp_path, model_name="deepseek-v4-pro")

    cost = agent._compute_cost_from_pricing(PROMPT, OUTPUT, CACHED)

    # Legacy behaviour: every prompt token billed at the input rate.
    uncached_at_input = (PROMPT - CACHED) * RATES_NO_CACHE["input_cost_per_token"]
    cache_at_input = CACHED * RATES_NO_CACHE["input_cost_per_token"]
    output_cost = OUTPUT * RATES_NO_CACHE["output_cost_per_token"]
    assert cost == pytest.approx(uncached_at_input + cache_at_input + output_cost)


def test_cursor_resolve_pricing_rates_uses_override(tmp_path, monkeypatch):
    # Use a non-Cursor model so the built-in Cursor pricing table is bypassed
    # and resolution falls through to the overrides file.
    _overrides(tmp_path, {"deepseek-v4-pro": RATES}, monkeypatch)
    agent = CursorCli(logs_dir=tmp_path, model_name="deepseek-v4-pro")

    result = agent._resolve_pricing_rates()

    assert result is not None
    rates, source = result
    assert source == "override"
    assert rates["cache_read"] == pytest.approx(3.625e-9)
    assert rates["input"] == pytest.approx(4.35e-7)


def test_cursor_resolve_pricing_rates_fails_loud(tmp_path, monkeypatch, caplog):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES_NO_CACHE}, monkeypatch)
    agent = CursorCli(logs_dir=tmp_path, model_name="deepseek-v4-pro")

    with caplog.at_level("ERROR"):
        result = agent._resolve_pricing_rates()

    assert result is None  # unpriced rather than silently billing cache at input
    assert any("cache-read price" in r.message for r in caplog.records)


def test_mini_swe_agent_prices_deepseek_via_supplementary(tmp_path):
    """mini-swe path: deepseek-v4-pro (absent from litellm) priced cache-aware
    out of the box via the shipped supplementary table -- no override needed."""
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openrouter/deepseek-v4-pro")

    cost = agent._compute_cost_from_pricing(PROMPT, OUTPUT, CACHED)

    # uncached 1.908M @ 4.35e-7 + cache 7.078M @ 3.625e-9 + out 43.11k @ 8.7e-7
    assert cost == pytest.approx(EXPECTED_DISCOUNTED, abs=1e-3)


def test_mini_swe_agent_fails_loud_on_missing_cache_rate(tmp_path, monkeypatch, caplog):
    # Override strips the cache rate -> mini-swe leaves cost unpriced (keeps reported).
    _overrides(tmp_path, {"deepseek-v4-pro": RATES_NO_CACHE}, monkeypatch)
    agent = MiniSweAgent(logs_dir=tmp_path, model_name="openrouter/deepseek-v4-pro")

    with caplog.at_level("ERROR"):
        cost = agent._compute_cost_from_pricing(PROMPT, OUTPUT, CACHED)

    assert cost is None
    assert any("cache-read price" in r.message for r in caplog.records)


def _viewer_client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from pier.viewer.server import create_app

    return TestClient(create_app(tmp_path, mode="jobs"))


def test_viewer_pricing_endpoint_uses_override(tmp_path, monkeypatch):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES}, monkeypatch)

    resp = _viewer_client(tmp_path).get(
        "/api/pricing", params={"model": "deepseek-v4-pro"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_read_input_token_cost"] == pytest.approx(3.625e-9)
    assert body["input_cost_per_token"] == pytest.approx(4.35e-7)


def test_viewer_pricing_reports_null_cache_rate_when_unlisted(tmp_path, monkeypatch):
    _overrides(tmp_path, {"deepseek-v4-pro": RATES_NO_CACHE}, monkeypatch)

    resp = _viewer_client(tmp_path).get(
        "/api/pricing", params={"model": "deepseek-v4-pro"}
    )

    assert resp.status_code == 200
    # The honest behaviour: null rather than mirroring the input rate.
    assert resp.json()["cache_read_input_token_cost"] is None


def test_viewer_pricing_unknown_model_returns_404(tmp_path):
    resp = _viewer_client(tmp_path).get(
        "/api/pricing", params={"model": "totally-unknown-model-xyz"}
    )
    assert resp.status_code == 404

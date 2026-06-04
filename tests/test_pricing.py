import json
from pathlib import Path

import pytest

from pier.utils.pricing import (
    PRICING_FALLBACK_ENV,
    PRICING_FILE_ENV,
    PricingConfigError,
    _OVERRIDES_CACHE,
    compute_cost,
    load_pricing_overrides,
    resolve_model_pricing,
)

# A model with a steep cache discount (DeepSeek-style: cache reads ~0.8% of input).
DEEPSEEK_RATES = {
    "input_cost_per_token": 4.35e-7,
    "output_cost_per_token": 8.7e-7,
    "cache_read_input_token_cost": 3.625e-9,
}
# Same model but *missing* the cache-read rate -- the trap this module guards.
DEEPSEEK_NO_CACHE = {
    "input_cost_per_token": 4.35e-7,
    "output_cost_per_token": 8.7e-7,
}


@pytest.fixture(autouse=True)
def _clear_overrides_cache(monkeypatch):
    """Each test starts with a clean overrides cache and no pricing env vars."""
    _OVERRIDES_CACHE.clear()
    monkeypatch.delenv(PRICING_FILE_ENV, raising=False)
    monkeypatch.delenv(PRICING_FALLBACK_ENV, raising=False)
    yield
    _OVERRIDES_CACHE.clear()


def _write_overrides(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_load_overrides_empty_when_unset():
    assert load_pricing_overrides() == {}


def test_resolve_prefers_override_file(tmp_path, monkeypatch):
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_RATES})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    pricing = resolve_model_pricing("deepseek-v4-pro")

    assert pricing is not None
    assert pricing.source == "override"
    assert pricing.cache_read_input_token_cost == pytest.approx(3.625e-9)


def test_resolve_strips_provider_prefix(tmp_path, monkeypatch):
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_RATES})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    pricing = resolve_model_pricing("deepseek/deepseek-v4-pro")

    assert pricing is not None
    assert pricing.input_cost_per_token == pytest.approx(4.35e-7)


def test_compute_cost_discounts_cache_reads(tmp_path, monkeypatch):
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_RATES})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    # 8.99M prompt tokens of which 7.08M are cache hits (the issue's trial).
    cost = compute_cost(
        "deepseek-v4-pro",
        prompt_tokens=8_986_237,
        completion_tokens=43_110,
        cached_tokens=7_078_144,
    )

    # Hand-computed: uncached 1.908M @ 4.35e-7 + cache 7.078M @ 3.625e-9
    # + output 43.11k @ 8.7e-7  ~= $0.89 -- not the ~$4.36 a miss-rate bill gives.
    assert cost == pytest.approx(0.8939, abs=1e-3)


def test_missing_cache_rate_with_cached_tokens_returns_none(
    tmp_path, monkeypatch, caplog
):
    """The core fix: do not silently bill cache reads at the input rate."""
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_NO_CACHE})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    with caplog.at_level("ERROR"):
        cost = compute_cost(
            "deepseek-v4-pro",
            prompt_tokens=8_986_237,
            completion_tokens=43_110,
            cached_tokens=7_078_144,
        )

    assert cost is None
    assert any("cache-read price" in r.message for r in caplog.records)


def test_missing_cache_rate_without_cached_tokens_is_fine(tmp_path, monkeypatch):
    """No cached tokens -> the absent cache rate is irrelevant; still prices."""
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_NO_CACHE})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    cost = compute_cost(
        "deepseek-v4-pro",
        prompt_tokens=1_000_000,
        completion_tokens=10_000,
        cached_tokens=0,
    )

    assert cost == pytest.approx(1_000_000 * 4.35e-7 + 10_000 * 8.7e-7)


def test_fallback_env_restores_legacy_input_rate(tmp_path, monkeypatch):
    path = _write_overrides(tmp_path, {"deepseek-v4-pro": DEEPSEEK_NO_CACHE})
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))
    monkeypatch.setenv(PRICING_FALLBACK_ENV, "input")

    cost = compute_cost(
        "deepseek-v4-pro",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=400_000,
    )

    # Legacy behaviour: cache reads billed at the full input rate.
    assert cost == pytest.approx(1_000_000 * 4.35e-7)


def test_unknown_model_returns_none(caplog):
    with caplog.at_level("ERROR"):
        cost = compute_cost(
            "totally-unknown-model-xyz",
            prompt_tokens=1000,
            completion_tokens=10,
            cached_tokens=500,
        )
    assert cost is None


def test_bad_overrides_file_raises(tmp_path, monkeypatch):
    path = tmp_path / "pricing.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setenv(PRICING_FILE_ENV, str(path))

    with pytest.raises(PricingConfigError):
        load_pricing_overrides()


def test_missing_overrides_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(PRICING_FILE_ENV, str(tmp_path / "does-not-exist.json"))
    with pytest.raises(PricingConfigError):
        load_pricing_overrides()

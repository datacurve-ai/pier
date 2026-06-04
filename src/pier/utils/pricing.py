"""Cache-aware model pricing resolution for token-based cost computation.

Rates are expressed in USD per token, matching the schema of
``litellm.model_cost`` entries (``input_cost_per_token``,
``output_cost_per_token``, ``cache_read_input_token_cost`` and
``cache_creation_input_token_cost``).

Two sources are consulted, in order:

1. A user-supplied JSON overrides file pointed to by the ``PIER_PRICING_FILE``
   environment variable. Use it to price models LiteLLM does not know about, or
   to supply a cache-read rate LiteLLM is missing, without waiting on a LiteLLM
   release.
2. LiteLLM's built-in ``model_cost`` table.

Fail-loud policy
----------------
Agent trajectories are dominated by cache *reads*. Historically each adapter
silently fell back to the full (cache-miss) input rate whenever a model lacked a
``cache_read_input_token_cost`` entry. Because cache reads are typically priced
at a steep discount (often 80-99% off the input rate) and make up the large
majority of prompt tokens, that silent fallback can overstate cost severalfold.

This module refuses to do that silently. When a run used cached tokens but no
cache-read price is known, :func:`compute_cost` logs an error and returns
``None`` (cost unavailable) rather than emitting a confidently-wrong number.
Supply the rate via ``PIER_PRICING_FILE``, or set ``PIER_PRICING_FALLBACK=input``
to explicitly opt back into the approximate (inflated) input-rate fallback.

Overrides file format
---------------------
A JSON object mapping model name to a subset of the LiteLLM rate fields::

    {
      "deepseek-v4-pro": {
        "input_cost_per_token": 4.35e-7,
        "output_cost_per_token": 8.7e-7,
        "cache_read_input_token_cost": 3.625e-9
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

PRICING_FILE_ENV = "PIER_PRICING_FILE"
PRICING_FALLBACK_ENV = "PIER_PRICING_FALLBACK"

_LOGGER = logging.getLogger(__name__)


class PricingConfigError(RuntimeError):
    """Raised when ``PIER_PRICING_FILE`` is set but cannot be read or parsed."""


@dataclass(frozen=True)
class ModelPricing:
    """Resolved per-token rates for a model (USD per token)."""

    model_name: str
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float | None
    cache_creation_input_token_cost: float | None
    source: str  # "override", "litellm", or "supplementary"


# Curated rates for models LiteLLM's registry is missing or lags on. This is a
# stopgap so cost is correct out of the box for known gaps -- the long-term fix
# is upstreaming these to litellm (model_prices_and_context_window.json). It is
# consulted only after the user overrides file and litellm, and anything here can
# be overridden or extended via PIER_PRICING_FILE. Rates are USD per token.
# deepseek-v4-* rates from datacurve-ai/deep-swe#21 and PR #4 (thanks @dephnor).
SUPPLEMENTARY_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {
        "input_cost_per_token": 0.435 / 1_000_000,
        "output_cost_per_token": 0.87 / 1_000_000,
        "cache_read_input_token_cost": 0.003625 / 1_000_000,
    },
    "deepseek-v4-flash": {
        "input_cost_per_token": 0.14 / 1_000_000,
        "output_cost_per_token": 0.28 / 1_000_000,
        "cache_read_input_token_cost": 0.0028 / 1_000_000,
    },
}


# Simple mtime-keyed cache so an edited overrides file is picked up without a
# restart, while avoiding a re-read on every cost computation.
_OVERRIDES_CACHE: dict[str, Any] = {}


def load_pricing_overrides() -> dict[str, dict[str, Any]]:
    """Load and cache the ``PIER_PRICING_FILE`` overrides (``{}`` if unset).

    Raises :class:`PricingConfigError` if the variable is set but the file is
    missing or not a JSON object -- a misconfiguration the caller asked for and
    should hear about loudly.
    """
    path = os.environ.get(PRICING_FILE_ENV)
    if not path:
        return {}

    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise PricingConfigError(
            f"{PRICING_FILE_ENV}={path!r} could not be read: {exc}"
        ) from exc

    if _OVERRIDES_CACHE.get("path") != path or _OVERRIDES_CACHE.get("mtime") != mtime:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PricingConfigError(
                f"{PRICING_FILE_ENV}={path!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise PricingConfigError(
                f"{PRICING_FILE_ENV}={path!r} must be a JSON object mapping "
                f"model name -> rate fields"
            )
        _OVERRIDES_CACHE.update(path=path, mtime=mtime, data=data)

    return _OVERRIDES_CACHE["data"]


def _lookup(table: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    """Look up a model by full name then by the part after the first ``/``."""
    for key in (model_name, model_name.split("/", 1)[-1]):
        entry = table.get(key)
        if entry:
            return entry
    return None


def resolve_model_pricing(
    model_name: str | None,
    *,
    logger: logging.Logger | None = None,
) -> ModelPricing | None:
    """Resolve pricing for ``model_name`` from overrides then LiteLLM.

    Returns ``None`` when no source knows the model, or when an entry lacks the
    mandatory input/output rates.
    """
    logger = logger or _LOGGER
    if not model_name:
        return None

    # Resolution order: user overrides -> litellm -> shipped supplementary gaps.
    entry = _lookup(load_pricing_overrides(), model_name)
    source = "override"
    if entry is None:
        try:
            import litellm

            entry = _lookup(litellm.model_cost, model_name)
            source = "litellm"
        except ImportError:
            logger.warning(
                "litellm is not installed; relying on %s and built-in "
                "supplementary pricing for %r",
                PRICING_FILE_ENV,
                model_name,
            )
    if entry is None:
        entry = _lookup(SUPPLEMENTARY_PRICING, model_name)
        source = "supplementary"

    if entry is None:
        return None

    input_rate = entry.get("input_cost_per_token")
    output_rate = entry.get("output_cost_per_token")
    if input_rate is None or output_rate is None:
        return None

    def _opt(field: str) -> float | None:
        value = entry.get(field)
        return None if value is None else float(value)

    return ModelPricing(
        model_name=model_name,
        input_cost_per_token=float(input_rate),
        output_cost_per_token=float(output_rate),
        cache_read_input_token_cost=_opt("cache_read_input_token_cost"),
        cache_creation_input_token_cost=_opt("cache_creation_input_token_cost"),
        source=source,
    )


def _fallback_to_input_rate() -> bool:
    return os.environ.get(PRICING_FALLBACK_ENV, "").strip().lower() == "input"


def resolve_cache_read_rate(
    pricing: ModelPricing,
    *,
    used_cached_tokens: bool,
    logger: logging.Logger | None = None,
) -> float | None:
    """Return the cache-read rate, applying the fail-loud policy.

    ``None`` means "do not price this": the run used cached tokens, no cache-read
    rate is known, and ``PIER_PRICING_FALLBACK=input`` was not set. When no
    cached tokens were used the rate is irrelevant and the input rate is returned.
    """
    logger = logger or _LOGGER
    if pricing.cache_read_input_token_cost is not None:
        return pricing.cache_read_input_token_cost
    if not used_cached_tokens:
        return pricing.input_cost_per_token
    if _fallback_to_input_rate():
        logger.warning(
            "No cache-read price for %r (source=%s); %s=input set, billing cached "
            "tokens at the full input rate -- this overstates cost.",
            pricing.model_name,
            pricing.source,
            PRICING_FALLBACK_ENV,
        )
        return pricing.input_cost_per_token
    logger.error(
        "No cache-read price for model %r (source=%s) but the run used cached "
        "input tokens. Billing them at the full input rate would overstate cost "
        "severalfold, so cost is left unpriced. Add "
        "'cache_read_input_token_cost' for this model via %s, or set %s=input to "
        "accept the approximate fallback.",
        pricing.model_name,
        pricing.source,
        PRICING_FILE_ENV,
        PRICING_FALLBACK_ENV,
    )
    return None


def compute_cost(
    model_name: str | None,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None = 0,
    cache_creation_tokens: int | None = 0,
    logger: logging.Logger | None = None,
) -> float | None:
    """Total USD cost from token counts, or ``None`` if it cannot be priced safely.

    ``prompt_tokens`` is the full prompt size *including* cache reads and writes
    (the OpenAI / LiteLLM convention); ``cached_tokens`` is the cache-read subset
    and ``cache_creation_tokens`` the cache-write subset. Returns ``None`` -- with
    an error log -- when pricing is unavailable, including the case where cached
    tokens were used but no cache-read rate is known (see module docstring).
    """
    logger = logger or _LOGGER
    pricing = resolve_model_pricing(model_name, logger=logger)
    if pricing is None:
        logger.error(
            "No pricing for model %r; leaving cost unpriced. Set %s to a JSON file "
            "mapping the model to its per-token rates.",
            model_name,
            PRICING_FILE_ENV,
        )
        return None

    cached = cached_tokens or 0
    created = cache_creation_tokens or 0
    completion = completion_tokens or 0
    uncached = max(0, (prompt_tokens or 0) - cached - created)

    cache_read_rate = resolve_cache_read_rate(
        pricing, used_cached_tokens=cached > 0, logger=logger
    )
    if cache_read_rate is None:
        return None

    cache_write_rate = (
        pricing.cache_creation_input_token_cost
        if pricing.cache_creation_input_token_cost is not None
        else pricing.input_cost_per_token
    )

    return (
        uncached * pricing.input_cost_per_token
        + cached * cache_read_rate
        + created * cache_write_rate
        + completion * pricing.output_cost_per_token
    )

"""Helpers for comparable trajectory aggregate metrics."""

from collections.abc import Iterable
from typing import Any

from pier.models.agent.context import AgentContext
from pier.models.trajectories.final_metrics import FinalMetrics
from pier.models.trajectories.step import Step


def sum_known[T: (int, float)](values: Iterable[T | None]) -> T | None:
    """Sum a counter only when every contributor reports it; empty is unknown."""
    total = None
    for value in values:
        if value is None:
            return None
        total = value if total is None else total + value
    return total


def peak_context_tokens_from_steps(steps: Iterable[Step]) -> int | None:
    """Return the largest prompt token count on any agent step."""
    peak: int | None = None
    for step in steps:
        if step.source != "agent" or step.metrics is None:
            continue
        prompt_tokens = step.metrics.prompt_tokens
        if prompt_tokens is None:
            continue
        peak = prompt_tokens if peak is None else max(peak, prompt_tokens)
    return peak


def extra_with_context_metrics(
    extra: dict[str, Any] | None,
    *,
    peak_context_tokens: int | None,
    summarization_count: int | None,
) -> dict[str, Any] | None:
    """Return final_metrics.extra with Pier context aggregate extensions."""
    result = dict(extra or {})
    if peak_context_tokens is not None:
        result["peak_context_tokens"] = peak_context_tokens
    if summarization_count is not None:
        result["summarization_count"] = summarization_count
    return result or None


def populate_context_from_final_metrics(
    context: AgentContext, metrics: FinalMetrics
) -> None:
    """Copy trajectory final metrics into the run-level agent context."""
    context.cost_usd = metrics.total_cost_usd
    context.n_input_tokens = metrics.total_prompt_tokens
    context.n_cache_tokens = metrics.total_cached_tokens
    context.n_output_tokens = metrics.total_completion_tokens
    extra = metrics.extra or {}
    peak_context_tokens = extra.get("peak_context_tokens")
    summarization_count = extra.get("summarization_count")
    context.peak_context_tokens = (
        peak_context_tokens if isinstance(peak_context_tokens, int) else None
    )
    context.summarization_count = (
        summarization_count if isinstance(summarization_count, int) else None
    )

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from pier.utils.logger import logger

EVENT_PREFIX = "PIER_EVENT "
CONCURRENCY_LOG_PREFIX = "CONCURRENCY "


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_event_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def concurrency_group_fields(
    provider: str | None, model: str | None, effort: str | None
) -> dict[str, Any]:
    return {"provider": provider, "model": model, "effort": effort}


def log_concurrency_event(event_type: str, **fields: Any) -> None:
    """Write one structured concurrency record through the existing job logger."""
    record = {"ts": utc_now(), "type": event_type, **fields}
    logger.getChild(__name__).debug(
        "%s%s",
        CONCURRENCY_LOG_PREFIX,
        json.dumps(record, separators=(",", ":"), sort_keys=True),
    )


@dataclass(frozen=True)
class PierEvent:
    """A small, generic telemetry event emitted by a running trial."""

    type: str
    trial_id: str
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
    retry_after_sec: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_event_id("event"))
    observed_at: str = field(default_factory=utc_now)
    evidence: str | None = None
    capacity_period: int | None = None


EventSink = Callable[[PierEvent], Awaitable[None]]


class EventBus:
    """In-process event bus used by rollout producers and job controllers."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PierEvent | None] = asyncio.Queue()

    async def publish(self, event: PierEvent) -> None:
        await self._queue.put(event)

    async def next(self) -> PierEvent | None:
        return await self._queue.get()

    async def close(self) -> None:
        await self._queue.put(None)


@dataclass(frozen=True)
class TelemetryContext:
    sink: EventSink
    trial_id: str
    provider: str | None
    model: str | None
    effort: str | None
    pause_messages: asyncio.Queue[str | None] | None = None


@dataclass(frozen=True)
class _RateLimitSignal:
    payload: dict[str, Any]
    retry_after_sec: float | None
    evidence: str | None
    capacity_period: int | None


_USAGE_METRICS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "agent_steps",
    "tool_calls",
    "cost_usd",
)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _structured_rate_limit(payload: dict[str, Any]) -> _RateLimitSignal | None:
    if payload.get("type") not in {
        "inference.rate_limited",
        "model.request.rate_limited",
    }:
        return None

    retry_after = _float_or_none(payload.get("retry_after_sec"))
    if retry_after is None:
        retry_after_ms = _float_or_none(payload.get("retry_after_ms"))
        if retry_after_ms is not None:
            retry_after = retry_after_ms / 1000

    status_code = _int_or_none(payload.get("status_code"))
    evidence = str(payload.get("evidence") or "") or None
    if status_code != 429 and evidence != "typed_rate_limit_error":
        return None
    if status_code == 429 and evidence is None:
        evidence = "http_status_429"

    return _RateLimitSignal(
        payload=payload,
        retry_after_sec=retry_after,
        evidence=evidence,
        capacity_period=_int_or_none(payload.get("capacity_period")),
    )


def _nonnegative_number(value: Any, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return int(value) if integer else float(value)


def _structured_request_completed(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("type") != "model.request.completed":
        return None
    producer_id = payload.get("producer_id")
    sequence = _nonnegative_number(payload.get("sequence"), integer=True)
    if (
        not isinstance(producer_id, str)
        or not producer_id
        or len(producer_id) > 128
        or sequence is None
        or sequence < 1
    ):
        return None

    sanitized: dict[str, Any] = {
        "producer_id": producer_id,
        "sequence": sequence,
    }
    for metric in _USAGE_METRICS:
        value = _nonnegative_number(
            payload.get(metric),
            integer=metric != "cost_usd",
        )
        if value is None:
            return None
        sanitized[metric] = value

    for timestamp_field in ("first_response_at", "last_response_at"):
        value = _nonnegative_number(payload.get(timestamp_field))
        if value is None:
            return None
        sanitized[timestamp_field] = value

    buckets = payload.get("buckets")
    if not isinstance(buckets, dict) or len(buckets) > 25:
        return None
    sanitized_buckets: dict[str, dict[str, int]] = {}
    for minute, bucket in buckets.items():
        if not isinstance(minute, str) or not minute.lstrip("-").isdigit():
            return None
        if not isinstance(bucket, dict):
            return None
        tokens = _nonnegative_number(bucket.get("tokens"), integer=True)
        requests = _nonnegative_number(bucket.get("requests"), integer=True)
        if tokens is None or requests is None:
            return None
        sanitized_buckets[minute] = {
            "tokens": tokens,
            "requests": requests,
        }
    sanitized["buckets"] = sanitized_buckets
    return sanitized


_CURRENT_CONTEXT: ContextVar[TelemetryContext | None] = ContextVar(
    "pier_telemetry_context", default=None
)


def current_telemetry_context() -> TelemetryContext | None:
    return _CURRENT_CONTEXT.get()


@contextmanager
def bind_telemetry(context: TelemetryContext) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


class TelemetryDecoder:
    """Incrementally converts a minimal agent telemetry stream into Pier events."""

    def __init__(self, context: TelemetryContext) -> None:
        self._context = context
        self._buffer = ""

    async def feed(self, chunk: str) -> None:
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            await self._decode_line(line)

    async def flush(self) -> None:
        if self._buffer:
            await self._decode_line(self._buffer)
            self._buffer = ""

    async def _decode_line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith(EVENT_PREFIX):
            return
        try:
            payload = json.loads(line[len(EVENT_PREFIX) :])
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        completed = _structured_request_completed(payload)
        if completed is not None:
            await self._context.sink(
                PierEvent(
                    type="model.request.completed",
                    trial_id=self._context.trial_id,
                    provider=self._context.provider,
                    model=self._context.model,
                    effort=self._context.effort,
                    payload=completed,
                )
            )
            return

        signal = _structured_rate_limit(payload)
        if signal is None:
            return

        await self._context.sink(
            PierEvent(
                type="inference.rate_limited",
                trial_id=self._context.trial_id,
                # Group capacity is acquired from the trial configuration, so events
                # must use that same key. Payload values fill only missing context.
                provider=str(
                    self._context.provider or signal.payload.get("provider") or ""
                )
                or None,
                model=str(self._context.model or signal.payload.get("model") or "")
                or None,
                effort=str(self._context.effort or signal.payload.get("effort") or "")
                or None,
                retry_after_sec=signal.retry_after_sec,
                evidence=signal.evidence,
                capacity_period=signal.capacity_period,
                # Do not forward the raw line: model output can contain secrets.
                payload={
                    key: value
                    for key, value in signal.payload.items()
                    if key
                    in {
                        "source",
                        "status_code",
                        "request_id",
                        "attempt",
                        "capacity_period",
                    }
                },
            )
        )


class TelemetrySnapshot:
    """Aggregate cumulative request events into one compact atomic snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._producers: dict[tuple[str, str], dict[str, Any]] = {}
        self._baseline_groups = self._load_baseline_groups()
        self._last_write = 0.0

    def _load_baseline_groups(self) -> list[dict[str, Any]]:
        try:
            document = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            return []
        return document["groups"]

    def record(self, event: PierEvent) -> None:
        if event.type != "model.request.completed":
            return
        producer_id = event.payload["producer_id"]
        sequence = event.payload["sequence"]
        key = (event.trial_id, producer_id)
        previous = self._producers.get(key)
        if previous is not None and previous["sequence"] >= sequence:
            return
        model_name = "/".join(value for value in (event.provider, event.model) if value)
        self._producers[key] = {
            **event.payload,
            "provider": event.provider,
            "model": event.model,
            "model_name": model_name,
            "effort": event.effort,
        }

    @staticmethod
    def _group_key(value: dict[str, Any]) -> tuple[Any, Any, Any]:
        return value["provider"], value["model"], value["effort"]

    def build(self, current_concurrency: int) -> dict[str, Any]:
        groups: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

        def merge_group(source: dict[str, Any]) -> None:
            key = self._group_key(source)
            target = groups.setdefault(
                key,
                {
                    "provider": source["provider"],
                    "model": source["model"],
                    "model_name": source["model_name"],
                    "effort": source["effort"],
                    **{
                        metric: 0.0 if metric == "cost_usd" else 0
                        for metric in _USAGE_METRICS
                    },
                    "buckets": {},
                },
            )
            for metric in _USAGE_METRICS:
                target[metric] += source[metric]
            for minute, bucket in source["buckets"].items():
                merged = target["buckets"].setdefault(
                    str(minute), {"tokens": 0, "requests": 0}
                )
                merged["tokens"] += int(bucket.get("tokens") or 0)
                merged["requests"] += int(bucket.get("requests") or 0)

        for baseline in self._baseline_groups:
            merge_group(baseline)
        for producer in self._producers.values():
            merge_group(producer)

        oldest_minute = int(datetime.now(timezone.utc).timestamp() // 60) - 20
        serialized_groups: list[dict[str, Any]] = []
        totals = {
            metric: 0.0 if metric == "cost_usd" else 0 for metric in _USAGE_METRICS
        }
        totals.update({"recent_tokens": 0, "recent_requests": 0})
        for group in groups.values():
            group["buckets"] = {
                minute: bucket
                for minute, bucket in group["buckets"].items()
                if int(minute) >= oldest_minute
            }
            for metric in _USAGE_METRICS:
                totals[metric] += group[metric]
            group["recent_tokens"] = sum(
                bucket["tokens"] for bucket in group["buckets"].values()
            )
            group["recent_requests"] = sum(
                bucket["requests"] for bucket in group["buckets"].values()
            )
            totals["recent_tokens"] += group["recent_tokens"]
            totals["recent_requests"] += group["recent_requests"]
            serialized_groups.append(group)

        return {
            "schema_version": 1,
            "observed_at": utc_now(),
            "current_concurrency": current_concurrency,
            "totals": totals,
            "groups": sorted(
                serialized_groups,
                key=lambda group: (
                    group["provider"] or "",
                    group["model"] or "",
                    group["effort"] or "",
                ),
            ),
        }

    def _write(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n"
        )
        temporary.replace(self.path)

    async def maybe_write(
        self,
        current_concurrency: int,
        *,
        force: bool = False,
    ) -> None:
        now = asyncio.get_running_loop().time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        try:
            await asyncio.to_thread(self._write, self.build(current_concurrency))
        except OSError as exc:
            logger.getChild(__name__).warning(
                "Failed to write telemetry snapshot %s: %s",
                self.path,
                exc,
            )

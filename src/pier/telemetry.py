from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def _structured_rate_limit(line: str) -> _RateLimitSignal | None:
    try:
        payload = json.loads(line[len(EVENT_PREFIX) :])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
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
        signal = _structured_rate_limit(line)
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

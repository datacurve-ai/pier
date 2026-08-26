from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from pier.concurrency import PierEvent

EVENT_PREFIX = "PIER_EVENT_V1 "
_RATE_LIMIT_RE = re.compile(
    r"(?:\b429\b|too many requests|"
    r"(?:rate[ _-]?limit(?:ed|ing|error)?).{0,80}"
    r"(?:error|exception|retry|backoff|throttl)|"
    r"(?:error|exception|retry|backoff|throttl).{0,80}"
    r"(?:rate[ _-]?limit(?:ed|ing|error)?))",
    re.I,
)
_RETRY_AFTER_RE = re.compile(
    r"retry[- _]?after(?:_ms| milliseconds| ms)?[\"'=:\s]+([0-9]+(?:\.[0-9]+)?)",
    re.I,
)

EventSink = Callable[[PierEvent], Awaitable[None]]


@dataclass(frozen=True)
class TelemetryContext:
    sink: EventSink
    trial_id: str
    provider: str | None
    model: str | None
    effort: str | None


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
    """Incrementally converts a minimal agent control stream into Pier events."""

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
        if not line:
            return

        payload: dict[str, Any] = {}
        event_type = ""
        retry_after: float | None = None

        if line.startswith(EVENT_PREFIX):
            try:
                payload = json.loads(line[len(EVENT_PREFIX) :])
            except (TypeError, ValueError):
                return
            event_type = str(payload.get("type", ""))
            retry_after_value = payload.get("retry_after_sec")
            if retry_after_value is None and payload.get("retry_after_ms") is not None:
                retry_after_value = float(payload["retry_after_ms"]) / 1000
            if retry_after_value is not None:
                retry_after = float(retry_after_value)
        elif _RATE_LIMIT_RE.search(line):
            event_type = "inference.rate_limited"
            match = _RETRY_AFTER_RE.search(line)
            if match:
                retry_after = float(match.group(1))
            payload = {"source": "agent_output_heuristic"}

        if event_type not in {"inference.rate_limited", "model.request.rate_limited"}:
            return

        await self._context.sink(
            PierEvent(
                type="inference.rate_limited",
                trial_id=self._context.trial_id,
                # Route capacity is acquired from the trial configuration, so events
                # must use that same key. Payload values fill only missing context.
                provider=str(self._context.provider or payload.get("provider") or "")
                or None,
                model=str(self._context.model or payload.get("model") or "") or None,
                effort=str(self._context.effort or payload.get("effort") or "")
                or None,
                retry_after_sec=retry_after,
                # Do not forward the raw line: model output can contain secrets.
                payload={
                    key: value
                    for key, value in payload.items()
                    if key
                    in {
                        "source",
                        "status_code",
                        "request_id",
                        "attempt",
                    }
                },
            )
        )

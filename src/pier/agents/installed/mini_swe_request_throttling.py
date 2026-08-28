from __future__ import annotations

import json
import os
import sys
import threading
import time

from minisweagent.models import get_model_class

_EVENT_PREFIX = "PIER_EVENT "


class _RequestGate:
    def __init__(self) -> None:
        # Fail closed until Pier sends the initial pause snapshot.
        self._paused = True
        self._last_control_index = -1
        self._capacity_period = 0
        self._not_before = 0.0
        self._condition = threading.Condition()
        threading.Thread(target=self._read_control, daemon=True).start()

    def _read_control(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            try:
                payload = json.loads(line)
                if payload.get("type") != "request.pause":
                    continue
                control_index = int(payload["control_index"])
                capacity_period = int(payload.get("capacity_period", 0))
                paused = bool(payload["paused"])
                not_before = float(payload.get("not_before") or 0.0)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            with self._condition:
                if control_index <= self._last_control_index:
                    continue
                self._last_control_index = control_index
                self._capacity_period = capacity_period
                self._paused = paused
                self._not_before = not_before
                self._condition.notify_all()

    def wait(self) -> int:
        with self._condition:
            while True:
                delay = self._not_before - time.time()
                if not self._paused and delay <= 0:
                    # Capture the period while holding the same lock used to
                    # apply pause updates. The returned value therefore names
                    # the capacity state under which this request starts.
                    return self._capacity_period
                self._condition.wait(timeout=delay if not self._paused else None)


def _status_code(error: Exception) -> int | None:
    for candidate in (
        error,
        getattr(error, "response", None),
        getattr(error, "__cause__", None),
    ):
        value = getattr(candidate, "status_code", None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _rate_limit_evidence(error: Exception) -> tuple[str, bool] | None:
    if _status_code(error) == 429:
        return "http_status_429", True
    name = type(error).__name__.lower()
    if "ratelimit" in name:
        return "typed_rate_limit_error", True
    message = str(error).lower()
    if (
        "rate limit" in message
        or "too many requests" in message
        or "http 429" in message
    ):
        return "exception_text_heuristic", False
    return None


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _emit_rate_limit(
    error: Exception,
    evidence: str,
    verified: bool,
    capacity_period: int,
) -> None:
    payload = {
        "type": "model.request.rate_limited",
        "evidence": evidence,
        "verified": verified,
        "capacity_period": capacity_period,
    }
    status_code = _status_code(error)
    if status_code is not None:
        payload["status_code"] = status_code
    retry_after = _retry_after(error)
    if retry_after is not None:
        payload["retry_after_sec"] = retry_after
    print(_EVENT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


_GATE = _RequestGate()


class _PierRequestThrottlingMixin:
    def _query(self, messages, **kwargs):
        capacity_period = _GATE.wait()
        try:
            return super()._query(messages, **kwargs)
        except Exception as error:
            classification = _rate_limit_evidence(error)
            if classification is not None:
                _emit_rate_limit(
                    error,
                    *classification,
                    capacity_period=capacity_period,
                )
            raise


_BASE_MODEL_CLASS = get_model_class(
    os.environ.get("PIER_MINISWE_MODEL_NAME", ""),
    os.environ.get("PIER_MINISWE_BASE_MODEL_CLASS", ""),
)


class PierModel(_PierRequestThrottlingMixin, _BASE_MODEL_CLASS):
    pass

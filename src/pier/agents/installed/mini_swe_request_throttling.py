from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from uuid import uuid4

from minisweagent.models import get_model_class

_EVENT_PREFIX = "PIER_EVENT "


class _RequestGate:
    def __init__(self, enabled: bool) -> None:
        self._paused = enabled
        self._last_control_index = -1
        self._capacity_period = 0
        self._not_before = 0.0
        self._condition = threading.Condition()
        if enabled:
            # Dynamic concurrency fails closed until Pier sends its initial state.
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


def _rate_limit_evidence(error: Exception) -> str | None:
    if _status_code(error) == 429:
        return "http_status_429"
    name = type(error).__name__.lower()
    if "ratelimit" in name:
        return "typed_rate_limit_error"
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
    capacity_period: int,
) -> None:
    payload = {
        "type": "model.request.rate_limited",
        "evidence": evidence,
        "capacity_period": capacity_period,
    }
    status_code = _status_code(error)
    if status_code is not None:
        payload["status_code"] = status_code
    retry_after = _retry_after(error)
    if retry_after is not None:
        payload["retry_after_sec"] = retry_after
    print(_EVENT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _mapping(value):
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    try:
        dumped = dict(value)
    except (TypeError, ValueError):
        return {}
    return dumped if isinstance(dumped, dict) else {}


def _number(value, default=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value if math.isfinite(float(value)) else default


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _request_usage(response):
    document = _mapping(response)
    usage = _mapping(document.get("usage"))
    input_tokens = int(
        _number(usage.get("input_tokens"), _number(usage.get("prompt_tokens")))
    )
    output_tokens = int(
        _number(
            usage.get("output_tokens"),
            _number(usage.get("completion_tokens")),
        )
    )
    input_details = _mapping(
        usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    )
    cached_input_tokens = int(
        next(
            (
                value
                for value in (
                    input_details.get("cached_tokens"),
                    usage.get("cache_read_input_tokens"),
                    usage.get("cache_read_tokens"),
                    usage.get("cached_tokens"),
                    usage.get("prompt_cache_hit_tokens"),
                )
                if _number(value) > 0
            ),
            0,
        )
    )

    choices = document.get("choices")
    tool_calls = 0
    if isinstance(choices, list) and choices:
        message = _mapping(_mapping(choices[0]).get("message"))
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            tool_calls = len(calls)
        elif isinstance(message.get("function_call"), dict):
            tool_calls = 1
    else:
        tool_types = {
            "computer_call",
            "custom_tool_call",
            "function_call",
            "image_generation_call",
            "tool_use",
            "web_search_call",
        }
        output = document.get("output")
        if isinstance(output, list):
            tool_calls = sum(
                1 for item in output if _mapping(item).get("type") in tool_types
            )

    cost_usd = next(
        (
            float(value)
            for value in (
                usage.get("cost"),
                _nested(usage, "cost_details", "upstream_inference_cost"),
            )
            if _number(value) > 0
        ),
        0.0,
    )
    return {
        "input_tokens": max(0, input_tokens),
        "cached_input_tokens": max(0, cached_input_tokens),
        "output_tokens": max(0, output_tokens),
        "tool_calls": max(0, tool_calls),
        "cost_usd": max(0.0, cost_usd),
    }


_GATE = _RequestGate(os.environ.get("PIER_REQUEST_THROTTLING") == "1")


class _PierModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pier_producer_id = uuid4().hex
        self._pier_sequence = 0
        self._pier_totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "agent_steps": 0,
            "tool_calls": 0,
            "cost_usd": 0.0,
        }
        self._pier_buckets = {}
        self._pier_first_response_at = None

    def _emit_request_completed(self, response) -> None:
        usage = _request_usage(response)
        if not usage["cost_usd"]:
            try:
                calculated = self._calculate_cost(response)
                usage["cost_usd"] = max(
                    0.0,
                    float(_number(_mapping(calculated).get("cost"))),
                )
            except Exception:
                pass

        completed_at = time.time()
        self._pier_sequence += 1
        self._pier_totals["agent_steps"] = self._pier_sequence
        for metric in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "tool_calls",
            "cost_usd",
        ):
            self._pier_totals[metric] += usage[metric]

        minute = str(int(completed_at // 60))
        bucket = self._pier_buckets.setdefault(
            minute,
            {"tokens": 0, "requests": 0},
        )
        bucket["tokens"] += usage["input_tokens"] + usage["output_tokens"]
        bucket["requests"] += 1
        oldest_minute = int(completed_at // 60) - 20
        self._pier_buckets = {
            key: value
            for key, value in self._pier_buckets.items()
            if int(key) >= oldest_minute
        }
        if self._pier_first_response_at is None:
            self._pier_first_response_at = completed_at

        payload = {
            "type": "model.request.completed",
            "producer_id": self._pier_producer_id,
            "sequence": self._pier_sequence,
            **self._pier_totals,
            "first_response_at": self._pier_first_response_at,
            "last_response_at": completed_at,
            "buckets": self._pier_buckets,
        }
        print(
            _EVENT_PREFIX + json.dumps(payload, separators=(",", ":")),
            flush=True,
        )

    def _query(self, messages, **kwargs):
        capacity_period = _GATE.wait()
        try:
            response = super()._query(messages, **kwargs)
        except Exception as error:
            classification = _rate_limit_evidence(error)
            if classification is not None:
                _emit_rate_limit(
                    error,
                    classification,
                    capacity_period=capacity_period,
                )
            raise
        self._emit_request_completed(response)
        return response


_BASE_MODEL_CLASS = get_model_class(
    os.environ.get("PIER_MINISWE_MODEL_NAME", ""),
    os.environ.get("PIER_MINISWE_BASE_MODEL_CLASS", ""),
)


class PierModel(_PierModelMixin, _BASE_MODEL_CLASS):
    pass

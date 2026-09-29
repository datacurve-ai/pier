#!/usr/bin/env python3
"""Reproduce sanitized GitHub Copilot CLI JSONL fixtures.

The committed fixtures in ``tests/fixtures/copilot_cli`` were produced from
local, real Copilot CLI event streams under ``~/.copilot/session-state``. This
script keeps that process reproducible while replacing private prompts, code,
paths, repository names, URLs, and token-like values with deterministic
synthetic data. Generic passthrough fields such as model, name, code, and type
use narrow value allowlists; remaining preserved schema labels still rely on
the leak checker for review.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import ipaddress
import json
import math
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_STRING_LENGTH = 400
ANCHOR_TIMESTAMP = datetime(1970, 1, 2, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    session_id: str
    start_line: int
    end_line: int
    truncate_last_line: bool = False

    @property
    def filename(self) -> str:
        return f"{self.name}.jsonl"


FIXTURES = [
    FixtureSpec(
        name="session_basic",
        session_id="3dc5ebd6-5bf4-4f86-8fba-85d46aa02e40",
        start_line=1,
        end_line=80,
    ),
    FixtureSpec(
        name="session_subagents",
        session_id="6e538374-2c67-4921-b6a8-4c53467ae366",
        start_line=1,
        end_line=149,
    ),
    FixtureSpec(
        name="session_compaction",
        session_id="b29b99d5-b945-4e27-a5bb-3bc4717e9b93",
        start_line=2605,
        end_line=2713,
    ),
    FixtureSpec(
        name="session_timeout",
        session_id="15d4b196-6679-4d9d-bd9d-2a990d6840a7",
        start_line=1,
        end_line=90,
        truncate_last_line=True,
    ),
    FixtureSpec(
        name="session_model_change",
        session_id="b9d91594-4c60-486a-9fab-0e08a88e23fc",
        start_line=1,
        end_line=148,
    ),
    FixtureSpec(
        name="session_system_events",
        session_id="b29b99d5-b945-4e27-a5bb-3bc4717e9b93",
        start_line=5607,
        end_line=5699,
    ),
]

ID_KEYS = {
    "agentId",
    "apiCallId",
    "checkpointPath",
    "clientRequestId",
    "id",
    "interactionId",
    "messageId",
    "parentId",
    "parentToolCallId",
    "reasoningId",
    "requestId",
    "serviceRequestId",
    "sessionId",
    "toolCallId",
    "turnId",
}
TIMESTAMP_KEYS = {"sessionStartTime", "startTime", "timestamp"}
OPAQUE_KEYS = {"encryptedContent", "reasoningOpaque"}
ARGUMENT_KEYS = {"arguments", "shellToolInfo"}
PRESERVE_STRING_KEYS = {
    "agentDisplayName",
    "agentMode",
    "agentName",
    "code",
    "contextTier",
    "copilotVersion",
    "currentModel",
    "delivery",
    "errorType",
    "mcpServerName",
    "mcpToolName",
    "model",
    "name",
    "newModel",
    "phase",
    "previousModel",
    "previousReasoningEffort",
    "producer",
    "reasoningEffort",
    "role",
    "rte",
    "shutdownType",
    "toolName",
    "type",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(
    r"(?:https?|ssh)://|(?:git@|[A-Za-z0-9._%+-]+@)[A-Za-z0-9.-]+:",
    re.IGNORECASE,
)
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9._-]+/?)+")
PLAIN_OBJECT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.(?P<fraction>\d{1,6}))?(?P<zone>Z|[+-]\d{2}:\d{2})?$"
)
CREDENTIAL_RE = re.compile(
    r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|sk-|sk-ant-|AIza|AKIA|xox[abprs]-)"
    r"[A-Za-z0-9_\-=]{8,}"
)
JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
IPV4_RE = re.compile(
    r"(?<![0-9.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?![0-9.])"
)
IPV6_RE = re.compile(
    r"(?<![A-Fa-f0-9:])"
    r"(?:"
    r"(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,7}:"
    r"|:(?::[A-Fa-f0-9]{1,4}){1,7}"
    r"|::1"
    r")"
    r"(?![A-Fa-f0-9:])"
)
WINDOWS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|\\\\[A-Za-z0-9_.-]+\\)[^\"'\r\n]+"
)
HOSTNAME_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z][A-Za-z0-9-]{1,62}"
    r"(?![A-Za-z0-9_-])"
)

MODEL_KEYS = {"model", "currentModel", "newModel", "previousModel"}
MODEL_REPLACEMENTS = {
    "claude-opus-4.7-1m-internal": "synthetic-cloud-model-1m",
    "qwen3-coder:30b-256k": "local-model-alpha:30b-256k",
    "qwen3-coder:30b-q8-128k": "local-model-beta:30b-q8-128k",
}
ALLOWED_MODEL_VALUES = {
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-sonnet-4.6",
    "synthetic-b706d4c",
    *MODEL_REPLACEMENTS.values(),
}
AGENT_NAME_REPLACEMENTS = {"Squad": "custom-agent"}
AGENT_DISPLAY_NAME_REPLACEMENTS = {"Squad": "Custom Agent"}
STRICT_PRESERVE_VALUE_ALLOWLIST = {
    "code": {"failure"},
    "name": {
        "ask_user",
        "bash",
        "create",
        "edit",
        "glob",
        "list_agents",
        "read_bash",
        "report_intent",
        "sql",
        "synt",
        "synthetic-54",
        "task",
        "view",
    },
    "type": {
        "abort",
        "assistant.message",
        "assistant.turn_end",
        "assistant.turn_start",
        "function",
        "hook.end",
        "hook.start",
        "permission.completed",
        "permission.requested",
        "session.compaction_complete",
        "session.compaction_start",
        "session.info",
        "session.model_change",
        "session.permissions_changed",
        "session.resume",
        "session.shutdown",
        "session.start",
        "shell_completed",
        "subagent.completed",
        "subagent.selected",
        "subagent.started",
        "system.message",
        "system.notification",
        "tool.execution_complete",
        "tool.execution_start",
        "user.message",
    },
}
HOSTNAME_LIKE_VALUE_PREFIX_ALLOWLIST = (
    "assistant.",
    "hook.",
    "permission.",
    "session.",
    "subagent.",
    "system.",
    "tool.",
    "user.",
)


class Sanitizer:
    def __init__(self, fixture_name: str, events: list[dict[str, Any]]) -> None:
        if not events:
            raise ValueError(f"{fixture_name}: no events selected")
        self.fixture_name = fixture_name
        self.first_timestamp = self._parse_timestamp(events[0]["timestamp"])

    def sanitize_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._sanitize_value(event, None) for event in events]

    def _sanitize_value(self, value: Any, key: str | None) -> Any:
        if isinstance(value, dict):
            if key == "result":
                return self._sanitize_tool_result(value)
            if key in ARGUMENT_KEYS:
                return self._sanitize_argument_value(value)
            return {
                self._sanitize_object_key(item_key, key): self._sanitize_value(
                    item_value,
                    item_key,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            if key in ARGUMENT_KEYS:
                return self._sanitize_argument_value(value)
            return [self._sanitize_value(item, key) for item in value]
        if isinstance(value, str):
            return self._sanitize_string(value, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in TIMESTAMP_KEYS:
                return self._shift_epoch(value, key)
        return value

    def _sanitize_object_key(self, value: Any, parent_key: str | None) -> str:
        if not isinstance(value, str):
            return str(value)
        if parent_key == "modelMetrics":
            return self._sanitize_model_name(value, parent_key)
        return value

    def _sanitize_string(self, value: str, key: str | None) -> str:
        if key in ARGUMENT_KEYS:
            return self._sanitize_argument_string(value)
        if key == "checkpointPath":
            return self._synthetic_path(value, "checkpoints")
        if key in ID_KEYS:
            return self._synthetic_id(value, key)
        if key in TIMESTAMP_KEYS or ISO_RE.match(value):
            parsed = self._try_parse_timestamp(value)
            if parsed is not None:
                return self._format_shifted_timestamp(parsed, value)
        if key in OPAQUE_KEYS:
            return f"<redacted:{self._digest(value, 8)}>"
        if key == "source":
            return self._sanitize_source(value)
        if key in PRESERVE_STRING_KEYS:
            return self._sanitize_preserved_string(value, key)
        if self._looks_like_absolute_path(value):
            return self._synthetic_path(value, "files")
        return self._synthetic_text(value)

    def _sanitize_preserved_string(self, value: str, key: str) -> str:
        if key in MODEL_KEYS:
            return self._sanitize_model_name(value, key)
        if key == "agentName":
            if value in AGENT_NAME_REPLACEMENTS:
                return AGENT_NAME_REPLACEMENTS[value]
            if value in {"custom-agent", "general-purpose", "task"}:
                return value
            raise ValueError(f"{self.fixture_name}: unrecognized agentName value")
        if key == "agentDisplayName":
            if value in AGENT_DISPLAY_NAME_REPLACEMENTS:
                return AGENT_DISPLAY_NAME_REPLACEMENTS[value]
            if value in {"Custom Agent", "General Purpose Agent", "Task Agent"}:
                return value
            raise ValueError(f"{self.fixture_name}: unrecognized agentDisplayName value")
        allowed_values = STRICT_PRESERVE_VALUE_ALLOWLIST.get(key)
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"{self.fixture_name}: unrecognized {key} value")
        return value

    def _sanitize_model_name(self, value: str, key: str | None) -> str:
        replacement = MODEL_REPLACEMENTS.get(value)
        if replacement is not None:
            return replacement
        if value in ALLOWED_MODEL_VALUES:
            return value
        raise ValueError(
            f"{self.fixture_name}: unrecognized model identifier for {key or 'key'}"
        )

    def _sanitize_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        content = result.get("content")
        detailed = result.get("detailedContent")
        content_text: str | None = None
        detailed_text: str | None = None

        if isinstance(content, str):
            content_cap = MAX_STRING_LENGTH
            if (
                isinstance(detailed, str)
                and content
                and content in detailed
                and content != detailed
            ):
                detailed_target = min(len(detailed), MAX_STRING_LENGTH)
                content_cap = max(1, detailed_target - 1)
            content_text = self._synthetic_text(content, max_length=content_cap)

        if isinstance(detailed, str):
            if isinstance(content, str) and detailed == content:
                detailed_text = content_text or ""
            elif isinstance(content, str) and content and content in detailed:
                if content_text is None:
                    content_text = self._synthetic_text(content)
                detailed_text = self._containing_text(detailed, content, content_text)
            else:
                detailed_text = self._synthetic_text(detailed)

        sanitized: dict[str, Any] = {}
        for item_key, item_value in result.items():
            if item_key == "content" and isinstance(item_value, str):
                sanitized[item_key] = content_text or ""
            elif item_key == "detailedContent" and isinstance(item_value, str):
                sanitized[item_key] = detailed_text or ""
            elif item_key == "displayContent" and isinstance(item_value, str):
                if isinstance(content, str) and item_value == content:
                    sanitized[item_key] = content_text or ""
                elif isinstance(detailed, str) and item_value == detailed:
                    sanitized[item_key] = detailed_text or ""
                else:
                    sanitized[item_key] = self._synthetic_text(item_value)
            else:
                sanitized[item_key] = self._sanitize_value(item_value, item_key)
        return sanitized

    def _sanitize_argument_string(self, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return self._synthetic_text(value)

        sanitized = self._sanitize_argument_value(parsed)
        rendered = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(rendered) <= MAX_STRING_LENGTH:
            return rendered
        return self._synthetic_text(value)

    def _sanitize_argument_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self._sanitize_argument_key(item_key): self._sanitize_argument_value(
                    item_value
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_argument_value(item) for item in value]
        if isinstance(value, str):
            if self._looks_like_absolute_path(value):
                return self._synthetic_path(value, "files")
            return self._synthetic_text(value)
        return value

    def _sanitize_argument_key(self, value: Any) -> str:
        key = str(value)
        if not PLAIN_OBJECT_KEY_RE.fullmatch(key):
            raise ValueError(f"{self.fixture_name}: unsafe argument object key")
        return key

    def _sanitize_source(self, value: str) -> str:
        if value.startswith("schedule-"):
            return "schedule-1"
        if value.startswith("skill-"):
            return "skill-sanitized"
        if value.startswith("agent-"):
            return f"agent-{self._synthetic_id(value.removeprefix('agent-'), 'agentId')}"
        if "-" in value:
            return value.split("-", 1)[0]
        return self._synthetic_text(value)

    def _containing_text(
        self,
        detailed: str,
        content: str,
        sanitized_content: str,
    ) -> str:
        target_length = min(len(detailed), MAX_STRING_LENGTH)
        if len(sanitized_content) >= target_length:
            sanitized_content = sanitized_content[: max(1, target_length - 1)]
        index = detailed.index(content)
        remaining = target_length - len(sanitized_content)
        prefix_raw = detailed[:index]
        suffix_raw = detailed[index + len(content) :]
        prefix_length = min(len(prefix_raw), remaining)
        suffix_length = remaining - prefix_length
        if prefix_length == 0 and suffix_length == 0:
            suffix_length = 1
        prefix = self._synthetic_text(
            prefix_raw or "prefix",
            max_length=prefix_length,
            namespace="prefix",
        )
        suffix = self._synthetic_text(
            suffix_raw or "suffix",
            max_length=suffix_length,
            namespace="suffix",
        )
        return f"{prefix}{sanitized_content}{suffix}"[:MAX_STRING_LENGTH]

    def _synthetic_text(
        self,
        value: str,
        max_length: int = MAX_STRING_LENGTH,
        namespace: str = "text",
    ) -> str:
        if not value or max_length <= 0:
            return ""
        target_length = min(len(value), max_length)
        seed = self._digest(f"{namespace}:{value}", 12)
        alphabet = (
            "synthetic "
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        )
        text = f"synthetic-{seed} {alphabet}"
        while len(text) < target_length:
            text += alphabet
        return text[:target_length]

    def _synthetic_path(self, value: str, group: str) -> str:
        suffix = Path(value).suffix
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix):
            suffix = ".txt"
        return f"/workspace/{group}/path_{self._digest(value, 12)}{suffix}"

    def _synthetic_id(self, value: str, key: str | None) -> str:
        prefix = {
            "agentId": "call",
            "apiCallId": "api",
            "clientRequestId": "req",
            "id": "evt",
            "interactionId": "int",
            "messageId": "msg",
            "parentId": "evt",
            "parentToolCallId": "call",
            "reasoningId": "reason",
            "requestId": "req",
            "serviceRequestId": "req",
            "sessionId": "session",
            "toolCallId": "call",
            "turnId": "turn",
        }.get(key or "", "id")
        return f"{prefix}_{self._digest(value, 16)}"

    def _digest(self, value: str, length: int) -> str:
        material = f"{self.fixture_name}\0{value}".encode()
        return hashlib.sha256(material).hexdigest()[:length]

    def _format_shifted_timestamp(self, timestamp: datetime, original: str) -> str:
        shifted = ANCHOR_TIMESTAMP + (timestamp - self.first_timestamp)
        shifted = shifted.astimezone(timezone.utc)
        match = ISO_RE.match(original)
        precision = len(match.group("fraction")) if match and match.group("fraction") else 3
        base = shifted.strftime("%Y-%m-%dT%H:%M:%S")
        fraction = f"{shifted.microsecond:06d}"[:precision].ljust(precision, "0")
        return f"{base}.{fraction}Z"

    def _shift_epoch(self, value: int | float, key: str | None) -> int | float:
        scale = self._epoch_scale(value, key)
        first_units = self._timestamp_units(self.first_timestamp, scale)
        anchor_units = self._timestamp_units(ANCHOR_TIMESTAMP, scale)
        shifted = value + (anchor_units - first_units)
        if isinstance(value, int):
            return int(shifted)
        return float(shifted)

    def _epoch_scale(self, value: int | float, key: str | None) -> int:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{self.fixture_name}: invalid numeric timestamp for {key}")
        if 1e8 <= numeric < 1e11:
            return 1
        if 1e11 <= numeric < 1e14:
            return 1_000
        if 1e14 <= numeric < 1e17:
            return 1_000_000
        raise ValueError(
            f"{self.fixture_name}: unsupported numeric timestamp magnitude for {key}"
        )

    @staticmethod
    def _timestamp_units(timestamp: datetime, scale: int) -> int:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = timestamp - epoch
        total_microseconds = (
            ((delta.days * 24 * 60 * 60) + delta.seconds) * 1_000_000
            + delta.microseconds
        )
        if scale == 1_000_000:
            return total_microseconds
        divisor = 1_000_000 // scale
        return int(round(total_microseconds / divisor))

    def _try_parse_timestamp(self, value: str) -> datetime | None:
        try:
            return self._parse_timestamp(value)
        except ValueError:
            return None

    def _parse_timestamp(self, value: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _looks_like_absolute_path(self, value: str) -> bool:
        return bool(ABSOLUTE_PATH_RE.search(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanitize local Copilot CLI events into committed fixtures.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("~/.copilot/session-state").expanduser(),
        help="Copilot session-state directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/copilot_cli"),
        help="Output directory for sanitized JSONL fixtures.",
    )
    return parser.parse_args()


def load_events(source: Path, spec: FixtureSpec) -> list[dict[str, Any]]:
    path = source / spec.session_id / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{spec.name}: missing source events file: {path}")

    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < spec.start_line:
                continue
            if line_number > spec.end_line:
                break
            events.append(json.loads(line))

    expected = spec.end_line - spec.start_line + 1
    if len(events) != expected:
        raise ValueError(f"{spec.name}: expected {expected} events, got {len(events)}")
    return events


def render_jsonl(events: list[dict[str, Any]], truncate_last_line: bool) -> str:
    lines = [
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        for event in events
    ]
    if truncate_last_line:
        lines[-1] = lines[-1][: max(1, len(lines[-1]) // 2)]
        return "\n".join(lines)
    return "\n".join(lines) + "\n"


def write_fixture(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_features(spec: FixtureSpec, events: list[dict[str, Any]]) -> None:
    types = [event["type"] for event in events]
    validators = {
        "session_basic": validate_basic,
        "session_subagents": validate_subagents,
        "session_compaction": validate_compaction,
        "session_timeout": validate_timeout,
        "session_model_change": validate_model_change,
        "session_system_events": validate_system_events,
    }
    validators[spec.name](events, types)


def validate_basic(events: list[dict[str, Any]], types: list[str]) -> None:
    shutdown = events[-1].get("data", {})
    if events[0]["type"] != "session.start" or events[-1]["type"] != "session.shutdown":
        raise ValueError("session_basic must start with session.start and end shutdown")
    if "user.message" not in types or "assistant.turn_start" not in types:
        raise ValueError("session_basic is missing core turn events")
    if not any(event.get("data", {}).get("toolRequests") for event in events):
        raise ValueError("session_basic is missing assistant tool requests")
    if not shutdown.get("modelMetrics") or "totalNanoAiu" not in shutdown:
        raise ValueError("session_basic shutdown is missing usage metrics")
    assert_tool_matches(events)


def validate_subagents(events: list[dict[str, Any]], types: list[str]) -> None:
    if events[-1]["type"] != "session.shutdown":
        raise ValueError("session_subagents must end with session.shutdown")
    task_calls = task_tool_calls(events)
    started = subagent_tool_calls(events, "subagent.started")
    completed = subagent_tool_calls(events, "subagent.completed")
    if not task_calls or not started or not completed:
        raise ValueError("session_subagents is missing task or subagent events")
    if not task_calls & started:
        raise ValueError("session_subagents task call does not match subagent start")
    if not started & completed:
        raise ValueError("session_subagents start does not match completion")
    if not any(event.get("agentId") for event in events):
        raise ValueError("session_subagents is missing agentId-tagged nested events")


def validate_compaction(events: list[dict[str, Any]], types: list[str]) -> None:
    if "session.compaction_start" not in types or "session.compaction_complete" not in types:
        raise ValueError("session_compaction is missing compaction events")
    if not any(
        event["type"] == "session.compaction_complete"
        and event.get("data", {}).get("success") is False
        for event in events
    ):
        raise ValueError("session_compaction should include a failed compaction")


def validate_timeout(events: list[dict[str, Any]], types: list[str]) -> None:
    if "session.shutdown" in types:
        raise ValueError("session_timeout must not contain session.shutdown")
    if events[0]["type"] != "session.start":
        raise ValueError("session_timeout must start with session.start")


def validate_model_change(events: list[dict[str, Any]], types: list[str]) -> None:
    if events[0]["type"] != "session.start":
        raise ValueError("session_model_change must start with session.start")
    changes = [
        (
            event.get("data", {}).get("newModel"),
            event.get("data", {}).get("reasoningEffort"),
            event.get("data", {}).get("contextTier"),
        )
        for event in events
        if event["type"] == "session.model_change"
    ]
    if len(set(changes)) < 2:
        raise ValueError("session_model_change needs multiple distinct model changes")
    change_positions = [
        index for index, event in enumerate(events) if event["type"] == "session.model_change"
    ]
    assistant_positions = [
        index for index, event in enumerate(events) if event["type"] == "assistant.message"
    ]
    if not any(
        any(position < change for position in assistant_positions)
        and any(position > change for position in assistant_positions)
        for change in change_positions
    ):
        raise ValueError("session_model_change needs assistant messages around a change")


def validate_system_events(events: list[dict[str, Any]], types: list[str]) -> None:
    required = {
        "hook.end",
        "hook.start",
        "permission.completed",
        "permission.requested",
        "session.info",
        "system.message",
        "system.notification",
    }
    missing = required - set(types)
    if missing:
        raise ValueError(f"session_system_events missing: {sorted(missing)}")
    if "abort" not in types and "session.error" not in types:
        raise ValueError("session_system_events needs abort or session.error")


def task_tool_calls(events: list[dict[str, Any]]) -> set[str]:
    calls: set[str] = set()
    for event in events:
        if event["type"] != "assistant.message":
            continue
        for request in event.get("data", {}).get("toolRequests") or []:
            if request.get("name") == "task" and isinstance(request.get("toolCallId"), str):
                calls.add(request["toolCallId"])
    return calls


def subagent_tool_calls(events: list[dict[str, Any]], event_type: str) -> set[str]:
    return {
        event.get("data", {}).get("toolCallId")
        for event in events
        if event["type"] == event_type and isinstance(event.get("data", {}).get("toolCallId"), str)
    }


def assert_tool_matches(events: list[dict[str, Any]]) -> None:
    requested: set[str] = set()
    starts: set[str] = set()
    completes: set[str] = set()
    for event in events:
        data = event.get("data", {})
        if event["type"] == "assistant.message":
            requested.update(
                request["toolCallId"]
                for request in data.get("toolRequests") or []
                if isinstance(request.get("toolCallId"), str)
            )
        elif event["type"] == "tool.execution_start" and isinstance(
            data.get("toolCallId"), str
        ):
            starts.add(data["toolCallId"])
        elif event["type"] == "tool.execution_complete" and isinstance(
            data.get("toolCallId"), str
        ):
            completes.add(data["toolCallId"])
    if not requested <= starts or not requested <= completes:
        raise ValueError("tool request/start/complete ids do not match")


def build_leak_patterns() -> list[tuple[str, re.Pattern[str]]]:
    identity_values = collect_identity_values()
    if not identity_values:
        raise ValueError("could not determine local identity for leak checks")

    leak_patterns = [
        ("URL", URL_RE),
        ("email address", EMAIL_RE),
        ("credential-like token", CREDENTIAL_RE),
        ("JWT", JWT_RE),
        ("IPv4 address", IPV4_RE),
        ("Windows path", WINDOWS_PATH_RE),
    ]
    for label, value in identity_values:
        leak_patterns.append((label, re.compile(identity_regex(value), re.IGNORECASE)))
    return leak_patterns


def collect_identity_values() -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, value: str | None) -> None:
        if value is None:
            return
        normalized = value.strip()
        if not normalized:
            return
        if len(normalized) < 3 and not any(char in normalized for char in ".@/\\"):
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        values.append((label, normalized))

    try:
        add("local username", getpass.getuser())
    except Exception:
        pass
    add("home directory name", Path.home().name)
    add("hostname", socket.gethostname())

    git_user_name = git_config_value("user.name")
    add("git user name", git_user_name)
    for token in identity_tokens(git_user_name):
        add("git user name component", token)

    git_user_email = git_config_value("user.email")
    add("git user email", git_user_email)
    if git_user_email and "@" in git_user_email:
        local_part = git_user_email.split("@", 1)[0]
        add("git email local part", local_part)
        for token in identity_tokens(local_part):
            add("git email component", token)

    for label, value in remote_origin_values(git_config_value("remote.origin.url")):
        add(label, value)

    return values


def git_config_value(name: str) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "config", "--get", name],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def identity_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    ignored = {"com", "git", "github", "noreply", "org", "users"}
    return [
        token
        for token in re.split(r"[^A-Za-z0-9_]+", value)
        if len(token) >= 3 and token.casefold() not in ignored
    ]


def remote_origin_values(remote_url: str | None) -> list[tuple[str, str]]:
    if not remote_url:
        return []
    host: str | None = None
    path = ""
    sanitized_url = re.sub(r"//[^/@]+@", "//", remote_url.strip())
    parsed = urlparse(sanitized_url)
    if parsed.scheme:
        host = parsed.hostname
        path = parsed.path
    elif "@" in sanitized_url and ":" in sanitized_url.rsplit("@", 1)[1]:
        after_user = sanitized_url.rsplit("@", 1)[1]
        host, path = after_user.split(":", 1)
    else:
        path = sanitized_url

    values: list[tuple[str, str]] = []
    if host:
        values.append(("git remote hostname", host))

    cleaned_path = path.strip("/")
    if cleaned_path.endswith(".git"):
        cleaned_path = cleaned_path[:-4]
    if cleaned_path:
        values.append(("git remote path", cleaned_path))
        for component in cleaned_path.split("/"):
            for token in identity_tokens(component):
                values.append(("git remote path component", token))
    return values


def identity_regex(value: str) -> str:
    escaped = re.escape(value)
    if re.fullmatch(r"[A-Za-z0-9_]+", value):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def leak_check(output: dict[FixtureSpec, str]) -> None:
    problems: list[str] = []
    leak_patterns = build_leak_patterns()

    for spec, text in output.items():
        for label, pattern in leak_patterns:
            if pattern.search(text):
                problems.append(f"{spec.filename}: leaked {label}")
        complete_lines = text.splitlines()
        if spec.truncate_last_line and complete_lines:
            complete_lines = complete_lines[:-1]
        for line_number, line in enumerate(complete_lines, start=1):
            event = json.loads(line)
            check_value_for_leaks(spec.filename, line_number, event, problems)
        check_raw_strings(spec.filename, text, problems)

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(f"leak check failed:\n  - {joined}")


def check_value_for_leaks(
    filename: str,
    line_number: int,
    value: Any,
    problems: list[str],
) -> None:
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            check_key_for_leaks(filename, line_number, item_key, problems)
            check_value_for_leaks(filename, line_number, item_value, problems)
    elif isinstance(value, list):
        for item in value:
            check_value_for_leaks(filename, line_number, item, problems)
    elif isinstance(value, str):
        check_string_for_leaks(filename, line_number, "string", value, problems)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if 1e11 <= value < 1e14:
            problems.append(f"{filename}:{line_number}: unshifted epoch-like value")


def check_key_for_leaks(
    filename: str,
    line_number: int,
    key: Any,
    problems: list[str],
) -> None:
    if not isinstance(key, str):
        problems.append(f"{filename}:{line_number}: non-string object key")
        return
    check_string_for_leaks(filename, line_number, "object key", key, problems)
    if not PLAIN_OBJECT_KEY_RE.fullmatch(key):
        problems.append(f"{filename}:{line_number}: unsafe object key")


def check_string_for_leaks(
    filename: str,
    line_number: int,
    label: str,
    value: str,
    problems: list[str],
) -> None:
    if len(value) > MAX_STRING_LENGTH:
        problems.append(
            f"{filename}:{line_number}: {label} length {len(value)} exceeds "
            f"{MAX_STRING_LENGTH}"
        )
    for match in ABSOLUTE_PATH_RE.finditer(value):
        path = match.group(0).rstrip("/")
        if not (path.startswith("/workspace/") or path.startswith("/home/agent/")):
            problems.append(f"{filename}:{line_number}: unsafe absolute path")
    if WINDOWS_PATH_RE.search(value):
        problems.append(f"{filename}:{line_number}: unsafe Windows path")
    if contains_ipv6_address(value):
        problems.append(f"{filename}:{line_number}: IPv6 address")
    for match in HOSTNAME_RE.finditer(value):
        hostname = match.group(0)
        if not hostname_like_value_is_allowed(hostname):
            problems.append(f"{filename}:{line_number}: bare hostname")


def hostname_like_value_is_allowed(value: str) -> bool:
    return value.startswith(HOSTNAME_LIKE_VALUE_PREFIX_ALLOWLIST)


def contains_ipv6_address(value: str) -> bool:
    for match in IPV6_RE.finditer(value):
        try:
            parsed = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if parsed.version == 6:
            return True
    return False


def check_raw_strings(filename: str, text: str, problems: list[str]) -> None:
    for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text):
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        check_string_for_leaks(filename, 0, "raw string", decoded, problems)


def main() -> int:
    args = parse_args()
    source = args.source.expanduser()
    out_dir = args.out
    rendered: dict[FixtureSpec, str] = {}
    sanitized_events_by_spec: dict[FixtureSpec, list[dict[str, Any]]] = {}

    for spec in FIXTURES:
        events = load_events(source, spec)
        sanitizer = Sanitizer(spec.name, events)
        sanitized = sanitizer.sanitize_events(events)
        validate_features(spec, sanitized)
        rendered[spec] = render_jsonl(sanitized, spec.truncate_last_line)
        sanitized_events_by_spec[spec] = sanitized

    leak_check(rendered)

    for spec, text in rendered.items():
        write_fixture(out_dir / spec.filename, text)
        suffix = " (last line truncated)" if spec.truncate_last_line else ""
        print(
            f"{spec.session_id} -> {spec.filename}: "
            f"{len(sanitized_events_by_spec[spec])} events{suffix}"
        )
    print("Leak check passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

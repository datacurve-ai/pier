"""Reusable ATIF v1.7 invariant assertions for agent trajectory conversions.

Pier emits an *augmented* ATIF v1.7: on top of the schema itself it promises
strict one step per API turn, strict reasoning versus agent message
separation, no fabricated assistant text, and honest `llm_call_count`,
`peak_context_tokens` and `summarization_count` values. Field-by-field
assertions in individual tests do not catch violations of those global
guarantees, so conversion tests call :func:`assert_valid_atif` on every
trajectory they build.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pier.models.trajectories import Trajectory
from pier.models.trajectories.step import Step

FABRICATED_MESSAGES = frozenset({"Tool call", "Tool result", "Tool use"})
FABRICATED_PREFIXES = ("Executed ",)


def assert_valid_atif(trajectory: Trajectory) -> None:
    """Assert a trajectory satisfies ATIF v1.7 and pier's augmented rules."""
    payload = trajectory.to_json_dict()
    assert Trajectory.model_validate(payload).to_json_dict() == payload, (
        "serialized trajectory does not round-trip through ATIF validation"
    )
    _assert_trajectory(trajectory, label="trajectory")


def assert_one_step_per_api_call(trajectory: Trajectory) -> None:
    """Assert no upstream API call is split across several agent steps.

    Producers record the upstream call identifier in ``step.extra`` under
    ``api_call_id``. Two agent steps sharing one is the "one step per API
    turn" violation pier's augmented ATIF forbids -- unless a turn boundary
    separates them, because providers only promise call ids unique within a
    turn and some (e.g. locally hosted OpenAI-compatible servers) hand out
    sequential ``chatcmpl-N`` ids that are reused once a tool round-trip has
    completed. A step carrying an observation, a step no inference produced
    (``llm_call_count`` 0, e.g. a user-dispatched tool), or any non-agent step
    marks such a boundary. The check is per trajectory, since ids are also
    reused across a parent and its subagents.
    """
    for label, current in _each_trajectory(trajectory, label="trajectory"):
        seen: dict[str, int] = {}
        for step in current.steps:
            if step.source != "agent":
                seen.clear()
                continue
            call_id = (step.extra or {}).get("api_call_id")
            if call_id:
                assert call_id not in seen, (
                    f"{label}: api_call_id {call_id!r} is split across steps "
                    f"{seen[call_id]} and {step.step_id}"
                )
                seen[call_id] = step.step_id
            if step.observation is not None or step.llm_call_count == 0:
                # A completed tool round-trip, or a step the user dispatched
                # without an inference, ends the API call the id referred to,
                # so the id may legitimately reappear after it.
                seen.clear()


def _assert_trajectory(trajectory: Trajectory, *, label: str) -> None:
    assert trajectory.schema_version == "ATIF-v1.7", (
        f"{label}: unexpected schema version {trajectory.schema_version}"
    )
    assert trajectory.agent.name, f"{label}: agent name is required"
    assert trajectory.agent.version, f"{label}: agent version is required"
    assert trajectory.steps, f"{label}: a trajectory must have at least one step"

    for index, step in enumerate(trajectory.steps):
        assert step.step_id == index + 1, (
            f"{label}: step ids must be sequential from 1, "
            f"got {step.step_id} at position {index}"
        )
        _assert_step(step, label=f"{label}.steps[{index}]")

    _assert_timestamps_non_decreasing(trajectory.steps, label=label)
    _assert_tool_call_ids_unique(trajectory.steps, label=label)
    _assert_final_metrics(trajectory, label=label)
    _assert_subagents(trajectory, label=label)


def _assert_step(step: Step, *, label: str) -> None:
    if step.source != "agent":
        for field in (
            "model_name",
            "reasoning_effort",
            "reasoning_content",
            "tool_calls",
            "metrics",
            "llm_call_count",
        ):
            assert getattr(step, field) is None, (
                f"{label}: {field} is only valid on agent steps, "
                f"but source is {step.source!r}"
            )
    else:
        assert step.llm_call_count is not None, (
            f"{label}: agent steps must declare llm_call_count"
        )
        if step.llm_call_count == 0:
            assert step.metrics is None and step.reasoning_content is None, (
                f"{label}: deterministic dispatch steps carry no LLM fields"
            )

    message = step.message
    if isinstance(message, str):
        assert message not in FABRICATED_MESSAGES, (
            f"{label}: {message!r} is fabricated assistant text"
        )
        assert not message.startswith(FABRICATED_PREFIXES), (
            f"{label}: {message!r} is fabricated assistant text"
        )
        if step.reasoning_content:
            assert message.strip() != step.reasoning_content.strip(), (
                f"{label}: reasoning content leaked into the visible message"
            )

    if step.timestamp is not None:
        _parse_timestamp(step.timestamp, label=label)

    if step.metrics is not None:
        for field in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            value = getattr(step.metrics, field, None)
            assert value is None or value >= 0, (
                f"{label}: metrics.{field} must not be negative, got {value}"
            )

    if step.observation is not None:
        tool_call_ids = {call.tool_call_id for call in step.tool_calls or []}
        for result in step.observation.results:
            if result.source_call_id is not None:
                assert result.source_call_id in tool_call_ids, (
                    f"{label}: observation references unknown call "
                    f"{result.source_call_id!r}"
                )


def _assert_timestamps_non_decreasing(steps: list[Step], *, label: str) -> None:
    previous: datetime | None = None
    previous_id: int | None = None
    for step in steps:
        if step.timestamp is None:
            continue
        current = _parse_timestamp(step.timestamp, label=label)
        if previous is not None:
            assert current >= previous, (
                f"{label}: timestamps must not go backwards between steps "
                f"{previous_id} and {step.step_id}"
            )
        previous, previous_id = current, step.step_id


def _assert_tool_call_ids_unique(steps: list[Step], *, label: str) -> None:
    seen: dict[str, int] = {}
    for step in steps:
        for call in step.tool_calls or []:
            if not call.tool_call_id:
                continue
            assert call.tool_call_id not in seen, (
                f"{label}: tool_call_id {call.tool_call_id!r} is reused by "
                f"steps {seen[call.tool_call_id]} and {step.step_id}"
            )
            seen[call.tool_call_id] = step.step_id


def _assert_final_metrics(trajectory: Trajectory, *, label: str) -> None:
    metrics = trajectory.final_metrics
    if metrics is None:
        return

    if metrics.total_steps is not None:
        assert metrics.total_steps == len(trajectory.steps), (
            f"{label}: total_steps {metrics.total_steps} does not match "
            f"{len(trajectory.steps)} steps"
        )
    for field in (
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
    ):
        value = getattr(metrics, field)
        assert value is None or value >= 0, (
            f"{label}: {field} must not be negative, got {value}"
        )
    if (
        metrics.total_cached_tokens is not None
        and metrics.total_prompt_tokens is not None
    ):
        assert metrics.total_cached_tokens <= metrics.total_prompt_tokens, (
            f"{label}: cached tokens are a subset of the prompt total"
        )

    extra: dict[str, Any] = metrics.extra or {}
    if (peak := extra.get("peak_context_tokens")) is not None:
        assert isinstance(peak, int) and peak > 0, (
            f"{label}: peak_context_tokens must be a positive int, got {peak!r}"
        )
    if (count := extra.get("summarization_count")) is not None:
        assert isinstance(count, int) and count >= 0, (
            f"{label}: summarization_count must be a non-negative int, "
            f"got {count!r}"
        )


def _assert_subagents(trajectory: Trajectory, *, label: str) -> None:
    subagents = trajectory.subagent_trajectories or []
    for index, subagent in enumerate(subagents):
        assert subagent.trajectory_id, (
            f"{label}: embedded subagents must set trajectory_id"
        )
        _assert_trajectory(subagent, label=f"{label}.subagent_trajectories[{index}]")

    known_ids = {subagent.trajectory_id for subagent in subagents}
    for step in trajectory.steps:
        if step.observation is None:
            continue
        for result in step.observation.results:
            for ref in result.subagent_trajectory_ref or []:
                assert ref.trajectory_path or ref.trajectory_id in known_ids, (
                    f"{label}: step {step.step_id} references unresolvable "
                    f"subagent trajectory {ref.trajectory_id!r}"
                )


def _each_trajectory(
    trajectory: Trajectory, *, label: str
) -> list[tuple[str, Trajectory]]:
    found = [(label, trajectory)]
    for index, subagent in enumerate(trajectory.subagent_trajectories or []):
        found.extend(
            _each_trajectory(
                subagent, label=f"{label}.subagent_trajectories[{index}]"
            )
        )
    return found


def _parse_timestamp(value: str, *, label: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - guarded by ATIF validation
        raise AssertionError(f"{label}: invalid ISO 8601 timestamp {value!r}") from exc

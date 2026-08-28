from __future__ import annotations

import asyncio
import json
import time

from pier.telemetry import concurrency_group_fields, log_concurrency_event

REQUEST_RESUME_WINDOW_SEC = 10.0

AgentPauseMessageQueue = asyncio.Queue[str | None]


class RequestThrottlingManager:
    """Pause model requests for in-flight rollouts that are beyond capacity."""

    def __init__(
        self,
        capacity: int,
        *,
        provider: str | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._members: dict[str, AgentPauseMessageQueue | None] = {}
        self._running: set[str] = set()
        self._control_index = 0
        self._capacity_period = 0
        self._not_before = 0.0
        self._lock = asyncio.Lock()
        self._concurrency_group = concurrency_group_fields(provider, model, effort)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def running(self) -> int:
        return len(self._running)

    @property
    def paused(self) -> int:
        return sum(
            pause_messages is not None and trial_id not in self._running
            for trial_id, pause_messages in self._members.items()
        )

    @property
    def control_index(self) -> int:
        return self._control_index

    @property
    def capacity_period(self) -> int:
        return self._capacity_period

    def _desired_running(self) -> set[str]:
        # Rollouts without a control queue cannot be paused, so reserve capacity
        # for them before selecting controllable rollouts in registration order.
        uncontrollable = {
            trial_id
            for trial_id, pause_messages in self._members.items()
            if pause_messages is None
        }
        available = max(0, self._capacity - len(uncontrollable))
        controllable = [
            trial_id
            for trial_id, pause_messages in self._members.items()
            if pause_messages is not None
        ]
        return uncontrollable | set(controllable[:available])

    def _serialize_pause_message(
        self, trial_id: str, *, resume_not_before: float = 0.0
    ) -> str:
        return (
            json.dumps(
                {
                    "type": "request.pause",
                    "control_index": self._control_index,
                    "capacity_period": self._capacity_period,
                    "paused": trial_id not in self._running,
                    "not_before": max(self._not_before, resume_not_before),
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    def _reconcile(
        self,
        *,
        force: set[str] | None = None,
        stagger_resumes: bool = False,
        reason: str,
        cause_id: str | None = None,
        removed_trial_ids: list[str] | None = None,
    ) -> None:
        previous = self._running
        desired = self._desired_running()
        state_changed = previous.symmetric_difference(desired)
        notified = state_changed | (force or set())
        resumed = [
            trial_id
            for trial_id in self._members
            if trial_id in desired and trial_id not in previous
        ]

        self._control_index += 1
        self._running = desired
        resume_times: dict[str, float] = {}
        if stagger_resumes and resumed:
            started_at = time.time()
            divisor = max(1, len(resumed) - 1)
            resume_times = {
                trial_id: started_at + REQUEST_RESUME_WINDOW_SEC * index / divisor
                for index, trial_id in enumerate(resumed)
            }
        for trial_id in notified:
            pause_messages = self._members.get(trial_id)
            if pause_messages is not None:
                pause_messages.put_nowait(
                    self._serialize_pause_message(
                        trial_id,
                        resume_not_before=resume_times.get(trial_id, 0.0),
                    )
                )

        newly_running = [
            trial_id
            for trial_id in self._members
            if trial_id in state_changed and trial_id in desired
        ]
        newly_paused = [
            trial_id
            for trial_id, pause_messages in self._members.items()
            if pause_messages is not None
            and trial_id in state_changed
            and trial_id not in desired
        ]
        if notified or removed_trial_ids:
            log_concurrency_event(
                "request.pause_changed",
                concurrency_group=self._concurrency_group,
                reason=reason,
                cause_id=cause_id,
                control_index=self._control_index,
                capacity_period=self._capacity_period,
                capacity=self._capacity,
                running=len(self._running),
                paused=self.paused,
                resumed_trial_ids=newly_running,
                paused_trial_ids=newly_paused,
                removed_trial_ids=removed_trial_ids or [],
                resume_not_before={
                    trial_id: max(
                        self._not_before,
                        resume_times.get(trial_id, 0.0),
                    )
                    for trial_id in newly_running
                    if max(
                        self._not_before,
                        resume_times.get(trial_id, 0.0),
                    )
                    > 0
                },
            )

    async def register(
        self,
        trial_id: str,
        *,
        pause_messages: AgentPauseMessageQueue | None,
    ) -> None:
        async with self._lock:
            if trial_id in self._members:
                raise ValueError(f"Trial {trial_id!r} is already registered")
            self._members[trial_id] = pause_messages
            self._reconcile(force={trial_id}, reason="trial_registered")

    async def unregister(self, trial_id: str) -> None:
        async with self._lock:
            if trial_id not in self._members:
                return
            del self._members[trial_id]
            self._running.discard(trial_id)
            # If a running rollout exits, resume the oldest paused rollout.
            self._reconcile(
                reason="trial_unregistered",
                removed_trial_ids=[trial_id],
            )

    async def resize(
        self,
        capacity: int,
        *,
        cause_id: str | None = None,
        capacity_period: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        async with self._lock:
            previous = self._capacity
            if capacity == previous:
                return
            if capacity_period is None:
                self._capacity_period += 1
            elif capacity_period <= self._capacity_period:
                raise ValueError("capacity_period must increase on every resize")
            else:
                self._capacity_period = capacity_period
            self._capacity = capacity
            self._reconcile(
                # Every controlled rollout must learn the new capacity period even
                # if its pause state did not change. Requests started afterward then
                # carry proof that they began after this resize.
                force=set(self._members),
                stagger_resumes=capacity > previous,
                reason="capacity_resized",
                cause_id=cause_id,
            )

    async def apply_retry_after(
        self, delay_sec: float, *, cause_id: str | None = None
    ) -> None:
        """Temporarily hold controlled request gates until Retry-After expires."""
        if delay_sec <= 0:
            return
        async with self._lock:
            self._not_before = max(self._not_before, time.time() + delay_sec)
            self._control_index += 1
            for trial_id, pause_messages in self._members.items():
                if pause_messages is not None:
                    pause_messages.put_nowait(self._serialize_pause_message(trial_id))
            log_concurrency_event(
                "request.retry_after_applied",
                concurrency_group=self._concurrency_group,
                cause_id=cause_id,
                control_index=self._control_index,
                capacity_period=self._capacity_period,
                retry_after_sec=delay_sec,
                not_before=self._not_before,
                capacity=self._capacity,
                running=len(self._running),
                paused=self.paused,
                controlled_trial_ids=[
                    trial_id
                    for trial_id, pause_messages in self._members.items()
                    if pause_messages is not None
                ],
            )

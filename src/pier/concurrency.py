from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from pier.request_throttling import (
    AgentPauseMessageQueue,
    RequestThrottlingManager,
)
from pier.telemetry import (
    EventBus,
    PierEvent,
    concurrency_group_fields as _concurrency_group,
    log_concurrency_event as _log_concurrency_event,
    new_event_id as _new_id,
)
from pier.utils.logger import logger

RAMP_UP_PERCENT = 10.0
BACKOFF_PERCENT = 33.0
BACKOFF_SIGNAL_THRESHOLD = 5
RECOVERY_CEILING_PERCENT = 90.0
STABILITY_WINDOW_SEC = 300.0
COOLDOWN_SEC = 300.0


@dataclass(frozen=True)
class ConcurrencySnapshot:
    """Point-in-time running/capacity state for one concurrency limiter."""

    running: int
    capacity: int
    queued: int
    paused: int = 0
    provider: str | None = None
    model: str | None = None
    effort: str | None = None


class ResizableLimiter:
    """Concurrency limiter whose capacity can change without cancelling holders."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._admitted = 0
        self._queued = 0
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def admitted(self) -> int:
        return self._admitted

    @property
    def queued(self) -> int:
        return self._queued

    async def resize(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        async with self._condition:
            previous = self._capacity
            self._capacity = capacity
            if capacity > previous:
                available = max(0, capacity - self._admitted)
                self._condition.notify(min(self._queued, available))

    async def acquire(self) -> None:
        async with self._condition:
            self._queued += 1
            try:
                while self._admitted >= self._capacity:
                    try:
                        await self._condition.wait()
                    except asyncio.CancelledError:
                        # If this waiter had been selected for an available slot,
                        # pass that opportunity to the next waiter.
                        if self._admitted < self._capacity:
                            self._condition.notify(1)
                        raise
                self._admitted += 1
            finally:
                self._queued -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._admitted == 0:
                raise RuntimeError("ResizableLimiter released without an acquisition")
            self._admitted -= 1
            if self._queued and self._admitted < self._capacity:
                self._condition.notify(1)

    async def __aenter__(self) -> "ResizableLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.release()


class DynamicConcurrencyController:
    """Adjust one concurrency group in response to observed rate-limit pressure."""

    def __init__(
        self,
        *,
        limiter: ResizableLimiter,
        request_throttling_manager: RequestThrottlingManager,
        provider: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        ramp_interval_sec: float = STABILITY_WINDOW_SEC,
        cooldown_sec: float = COOLDOWN_SEC,
    ) -> None:
        self.limiter = limiter
        self.request_throttling_manager = request_throttling_manager
        self.ramp_interval_sec = ramp_interval_sec
        self.cooldown_sec = cooldown_sec
        self.recovery_floor = limiter.capacity
        self.known_bad_capacity: int | None = None
        self.current_probe_capacity: int | None = None
        self._last_reduction_period: int | None = None
        self._rate_limit_count = 0
        self._backoff_allowed_at = 0.0
        self._next_probe_at = time.monotonic() + ramp_interval_sec
        self._logger = logger.getChild(__name__)
        self._concurrency_group = _concurrency_group(provider, model, effort)

    def _state(self) -> dict[str, Any]:
        return {
            "running": self.request_throttling_manager.running,
            "capacity": self.limiter.capacity,
            "queued": self.limiter.queued,
            "paused": self.request_throttling_manager.paused,
            "recovery_floor": self.recovery_floor,
            "known_bad_capacity": self.known_bad_capacity,
            "recovery_ceiling": self.recovery_ceiling,
            "current_probe_capacity": self.current_probe_capacity,
            "capacity_period": self.request_throttling_manager.capacity_period,
            "control_index": self.request_throttling_manager.control_index,
            "last_reduction_period": self._last_reduction_period,
            "rate_limit_count": self._rate_limit_count,
            "backoff_signal_threshold": BACKOFF_SIGNAL_THRESHOLD,
        }

    @property
    def recovery_ceiling(self) -> int | None:
        """Highest recovery capacity allowed below the smallest known-bad value."""
        if self.known_bad_capacity is None:
            return None
        ceiling = math.floor(self.known_bad_capacity * RECOVERY_CEILING_PERCENT / 100)
        if self.known_bad_capacity > 1:
            ceiling = min(ceiling, self.known_bad_capacity - 1)
        return max(1, ceiling)

    def audit_initialized(self) -> None:
        _log_concurrency_event(
            "concurrency.group_initialized",
            concurrency_group=self._concurrency_group,
            **self._state(),
        )

    def audit_finalized(self) -> None:
        _log_concurrency_event(
            "concurrency.group_finalized",
            concurrency_group=self._concurrency_group,
            **self._state(),
        )

    def _event_group(self, event: PierEvent) -> dict[str, Any]:
        if any(self._concurrency_group.values()):
            return self._concurrency_group
        return _concurrency_group(event.provider, event.model, event.effort)

    def _audit_rate_limit(
        self,
        event: PierEvent,
        *,
        decision_id: str,
        action: str,
        capacity_before: int,
        trigger_rate_limit_count: int,
        **fields: Any,
    ) -> None:
        signal = {
            key: value
            for key, value in event.payload.items()
            if key in {"source", "status_code", "request_id", "attempt"}
        }
        _log_concurrency_event(
            "inference.rate_limit",
            event_id=event.event_id,
            decision_id=decision_id,
            observed_at=event.observed_at,
            evidence=event.evidence,
            trial_id=event.trial_id,
            concurrency_group=self._event_group(event),
            action=action,
            capacity_before=capacity_before,
            capacity_after=self.limiter.capacity,
            event_capacity_period=event.capacity_period,
            trigger_rate_limit_count=trigger_rate_limit_count,
            retry_after_sec=event.retry_after_sec,
            signal=signal,
            **fields,
            **self._state(),
        )

    def _ignored_action(self, event: PierEvent) -> str | None:
        if self._last_reduction_period is None:
            return None
        if event.capacity_period is None:
            return "ignored_missing_capacity_period"
        if event.capacity_period < self._last_reduction_period:
            return "ignored_stale_capacity_period"
        if event.capacity_period > self.request_throttling_manager.capacity_period:
            return "ignored_unknown_capacity_period"
        return None

    def _audit_ramp(self, *, decision_id: str, capacity_before: int) -> None:
        _log_concurrency_event(
            "concurrency.ramp",
            decision_id=decision_id,
            concurrency_group=self._concurrency_group,
            capacity_before=capacity_before,
            capacity_after=self.limiter.capacity,
            **self._state(),
        )

    async def _resize(self, target: int, *, cause_id: str | None = None) -> None:
        """Apply one target to admission and request-pause enforcement."""
        current = self.limiter.capacity
        if target == current:
            return
        next_period = self.request_throttling_manager.capacity_period + 1
        if target < current:
            # Stop new admission before pausing additional in-flight rollouts.
            await self.limiter.resize(target)
            await self.request_throttling_manager.resize(
                target,
                cause_id=cause_id,
                capacity_period=next_period,
            )
        else:
            # Resume paused work before admitting new rollouts.
            await self.request_throttling_manager.resize(
                target,
                cause_id=cause_id,
                capacity_period=next_period,
            )
            await self.limiter.resize(target)
        self._rate_limit_count = 0
        if target < current:
            self._last_reduction_period = next_period

    def _backoff_target(self, current: int) -> int:
        self.known_bad_capacity = (
            current
            if self.known_bad_capacity is None
            else min(self.known_bad_capacity, current)
        )
        reduction = max(1, math.floor(current * BACKOFF_PERCENT / 100 + 0.5))
        percentage_target = max(1, current - reduction)

        failed_probe = (
            self.current_probe_capacity == current or current > self.recovery_floor
        )
        self.current_probe_capacity = None
        if failed_probe:
            return self.recovery_floor

        self.recovery_floor = min(self.recovery_floor, percentage_target)
        return percentage_target

    async def handle(self, event: PierEvent, *, now: float | None = None) -> None:
        if event.type != "inference.rate_limited":
            return

        now = time.monotonic() if now is None else now
        current = self.limiter.capacity
        decision_id = _new_id("decision")

        if ignored_action := self._ignored_action(event):
            self._audit_rate_limit(
                event,
                decision_id=decision_id,
                action=ignored_action,
                capacity_before=current,
                trigger_rate_limit_count=self._rate_limit_count,
                required_capacity_period=self._last_reduction_period,
            )
            return

        delay = max(self.cooldown_sec, event.retry_after_sec or 0.0)
        # A fresh rate-limit signal restarts the quiet window, even when the next
        # capacity reduction is still cooling down.
        self._next_probe_at = max(
            self._next_probe_at,
            now + max(self.ramp_interval_sec, delay),
        )
        self._rate_limit_count = min(
            self._rate_limit_count + 1,
            BACKOFF_SIGNAL_THRESHOLD,
        )

        trigger_rate_limit_count = self._rate_limit_count
        threshold_reached = self._rate_limit_count >= BACKOFF_SIGNAL_THRESHOLD
        action = (
            "backoff_interval_hold"
            if threshold_reached
            else "backoff_signal_accumulating"
        )
        if threshold_reached and now >= self._backoff_allowed_at:
            target = self._backoff_target(current)
            if target != current:
                await self._resize(target, cause_id=decision_id)
                action = "backoff"
                self._logger.warning(
                    "Rate limit detected for %s/%s; reducing trial concurrency %s -> %s",
                    event.provider or "unknown",
                    event.model or "unknown",
                    current,
                    target,
                )
            else:
                action = "capacity_floor_hold"
                self._rate_limit_count = 0
            self._backoff_allowed_at = now + delay

        if event.retry_after_sec:
            await self.request_throttling_manager.apply_retry_after(
                event.retry_after_sec,
                cause_id=event.event_id,
            )
        self._audit_rate_limit(
            event,
            decision_id=decision_id,
            action=action,
            capacity_before=current,
            trigger_rate_limit_count=trigger_rate_limit_count,
            cooldown_sec=delay,
        )

    async def maybe_ramp(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now < self._next_probe_at:
            return False

        current = self.limiter.capacity
        paused = self.request_throttling_manager.paused
        probe_survived = (
            self.current_probe_capacity == current or current > self.recovery_floor
        )
        if probe_survived and (
            self.limiter.admitted >= current or self.limiter.queued or paused
        ):
            self.recovery_floor = current
            self.current_probe_capacity = None

        ceiling = self.recovery_ceiling
        if (
            ceiling is not None
            and current >= ceiling
            and self.current_probe_capacity is None
        ):
            self._next_probe_at = now + self.ramp_interval_sec
            return False

        if self.limiter.queued == 0 and paused == 0:
            self._next_probe_at = now + self.ramp_interval_sec
            return False
        if self.limiter.admitted < current and paused == 0:
            self._next_probe_at = now + self.ramp_interval_sec
            return False

        increase = max(1, math.ceil(current * RAMP_UP_PERCENT / 100))
        target = current + increase
        if ceiling is not None:
            target = min(target, ceiling)

        decision_id = _new_id("decision")
        await self._resize(target, cause_id=decision_id)
        self.current_probe_capacity = target
        self._logger.info(
            "No rate limits observed; increasing trial concurrency to %s",
            self.limiter.capacity,
        )
        self._audit_ramp(
            decision_id=decision_id,
            capacity_before=current,
        )
        self._next_probe_at = now + self.ramp_interval_sec
        return True


ConcurrencyGroupKey = tuple[str | None, str | None, str | None]


class DynamicConcurrencyPool:
    """Own an independent adaptive controller for each concurrency group."""

    def __init__(
        self,
        *,
        events: EventBus,
        initial_capacity: int,
        ramp_interval_sec: float = STABILITY_WINDOW_SEC,
        cooldown_sec: float = COOLDOWN_SEC,
    ) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")
        self.events = events
        self.initial_capacity = initial_capacity
        self.ramp_interval_sec = ramp_interval_sec
        self.cooldown_sec = cooldown_sec
        self._controllers: dict[ConcurrencyGroupKey, DynamicConcurrencyController] = {}

    def _controller_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> DynamicConcurrencyController:
        key = (provider, model, effort)
        controller = self._controllers.get(key)
        if controller is not None:
            return controller

        limiter = ResizableLimiter(self.initial_capacity)
        request_throttling_manager = RequestThrottlingManager(
            self.initial_capacity,
            provider=provider,
            model=model,
            effort=effort,
        )
        controller = DynamicConcurrencyController(
            limiter=limiter,
            request_throttling_manager=request_throttling_manager,
            provider=provider,
            model=model,
            effort=effort,
            ramp_interval_sec=self.ramp_interval_sec,
            cooldown_sec=self.cooldown_sec,
        )
        self._controllers[key] = controller
        controller.audit_initialized()
        return controller

    def limiter_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> ResizableLimiter:
        return self._controller_for(provider, model, effort).limiter

    def request_throttling_manager_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> RequestThrottlingManager:
        return self._controller_for(provider, model, effort).request_throttling_manager

    async def register_trial(
        self,
        provider: str | None,
        model: str | None,
        effort: str | None,
        *,
        trial_id: str,
        pause_messages: AgentPauseMessageQueue | None,
    ) -> None:
        await self.request_throttling_manager_for(provider, model, effort).register(
            trial_id,
            pause_messages=pause_messages,
        )

    async def unregister_trial(
        self,
        provider: str | None,
        model: str | None,
        effort: str | None,
        *,
        trial_id: str,
    ) -> None:
        await self.request_throttling_manager_for(provider, model, effort).unregister(
            trial_id
        )

    def capacity_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> int:
        return self.limiter_for(provider, model, effort).capacity

    def snapshots(self) -> list[ConcurrencySnapshot]:
        """Return a display-oriented snapshot of every known concurrency group."""
        snapshots = [
            ConcurrencySnapshot(
                running=controller.request_throttling_manager.running,
                capacity=controller.limiter.capacity,
                queued=controller.limiter.queued,
                paused=controller.request_throttling_manager.paused,
                provider=provider,
                model=model,
                effort=effort,
            )
            for (provider, model, effort), controller in self._controllers.items()
        ]
        return sorted(
            snapshots,
            key=lambda item: (
                item.provider or "",
                item.model or "",
                item.effort or "",
            ),
        )

    async def run(self) -> None:
        poll_interval = min(1.0, max(0.05, self.ramp_interval_sec))
        while True:
            try:
                event = await asyncio.wait_for(
                    self.events.next(), timeout=poll_interval
                )
            except TimeoutError:
                for controller in tuple(self._controllers.values()):
                    await controller.maybe_ramp()
                continue
            if event is None:
                for controller in self._controllers.values():
                    controller.audit_finalized()
                return
            controller = self._controller_for(
                event.provider,
                event.model,
                event.effort,
            )
            await controller.handle(event)

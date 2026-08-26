from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from pier.utils.logger import logger

DYNAMIC_CONCURRENCY_RAMP_UP_PERCENT = 10.0
DYNAMIC_CONCURRENCY_BACKOFF_PERCENT = 20.0


@dataclass(frozen=True)
class PierEvent:
    """A small, generic control-plane event emitted by a running trial."""

    type: str
    trial_id: str
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
    retry_after_sec: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


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


class ResizableLimiter:
    """Concurrency limiter whose capacity can change without cancelling holders."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._in_use = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def waiting(self) -> int:
        return self._waiting

    async def resize(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        async with self._condition:
            self._capacity = capacity
            self._condition.notify_all()

    async def acquire(self) -> None:
        async with self._condition:
            self._waiting += 1
            try:
                await self._condition.wait_for(
                    lambda: self._in_use < self._capacity
                )
                self._in_use += 1
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._in_use == 0:
                raise RuntimeError("ResizableLimiter released without an acquisition")
            self._in_use -= 1
            self._condition.notify_all()

    async def __aenter__(self) -> "ResizableLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.release()


class DynamicConcurrencyController:
    """A conservative percentage-based controller for rollout concurrency."""

    def __init__(
        self,
        *,
        limiter: ResizableLimiter,
        events: EventBus,
        ramp_interval_sec: float = 30.0,
        cooldown_sec: float = 30.0,
    ) -> None:
        self.limiter = limiter
        self.events = events
        self.ramp_interval_sec = ramp_interval_sec
        self.cooldown_sec = cooldown_sec
        self.last_known_good = limiter.capacity
        self.last_known_bad: int | None = None
        self._cooldown_until = 0.0
        self._next_ramp_at = time.monotonic() + ramp_interval_sec
        self._logger = logger.getChild(__name__)

    async def handle(self, event: PierEvent, *, now: float | None = None) -> None:
        if event.type != "inference.rate_limited":
            return

        now = time.monotonic() if now is None else now
        current = self.limiter.capacity
        # A provider commonly returns a burst of 429s to concurrent requests. Treat
        # that burst as one pressure signal instead of decrementing once per request.
        if now >= self._cooldown_until:
            self.last_known_bad = (
                current
                if self.last_known_bad is None
                else min(self.last_known_bad, current)
            )
            percentage_target = max(
                1,
                current
                - math.ceil(current * DYNAMIC_CONCURRENCY_BACKOFF_PERCENT / 100),
            )
            if current > self.last_known_good:
                # A known-good capacity is a floor for the first response to a
                # speculative ramp. The policy backoff may require more than one
                # pressure period to return to it.
                target = max(self.last_known_good, percentage_target)
            else:
                target = percentage_target
                self.last_known_good = min(self.last_known_good, target)

            if target != current:
                await self.limiter.resize(target)
                self._logger.warning(
                    "Rate limit detected for %s/%s; reducing trial concurrency %s -> %s",
                    event.provider or "unknown",
                    event.model or "unknown",
                    current,
                    target,
                )

        delay = max(self.cooldown_sec, event.retry_after_sec or 0.0)
        self._cooldown_until = max(self._cooldown_until, now + delay)
        self._next_ramp_at = self._cooldown_until + self.ramp_interval_sec

    async def maybe_ramp(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now < self._cooldown_until or now < self._next_ramp_at:
            return False
        if self.limiter.waiting == 0 or self.limiter.in_use < self.limiter.capacity:
            self._next_ramp_at = now + self.ramp_interval_sec
            return False

        current = self.limiter.capacity
        self.last_known_good = current
        increase = max(
            1, math.ceil(current * DYNAMIC_CONCURRENCY_RAMP_UP_PERCENT / 100)
        )
        target = current + increase
        if self.last_known_bad is not None:
            gap = self.last_known_bad - current
            if gap <= 1:
                self._next_ramp_at = now + self.ramp_interval_sec
                return False
            # Continue percentage-based growth when far from the bound, but switch
            # to midpoint probes as we approach it. The target always remains
            # strictly below the lowest capacity that produced pressure.
            midpoint = current + max(1, gap // 2)
            target = min(target, midpoint, self.last_known_bad - 1)

        await self.limiter.resize(target)
        self._logger.info(
            "No rate limits observed; increasing trial concurrency to %s",
            self.limiter.capacity,
        )
        self._next_ramp_at = now + self.ramp_interval_sec
        return True

    async def run(self) -> None:
        while True:
            timeout = max(0.05, self._next_ramp_at - time.monotonic())
            try:
                event = await asyncio.wait_for(self.events.next(), timeout=timeout)
            except TimeoutError:
                await self.maybe_ramp()
                continue
            if event is None:
                return
            await self.handle(event)


RouteKey = tuple[str | None, str | None, str | None]


class DynamicConcurrencyPool:
    """Owns an independent adaptive limiter for each inference route."""

    def __init__(
        self,
        *,
        events: EventBus,
        initial_capacity: int,
        ramp_interval_sec: float = 30.0,
        cooldown_sec: float = 30.0,
    ) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")
        self.events = events
        self.initial_capacity = initial_capacity
        self.ramp_interval_sec = ramp_interval_sec
        self.cooldown_sec = cooldown_sec
        self._controllers: dict[RouteKey, DynamicConcurrencyController] = {}

    @staticmethod
    def route_key(
        provider: str | None, model: str | None, effort: str | None
    ) -> RouteKey:
        return provider, model, effort

    def limiter_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> ResizableLimiter:
        key = self.route_key(provider, model, effort)
        controller = self._controllers.get(key)
        if controller is None:
            limiter = ResizableLimiter(self.initial_capacity)
            controller = DynamicConcurrencyController(
                limiter=limiter,
                # The pool consumes the shared bus and dispatches directly.
                events=self.events,
                ramp_interval_sec=self.ramp_interval_sec,
                cooldown_sec=self.cooldown_sec,
            )
            self._controllers[key] = controller
        return controller.limiter

    def capacity_for(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> int:
        return self.limiter_for(provider, model, effort).capacity

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
                return
            key = self.route_key(event.provider, event.model, event.effort)
            controller = self._controllers.get(key)
            if controller is None:
                # A signal can race slightly ahead of route registration in custom
                # agents. Registering here keeps the event bus generic.
                self.limiter_for(*key)
                controller = self._controllers[key]
            await controller.handle(event)

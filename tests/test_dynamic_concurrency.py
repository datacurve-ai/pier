import asyncio
import json

from pier.concurrency import (
    DYNAMIC_CONCURRENCY_BACKOFF_PERCENT,
    DYNAMIC_CONCURRENCY_RAMP_UP_PERCENT,
    DynamicConcurrencyController,
    DynamicConcurrencyPool,
    EventBus,
    PierEvent,
    ResizableLimiter,
)
from pier.models.job.config import JobConfig
from pier.telemetry import EVENT_PREFIX, TelemetryContext, TelemetryDecoder
from pier.trial.queue import TrialQueue


def run(coro):
    return asyncio.run(coro)


def test_fixed_concurrency_remains_the_default():
    config = JobConfig(n_concurrent_trials=7)
    queue = TrialQueue(n_concurrent=config.n_concurrent_trials)

    assert config.dynamic_concurrency is False
    assert queue.concurrency_limit == 7
    assert queue._dynamic_pool is None


def test_dynamic_concurrency_uses_configured_value_as_route_start_capacity():
    queue = TrialQueue(n_concurrent=7, dynamic_concurrency=True)

    assert queue.concurrency_limit == 7
    assert queue.route_concurrency_limit("openai", "gpt-5", "high") == 7
    assert queue._dynamic_pool is not None
    assert queue._dynamic_pool.initial_capacity == 7
    assert DYNAMIC_CONCURRENCY_RAMP_UP_PERCENT == 10
    assert DYNAMIC_CONCURRENCY_BACKOFF_PERCENT == 20


def test_resizing_down_does_not_cancel_current_holders():
    async def scenario():
        limiter = ResizableLimiter(2)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.resize(1)

        acquired = asyncio.Event()

        async def wait_for_slot():
            async with limiter:
                acquired.set()

        waiter = asyncio.create_task(wait_for_slot())
        await asyncio.sleep(0)
        assert limiter.in_use == 2
        assert not acquired.is_set()

        await limiter.release()
        await asyncio.sleep(0)
        assert limiter.in_use == 1
        assert not acquired.is_set()

        await limiter.release()
        await asyncio.wait_for(acquired.wait(), timeout=1)
        await waiter

    run(scenario())


def test_controller_probes_up_then_returns_to_last_known_good_on_429():
    async def scenario():
        limiter = ResizableLimiter(2)
        bus = EventBus()
        controller = DynamicConcurrencyController(
            limiter=limiter,
            events=bus,
            ramp_interval_sec=1,
            cooldown_sec=10,
        )

        await limiter.acquire()
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert limiter.waiting == 1

        assert await controller.maybe_ramp(now=10**9)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 3
        assert controller.last_known_good == 2

        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="trial-1",
                provider="openai",
                model="gpt-5",
            ),
            now=10**9 + 1,
        )
        assert limiter.capacity == 2
        assert limiter.in_use == 3

        # A burst from requests that were already in flight is one pressure event.
        await controller.handle(
            PierEvent(type="inference.rate_limited", trial_id="trial-2"),
            now=10**9 + 2,
        )
        assert limiter.capacity == 2

        await limiter.release()
        await limiter.release()
        await limiter.release()

    run(scenario())


def test_controller_uses_percentage_steps_at_high_concurrency():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = DynamicConcurrencyController(
            limiter=limiter,
            events=EventBus(),
            ramp_interval_sec=1,
        )

        for _ in range(100):
            await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        assert await controller.maybe_ramp(now=10**9)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 110
        assert controller.last_known_good == 100

        # The first pressure response returns to the stable capacity instead of
        # overshooting the 20% target below it.
        await controller.handle(
            PierEvent(type="inference.rate_limited", trial_id="trial-1"),
            now=10**9 + 1,
        )
        assert limiter.capacity == 100
        assert controller.last_known_bad == 110

        # Recovery uses a bounded probe and never returns to the known-bad value.
        await limiter.release()
        bounded_waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert await controller.maybe_ramp(now=10**9 + 33)
        await asyncio.wait_for(bounded_waiter, timeout=1)
        assert limiter.capacity == 105
        assert limiter.capacity < controller.last_known_bad

        for _ in range(101):
            await limiter.release()

    run(scenario())


def test_pressure_at_current_capacity_sets_permanent_bad_bound():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = DynamicConcurrencyController(
            limiter=limiter,
            events=EventBus(),
        )

        await controller.handle(
            PierEvent(type="inference.rate_limited", trial_id="trial-1"),
            now=10**9,
        )

        assert controller.last_known_bad == 100
        assert limiter.capacity == 80

    run(scenario())


def test_dynamic_pool_isolates_provider_model_effort_routes():
    async def scenario():
        bus = EventBus()
        pool = DynamicConcurrencyPool(
            events=bus,
            initial_capacity=1,
            ramp_interval_sec=1,
        )
        high = pool.limiter_for("openai", "gpt-5", "high")
        low = pool.limiter_for("openai", "gpt-5", "low")
        await high.resize(3)
        await low.resize(4)
        pool._controllers[("openai", "gpt-5", "high")].last_known_good = 2

        task = asyncio.create_task(pool.run())
        await bus.publish(
            PierEvent(
                type="inference.rate_limited",
                trial_id="trial-high",
                provider="openai",
                model="gpt-5",
                effort="high",
            )
        )
        for _ in range(10):
            if high.capacity == 2:
                break
            await asyncio.sleep(0)
        await bus.close()
        await task

        assert high.capacity == 2
        assert low.capacity == 4

    run(scenario())


def test_telemetry_decoder_accepts_protocol_and_text_fallback():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)

        context = TelemetryContext(
            sink=sink,
            trial_id="trial-1",
            provider="openai",
            model="gpt-5",
            effort="high",
        )
        decoder = TelemetryDecoder(context)
        record = {
            "type": "model.request.rate_limited",
            "retry_after_ms": 2500,
            "status_code": 429,
        }
        encoded = EVENT_PREFIX + json.dumps(record) + "\n"
        await decoder.feed(encoded[:12])
        await decoder.feed(encoded[12:])
        await decoder.feed("litellm.RateLimitError: HTTP 429 retry-after: 4\n")
        await decoder.feed("normal verbose model output\n")
        await decoder.flush()

        assert len(events) == 2
        assert events[0].retry_after_sec == 2.5
        assert events[0].provider == "openai"
        assert events[0].effort == "high"
        assert events[1].retry_after_sec == 4
        assert events[1].payload == {"source": "agent_output_heuristic"}

    run(scenario())

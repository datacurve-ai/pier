import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from pier.concurrency import (
    BACKOFF_PERCENT,
    BACKOFF_SIGNAL_THRESHOLD,
    COOLDOWN_SEC,
    DynamicConcurrencyController,
    DynamicConcurrencyPool,
    RAMP_UP_PERCENT,
    RECOVERY_CEILING_PERCENT,
    ResizableLimiter,
    STABILITY_WINDOW_SEC,
)
from pier.job import _ConcurrencyColumn
from pier.models.environment_type import EnvironmentType
from pier.models.job.config import JobConfig
from pier.models.trial.config import AgentConfig, EnvironmentConfig
from pier.request_throttling import (
    REQUEST_RESUME_WINDOW_SEC,
    RequestThrottlingManager,
)
from pier.telemetry import (
    CONCURRENCY_LOG_PREFIX,
    EVENT_PREFIX,
    PierEvent,
    TelemetryContext,
    TelemetryDecoder,
    TelemetrySnapshot,
    current_telemetry_context,
)
from pier.trial.queue import TrialQueue
from pier.utils.logger import logger


def run(coro):
    return asyncio.run(coro)


def decode_control(message: str) -> dict:
    return json.loads(message)


def decode_audit_lines(text: str) -> list[dict]:
    return [
        json.loads(line[len(CONCURRENCY_LOG_PREFIX) :])
        for line in text.splitlines()
        if line.startswith(CONCURRENCY_LOG_PREFIX)
    ]


async def deliver_backoff_threshold(
    controller: DynamicConcurrencyController,
    *,
    now: float,
    capacity_period: int | None = None,
    final_event: PierEvent | None = None,
) -> None:
    for index in range(BACKOFF_SIGNAL_THRESHOLD - 1):
        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id=f"pressure-{index}",
                capacity_period=capacity_period,
            ),
            now=now,
        )
    await controller.handle(
        final_event
        or PierEvent(
            type="inference.rate_limited",
            trial_id="pressure-final",
            capacity_period=capacity_period,
        ),
        now=now,
    )


def make_controller(
    *,
    limiter: ResizableLimiter,
    request_throttling_manager: RequestThrottlingManager | None = None,
    **kwargs,
) -> DynamicConcurrencyController:
    return DynamicConcurrencyController(
        limiter=limiter,
        request_throttling_manager=request_throttling_manager
        or RequestThrottlingManager(limiter.capacity),
        **kwargs,
    )


def test_fixed_concurrency_remains_the_default():
    config = JobConfig(n_concurrent_trials=7)
    queue = TrialQueue(n_concurrent=config.n_concurrent_trials)

    assert config.optimize_concurrency is False
    assert queue.concurrency_limit == 7
    assert queue._dynamic_pool is None
    assert isinstance(queue._fixed_limiter, ResizableLimiter)
    assert queue._fixed_limiter.capacity == 7


def test_fixed_queue_uses_resizable_limiter_as_a_fixed_global_limit():
    async def scenario():
        queue = TrialQueue(n_concurrent=2)
        running = 0
        peak = 0

        async def execute(config):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return config

        queue._execute_trial_with_retries = execute
        configs = [object() for _ in range(6)]
        results = await asyncio.gather(*(queue.submit(config) for config in configs))

        assert results == configs
        assert peak == 2
        assert queue._fixed_limiter.capacity == 2

    run(scenario())


def test_optimized_concurrency_uses_configured_value_as_group_start_capacity():
    queue = TrialQueue(n_concurrent=7, optimize_concurrency=True)

    assert queue.concurrency_limit == 7
    assert queue.concurrency_group_capacity("openai", "gpt-5", "high") == 7
    assert queue._dynamic_pool is not None
    assert queue._dynamic_pool.initial_capacity == 7
    assert RAMP_UP_PERCENT == 10
    assert BACKOFF_PERCENT == 33
    assert BACKOFF_SIGNAL_THRESHOLD == 5
    assert RECOVERY_CEILING_PERCENT == 90
    assert queue._dynamic_pool.ramp_interval_sec == 300
    assert queue._dynamic_pool.cooldown_sec == 300
    assert STABILITY_WINDOW_SEC == 300
    assert COOLDOWN_SEC == 300


def test_dynamic_mini_swe_modal_trial_gets_initial_request_pause_state():
    async def scenario():
        queue = TrialQueue(n_concurrent=2, optimize_concurrency=True)
        config = SimpleNamespace(
            trial_name="trial-one",
            agent=AgentConfig(
                name="mini-swe-agent",
                model_name="openai/gpt-5.5",
            ),
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
        )
        observed = {}

        async def execute(_config):
            context = current_telemetry_context()
            assert context is not None
            assert context.pause_messages is not None
            observed["message"] = decode_control(await context.pause_messages.get())
            observed["manager"] = queue._dynamic_pool.request_throttling_manager_for(
                "openai", "gpt-5.5", None
            )
            return "done"

        queue._execute_trial_with_retries = execute
        assert await queue.submit(config) == "done"
        assert observed["message"]["paused"] is False
        assert observed["message"]["capacity_period"] == 0
        assert observed["manager"].running == 0
        assert observed["manager"].paused == 0

    run(scenario())


def test_fixed_concurrency_still_collects_telemetry(tmp_path):
    async def scenario():
        telemetry_path = tmp_path / "logs" / "telemetry.json"
        queue = TrialQueue(
            n_concurrent=3,
            telemetry=TelemetrySnapshot(telemetry_path),
        )
        config = SimpleNamespace(
            trial_name="trial-one",
            agent=AgentConfig(
                name="mini-swe-agent",
                model_name="openai/gpt-5.5",
            ),
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
        )

        async def execute(_config):
            context = current_telemetry_context()
            assert context is not None
            assert context.pause_messages is None
            await context.sink(
                PierEvent(
                    type="model.request.completed",
                    trial_id="trial-one",
                    provider="openai",
                    model="gpt-5.5",
                    payload={
                        "producer_id": "producer-one",
                        "sequence": 1,
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 30,
                        "agent_steps": 1,
                        "tool_calls": 1,
                        "cost_usd": 0.05,
                        "first_response_at": 1_800_000_000.0,
                        "last_response_at": 1_800_000_001.0,
                        "buckets": {},
                    },
                )
            )
            return "done"

        queue._execute_trial_with_retries = execute
        await queue.start()
        assert await queue.submit(config) == "done"
        await queue.stop()

        document = json.loads(telemetry_path.read_text())
        assert document["current_concurrency"] == 3
        assert document["totals"]["input_tokens"] == 100

    run(scenario())


def test_ramp_timer_is_not_starved_by_continuous_events():
    async def scenario():
        queue = TrialQueue(n_concurrent=1, optimize_concurrency=True)
        assert queue._dynamic_pool is not None
        queue._dynamic_pool.ramp_interval_sec = 0.01
        limiter = queue._dynamic_pool.limiter_for("openai", "gpt-5", "high")
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert limiter.queued == 1

        await queue.start()
        publishing = True

        async def publish_continuously():
            while publishing:
                assert queue._event_bus is not None
                await queue._event_bus.publish(
                    PierEvent(
                        type="model.request.completed",
                        trial_id="busy-trial",
                        provider="openai",
                        model="gpt-5",
                        effort="high",
                    )
                )
                await asyncio.sleep(0.001)

        producer = asyncio.create_task(publish_continuously())
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 2

        publishing = False
        await producer
        await queue.stop()
        assert queue._event_task is None
        await limiter.release()
        await limiter.release()

    run(scenario())


def test_live_concurrency_column_shows_each_concurrency_group():
    async def scenario():
        queue = TrialQueue(n_concurrent=4, optimize_concurrency=True)
        assert queue._dynamic_pool is not None
        high = queue._dynamic_pool.limiter_for("openai", "gpt-5", "high")
        low = queue._dynamic_pool.limiter_for("anthropic", "claude", "low")
        high_manager = queue._dynamic_pool.request_throttling_manager_for(
            "openai", "gpt-5", "high"
        )
        low_manager = queue._dynamic_pool.request_throttling_manager_for(
            "anthropic", "claude", "low"
        )
        await high.resize(5)
        await high.acquire()
        await high_manager.register("high-1", pause_messages=None)
        await high.acquire()
        await high_manager.register("high-2", pause_messages=None)
        await low.acquire()
        await low_manager.register("low-1", pause_messages=None)

        rendered = _ConcurrencyColumn(queue).render(None).plain

        assert queue.current_concurrency == 9
        assert "anthropic/claude/low running=1 capacity=4 queued=0 paused=0" in rendered
        assert "openai/gpt-5/high running=2 capacity=5 queued=0 paused=0" in rendered

        await high_manager.unregister("high-1")
        await high_manager.unregister("high-2")
        await high.release()
        await high.release()

        assert queue.current_concurrency == 4

        await low_manager.unregister("low-1")
        await low.release()

    run(scenario())


def test_live_concurrency_column_shows_fixed_global_limiter():
    async def scenario():
        queue = TrialQueue(n_concurrent=4)
        await queue._fixed_limiter.acquire()
        await queue._fixed_limiter.acquire()

        rendered = _ConcurrencyColumn(queue).render(None).plain

        assert rendered == "Concurrency: running=2 capacity=4 queued=0 paused=0"
        await queue._fixed_limiter.release()
        await queue._fixed_limiter.release()

    run(scenario())


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
        assert limiter.admitted == 2
        assert not acquired.is_set()

        await limiter.release()
        await asyncio.sleep(0)
        assert limiter.admitted == 1
        assert not acquired.is_set()

        await limiter.release()
        await asyncio.wait_for(acquired.wait(), timeout=1)
        await waiter

    run(scenario())


def test_resize_up_only_admits_newly_available_capacity():
    async def scenario():
        limiter = ResizableLimiter(1)
        await limiter.acquire()
        entered: list[int] = []
        gates = [asyncio.Event() for _ in range(3)]

        async def worker(index: int):
            async with limiter:
                entered.append(index)
                await gates[index].wait()

        tasks = [asyncio.create_task(worker(index)) for index in range(3)]
        await asyncio.sleep(0)
        assert limiter.queued == 3

        await limiter.resize(3)
        for _ in range(10):
            if len(entered) == 2:
                break
            await asyncio.sleep(0)

        assert len(entered) == 2
        assert limiter.admitted == 3
        assert limiter.queued == 1

        for gate in gates:
            gate.set()
        await asyncio.gather(*tasks)
        await limiter.release()

    run(scenario())


def test_cancelled_awakened_waiter_hands_slot_to_next_waiter():
    async def scenario():
        limiter = ResizableLimiter(1)
        await limiter.acquire()
        second_acquired = asyncio.Event()

        async def first_worker():
            async with limiter:
                raise AssertionError("cancelled waiter should not acquire")

        async def second_worker():
            async with limiter:
                second_acquired.set()

        first = asyncio.create_task(first_worker())
        second = asyncio.create_task(second_worker())
        await asyncio.sleep(0)
        assert limiter.queued == 2

        # Simulate a release while retaining the condition lock so the selected
        # waiter can be cancelled before it reacquires the lock.
        async with limiter._condition:
            limiter._admitted -= 1
            limiter._condition.notify(1)
            first.cancel()

        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(second_acquired.wait(), timeout=1)
        await second
        assert limiter.admitted == 0

    run(scenario())


def test_request_throttling_manager_pauses_newest_and_resumes_oldest():
    async def scenario():
        manager = RequestThrottlingManager(capacity=2)
        pause_queues = {name: asyncio.Queue() for name in ("one", "two", "three")}

        await manager.register("one", pause_messages=pause_queues["one"])
        await manager.register("two", pause_messages=pause_queues["two"])
        await manager.register("three", pause_messages=pause_queues["three"])
        assert manager.running == 2
        assert manager.paused == 1
        initial = {
            name: decode_control(pause_queue.get_nowait())
            for name, pause_queue in pause_queues.items()
        }
        assert initial["one"]["paused"] is False
        assert initial["two"]["paused"] is False
        assert initial["three"]["paused"] is True

        await manager.resize(1)
        assert manager.running == 1
        assert manager.paused == 2
        assert decode_control(pause_queues["two"].get_nowait())["paused"] is True

        await manager.unregister("one")
        assert manager.running == 1
        assert manager.paused == 1
        assert decode_control(pause_queues["two"].get_nowait())["paused"] is False

    run(scenario())


def test_request_throttling_manager_reserves_capacity_for_uncontrollable_rollouts():
    async def scenario():
        manager = RequestThrottlingManager(capacity=2)
        first_pause_queue = asyncio.Queue()
        second_pause_queue = asyncio.Queue()

        await manager.register("first", pause_messages=first_pause_queue)
        await manager.register("opaque", pause_messages=None)
        await manager.register("second", pause_messages=second_pause_queue)

        assert manager.running == 2
        assert manager.paused == 1
        assert decode_control(second_pause_queue.get_nowait())["paused"] is True

    run(scenario())


def test_request_throttling_manager_staggers_resumes_within_window(monkeypatch):
    async def scenario():
        manager = RequestThrottlingManager(capacity=1)
        pause_queues = [asyncio.Queue() for _ in range(3)]
        for index, pause_queue in enumerate(pause_queues):
            await manager.register(f"trial-{index}", pause_messages=pause_queue)
            pause_queue.get_nowait()
        monkeypatch.setattr("pier.concurrency.time.time", lambda: 100.0)

        await manager.resize(3)

        second = decode_control(pause_queues[1].get_nowait())
        third = decode_control(pause_queues[2].get_nowait())
        assert second["not_before"] == 100.0
        assert third["not_before"] == (100.0 + REQUEST_RESUME_WINDOW_SEC)

    run(scenario())


def test_capacity_resize_broadcasts_period_to_every_controlled_rollout():
    async def scenario():
        manager = RequestThrottlingManager(capacity=3)
        pause_queues = [asyncio.Queue() for _ in range(3)]
        for index, pause_queue in enumerate(pause_queues):
            await manager.register(f"trial-{index}", pause_messages=pause_queue)
            assert decode_control(pause_queue.get_nowait())["capacity_period"] == 0

        await manager.resize(2)

        messages = [
            decode_control(pause_queue.get_nowait()) for pause_queue in pause_queues
        ]
        assert [message["capacity_period"] for message in messages] == [1, 1, 1]
        assert [message["paused"] for message in messages] == [False, False, True]
        assert manager.capacity_period == 1

    run(scenario())


def test_controller_resizes_request_throttling_manager_in_lockstep():
    async def scenario():
        limiter = ResizableLimiter(3)
        manager = RequestThrottlingManager(3)
        pause_queues = [asyncio.Queue() for _ in range(3)]
        for index, pause_queue in enumerate(pause_queues):
            await limiter.acquire()
            await manager.register(f"trial-{index}", pause_messages=pause_queue)
            pause_queue.get_nowait()

        controller = make_controller(
            limiter=limiter,
            request_throttling_manager=manager,
        )
        await deliver_backoff_threshold(controller, now=10**9)

        assert limiter.capacity == 2
        assert manager.capacity == 2
        assert manager.running == 2
        assert manager.paused == 1
        assert decode_control(pause_queues[-1].get_nowait())["paused"] is True

        for _ in range(3):
            await limiter.release()

    run(scenario())


def test_audit_log_causally_links_429_backoff_and_paused_trials(tmp_path):
    async def scenario():
        audit_path = tmp_path / "job.log"
        handler = logging.FileHandler(audit_path)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            limiter = ResizableLimiter(3)
            manager = RequestThrottlingManager(
                3,
                provider="openai",
                model="gpt-5",
                effort="high",
            )
            pause_queues = [asyncio.Queue() for _ in range(3)]
            for index, pause_queue in enumerate(pause_queues):
                await limiter.acquire()
                await manager.register(f"trial-{index}", pause_messages=pause_queue)
                pause_queue.get_nowait()

            controller = make_controller(
                limiter=limiter,
                request_throttling_manager=manager,
                provider="openai",
                model="gpt-5",
                effort="high",
            )
            event = PierEvent(
                type="inference.rate_limited",
                trial_id="trial-0",
                provider="openai",
                model="gpt-5",
                effort="high",
                event_id="rate-limit-17",
                observed_at="2026-08-26T18:42:10Z",
                evidence="http_status_429",
                payload={"status_code": 429, "request_id": "request-8"},
            )

            await deliver_backoff_threshold(
                controller,
                now=10**9,
                final_event=event,
            )
            for _ in range(3):
                await limiter.release()
        finally:
            logger.removeHandler(handler)
            handler.close()

        records = decode_audit_lines(audit_path.read_text())
        rate_limit = next(
            record
            for record in records
            if record["type"] == "inference.rate_limit"
            and record["event_id"] == "rate-limit-17"
        )
        pause_change = next(
            record
            for record in records
            if record["type"] == "request.pause_changed"
            and record["reason"] == "capacity_resized"
        )

        assert rate_limit["event_id"] == "rate-limit-17"
        assert "verified" not in rate_limit
        assert rate_limit["evidence"] == "http_status_429"
        assert rate_limit["signal"] == {
            "request_id": "request-8",
            "status_code": 429,
        }
        expected_group = {
            "provider": "openai",
            "model": "gpt-5",
            "effort": "high",
        }
        assert rate_limit["concurrency_group"] == expected_group
        assert rate_limit["action"] == "backoff"
        assert rate_limit["capacity_before"] == 3
        assert rate_limit["capacity_after"] == 2
        assert rate_limit["trigger_rate_limit_count"] == 5
        assert rate_limit["rate_limit_count"] == 0
        assert pause_change["cause_id"] == rate_limit["decision_id"]
        assert pause_change["concurrency_group"] == expected_group
        assert pause_change["paused_trial_ids"] == ["trial-2"]
        assert pause_change["running"] == 2
        assert pause_change["paused"] == 1
        assert len(records) == 6
        assert sum(record["type"] == "inference.rate_limit" for record in records) == 5
        assert sum(record["type"] == "request.pause_changed" for record in records) == 1

    run(scenario())


def test_group_lifecycle_and_stable_ramp_are_audited(tmp_path):
    async def scenario():
        audit_path = tmp_path / "job.log"
        handler = logging.FileHandler(audit_path)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            pool = DynamicConcurrencyPool(
                initial_capacity=1,
                ramp_interval_sec=1,
            )
            group = ("openai", "gpt-5", "high")
            limiter = pool.limiter_for(*group)
            await limiter.acquire()
            waiter = asyncio.create_task(limiter.acquire())
            await asyncio.sleep(0)

            assert await pool._controllers[group].maybe_ramp(now=10**9)
            await asyncio.wait_for(waiter, timeout=1)
            await limiter.release()
            await limiter.release()

            pool.finalize()
        finally:
            logger.removeHandler(handler)
            handler.close()

        records = decode_audit_lines(audit_path.read_text())
        assert [record["type"] for record in records] == [
            "concurrency.group_initialized",
            "concurrency.ramp",
            "concurrency.group_finalized",
        ]
        ramp = next(
            record for record in records if record["type"] == "concurrency.ramp"
        )
        assert ramp["capacity_before"] == 1
        assert ramp["capacity_after"] == 2

    run(scenario())


def test_paused_rollouts_count_as_demand_for_ramping():
    async def scenario():
        limiter = ResizableLimiter(3)
        manager = RequestThrottlingManager(3)
        for index in range(3):
            await limiter.acquire()
            await manager.register(f"trial-{index}", pause_messages=asyncio.Queue())

        controller = make_controller(
            limiter=limiter,
            request_throttling_manager=manager,
            ramp_interval_sec=1,
        )
        await controller._resize(2)
        assert limiter.queued == 0
        assert manager.paused == 1

        assert await controller.maybe_ramp(now=10**9)
        assert limiter.capacity == 3
        assert manager.paused == 0

        for _ in range(3):
            await limiter.release()

    run(scenario())


def test_retry_after_is_broadcast_and_audited_without_changing_capacity(
    monkeypatch, tmp_path
):
    async def scenario():
        audit_path = tmp_path / "job.log"
        handler = logging.FileHandler(audit_path)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        manager = RequestThrottlingManager(2)
        pause_queue = asyncio.Queue()
        try:
            await manager.register("trial", pause_messages=pause_queue)
            pause_queue.get_nowait()
            monkeypatch.setattr("pier.concurrency.time.time", lambda: 100.0)

            await manager.apply_retry_after(7.5, cause_id="rate-limit-1")
        finally:
            logger.removeHandler(handler)
            handler.close()

        message = decode_control(pause_queue.get_nowait())
        assert message["paused"] is False
        assert message["not_before"] == 107.5
        assert manager.capacity == 2
        records = decode_audit_lines(audit_path.read_text())
        assert len(records) == 1
        assert records[0]["type"] == "request.retry_after_applied"
        assert records[0]["cause_id"] == "rate-limit-1"
        assert records[0]["controlled_trial_ids"] == ["trial"]

    run(scenario())


def test_controller_probes_up_then_returns_to_recovery_floor_on_429():
    async def scenario():
        limiter = ResizableLimiter(2)
        controller = make_controller(
            limiter=limiter,
            ramp_interval_sec=1,
            cooldown_sec=10,
        )

        await limiter.acquire()
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert limiter.queued == 1

        assert await controller.maybe_ramp(now=10**9)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 3
        assert controller.recovery_floor == 2
        assert controller.current_probe_capacity == 3

        await deliver_backoff_threshold(
            controller,
            now=10**9 + 1,
            final_event=PierEvent(
                type="inference.rate_limited",
                trial_id="trial-1",
                provider="openai",
                model="gpt-5",
            ),
        )
        assert limiter.capacity == 2
        assert limiter.admitted == 3
        assert controller.recovery_floor == 2
        assert controller.current_probe_capacity is None
        assert controller.recovery_ceiling == 2
        assert not await controller.maybe_ramp(now=10**9 + 10_000)

        # A burst from requests that were already in flight is one pressure event.
        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="trial-2",
            ),
            now=10**9 + 2,
        )
        assert limiter.capacity == 2

        await limiter.release()
        await limiter.release()
        await limiter.release()

    run(scenario())


def test_controller_waits_for_full_stability_window_between_probes(monkeypatch):
    async def scenario():
        monkeypatch.setattr("pier.concurrency.time.monotonic", lambda: 100.0)
        limiter = ResizableLimiter(2)
        controller = make_controller(
            limiter=limiter,
            ramp_interval_sec=120,
        )

        await limiter.acquire()
        await limiter.acquire()
        first_waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        assert not await controller.maybe_ramp(now=219.9)
        assert await controller.maybe_ramp(now=220.0)
        await asyncio.wait_for(first_waiter, timeout=1)
        assert limiter.capacity == 3
        assert controller.recovery_floor == 2
        assert controller.current_probe_capacity == 3

        second_waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert not await controller.maybe_ramp(now=339.9)
        assert await controller.maybe_ramp(now=340.0)
        await asyncio.wait_for(second_waiter, timeout=1)
        assert limiter.capacity == 4
        assert controller.recovery_floor == 3
        assert controller.current_probe_capacity == 4

        for _ in range(4):
            await limiter.release()

    run(scenario())


def test_controller_uses_percentage_steps_at_high_concurrency():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = make_controller(
            limiter=limiter,
            ramp_interval_sec=1,
        )

        for _ in range(100):
            await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        assert await controller.maybe_ramp(now=10**9)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 110
        assert controller.recovery_floor == 100

        # A failed startup probe returns to the stable capacity instead of
        # applying the larger percentage backoff below it.
        await deliver_backoff_threshold(controller, now=10**9 + 1)
        assert limiter.capacity == 100
        assert controller.known_bad_capacity == 110
        assert controller.recovery_ceiling == 99

        # The learned ceiling prevents immediately retrying the failed capacity.
        assert not await controller.maybe_ramp(now=10**9 + 10_000)
        assert limiter.capacity == 100

        for _ in range(101):
            await limiter.release()

    run(scenario())


def test_pressure_at_current_capacity_starts_bounded_recovery():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = make_controller(
            limiter=limiter,
        )

        await deliver_backoff_threshold(controller, now=10**9)

        assert controller.known_bad_capacity == 100
        assert controller.recovery_ceiling == 90
        assert limiter.capacity == 67

    run(scenario())


def test_backoff_requires_five_rate_limit_signals_at_the_current_capacity():
    async def scenario():
        limiter = ResizableLimiter(80)
        controller = make_controller(
            limiter=limiter,
        )

        for index in range(BACKOFF_SIGNAL_THRESHOLD - 1):
            await controller.handle(
                PierEvent(
                    type="inference.rate_limited",
                    trial_id=f"trial-{index}",
                    capacity_period=0,
                ),
                now=10**9,
            )
            assert limiter.capacity == 80

        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="trial-final",
                capacity_period=0,
            ),
            now=10**9,
        )
        assert limiter.capacity == 54

    run(scenario())


def test_rate_limit_pressure_restarts_the_quiet_window_without_forcing_backoff():
    async def scenario():
        base = 10**9
        limiter = ResizableLimiter(100)
        controller = make_controller(
            limiter=limiter,
        )
        await deliver_backoff_threshold(controller, now=base, capacity_period=0)

        for _ in range(67):
            await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="single-pressure-signal",
                capacity_period=1,
            ),
            now=base + 200,
        )
        assert limiter.capacity == 67
        assert not await controller.maybe_ramp(now=base + 499.9)
        assert await controller.maybe_ramp(now=base + 500)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 74

        for _ in range(68):
            await limiter.release()

    run(scenario())


def test_backoff_recovers_in_bounded_steps_below_known_bad():
    async def scenario():
        base = 10**9
        limiter = ResizableLimiter(80)
        controller = make_controller(
            limiter=limiter,
        )

        await deliver_backoff_threshold(controller, now=base, capacity_period=0)
        assert limiter.capacity == 54
        assert controller.recovery_ceiling == 72

        for _ in range(54):
            await limiter.acquire()
        waiters = [asyncio.create_task(limiter.acquire()) for _ in range(18)]
        await asyncio.sleep(0)

        assert not await controller.maybe_ramp(now=base + 299.9)
        assert await controller.maybe_ramp(now=base + 300)
        assert limiter.capacity == 60
        await asyncio.sleep(0)
        assert await controller.maybe_ramp(now=base + 600)
        assert limiter.capacity == 66
        await asyncio.sleep(0)
        assert await controller.maybe_ramp(now=base + 900)
        assert limiter.capacity == 72
        await asyncio.wait_for(asyncio.gather(*waiters), timeout=1)

        # The ceiling itself gets one full quiet observation window, then holds
        # without permanently disabling later recovery from a lower capacity.
        assert not await controller.maybe_ramp(now=base + 1200)
        assert controller.recovery_floor == 72
        assert controller.current_probe_capacity is None
        assert not await controller.maybe_ramp(now=base + 1500)

        for _ in range(72):
            await limiter.release()

    run(scenario())


def test_failed_recovery_probe_can_recover_again_after_a_later_backoff():
    async def scenario():
        base = 10**9
        limiter = ResizableLimiter(80)
        controller = make_controller(
            limiter=limiter,
        )

        await deliver_backoff_threshold(controller, now=base, capacity_period=0)
        for _ in range(54):
            await limiter.acquire()
        waiters = [asyncio.create_task(limiter.acquire()) for _ in range(6)]
        await asyncio.sleep(0)

        assert await controller.maybe_ramp(now=base + 300)
        await asyncio.wait_for(asyncio.gather(*waiters), timeout=1)
        assert limiter.capacity == 60
        assert controller.recovery_floor == 54
        assert controller.current_probe_capacity == 60

        await deliver_backoff_threshold(
            controller,
            now=base + 301,
            capacity_period=2,
        )
        assert limiter.capacity == 54
        assert limiter.admitted == 60
        assert controller.known_bad_capacity == 60
        assert controller.recovery_floor == 54
        assert controller.current_probe_capacity is None
        assert not await controller.maybe_ramp(now=base + 600)

        # More pressure can establish a lower floor. Recovery remains available
        # from that lower capacity rather than being disabled for the run.
        await deliver_backoff_threshold(
            controller,
            now=base + 601,
            capacity_period=3,
        )
        assert limiter.capacity == 36
        assert controller.recovery_floor == 36
        assert controller.recovery_ceiling == 48

        for _ in range(24):
            await limiter.release()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        assert await controller.maybe_ramp(now=base + 901)
        await asyncio.wait_for(waiter, timeout=1)
        assert limiter.capacity == 40

        for _ in range(37):
            await limiter.release()

    run(scenario())


def test_continuous_fresh_429s_can_backoff_once_per_fixed_interval():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = make_controller(
            limiter=limiter,
            cooldown_sec=60,
        )

        await deliver_backoff_threshold(controller, now=100, capacity_period=0)
        assert limiter.capacity == 67
        assert controller.request_throttling_manager.capacity_period == 1

        # Five current-period signals satisfy the pressure threshold, but cannot
        # reduce again before the independent deadline established at t=100.
        for now in (110, 120, 130, 140, 150):
            await controller.handle(
                PierEvent(
                    type="inference.rate_limited",
                    trial_id=f"fresh-{now}",
                    capacity_period=1,
                ),
                now=now,
            )
            assert limiter.capacity == 67

        # Continued pressure after the deadline can use the already-satisfied
        # threshold and starts a new count in the resized capacity period.
        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="fresh-160",
                capacity_period=1,
            ),
            now=160,
        )
        assert limiter.capacity == 45
        assert controller.request_throttling_manager.capacity_period == 2

    run(scenario())


def test_only_requests_started_after_latest_reduction_can_backoff_again():
    async def scenario():
        limiter = ResizableLimiter(100)
        controller = make_controller(
            limiter=limiter,
            cooldown_sec=60,
        )
        await deliver_backoff_threshold(controller, now=100, capacity_period=0)
        assert limiter.capacity == 67

        # Both arrive after the interval, but neither proves that its request
        # began after capacity period 1 was installed.
        for trial_id, period in (("stale", 0), ("untagged", None)):
            await controller.handle(
                PierEvent(
                    type="inference.rate_limited",
                    trial_id=trial_id,
                    capacity_period=period,
                ),
                now=200,
            )
            assert limiter.capacity == 67

        await deliver_backoff_threshold(controller, now=200, capacity_period=1)
        assert limiter.capacity == 45

        # Capacity period 1 is now stale because the second reduction installed 2.
        await controller.handle(
            PierEvent(
                type="inference.rate_limited",
                trial_id="formerly-fresh",
                capacity_period=1,
            ),
            now=300,
        )
        assert limiter.capacity == 45

    run(scenario())


def test_dynamic_pool_isolates_provider_model_effort_groups():
    async def scenario():
        pool = DynamicConcurrencyPool(
            initial_capacity=2,
            ramp_interval_sec=1,
        )
        high = pool.limiter_for("openai", "gpt-5", "high")
        low = pool.limiter_for("openai", "gpt-5", "low")
        await high.resize(3)
        await low.resize(4)

        for index in range(BACKOFF_SIGNAL_THRESHOLD):
            await pool.handle(
                PierEvent(
                    type="inference.rate_limited",
                    trial_id=f"trial-high-{index}",
                    provider="openai",
                    model="gpt-5",
                    effort="high",
                )
            )
        assert high.capacity == 2
        assert low.capacity == 4

    run(scenario())


def test_telemetry_decoder_accepts_structured_evidence_and_ignores_plain_text():
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
            "capacity_period": 7,
        }
        encoded = EVENT_PREFIX + json.dumps(record) + "\n"
        await decoder.feed(encoded[:12])
        await decoder.feed(encoded[12:])
        await decoder.feed(
            EVENT_PREFIX
            + json.dumps(
                {
                    "type": "model.request.rate_limited",
                    "evidence": "typed_rate_limit_error",
                    "capacity_period": 8,
                }
            )
            + "\n"
        )
        await decoder.feed("litellm.RateLimitError: HTTP 429 retry-after: 4\n")
        await decoder.feed("normal verbose model output\n")
        await decoder.flush()

        assert len(events) == 2
        assert events[0].retry_after_sec == 2.5
        assert events[0].provider == "openai"
        assert events[0].effort == "high"
        assert events[0].evidence == "http_status_429"
        assert events[0].capacity_period == 7
        assert events[1].evidence == "typed_rate_limit_error"
        assert events[1].capacity_period == 8

    run(scenario())


def test_telemetry_decoder_ignores_malformed_fields_without_stopping_stream():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)

        decoder = TelemetryDecoder(
            TelemetryContext(
                sink=sink,
                trial_id="trial-1",
                provider="openai",
                model="gpt-5",
                effort=None,
            )
        )
        await decoder.feed(f"{EVENT_PREFIX}[]\n")
        await decoder.feed(
            EVENT_PREFIX
            + json.dumps(
                {
                    "type": "model.request.rate_limited",
                    "status_code": 429,
                    "retry_after_sec": "invalid",
                    "capacity_period": "invalid",
                }
            )
            + "\n"
        )

        assert len(events) == 1
        assert events[0].retry_after_sec is None
        assert events[0].capacity_period is None

    run(scenario())


def test_telemetry_decoder_accepts_cumulative_request_usage():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)

        decoder = TelemetryDecoder(
            TelemetryContext(
                sink=sink,
                trial_id="trial-1",
                provider="openai",
                model="gpt-5",
                effort="high",
            )
        )
        record = {
            "type": "model.request.completed",
            "producer_id": "producer-1",
            "sequence": 2,
            "input_tokens": 120,
            "cached_input_tokens": 20,
            "output_tokens": 30,
            "agent_steps": 2,
            "tool_calls": 1,
            "cost_usd": 0.04,
            "first_response_at": 1_800_000_000.0,
            "last_response_at": 1_800_000_010.0,
            "buckets": {
                "30000000": {"tokens": 150, "requests": 2},
            },
        }

        await decoder.feed(EVENT_PREFIX + json.dumps(record) + "\n")

        assert len(events) == 1
        assert events[0].type == "model.request.completed"
        assert events[0].trial_id == "trial-1"
        assert events[0].provider == "openai"
        assert events[0].payload["sequence"] == 2
        assert events[0].payload["input_tokens"] == 120

    run(scenario())


def test_telemetry_snapshot_deduplicates_and_uses_job_capacity(tmp_path):
    snapshot = TelemetrySnapshot(
        tmp_path / "logs" / "telemetry.json",
    )
    payload = {
        "producer_id": "producer-1",
        "sequence": 1,
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 40,
        "agent_steps": 1,
        "tool_calls": 2,
        "cost_usd": 0.03,
        "first_response_at": 1_800_000_000.0,
        "last_response_at": 1_800_000_001.0,
        "buckets": {},
    }
    event = PierEvent(
        type="model.request.completed",
        trial_id="trial-1",
        provider="openai",
        model="gpt-5",
        effort="high",
        payload=payload,
    )

    snapshot.record(event)
    snapshot.record(event)
    assert snapshot.build(128)["current_concurrency"] == 128

    document = snapshot.build(126)

    assert document["current_concurrency"] == 126
    assert document["totals"]["input_tokens"] == 100
    assert document["totals"]["agent_steps"] == 1
    assert document["groups"][0]["model_name"] == "openai/gpt-5"

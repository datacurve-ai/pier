import asyncio
import shutil
from collections.abc import Coroutine
from typing import Any

from pier.concurrency import (
    ConcurrencySnapshot,
    DynamicConcurrencyPool,
    ResizableLimiter,
)
from pier.agents.factory import AgentFactory
from pier.environments.factory import EnvironmentFactory
from pier.models.job.config import RetryConfig
from pier.models.trial.config import TrialConfig
from pier.models.trial.result import TrialResult
from pier.request_throttling import AgentPauseMessageQueue
from pier.telemetry import (
    EventBus,
    TelemetryContext,
    TelemetrySnapshot,
    bind_telemetry,
)
from pier.trial.hooks import HookCallback, TrialEvent
from pier.utils.logger import logger


class TrialQueue:
    """
    Handles orchestration of concurrent trials.

    Receives TrialConfigs, creates Trial objects internally, runs them
    with retry logic, and returns TrialResult tasks. Concurrency is
    bounded by a ResizableLimiter. Hooks are wired to each Trial
    instance — Trial handles all event invocations.
    """

    def __init__(
        self,
        n_concurrent: int,
        optimize_concurrency: bool = False,
        telemetry: TelemetrySnapshot | None = None,
        retry_config: RetryConfig | None = None,
        hooks: dict[TrialEvent, list[HookCallback]] | None = None,
    ):
        if hooks is None:
            hooks = {event: [] for event in TrialEvent}
        else:
            for event in TrialEvent:
                hooks.setdefault(event, [])

        if n_concurrent < 1:
            raise ValueError("n_concurrent must be at least 1")
        self._n_concurrent = n_concurrent
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._hooks = hooks
        self._logger = logger.getChild(__name__)
        self._fixed_limiter = ResizableLimiter(n_concurrent)
        self._telemetry = telemetry
        self._event_bus = (
            EventBus() if optimize_concurrency or self._telemetry else None
        )
        self._dynamic_pool: DynamicConcurrencyPool | None = None
        self._event_task: asyncio.Task[None] | None = None
        if optimize_concurrency:
            self._dynamic_pool = DynamicConcurrencyPool(
                initial_capacity=n_concurrent,
            )

    @property
    def concurrency_limit(self) -> int:
        """Fixed limit, or the starting capacity for each concurrency group."""
        return self._n_concurrent

    def concurrency_group_capacity(
        self, provider: str | None, model: str | None, effort: str | None
    ) -> int:
        """Return the current capacity for a concurrency group."""
        if self._dynamic_pool is None:
            return self._n_concurrent
        return self._dynamic_pool.capacity_for(provider, model, effort)

    @property
    def optimize_concurrency(self) -> bool:
        return self._dynamic_pool is not None

    def concurrency_snapshots(self) -> list[ConcurrencySnapshot]:
        """Return fixed or per-group concurrency for live displays."""
        if self._dynamic_pool is not None:
            return self._dynamic_pool.snapshots()
        return [
            ConcurrencySnapshot(
                running=self._fixed_limiter.admitted,
                capacity=self._fixed_limiter.capacity,
                queued=self._fixed_limiter.queued,
            )
        ]

    @property
    def current_concurrency(self) -> int:
        """Return Pier's single job-level capacity."""
        snapshots = self.concurrency_snapshots()
        active = [
            snapshot
            for snapshot in snapshots
            if snapshot.running or snapshot.paused or snapshot.queued
        ]
        if active:
            return sum(snapshot.capacity for snapshot in active)
        if snapshots:
            return sum(snapshot.capacity for snapshot in snapshots)
        return self._n_concurrent

    @staticmethod
    def _concurrency_group(
        trial_config: TrialConfig,
    ) -> tuple[str | None, str | None, str | None]:
        model_name = trial_config.agent.model_name or ""
        provider, separator, model = model_name.partition("/")
        if not separator:
            provider, model = None, model_name or None
        effort = trial_config.agent.kwargs.get("reasoning_effort")
        if effort is None:
            effort = trial_config.agent.kwargs.get("effort")
        return provider, model, str(effort) if effort is not None else None

    async def start(self) -> None:
        """Start the job event processor."""
        if self._event_bus is None or self._event_task is not None:
            return
        await self._write_telemetry(force=True)
        self._event_task = asyncio.create_task(
            self._process_events(), name="pier-events"
        )

    async def stop(self) -> None:
        """Drain job events and write the final telemetry snapshot."""
        if self._event_bus is None or self._event_task is None:
            return
        await self._event_bus.close()
        await self._event_task
        self._event_task = None

    async def _write_telemetry(self, *, force: bool = False) -> None:
        if self._telemetry is not None:
            await self._telemetry.maybe_write(
                self.current_concurrency,
                force=force,
            )

    async def _process_events(self) -> None:
        assert self._event_bus is not None
        event_task = asyncio.create_task(self._event_bus.next(), name="pier-next-event")
        ramp_interval = (
            min(1.0, max(0.05, self._dynamic_pool.ramp_interval_sec))
            if self._dynamic_pool is not None
            else None
        )
        ramp_timer = (
            asyncio.create_task(
                asyncio.sleep(ramp_interval), name="pier-concurrency-ramp-timer"
            )
            if ramp_interval is not None
            else None
        )
        try:
            while True:
                waiters = {event_task}
                if ramp_timer is not None:
                    waiters.add(ramp_timer)
                completed, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if event_task in completed:
                    event = event_task.result()
                    if event is None:
                        if self._dynamic_pool is not None:
                            self._dynamic_pool.finalize()
                        await self._write_telemetry(force=True)
                        return
                    event_task = asyncio.create_task(
                        self._event_bus.next(), name="pier-next-event"
                    )

                    if self._telemetry is not None:
                        self._telemetry.record(event)
                    capacity_changed = (
                        await self._dynamic_pool.handle(event)
                        if self._dynamic_pool is not None
                        else False
                    )
                    if self._telemetry is not None and (
                        event.type == "model.request.completed" or capacity_changed
                    ):
                        await self._write_telemetry(force=capacity_changed)

                if ramp_timer is not None and ramp_timer in completed:
                    assert self._dynamic_pool is not None
                    if await self._dynamic_pool.maybe_ramp():
                        await self._write_telemetry()
                    ramp_timer = asyncio.create_task(
                        asyncio.sleep(ramp_interval),
                        name="pier-concurrency-ramp-timer",
                    )
        finally:
            pending = [
                task
                for task in (event_task, ramp_timer)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "TrialQueue":
        """Register a callback for a trial lifecycle event and return the queue."""
        self._hooks[event].append(callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _should_retry_exception(self, exception_type: str) -> bool:
        """Check if an exception should trigger a retry."""
        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is in exclude_exceptions, not retrying"
            )
            return False

        if (
            self._retry_config.include_exceptions
            and exception_type not in self._retry_config.include_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is not in include_exceptions, not retrying"
            )
            return False

        return True

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a retry attempt."""
        delay = self._retry_config.min_wait_sec * (
            self._retry_config.wait_multiplier**attempt
        )
        return min(delay, self._retry_config.max_wait_sec)

    def _setup_hooks(self, trial) -> None:
        """Wire queue-level hooks to the trial."""
        for event, hooks in self._hooks.items():
            for hook in hooks:
                trial.add_hook(event, hook)

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        """Execute a trial with retry logic."""
        from pier.trial.trial import Trial

        for attempt in range(self._retry_config.max_retries + 1):
            trial = await Trial.create(trial_config)
            self._setup_hooks(trial)
            result = await trial.run()

            if result.exception_info is None:
                return result

            if not self._should_retry_exception(result.exception_info.exception_type):
                self._logger.debug(
                    "Not retrying trial because the exception is not in "
                    "include_exceptions or the maximum number of retries has been "
                    "reached"
                )
                return result
            if attempt == self._retry_config.max_retries:
                self._logger.debug(
                    "Not retrying trial because the maximum number of retries has been "
                    "reached"
                )
                return result

            shutil.rmtree(trial.trial_dir, ignore_errors=True)

            delay = self._calculate_backoff_delay(attempt)

            self._logger.debug(
                f"Trial {trial_config.trial_name} failed with exception "
                f"{result.exception_info.exception_type}. Retrying in "
                f"{delay:.2f} seconds..."
            )

            await asyncio.sleep(delay)

        raise RuntimeError(
            f"Trial {trial_config.trial_name} produced no result. This should never "
            "happen."
        )

    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single trial through the selected concurrency limiter."""
        limiter = self._fixed_limiter
        concurrency_group: tuple[str | None, str | None, str | None] | None = None

        if self._dynamic_pool is not None:
            concurrency_group = self._concurrency_group(trial_config)
            limiter = self._dynamic_pool.limiter_for(*concurrency_group)

        async with limiter:
            if self._event_bus is None:
                return await self._execute_trial_with_retries(trial_config)
            if concurrency_group is None:
                concurrency_group = self._concurrency_group(trial_config)

            supports_request_throttling = (
                self._dynamic_pool is not None
                and AgentFactory.supports_request_throttling(trial_config.agent)
                and EnvironmentFactory.supports_exec_stdin(
                    trial_config.environment.type,
                    trial_config.environment.import_path,
                )
            )
            pause_messages: AgentPauseMessageQueue | None = (
                asyncio.Queue() if supports_request_throttling else None
            )
            if self._dynamic_pool is not None:
                await self._dynamic_pool.register_trial(
                    *concurrency_group,
                    trial_id=trial_config.trial_name,
                    pause_messages=pause_messages,
                )
                await self._write_telemetry()
            try:
                context = TelemetryContext(
                    sink=self._event_bus.publish,
                    trial_id=trial_config.trial_name,
                    provider=concurrency_group[0],
                    model=concurrency_group[1],
                    effort=concurrency_group[2],
                    pause_messages=pause_messages,
                )
                with bind_telemetry(context):
                    return await self._execute_trial_with_retries(trial_config)
            finally:
                if self._dynamic_pool is not None:
                    previous_concurrency = self.current_concurrency
                    await self._dynamic_pool.unregister_trial(
                        *concurrency_group,
                        trial_id=trial_config.trial_name,
                    )
                    await self._write_telemetry(
                        force=self.current_concurrency != previous_concurrency
                    )

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        """
        Return a coroutine that executes one trial.

        The caller decides how to schedule it (await, gather, TaskGroup).
        """
        return self._run_trial(trial_config)

    def submit_batch(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """
        Return coroutines for multiple trials, ordered to match `configs`.
        """
        return [self.submit(config) for config in configs]

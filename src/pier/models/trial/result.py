import traceback
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from pier.models.agent.context import AgentContext
from pier.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from pier.models.trial.config import TrialConfig
from pier.models.verifier.result import VerifierResult


class TimingInfo(BaseModel):
    """Timing information for a phase of trial execution."""

    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int | None = None
    max_attempts: int | None = None


class ExceptionInfo(BaseModel):
    """Information about an exception that occurred during trial execution."""

    exception_type: str
    exception_message: str
    exception_traceback: str
    occurred_at: datetime

    @classmethod
    def from_exception(cls, e: BaseException) -> "ExceptionInfo":
        return cls(
            exception_type=type(e).__name__,
            exception_message=str(e),
            exception_traceback=traceback.format_exc(),
            occurred_at=datetime.now(),
        )


class ModelInfo(BaseModel):
    """Information about a model that participated in a trial.

    ``provider`` is optional: when the user runs e.g. ``-m gpt-5.4`` with no
    ``<provider>/`` prefix, the CLI records the model name without a
    provider. Downstream writes to the ``model`` table omit the column so
    the DB default (``'unknown'``) takes over, keeping both sides honest
    about "not specified" vs "explicitly unknown".
    """

    name: str
    provider: str | None = None


class AgentInfo(BaseModel):
    """Information about an agent that participated in a trial."""

    name: str
    version: str
    model_info: ModelInfo | None = None


class StepResult(BaseModel):
    step_name: str
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None
    verifier_attempt: int | None = None
    max_verifier_attempts: int | None = None


class TrialResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task_name: str
    trial_name: str
    trial_uri: str
    task_id: LocalTaskId | GitTaskId | PackageTaskId
    source: str | None = None
    task_checksum: str
    config: TrialConfig
    agent_info: AgentInfo
    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None
    verifier_attempt: int | None = None
    max_verifier_attempts: int | None = None
    n_agent_steps: int | None = None
    step_results: list[StepResult] | None = None

    def agent_step_count(self) -> int | None:
        if self.n_agent_steps is not None:
            return self.n_agent_steps
        if (
            self.agent_result is not None
            and self.agent_result.n_agent_steps is not None
        ):
            return self.agent_result.n_agent_steps
        if not self.step_results:
            return None

        total = 0
        found = False
        for step_result in self.step_results:
            agent_result = step_result.agent_result
            if agent_result is None or agent_result.n_agent_steps is None:
                continue
            found = True
            total += agent_result.n_agent_steps
        return total if found else None

    def compute_token_cost_totals(
        self,
    ) -> tuple[int | None, int | None, int | None, float | None]:
        """Sum (n_input_tokens, n_cache_tokens, n_output_tokens, cost_usd)."""
        if self.agent_result is not None:
            contexts = [self.agent_result]
        elif self.step_results:
            contexts = [
                step_result.agent_result
                for step_result in self.step_results
                if step_result.agent_result is not None
            ]
        else:
            contexts = []

        if not contexts:
            return None, None, None, None

        n_input: int | None = None
        n_cache: int | None = None
        n_output: int | None = None
        cost: float | None = None
        for ctx in contexts:
            if ctx.n_input_tokens is not None:
                n_input = (n_input or 0) + ctx.n_input_tokens
            if ctx.n_cache_tokens is not None:
                n_cache = (n_cache or 0) + ctx.n_cache_tokens
            if ctx.n_output_tokens is not None:
                n_output = (n_output or 0) + ctx.n_output_tokens
            if ctx.cost_usd is not None:
                cost = (cost or 0.0) + ctx.cost_usd

        return n_input, n_cache, n_output, cost

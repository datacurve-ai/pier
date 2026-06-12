"""FastAPI server for the Pier Viewer."""

import json
import math
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pier.critique.runner import (
    CRITIQUE_MARKDOWN_FILENAME,
    CRITIQUE_RESULT_FILENAME,
    CRITIQUE_RUNS_DIRNAME,
)
from pier.models.job.config import (
    JobConfig,
)
from pier.models.job.result import JobStats
from pier.models.trial.result import TrialResult
from pier.utils.pricing import resolve_model_pricing
from pier.viewer.models import (
    CritiqueHeatmapCell,
    CritiqueHeatmapColumn,
    CritiqueHeatmapData,
    CritiqueHeatmapRow,
    CritiqueItemFilters,
    CritiqueItemSummary,
    CritiqueRunDetail,
    CritiqueRunSummary,
    EvalSummary,
    FileInfo,
    FilterOption,
    JobFilters,
    JobHeatmapCell,
    JobHeatmapColumn,
    JobHeatmapData,
    JobHeatmapRouteParams,
    JobHeatmapRow,
    JobSummary,
    ModelPricing,
    PaginatedResponse,
    TaskDefinitionDetail,
    TaskDefinitionFilters,
    TaskDefinitionSummary,
    TaskFilters,
    TaskSummary,
    TrialCritiqueDetail,
    TrialSummary,
)
from pier.viewer.scanner import JobScanner
from pier.viewer.task_scanner import TaskDefinitionScanner


class SummarizeRequest(BaseModel):
    """Request body for job summarization."""

    model: str = "haiku"
    n_concurrent: int = 32
    only_failed: bool = False
    overwrite: bool = False


class TrialSummarizeRequest(BaseModel):
    """Request body for single trial summarization."""

    model: str = "haiku"


class TaskGroupStats(TypedDict):
    """Stats accumulated for a task group."""

    n_trials: int
    n_completed: int
    n_errors: int
    exception_types: set[str]
    total_reward: float
    reward_count: int
    total_duration_ms: float
    duration_count: int
    total_input_tokens: int
    input_tokens_count: int
    total_cached_input_tokens: int
    cached_input_tokens_count: int
    total_output_tokens: int
    output_tokens_count: int
    total_cost_usd: float
    cost_usd_count: int
    total_peak_context_tokens: int
    peak_context_tokens_count: int
    total_agent_steps: int
    agent_steps_count: int


class HeatmapGroupStats(TypedDict):
    """Stats accumulated for one heatmap cell."""

    n_trials: int
    n_completed: int
    n_errors: int
    exception_counts: dict[str, int]
    total_reward: float
    reward_count: int
    total_duration_ms: float
    duration_count: int
    total_input_tokens: int
    input_tokens_count: int
    total_cached_input_tokens: int
    cached_input_tokens_count: int
    total_output_tokens: int
    output_tokens_count: int
    total_cost_usd: float
    cost_usd_count: int
    total_peak_context_tokens: int
    peak_context_tokens_count: int
    total_agent_steps: int
    agent_steps_count: int


def _uncached_input(n_input: int | None, n_cache: int | None) -> int | None:
    """Derive uncached input token count from raw input + cache totals.

    ``AgentContext.n_input_tokens`` is documented to include cached tokens,
    so the "uncached" portion the viewer surfaces is total minus cached.
    """
    if n_input is None:
        return None
    if n_cache is None:
        return n_input
    return max(0, n_input - n_cache)


def _agent_step_count_from_result(result: TrialResult) -> int | None:
    """Read promoted agent step counts from trial result.json.

    Older trials may not have this field; those intentionally surface as
    missing until a backfill rewrites their result.json files.
    """
    return result.agent_step_count()


def _extract_reasoning_effort(result: TrialResult) -> str | None:
    """Best-effort recovery of a trial's reasoning/thinking effort.

    Effort is not a first-class field; it lives inside the agent's free-form
    ``kwargs``, with agent/provider-specific shapes:
      - top-level ``reasoning_effort`` (claude-code, codex, gemini-cli, mini-swe
        on Gemini)
      - ``model_kwargs.output_config.effort`` (mini-swe on Anthropic)
      - ``model_kwargs.reasoning.effort`` (mini-swe on OpenAI)
    Only explicitly configured efforts are surfaced; an unset value reports as
    missing rather than inferring agent-specific defaults.
    """
    kwargs = result.config.agent.kwargs or {}

    value = kwargs.get("reasoning_effort")
    if value is not None:
        return str(value)

    model_kwargs = kwargs.get("model_kwargs")
    if isinstance(model_kwargs, dict):
        for outer_key in ("output_config", "reasoning"):
            outer = model_kwargs.get(outer_key)
            if isinstance(outer, dict) and outer.get("effort") is not None:
                return str(outer["effort"])

    return None


# Maximum file size to serve (1MB)
MAX_FILE_SIZE = 1024 * 1024


def _format_size(size_bytes: int) -> str:
    """Format bytes as a compact human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def create_app(
    folder: Path,
    mode: str = "jobs",
    static_dir: Path | None = None,
) -> FastAPI:
    """Create the FastAPI application with routes configured for the given directory.

    Args:
        folder: Directory containing job/trial data or task definitions
        mode: "jobs" for job viewer, "tasks" for task definition browser
        static_dir: Optional directory containing static viewer files (index.html, assets/)
    """
    # Store cleanup callbacks for lifespan
    cleanup_callbacks: list = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        for callback in cleanup_callbacks:
            await callback()

    app = FastAPI(
        title="Pier Viewer",
        description="API for browsing Pier jobs and trials",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow CORS for local development
    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/api/config")
    def get_config() -> dict[str, str]:
        """Get viewer configuration."""
        return {"folder": str(folder), "mode": mode}

    @app.get("/api/pricing", response_model=ModelPricing)
    def get_model_pricing(
        model: str = Query(
            ..., description="Model name, e.g. 'gpt-4' or 'openai/gpt-4'"
        ),
    ) -> ModelPricing:
        """Look up per-token pricing for a model.

        Resolved from the ``PIER_PRICING_FILE`` overrides then LiteLLM, falling
        back to the bare model name when the provider-prefixed form is not in the
        table (e.g. ``openai/gpt-4`` -> ``gpt-4``). The cache-read rate is
        reported as ``null`` when not separately listed, rather than mirroring
        the input rate (which would misrepresent the true cost of cache hits).
        """
        pricing = resolve_model_pricing(model)
        if pricing is None:
            raise HTTPException(
                status_code=404, detail=f"No pricing entry for model '{model}'"
            )

        return ModelPricing(
            model_name=model,
            input_cost_per_token=pricing.input_cost_per_token,
            cache_read_input_token_cost=pricing.cache_read_input_token_cost,
            output_cost_per_token=pricing.output_cost_per_token,
        )

    if mode == "tasks":
        _register_task_endpoints(app, folder, cleanup_callbacks)
    else:
        _register_job_endpoints(app, folder)

    # Serve static viewer files if provided
    if static_dir and static_dir.exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets", StaticFiles(directory=assets_dir), name="static_assets"
            )

        fonts_dir = static_dir / "fonts"
        if fonts_dir.exists():
            app.mount("/fonts", StaticFiles(directory=fonts_dir), name="static_fonts")

        @app.get("/favicon.ico")
        def favicon() -> FileResponse:
            """Serve favicon."""
            return FileResponse(static_dir / "favicon.ico")

        @app.get("/{path:path}")
        def serve_spa(path: str) -> FileResponse:
            """Serve the SPA for all non-API routes."""
            return FileResponse(static_dir / "index.html")

    return app


def _register_task_endpoints(
    app: FastAPI, tasks_dir: Path, cleanup_callbacks: list
) -> None:
    """Register API endpoints for task definition browsing."""
    from collections import Counter

    from pier.viewer.chat import ChatSessionManager, stream_chat_response

    task_scanner = TaskDefinitionScanner(tasks_dir)
    chat_manager = ChatSessionManager(tasks_dir)
    cleanup_callbacks.append(chat_manager.close_all)

    resolved_tasks_dir = tasks_dir.resolve()

    def _validate_task_name(name: str) -> Path:
        """Validate task name and return the resolved task directory."""
        task_dir = (tasks_dir / name).resolve()
        if resolved_tasks_dir not in task_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid task name")
        return task_dir

    def _get_all_task_definition_summaries() -> list[TaskDefinitionSummary]:
        """Build summaries for all task definitions."""
        task_names = task_scanner.list_tasks()
        summaries = []
        for name in task_names:
            config = task_scanner.get_task_config(name)
            paths_info = task_scanner.get_task_paths_info(name)
            if config:
                summaries.append(
                    TaskDefinitionSummary(
                        name=name,
                        version=config.schema_version,
                        source=config.source,
                        metadata=config.metadata,
                        has_instruction=paths_info["has_instruction"],
                        has_environment=paths_info["has_environment"],
                        has_tests=paths_info["has_tests"],
                        has_solution=paths_info["has_solution"],
                        agent_timeout_sec=config.agent.timeout_sec,
                        verifier_timeout_sec=config.verifier.timeout_sec,
                        os=config.environment.os.value,
                        cpus=config.environment.cpus,
                        memory_mb=config.environment.memory_mb,
                        storage_mb=config.environment.storage_mb,
                        gpus=config.environment.gpus,
                    )
                )
            else:
                summaries.append(
                    TaskDefinitionSummary(
                        name=name,
                        has_instruction=paths_info["has_instruction"],
                        has_environment=paths_info["has_environment"],
                        has_tests=paths_info["has_tests"],
                        has_solution=paths_info["has_solution"],
                    )
                )
        return summaries

    @app.get("/api/task-definitions/filters", response_model=TaskDefinitionFilters)
    def get_task_definition_filters() -> TaskDefinitionFilters:
        """Get available filter options for task definitions."""
        summaries = _get_all_task_definition_summaries()

        difficulty_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()

        for s in summaries:
            meta = s.metadata
            if "difficulty" in meta and isinstance(meta["difficulty"], str):
                difficulty_counts[meta["difficulty"]] += 1
            if "category" in meta and isinstance(meta["category"], str):
                category_counts[meta["category"]] += 1
            if "tags" in meta and isinstance(meta["tags"], list):
                for tag in meta["tags"]:
                    if isinstance(tag, str):
                        tag_counts[tag] += 1

        return TaskDefinitionFilters(
            difficulties=[
                FilterOption(value=v, count=c)
                for v, c in sorted(difficulty_counts.items())
            ],
            categories=[
                FilterOption(value=v, count=c)
                for v, c in sorted(category_counts.items())
            ],
            tags=[
                FilterOption(value=v, count=c) for v, c in sorted(tag_counts.items())
            ],
        )

    @app.get(
        "/api/task-definitions",
        response_model=PaginatedResponse[TaskDefinitionSummary],
    )
    def list_task_definitions(
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=100, description="Number of items per page"
        ),
        q: str | None = Query(default=None, description="Search query"),
        difficulty: list[str] = Query(default=[], description="Filter by difficulty"),
        category: list[str] = Query(default=[], description="Filter by category"),
        tag: list[str] = Query(default=[], description="Filter by tags"),
    ) -> PaginatedResponse[TaskDefinitionSummary]:
        """List all task definitions with summary information."""
        summaries = _get_all_task_definition_summaries()

        # Search filter
        if q:
            query = q.lower()
            summaries = [
                s
                for s in summaries
                if query in s.name.lower()
                or (s.source and query in s.source.lower())
                or any(query in str(v).lower() for v in s.metadata.values())
            ]

        # Difficulty filter
        if difficulty:
            summaries = [
                s for s in summaries if s.metadata.get("difficulty") in difficulty
            ]

        # Category filter
        if category:
            summaries = [s for s in summaries if s.metadata.get("category") in category]

        # Tag filter
        if tag:
            summaries = [
                s
                for s in summaries
                if any(t in s.metadata.get("tags", []) for t in tag)
            ]

        # Paginate
        total = len(summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_summaries = summaries[start_idx:end_idx]

        return PaginatedResponse(
            items=page_summaries,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @app.get(
        "/api/task-definitions/{name}",
        response_model=TaskDefinitionDetail,
    )
    def get_task_definition(name: str) -> TaskDefinitionDetail:
        """Get full detail for a task definition."""
        task_dir = _validate_task_name(name)
        if not task_dir.exists() or not (task_dir / "task.toml").exists():
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")

        config = task_scanner.get_task_config(name)
        instruction = task_scanner.get_instruction(name)
        paths_info = task_scanner.get_task_paths_info(name)

        return TaskDefinitionDetail(
            name=name,
            task_dir=str(task_dir.resolve()),
            config=config.model_dump(mode="json") if config else {},
            instruction=instruction,
            has_instruction=paths_info["has_instruction"],
            has_environment=paths_info["has_environment"],
            has_tests=paths_info["has_tests"],
            has_solution=paths_info["has_solution"],
        )

    @app.get("/api/task-definitions/{name}/files")
    def list_task_definition_files(name: str) -> list[FileInfo]:
        """List all files in a task definition directory."""
        task_dir = _validate_task_name(name)
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")

        raw_files = task_scanner.list_files(name)
        return [
            FileInfo(
                path=f["path"],  # type: ignore[arg-type]
                name=f["name"],  # type: ignore[arg-type]
                is_dir=f["is_dir"],  # type: ignore[arg-type]
                size=f["size"],  # type: ignore[arg-type]
            )
            for f in raw_files
        ]

    @app.get(
        "/api/task-definitions/{name}/files/{file_path:path}",
        response_model=None,
    )
    def get_task_definition_file(
        name: str, file_path: str
    ) -> PlainTextResponse | FileResponse:
        """Get content of a file in a task definition directory."""
        task_dir = _validate_task_name(name)
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")

        # Resolve the path and ensure it's within the task directory
        try:
            full_path = (task_dir / file_path).resolve()
            if (
                task_dir.resolve() not in full_path.parents
                and full_path != task_dir.resolve()
            ):
                raise HTTPException(status_code=403, detail="Access denied")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if full_path.is_dir():
            raise HTTPException(status_code=400, detail="Cannot read directory")

        def _format_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"

        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {_format_size(file_size)} (max {_format_size(MAX_FILE_SIZE)})",
            )

        image_extensions = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        suffix = full_path.suffix.lower()
        if suffix in image_extensions:
            return FileResponse(
                path=full_path,
                media_type=image_extensions[suffix],
                filename=full_path.name,
            )

        try:
            content = full_path.read_text()
            return PlainTextResponse(content)
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=415, detail="File is binary and cannot be displayed"
            )

    class ChatRequest(BaseModel):
        message: str

    @app.post("/api/task-definitions/{name}/chat")
    async def chat_with_task(name: str, request: ChatRequest) -> StreamingResponse:
        """Chat with Claude about a task definition."""
        task_dir = _validate_task_name(name)
        if not task_dir.exists() or not (task_dir / "task.toml").exists():
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")

        try:
            session = await chat_manager.get_or_create(name)
        except Exception as e:
            error_name = type(e).__name__
            if error_name == "CLINotFoundError":
                raise HTTPException(
                    status_code=503,
                    detail="Claude CLI is not installed. Install it to use chat.",
                )
            raise HTTPException(
                status_code=500, detail=f"Failed to start chat session: {str(e)}"
            )

        return StreamingResponse(
            stream_chat_response(session, request.message),
            media_type="text/event-stream",
        )

    @app.delete("/api/task-definitions/{name}/chat")
    async def reset_chat(name: str) -> dict[str, str]:
        """Reset (close) a chat session for a task."""
        await chat_manager.close(name)
        return {"status": "ok"}


def _register_job_endpoints(app: FastAPI, jobs_dir: Path) -> None:
    """Register API endpoints for job browsing."""

    scanner = JobScanner(jobs_dir)
    resolved_jobs_dir = jobs_dir.resolve()

    def _validate_job_path(job_name: str) -> Path:
        """Validate job name and return the resolved job directory."""
        job_dir = (jobs_dir / job_name).resolve()
        if resolved_jobs_dir not in job_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid job name")
        return job_dir

    def _validate_trial_path(job_name: str, trial_name: str) -> Path:
        """Validate trial path and return the resolved trial directory."""
        job_dir = _validate_job_path(job_name)
        trial_dir = (job_dir / trial_name).resolve()
        if job_dir not in trial_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid trial name")
        return trial_dir

    def _resolve_step_root(trial_dir: Path, step: str | None) -> Path:
        """Return the directory to read step-scoped files from.

        When ``step`` is set, resolves to ``trial_dir/steps/{step}`` after
        validating the path stays inside the trial directory. When it's
        ``None``, returns ``trial_dir`` unchanged.
        """
        if step is None:
            return trial_dir
        trial_resolved = trial_dir.resolve()
        step_dir = (trial_dir / "steps" / step).resolve()
        if trial_resolved not in step_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid step name")
        if not step_dir.exists():
            raise HTTPException(status_code=404, detail=f"Step '{step}' not found")
        return step_dir

    def _critiques_root(job_name: str) -> Path:
        job_dir = _validate_job_path(job_name)
        return (job_dir / CRITIQUE_RUNS_DIRNAME).resolve()

    def _validate_critique_run_path(job_name: str, critique_run_name: str) -> Path:
        critiques_root = _critiques_root(job_name)
        run_dir = (critiques_root / critique_run_name).resolve()
        if critiques_root not in run_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid critique run name")
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Critique run '{critique_run_name}' not found",
            )
        return run_dir

    def _validate_critique_item_path(
        job_name: str, critique_run_name: str, trial_name: str
    ) -> Path:
        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        item_dir = (run_dir / trial_name).resolve()
        if run_dir not in item_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid critique item name")
        return item_dir

    def _read_json_file(path: Path) -> Any | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _read_text_file(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            file_size = path.stat().st_size
        except OSError:
            return "[Error reading file]"
        if file_size > MAX_FILE_SIZE:
            return (
                f"[File too large: {_format_size(file_size)} "
                f"(max {_format_size(MAX_FILE_SIZE)})]"
            )
        try:
            return path.read_text()
        except Exception:
            return "[Error reading file]"

    def _split_model_name(model_name: str | None) -> tuple[str | None, str | None]:
        if not model_name:
            return None, None
        parts = model_name.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, model_name

    def _critique_run_dirs(job_name: str) -> list[Path]:
        critiques_root = _critiques_root(job_name)
        if not critiques_root.exists():
            return []
        return sorted(
            [d for d in critiques_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )

    def _critique_item_dirs(run_dir: Path) -> list[Path]:
        return sorted(
            [d for d in run_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    def _critique_item_status(
        item_dir: Path,
        metadata: dict[str, Any] | None,
        *,
        run_finished: bool = False,
    ) -> str:
        if metadata:
            if metadata.get("exception_info") is not None:
                return "failed"
            if metadata.get("finished_at") is not None:
                return "completed"
            if metadata.get("started_at") is not None:
                return "running"
        if item_dir.exists():
            if run_finished:
                return "missing"
            if (
                (item_dir / "critique.log").exists()
                or (item_dir / "agent").exists()
                or (item_dir / "artifacts").exists()
            ):
                return "running"
            return "pending"
        return "pending"

    def _critique_result_for_item(
        item_dir: Path, metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        artifacts_result = _read_json_file(
            item_dir / "artifacts" / CRITIQUE_RESULT_FILENAME
        )
        if isinstance(artifacts_result, dict):
            return artifacts_result
        return None

    def _critique_tags(result: dict[str, Any] | None) -> list[str]:
        if not result:
            return []

        tags: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value and value not in tags:
                tags.append(value)

        add(result.get("tag"))
        raw_tags = result.get("tags")
        if isinstance(raw_tags, str):
            add(raw_tags)
        elif isinstance(raw_tags, list):
            for tag in raw_tags:
                add(tag)

        return tags

    def _critique_rating(result: dict[str, Any] | None) -> str | None:
        if not result:
            return None
        rating = result.get("rating")
        return rating if rating in {"good", "bad"} else None

    def _critique_feedback(result: dict[str, Any] | None) -> str | None:
        if not result:
            return None
        feedback = result.get("feedback")
        return feedback if isinstance(feedback, str) else None

    def _number_or_none(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    def _flatten_critique_values(
        result: dict[str, Any] | None,
        *,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        if not result:
            return {}

        values: dict[str, Any] = {}

        def visit(prefix: str, value: Any, depth: int) -> None:
            if isinstance(value, str | int | float | bool) or value is None:
                values[prefix] = value
                return
            if depth >= max_depth or not isinstance(value, dict):
                return
            for key, child in value.items():
                if not isinstance(key, str):
                    continue
                path = f"{prefix}.{key}" if prefix else key
                visit(path, child, depth + 1)

        for key, value in result.items():
            if isinstance(key, str):
                visit(key, value, 0)
        tags = _critique_tags(result)
        if tags:
            values["tags"] = ", ".join(tags)
        return values

    def _critique_item_summary(
        item_dir: Path,
        *,
        job_name: str | None = None,
        run_finished: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CritiqueItemSummary:
        metadata = metadata or _read_json_file(item_dir / "critique-metadata.json")
        if not isinstance(metadata, dict):
            metadata = None

        artifacts_dir = item_dir / "artifacts"
        result_json = artifacts_dir / CRITIQUE_RESULT_FILENAME
        result_md = artifacts_dir / CRITIQUE_MARKDOWN_FILENAME
        exception_info = metadata.get("exception_info") if metadata else None
        error_type = (
            exception_info.get("exception_type")
            if isinstance(exception_info, dict)
            else None
        )
        source_trial_name = (
            str(metadata.get("source_trial_name"))
            if metadata and metadata.get("source_trial_name") is not None
            else item_dir.name
        )
        source_result = (
            scanner.get_trial_result(job_name, source_trial_name) if job_name else None
        )
        model_info = source_result.agent_info.model_info if source_result else None
        source_reward = (
            source_result.verifier_result.rewards.get("reward")
            if source_result
            and source_result.verifier_result
            and source_result.verifier_result.rewards
            else None
        )
        agent_result = metadata.get("agent_result") if metadata else None
        cost_usd = (
            _number_or_none(agent_result.get("cost_usd"))
            if isinstance(agent_result, dict)
            else None
        )
        critique_result = _critique_result_for_item(item_dir, metadata)

        return CritiqueItemSummary(
            source_trial_name=source_trial_name,
            critique_trial_name=(
                str(metadata.get("critique_trial_name"))
                if metadata and metadata.get("critique_trial_name") is not None
                else None
            ),
            task_name=(
                source_result.task_name
                if source_result
                else (
                    str(metadata.get("task_name"))
                    if metadata and metadata.get("task_name") is not None
                    else None
                )
            ),
            source=source_result.source if source_result else None,
            agent_name=source_result.agent_info.name if source_result else None,
            model_provider=model_info.provider if model_info else None,
            model_name=model_info.name if model_info else None,
            source_reward=source_reward,
            source_error_type=(
                source_result.exception_info.exception_type
                if source_result and source_result.exception_info
                else None
            ),
            cost_usd=cost_usd,
            rating=_critique_rating(critique_result),
            tags=_critique_tags(critique_result),
            feedback=_critique_feedback(critique_result),
            critique_values=_flatten_critique_values(critique_result),
            status=_critique_item_status(
                item_dir, metadata, run_finished=run_finished
            ),
            started_at=metadata.get("started_at") if metadata else None,
            finished_at=metadata.get("finished_at") if metadata else None,
            error_type=error_type,
            has_metadata=metadata is not None,
            has_result_json=result_json.exists(),
            has_result_md=result_md.exists(),
        )

    def _critique_item_summaries(
        run_dir: Path, *, job_name: str, run_finished: bool
    ) -> list[CritiqueItemSummary]:
        return [
            _critique_item_summary(
                item_dir, job_name=job_name, run_finished=run_finished
            )
            for item_dir in _critique_item_dirs(run_dir)
        ]

    def _filter_critique_items(
        items: list[CritiqueItemSummary],
        *,
        q: str | None = None,
        agent: list[str] | None = None,
        provider: list[str] | None = None,
        model: list[str] | None = None,
        source: list[str] | None = None,
        task: list[str] | None = None,
        rating: list[str] | None = None,
        tag: list[str] | None = None,
        source_trials: str = "all",
        status: list[str] | None = None,
    ) -> list[CritiqueItemSummary]:
        filtered = items
        if q:
            query = q.lower()
            filtered = [
                item
                for item in filtered
                if query in item.source_trial_name.lower()
                or (
                    item.critique_trial_name
                    and query in item.critique_trial_name.lower()
                )
                or (item.task_name and query in item.task_name.lower())
                or (item.source and query in item.source.lower())
                or (item.rating and query in item.rating.lower())
                or query in item.status.lower()
                or (item.error_type and query in item.error_type.lower())
                or (
                    item.source_error_type
                    and query in item.source_error_type.lower()
                )
                or (item.feedback and query in item.feedback.lower())
                or any(query in item_tag.lower() for item_tag in item.tags)
            ]
        if agent:
            agents = set(agent)
            filtered = [item for item in filtered if item.agent_name in agents]
        if provider:
            providers = set(provider)
            filtered = [
                item for item in filtered if item.model_provider in providers
            ]
        if model:
            models = set(model)
            filtered = [item for item in filtered if item.model_name in models]
        if source:
            sources = set(source)
            filtered = [item for item in filtered if item.source in sources]
        if task:
            tasks = set(task)
            filtered = [item for item in filtered if item.task_name in tasks]
        if rating:
            ratings = set(rating)
            filtered = [
                item for item in filtered if (item.rating or "none") in ratings
            ]
        if tag:
            tags = set(tag)
            filtered = [
                item for item in filtered if any(item_tag in tags for item_tag in item.tags)
            ]
        if source_trials != "all":
            filtered = [
                item
                for item in filtered
                if (
                    (source_trials == "errored" and item.source_error_type)
                    or (
                        source_trials == "non_errored"
                        and not item.source_error_type
                    )
                    or (
                        source_trials == "successful"
                        and not item.source_error_type
                        and item.source_reward is not None
                        and item.source_reward >= 1
                    )
                )
            ]
        if status:
            statuses = set(status)
            filtered = [item for item in filtered if item.status in statuses]
        return filtered

    def _sort_critique_items(
        items: list[CritiqueItemSummary],
        *,
        sort_by: str | None,
        sort_order: str,
    ) -> None:
        if not sort_by:
            return
        reverse = sort_order == "desc"

        def value(item: CritiqueItemSummary) -> Any:
            if sort_by == "source_outcome":
                return item.source_error_type or item.source_reward
            if sort_by == "tags":
                return ", ".join(item.tags)
            return getattr(item, sort_by, None)

        def sort_key(item: CritiqueItemSummary) -> tuple[bool, int, float | str]:
            raw_value = value(item)
            if raw_value is None:
                return (True, 0, "")
            if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
                return (False, 0, float(raw_value))
            return (False, 1, str(raw_value))

        items.sort(key=sort_key, reverse=reverse)

    def _critique_run_summary(run_dir: Path) -> CritiqueRunSummary:
        config = _read_json_file(run_dir / "config.json")
        result = _read_json_file(run_dir / "result.json")
        if not isinstance(config, dict):
            config = None
        if not isinstance(result, dict):
            result = None

        source = result or {}
        run_config = (
            source.get("config") if isinstance(source.get("config"), dict) else config
        )
        agent = (
            run_config.get("agent")
            if isinstance(run_config, dict)
            and isinstance(run_config.get("agent"), dict)
            else {}
        )
        model_provider, model_name = _split_model_name(agent.get("model_name"))

        environment = (
            run_config.get("environment")
            if isinstance(run_config, dict)
            and isinstance(run_config.get("environment"), dict)
            else {}
        )
        environment_type = environment.get("type")

        failed_items = (
            source.get("failed_items")
            if isinstance(source.get("failed_items"), list)
            else []
        )
        run_finished = result is not None and source.get("finished_at") is not None
        item_dirs = _critique_item_dirs(run_dir)
        item_summaries = [
            _critique_item_summary(item_dir, run_finished=run_finished)
            for item_dir in item_dirs
        ]

        n_completed = sum(1 for item in item_summaries if item.finished_at is not None)
        n_running = sum(1 for item in item_summaries if item.status == "running")

        filesystem_failures = sum(
            1 for item in item_summaries if item.status == "failed"
        )
        root_item_count = len(failed_items)

        if result is not None:
            n_items = max(
                int(source.get("n_items") or 0), root_item_count, len(item_dirs)
            )
            n_failed = max(
                int(source.get("n_failed") or 0),
                len(failed_items),
                filesystem_failures,
            )
            status = "completed"
            if not run_finished:
                status = "running"
        else:
            n_items = len(item_dirs)
            n_failed = filesystem_failures
            status = "running" if item_dirs else "pending"
        terminal_item_dirs = sum(
            1
            for item in item_summaries
            if item.status in {"completed", "failed", "missing"}
        )
        failed_items_without_dirs = max(n_failed - filesystem_failures, 0)
        n_unaccounted = max(
            n_items - terminal_item_dirs - failed_items_without_dirs - n_running, 0
        )
        n_missing = n_unaccounted if run_finished else 0
        n_pending = 0 if run_finished else n_unaccounted

        return CritiqueRunSummary(
            name=run_dir.name,
            id=source.get("id") if result else None,
            status=status,
            started_at=source.get("started_at") if result else None,
            finished_at=source.get("finished_at") if result else None,
            n_items=n_items,
            n_completed_items=n_completed,
            n_running_items=n_running,
            n_pending_items=n_pending,
            n_missing_items=n_missing,
            n_failed_items=n_failed,
            agent_name=agent.get("name"),
            model_provider=model_provider,
            model_name=model_name,
            environment_type=environment_type,
            critique_uri=run_dir.resolve().as_uri(),
            has_config=config is not None,
            has_result=result is not None,
        )

    def _critique_artifacts(artifacts_dir: Path) -> tuple[list[FileInfo], Any | None]:
        manifest = _read_json_file(artifacts_dir / "manifest.json")
        files: list[FileInfo] = []
        if not artifacts_dir.exists():
            return files, manifest

        def scan_dir(dir_path: Path, relative_base: str = "") -> None:
            try:
                for item in sorted(dir_path.iterdir()):
                    relative_path = (
                        f"{relative_base}/{item.name}" if relative_base else item.name
                    )
                    if item.name == "manifest.json" and not relative_base:
                        continue
                    if item.is_dir():
                        scan_dir(item, relative_path)
                    else:
                        files.append(
                            FileInfo(
                                path=relative_path,
                                name=item.name,
                                is_dir=False,
                                size=item.stat().st_size,
                            )
                        )
            except PermissionError:
                pass

        scan_dir(artifacts_dir)
        return files, manifest

    def _trial_critique_detail(
        job_name: str, trial_name: str, critique_run_name: str
    ) -> TrialCritiqueDetail:
        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        item_dir = _validate_critique_item_path(
            job_name, critique_run_name, trial_name
        )
        run_config = _read_json_file(run_dir / "config.json")
        metadata = _read_json_file(item_dir / "critique-metadata.json")
        if not isinstance(run_config, dict):
            run_config = None
        if not isinstance(metadata, dict):
            metadata = None

        artifacts_dir = item_dir / "artifacts"
        result_path = artifacts_dir / CRITIQUE_RESULT_FILENAME
        markdown_path = artifacts_dir / CRITIQUE_MARKDOWN_FILENAME
        critique_result = _read_json_file(result_path)
        if not isinstance(critique_result, dict):
            critique_result = None

        files, manifest = _critique_artifacts(artifacts_dir)
        run_summary = _critique_run_summary(run_dir)

        return TrialCritiqueDetail(
            run_name=critique_run_name,
            status=_critique_item_status(
                item_dir,
                metadata,
                run_finished=run_summary.finished_at is not None,
            ),
            critique_uri=item_dir.resolve().as_uri() if item_dir.exists() else None,
            run_config=run_config,
            metadata=metadata,
            critique_result=critique_result,
            markdown=_read_text_file(markdown_path),
            log=_read_text_file(item_dir / "critique.log"),
            exception_text=_read_text_file(item_dir / "exception.txt"),
            files=files,
            manifest=manifest,
            has_item_dir=item_dir.exists(),
            has_artifacts_dir=artifacts_dir.exists(),
            has_result_json=result_path.exists(),
            has_result_md=markdown_path.exists(),
        )

    def _get_all_job_summaries() -> list[JobSummary]:
        """Get all job summaries (used by both list_jobs and get_job_filters)."""
        job_names = scanner.list_jobs()
        summaries = []

        for name in job_names:
            result = scanner.get_job_result(name)
            config = scanner.get_job_config(name)

            # Extract unique agents, providers, models, datasets, and environment type from config
            agents: list[str] = []
            providers: list[str] = []
            models: list[str] = []
            datasets: list[str] = []
            environment_type: str | None = None
            if config:
                agents = sorted(
                    set(agent.name for agent in config.agents if agent.name is not None)
                )
                # Extract dataset names
                for ds in config.datasets:
                    if ds.is_local():
                        assert ds.path is not None
                        datasets.append(ds.path.name)
                    elif ds.is_package() or ds.is_registry():
                        assert ds.name is not None
                        datasets.append(ds.name)
                datasets = sorted(set(datasets))
                # Extract provider from model_name (format: "provider/model")
                for agent in config.agents:
                    if agent.model_name:
                        parts = agent.model_name.split("/", 1)
                        if len(parts) == 2:
                            providers.append(parts[0])
                            models.append(parts[1])
                        else:
                            models.append(agent.model_name)
                providers = sorted(set(providers))
                models = sorted(set(models))
                if config.environment.type:
                    environment_type = config.environment.type.value

            if result:
                total_agent_steps: int | None = None
                for trial_name in scanner.list_trials(name):
                    trial_result = scanner.get_trial_result(name, trial_name)
                    if trial_result is None:
                        continue
                    agent_steps = _agent_step_count_from_result(trial_result)
                    if agent_steps is not None:
                        total_agent_steps = (total_agent_steps or 0) + agent_steps

                # Extract evals from stats
                evals = {
                    key: EvalSummary(metrics=eval_stats.metrics)
                    for key, eval_stats in result.stats.evals.items()
                    if eval_stats.metrics
                }
                summaries.append(
                    JobSummary(
                        name=name,
                        id=result.id,
                        started_at=result.started_at,
                        updated_at=result.updated_at,
                        finished_at=result.finished_at,
                        n_total_trials=result.n_total_trials,
                        n_completed_trials=result.stats.n_completed_trials,
                        n_errored_trials=result.stats.n_errored_trials,
                        datasets=datasets,
                        agents=agents,
                        providers=providers,
                        models=models,
                        environment_type=environment_type,
                        evals=evals,
                        total_input_tokens=_uncached_input(
                            result.stats.n_input_tokens, result.stats.n_cache_tokens
                        ),
                        total_cached_input_tokens=result.stats.n_cache_tokens,
                        total_output_tokens=result.stats.n_output_tokens,
                        total_cost_usd=result.stats.cost_usd,
                        total_agent_steps=total_agent_steps,
                    )
                )
            else:
                summaries.append(
                    JobSummary(
                        name=name,
                        datasets=datasets,
                        agents=agents,
                        providers=providers,
                        models=models,
                        environment_type=environment_type,
                    )
                )

        # Sort by started_at descending (most recent first), jobs without started_at go last.
        # Match the existing date-filter path: strip tzinfo when persisted jobs mix forms.
        summaries.sort(
            key=lambda s: (
                s.started_at is not None,
                s.started_at.replace(tzinfo=None) if s.started_at is not None else None,
            ),
            reverse=True,
        )
        return summaries

    @app.get("/api/jobs/filters", response_model=JobFilters)
    def get_job_filters() -> JobFilters:
        """Get available filter options for jobs list."""
        from collections import Counter

        summaries = _get_all_job_summaries()

        # Count occurrences of agents, providers, and models
        agent_counts: Counter[str] = Counter()
        provider_counts: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()

        for summary in summaries:
            for agent in summary.agents:
                agent_counts[agent] += 1
            for provider in summary.providers:
                provider_counts[provider] += 1
            for model in summary.models:
                model_counts[model] += 1

        return JobFilters(
            agents=[
                FilterOption(value=v, count=c) for v, c in sorted(agent_counts.items())
            ],
            providers=[
                FilterOption(value=v, count=c)
                for v, c in sorted(provider_counts.items())
            ],
            models=[
                FilterOption(value=v, count=c) for v, c in sorted(model_counts.items())
            ],
        )

    @app.get("/api/jobs", response_model=PaginatedResponse[JobSummary])
    def list_jobs(
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=100, description="Number of items per page"
        ),
        q: str | None = Query(default=None, description="Search query"),
        agent: list[str] = Query(default=[], description="Filter by agent names"),
        provider: list[str] = Query(default=[], description="Filter by provider names"),
        model: list[str] = Query(default=[], description="Filter by model names"),
        date: list[str] = Query(
            default=[],
            description="Filter by date ranges (today, week, month)",
        ),
    ) -> PaginatedResponse[JobSummary]:
        """List all jobs with summary information."""
        from datetime import datetime, timedelta

        summaries = _get_all_job_summaries()

        # Filter by search query
        if q:
            query = q.lower()
            summaries = [
                s
                for s in summaries
                if query in s.name.lower()
                or any(query in agent_name.lower() for agent_name in s.agents)
                or any(query in provider_name.lower() for provider_name in s.providers)
                or any(query in model_name.lower() for model_name in s.models)
            ]

        # Filter by agents (OR within agents)
        if agent:
            summaries = [s for s in summaries if any(a in s.agents for a in agent)]

        # Filter by providers (OR within providers)
        if provider:
            summaries = [
                s for s in summaries if any(p in s.providers for p in provider)
            ]

        # Filter by models (OR within models)
        if model:
            summaries = [s for s in summaries if any(m in s.models for m in model)]

        # Filter by date (OR within dates - use the most permissive)
        if date:
            now = datetime.now()
            cutoffs = []
            for d in date:
                if d == "today":
                    cutoffs.append(now - timedelta(days=1))
                elif d == "week":
                    cutoffs.append(now - timedelta(weeks=1))
                elif d == "month":
                    cutoffs.append(now - timedelta(days=30))

            if cutoffs:
                # Use the earliest cutoff (most permissive)
                cutoff = min(cutoffs)
                summaries = [
                    s
                    for s in summaries
                    if s.started_at is not None
                    and s.started_at.replace(tzinfo=None) >= cutoff
                ]

        # Paginate
        total = len(summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_summaries = summaries[start_idx:end_idx]

        return PaginatedResponse(
            items=page_summaries,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @app.get("/api/jobs/{job_name}")
    def get_job(job_name: str) -> dict[str, Any]:
        """Get full job result details."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        result = scanner.get_job_result(job_name)
        if result is None:
            # Return minimal info for jobs without result.json (incomplete jobs)
            # Count trials from subdirectories
            n_trials = sum(1 for d in job_dir.iterdir() if d.is_dir())
            stats = JobStats.from_counts(
                n_total_trials=n_trials,
            )
            return {
                "id": job_name,
                "started_at": None,
                "updated_at": None,
                "finished_at": None,
                "n_total_trials": n_trials,
                "stats": stats.model_dump(mode="json"),
                "job_uri": job_dir.resolve().as_uri(),
            }

        # Convert to dict and add job_uri
        result_dict = result.model_dump(mode="json")
        result_dict["job_uri"] = job_dir.resolve().as_uri()
        return result_dict

    @app.get("/api/jobs/{job_name}/summary")
    def get_job_summary(job_name: str) -> dict[str, str | None]:
        """Get job analysis (analysis.md file at job root)."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        analysis_path = job_dir / "analysis.md"
        if analysis_path.exists():
            try:
                return {"summary": analysis_path.read_text()}
            except Exception:
                return {"summary": "[Error reading file]"}
        return {"summary": None}

    @app.get("/api/jobs/{job_name}/analysis")
    def get_job_analysis(job_name: str) -> dict[str, Any]:
        """Get full structured analysis (analysis.json) for a job."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        analysis_path = job_dir / "analysis.json"
        if analysis_path.exists():
            try:
                return json.loads(analysis_path.read_text())
            except Exception:
                raise HTTPException(
                    status_code=500, detail="Error reading analysis.json"
                )
        return {}

    @app.get(
        "/api/jobs/{job_name}/critiques",
        response_model=PaginatedResponse[CritiqueRunSummary],
    )
    def list_critique_runs(
        job_name: str,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=100, description="Number of items per page"
        ),
        q: str | None = Query(default=None, description="Search query"),
        status: list[str] | None = Query(
            default=None, description="Filter by lifecycle status"
        ),
        has_failures: bool = Query(
            default=False, description="Only include runs with failed critique items"
        ),
    ) -> PaginatedResponse[CritiqueRunSummary]:
        """List critique runs stored under a source job."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        summaries = [_critique_run_summary(d) for d in _critique_run_dirs(job_name)]
        if q:
            query = q.lower()
            summaries = [
                s
                for s in summaries
                if query in s.name.lower()
                or (s.agent_name and query in s.agent_name.lower())
                or (s.model_provider and query in s.model_provider.lower())
                or (s.model_name and query in s.model_name.lower())
                or (s.environment_type and query in s.environment_type.lower())
            ]
        if status:
            statuses = set(status)
            summaries = [s for s in summaries if s.status in statuses]
        if has_failures:
            summaries = [s for s in summaries if s.n_failed_items > 0]

        total = len(summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_summaries = summaries[start_idx:end_idx]

        return PaginatedResponse(
            items=page_summaries,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @app.get(
        "/api/jobs/{job_name}/critiques/{critique_run_name}",
        response_model=CritiqueRunDetail,
    )
    def get_critique_run(
        job_name: str,
        critique_run_name: str,
        include_items: bool = Query(
            default=True,
            description="Include per-source-trial item summaries in the response",
        ),
    ) -> CritiqueRunDetail:
        """Get a critique run and its per-source-trial items."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        run_summary = _critique_run_summary(run_dir)
        run_finished = run_summary.finished_at is not None
        config = _read_json_file(run_dir / "config.json")
        result = _read_json_file(run_dir / "result.json")
        return CritiqueRunDetail(
            run=run_summary,
            config=config if isinstance(config, dict) else None,
            result=result if isinstance(result, dict) else None,
            items=[
                _critique_item_summary(
                    item_dir, job_name=job_name, run_finished=run_finished
                )
                for item_dir in _critique_item_dirs(run_dir)
            ]
            if include_items
            else [],
        )

    @app.get(
        "/api/jobs/{job_name}/critiques/{critique_run_name}/items/filters",
        response_model=CritiqueItemFilters,
    )
    def get_critique_item_filters(
        job_name: str, critique_run_name: str
    ) -> CritiqueItemFilters:
        """Get available filter options for critique items within a critique run."""
        from collections import Counter

        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        run_summary = _critique_run_summary(run_dir)
        items = _critique_item_summaries(
            run_dir, job_name=job_name, run_finished=run_summary.finished_at is not None
        )

        agent_counts: Counter[str] = Counter()
        provider_counts: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        rating_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for item in items:
            if item.agent_name:
                agent_counts[item.agent_name] += 1
            if item.model_provider:
                provider_counts[item.model_provider] += 1
            if item.model_name:
                model_counts[item.model_name] += 1
            if item.source:
                source_counts[item.source] += 1
            if item.task_name:
                task_counts[item.task_name] += 1
            rating_counts[item.rating or "none"] += 1
            for item_tag in item.tags:
                tag_counts[item_tag] += 1
            status_counts[item.status] += 1

        return CritiqueItemFilters(
            agents=[
                FilterOption(value=v, count=c) for v, c in sorted(agent_counts.items())
            ],
            providers=[
                FilterOption(value=v, count=c)
                for v, c in sorted(provider_counts.items())
            ],
            models=[
                FilterOption(value=v, count=c) for v, c in sorted(model_counts.items())
            ],
            sources=[
                FilterOption(value=v, count=c) for v, c in sorted(source_counts.items())
            ],
            tasks=[
                FilterOption(value=v, count=c) for v, c in sorted(task_counts.items())
            ],
            ratings=[
                FilterOption(value=v, count=c) for v, c in sorted(rating_counts.items())
            ],
            tags=[
                FilterOption(value=v, count=c) for v, c in sorted(tag_counts.items())
            ],
            statuses=[
                FilterOption(value=v, count=c) for v, c in sorted(status_counts.items())
            ],
        )

    @app.get(
        "/api/jobs/{job_name}/critiques/{critique_run_name}/items",
        response_model=PaginatedResponse[CritiqueItemSummary],
    )
    def list_critique_items(
        job_name: str,
        critique_run_name: str,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=50000, description="Number of items per page"
        ),
        q: str | None = Query(default=None, description="Search query"),
        agent: list[str] | None = Query(default=None, description="Filter by source agent"),
        provider: list[str] | None = Query(
            default=None, description="Filter by source model provider"
        ),
        model: list[str] | None = Query(default=None, description="Filter by source model"),
        source: list[str] | None = Query(default=None, description="Filter by dataset"),
        task: list[str] | None = Query(default=None, description="Filter by task"),
        rating: list[str] | None = Query(default=None, description="Filter by rating"),
        tag: list[str] | None = Query(default=None, description="Filter by tag"),
        source_trials: str = Query(
            default="all",
            pattern="^(all|non_errored|errored|successful)$",
            description="Filter by source trial outcome",
        ),
        status: list[str] | None = Query(default=None, description="Filter by item state"),
        sort_by: str | None = Query(default=None, description="Field to sort by"),
        sort_order: str = Query(default="asc", description="Sort order (asc or desc)"),
    ) -> PaginatedResponse[CritiqueItemSummary]:
        """List critique items for a critique run with server-side filtering and paging."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        run_summary = _critique_run_summary(run_dir)
        summaries = _critique_item_summaries(
            run_dir, job_name=job_name, run_finished=run_summary.finished_at is not None
        )
        summaries = _filter_critique_items(
            summaries,
            q=q,
            agent=agent,
            provider=provider,
            model=model,
            source=source,
            task=task,
            rating=rating,
            tag=tag,
            source_trials=source_trials,
            status=status,
        )
        _sort_critique_items(
            summaries, sort_by=sort_by, sort_order=sort_order
        )

        total = len(summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        return PaginatedResponse(
            items=summaries[start_idx:end_idx],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @app.get(
        "/api/jobs/{job_name}/critiques/{critique_run_name}/heatmap",
        response_model=CritiqueHeatmapData,
    )
    def get_critique_heatmap(
        job_name: str,
        critique_run_name: str,
        row_by: str = Query(default="rating", pattern="^(rating|tag)$"),
        column_by: str = Query(default="task", pattern="^(task|dataset)$"),
        source_trials: str = Query(
            default="all", pattern="^(all|non_errored|errored|successful)$"
        ),
        rating: str = Query(default="all", pattern="^(all|good|bad)$"),
        rating_value: list[str] | None = Query(
            default=None, description="Filter by critique rating values"
        ),
        tag: list[str] | None = Query(default=None, description="Filter by critique tags"),
        status: list[str] | None = Query(
            default=None, description="Filter by critique item lifecycle status"
        ),
        agent: list[str] | None = Query(default=None, description="Filter by source agent"),
        provider: list[str] | None = Query(
            default=None, description="Filter by source model provider"
        ),
        model: list[str] | None = Query(default=None, description="Filter by source model"),
        source: list[str] | None = Query(default=None, description="Filter by dataset"),
        task: list[str] | None = Query(default=None, description="Filter by task"),
        q: str | None = Query(default=None, description="Search query"),
    ) -> CritiqueHeatmapData:
        """Build a heatmap from per-item critique artifact results."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        run_dir = _validate_critique_run_path(job_name, critique_run_name)
        run_summary = _critique_run_summary(run_dir)
        run_finished = run_summary.finished_at is not None

        rows: dict[str, CritiqueHeatmapRow] = {}
        columns: dict[str, CritiqueHeatmapColumn] = {}
        cell_stats: dict[tuple[str, str], dict[str, Any]] = {}

        def column_for(item: CritiqueItemSummary) -> CritiqueHeatmapColumn:
            if column_by == "dataset":
                key = f"dataset::{item.source or ''}"
                return CritiqueHeatmapColumn(
                    key=key,
                    label=item.source or "No dataset",
                    source=item.source,
                )
            key = f"task::{item.source or ''}::{item.task_name or ''}"
            label = (
                f"{item.source} / {item.task_name}"
                if item.source and item.task_name
                else item.task_name or "Unknown task"
            )
            return CritiqueHeatmapColumn(
                key=key,
                label=label,
                source=item.source,
                task_name=item.task_name,
            )

        def row_values(item: CritiqueItemSummary) -> list[CritiqueHeatmapRow]:
            if row_by == "tag":
                tags = item.tags or ["No tags"]
                return [
                    CritiqueHeatmapRow(
                        key=f"tag::{tag}",
                        label=tag,
                        kind="tag",
                        value=None if tag == "No tags" else tag,
                    )
                    for tag in tags
                ]

            rating = item.rating or "No rating"
            return [
                CritiqueHeatmapRow(
                    key=f"rating::{rating}",
                    label=rating,
                    kind="rating",
                    value=None if rating == "No rating" else rating,
                )
            ]

        def new_stats() -> dict[str, Any]:
            return {
                "n_items": 0,
                "n_good": 0,
                "n_bad": 0,
                "n_errors": 0,
                "rating_counts": {},
                "tag_counts": {},
            }

        def increment(counter: dict[str, int], key: str) -> None:
            counter[key] = counter.get(key, 0) + 1

        items = _filter_critique_items(
            _critique_item_summaries(
                run_dir, job_name=job_name, run_finished=run_finished
            ),
            q=q,
            agent=agent,
            provider=provider,
            model=model,
            source=source,
            task=task,
            rating=rating_value or ([rating] if rating != "all" else None),
            tag=tag,
            source_trials=source_trials,
            status=status,
        )
        for item in items:
            column = column_for(item)
            columns[column.key] = column
            for row in row_values(item):
                rows[row.key] = row
                stats = cell_stats.setdefault((row.key, column.key), new_stats())
                stats["n_items"] += 1
                if item.rating == "good":
                    stats["n_good"] += 1
                elif item.rating == "bad":
                    stats["n_bad"] += 1
                if item.error_type:
                    stats["n_errors"] += 1
                if item.rating:
                    increment(stats["rating_counts"], item.rating)
                for tag in item.tags:
                    increment(stats["tag_counts"], tag)

        cells: dict[str, dict[str, CritiqueHeatmapCell]] = {}
        column_bad_rates: dict[str, list[float]] = {}

        for (row_key, column_key), stats in cell_stats.items():
            n_items = int(stats["n_items"])
            n_good = int(stats["n_good"])
            n_bad = int(stats["n_bad"])
            good_rate = n_good / n_items if n_items else None
            bad_rate = n_bad / n_items if n_items else None
            if bad_rate is not None:
                column_bad_rates.setdefault(column_key, []).append(bad_rate)
            cell = CritiqueHeatmapCell(
                row_key=row_key,
                column_key=column_key,
                n_items=n_items,
                n_good=n_good,
                n_bad=n_bad,
                n_errors=int(stats["n_errors"]),
                good_rate=good_rate,
                bad_rate=bad_rate,
                rating_counts=stats["rating_counts"],
                tag_counts=stats["tag_counts"],
            )
            cells.setdefault(row_key, {})[column_key] = cell

        rating_order = {"bad": 0, "good": 1, "No rating": 2}
        sorted_rows = sorted(
            rows.values(),
            key=lambda row: (
                rating_order.get(row.label, 99) if row_by == "rating" else 0,
                row.label.lower(),
            ),
        )

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        sorted_columns = sorted(
            columns.values(),
            key=lambda column: (
                -mean(column_bad_rates.get(column.key, []))
                if column_by == "task"
                else 0,
                column.label.lower(),
            ),
        )

        return CritiqueHeatmapData(
            rows=sorted_rows,
            columns=sorted_columns,
            cells=cells,
        )

    @app.post("/api/jobs/{job_name}/summarize")
    async def summarize_job(
        job_name: str, request: SummarizeRequest
    ) -> dict[str, str | int | bool | None]:
        """Generate an analysis for a job using pier analyze."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        from pier.analyze.analyzer import Analyzer

        # Map only_failed to filter_passing
        filter_passing: bool | None = None
        if request.only_failed:
            filter_passing = False

        # Respect overwrite flag — return existing analysis if available
        analysis_path = job_dir / "analysis.md"
        if not request.overwrite and analysis_path.exists():
            try:
                return {
                    "summary": analysis_path.read_text(),
                    "n_trials_summarized": 0,
                    "job_summary_created": False,
                }
            except Exception:
                pass  # Fall through to re-analyze

        analyzer = Analyzer(
            model=request.model,
            n_concurrent=request.n_concurrent,
        )

        try:
            result, _failed = await analyzer.analyze_job(
                job_dir, filter_passing=filter_passing
            )
        except ValueError as e:
            if "trial directories found" in str(e):
                return {
                    "summary": None,
                    "n_trials_summarized": 0,
                    "job_summary_created": False,
                }
            raise

        return {
            "summary": result.job_summary,
            "n_trials_summarized": len(result.trials),
            "job_summary_created": True,
        }

    @app.delete("/api/jobs/{job_name}")
    def delete_job(job_name: str) -> dict[str, str]:
        """Delete a job and all its trials."""
        job_dir = _validate_job_path(job_name)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        try:
            shutil.rmtree(job_dir)
            return {"status": "ok", "message": f"Job '{job_name}' deleted"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete job: {str(e)}"
            )

    @app.get("/api/jobs/{job_name}/config", response_model=JobConfig)
    def get_job_config(job_name: str) -> JobConfig:
        """Get job configuration."""
        config = scanner.get_job_config(job_name)
        if not config:
            raise HTTPException(
                status_code=404, detail=f"Config for job '{job_name}' not found"
            )
        return config

    def _get_all_task_summaries(job_name: str) -> list[TaskSummary]:
        """Get all task summaries for a job (used by list_tasks and get_task_filters)."""
        trial_names = scanner.list_trials(job_name)
        if not trial_names:
            return []

        # Group trials by (agent_name, model_provider, model_name, source, task_name)
        groups: dict[
            tuple[str | None, str | None, str | None, str | None, str],
            TaskGroupStats,
        ] = {}

        for name in trial_names:
            result = scanner.get_trial_result(job_name, name)
            if not result:
                continue

            agent_name = result.agent_info.name
            model_info = result.agent_info.model_info
            model_name = model_info.name if model_info else None
            model_provider = model_info.provider if model_info else None
            source = result.source
            task_name = result.task_name

            key = (
                agent_name,
                model_provider,
                model_name,
                source,
                task_name,
            )

            if key not in groups:
                groups[key] = {
                    "n_trials": 0,
                    "n_completed": 0,
                    "n_errors": 0,
                    "exception_types": set(),
                    "total_reward": 0.0,
                    "reward_count": 0,
                    "total_duration_ms": 0.0,
                    "duration_count": 0,
                    "total_input_tokens": 0,
                    "input_tokens_count": 0,
                    "total_cached_input_tokens": 0,
                    "cached_input_tokens_count": 0,
                    "total_output_tokens": 0,
                    "output_tokens_count": 0,
                    "total_cost_usd": 0.0,
                    "cost_usd_count": 0,
                    "total_peak_context_tokens": 0,
                    "peak_context_tokens_count": 0,
                    "total_agent_steps": 0,
                    "agent_steps_count": 0,
                }

            groups[key]["n_trials"] += 1

            if result.finished_at:
                groups[key]["n_completed"] += 1
                if result.started_at:
                    duration_ms = (
                        result.finished_at - result.started_at
                    ).total_seconds() * 1000
                    groups[key]["total_duration_ms"] += duration_ms
                    groups[key]["duration_count"] += 1

            if result.exception_info:
                groups[key]["n_errors"] += 1
                groups[key]["exception_types"].add(result.exception_info.exception_type)

            # Get reward, defaulting to 0 if missing (evaluated but no reward)
            reward = (
                result.verifier_result.rewards.get("reward", 0)
                if result.verifier_result and result.verifier_result.rewards
                else 0
            )
            groups[key]["total_reward"] += reward
            groups[key]["reward_count"] += 1

            n_input, n_cache, n_output, cost = result.compute_token_cost_totals()
            uncached = _uncached_input(n_input, n_cache)
            if uncached is not None:
                groups[key]["total_input_tokens"] += uncached
                groups[key]["input_tokens_count"] += 1
            if n_cache is not None:
                groups[key]["total_cached_input_tokens"] += n_cache
                groups[key]["cached_input_tokens_count"] += 1
            if n_output is not None:
                groups[key]["total_output_tokens"] += n_output
                groups[key]["output_tokens_count"] += 1
            if cost is not None:
                groups[key]["total_cost_usd"] += cost
                groups[key]["cost_usd_count"] += 1

            if (
                result.agent_result is not None
                and result.agent_result.peak_context_tokens is not None
            ):
                groups[key]["total_peak_context_tokens"] += (
                    result.agent_result.peak_context_tokens
                )
                groups[key]["peak_context_tokens_count"] += 1

            agent_steps = _agent_step_count_from_result(result)
            if agent_steps is not None:
                groups[key]["total_agent_steps"] += agent_steps
                groups[key]["agent_steps_count"] += 1

        # Convert to TaskSummary list
        summaries = []
        for (
            agent_name,
            model_provider,
            model_name,
            source,
            task_name,
        ), stats in groups.items():
            avg_reward = (
                stats["total_reward"] / stats["reward_count"]
                if stats["reward_count"] > 0
                else 0.0
            )
            avg_duration_ms = (
                stats["total_duration_ms"] / stats["duration_count"]
                if stats["duration_count"] > 0
                else None
            )
            avg_input_tokens = (
                stats["total_input_tokens"] / stats["input_tokens_count"]
                if stats["input_tokens_count"] > 0
                else None
            )
            avg_cached_input_tokens = (
                stats["total_cached_input_tokens"] / stats["cached_input_tokens_count"]
                if stats["cached_input_tokens_count"] > 0
                else None
            )
            avg_output_tokens = (
                stats["total_output_tokens"] / stats["output_tokens_count"]
                if stats["output_tokens_count"] > 0
                else None
            )
            avg_cost_usd = (
                stats["total_cost_usd"] / stats["cost_usd_count"]
                if stats["cost_usd_count"] > 0
                else None
            )
            avg_peak_context_tokens = (
                stats["total_peak_context_tokens"] / stats["peak_context_tokens_count"]
                if stats["peak_context_tokens_count"] > 0
                else None
            )
            avg_agent_steps = (
                stats["total_agent_steps"] / stats["agent_steps_count"]
                if stats["agent_steps_count"] > 0
                else None
            )

            summaries.append(
                TaskSummary(
                    task_name=task_name,
                    source=source,
                    agent_name=agent_name,
                    model_provider=model_provider,
                    model_name=model_name,
                    n_trials=int(stats["n_trials"]),
                    n_completed=int(stats["n_completed"]),
                    n_errors=int(stats["n_errors"]),
                    exception_types=sorted(stats["exception_types"]),
                    avg_reward=avg_reward,
                    avg_duration_ms=avg_duration_ms,
                    avg_input_tokens=avg_input_tokens,
                    avg_cached_input_tokens=avg_cached_input_tokens,
                    avg_output_tokens=avg_output_tokens,
                    avg_cost_usd=avg_cost_usd,
                    avg_peak_context_tokens=avg_peak_context_tokens,
                    avg_agent_steps=avg_agent_steps,
                )
            )

        return summaries

    @app.get("/api/jobs/{job_name}/tasks/filters", response_model=TaskFilters)
    def get_task_filters(job_name: str) -> TaskFilters:
        """Get available filter options for tasks list within a job."""
        from collections import Counter

        if job_name not in scanner.list_jobs():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        summaries = _get_all_task_summaries(job_name)

        # Count occurrences of each filter value
        agent_counts: Counter[str] = Counter()
        provider_counts: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()

        for summary in summaries:
            if summary.agent_name:
                agent_counts[summary.agent_name] += 1
            if summary.model_provider:
                provider_counts[summary.model_provider] += 1
            if summary.model_name:
                model_counts[summary.model_name] += 1
            if summary.source:
                source_counts[summary.source] += 1
            if summary.task_name:
                task_counts[summary.task_name] += 1

        return TaskFilters(
            agents=[
                FilterOption(value=v, count=c) for v, c in sorted(agent_counts.items())
            ],
            providers=[
                FilterOption(value=v, count=c)
                for v, c in sorted(provider_counts.items())
            ],
            models=[
                FilterOption(value=v, count=c) for v, c in sorted(model_counts.items())
            ],
            sources=[
                FilterOption(value=v, count=c)
                for v, c in sorted(source_counts.items())
            ],
            tasks=[
                FilterOption(value=v, count=c) for v, c in sorted(task_counts.items())
            ],
        )

    @app.get(
        "/api/jobs/{job_name}/tasks", response_model=PaginatedResponse[TaskSummary]
    )
    def list_tasks(
        job_name: str,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=100, description="Number of items per page"
        ),
        q: str | None = Query(default=None, description="Search query"),
        agent: list[str] = Query(default=[], description="Filter by agent names"),
        provider: list[str] = Query(default=[], description="Filter by provider names"),
        model: list[str] = Query(default=[], description="Filter by model names"),
        source: list[str] = Query(default=[], description="Filter by datasets/sources"),
        task: list[str] = Query(default=[], description="Filter by task names"),
        sort_by: str | None = Query(
            default=None,
            description="Field to sort by (task_name, agent_name, model_provider, model_name, source, n_trials, n_errors, avg_duration_ms, avg_reward, avg_input_tokens, avg_cached_input_tokens, avg_output_tokens, avg_cost_usd, avg_peak_context_tokens, avg_agent_steps)",
        ),
        sort_order: str = Query(default="asc", description="Sort order (asc or desc)"),
    ) -> PaginatedResponse[TaskSummary]:
        """List tasks in a job, grouped by agent + model + source + task_name."""
        if job_name not in scanner.list_jobs():
            raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found")

        summaries = _get_all_task_summaries(job_name)

        # Filter by search query (searches task, agent, provider, model, dataset)
        if q:
            query = q.lower()
            summaries = [
                s
                for s in summaries
                if query in s.task_name.lower()
                or (s.agent_name and query in s.agent_name.lower())
                or (s.model_provider and query in s.model_provider.lower())
                or (s.model_name and query in s.model_name.lower())
                or (s.source and query in s.source.lower())
            ]

        # Filter by agents
        if agent:
            summaries = [s for s in summaries if s.agent_name in agent]

        # Filter by providers
        if provider:
            summaries = [s for s in summaries if s.model_provider in provider]

        # Filter by models
        if model:
            summaries = [s for s in summaries if s.model_name in model]

        # Filter by datasets/sources
        if source:
            summaries = [s for s in summaries if s.source in source]

        # Filter by task names
        if task:
            summaries = [s for s in summaries if s.task_name in task]

        # Sort
        if sort_by:
            reverse = sort_order == "desc"
            if sort_by == "task_name":
                summaries.sort(key=lambda s: s.task_name or "", reverse=reverse)
            elif sort_by == "agent_name":
                summaries.sort(key=lambda s: s.agent_name or "", reverse=reverse)
            elif sort_by == "model_provider":
                summaries.sort(key=lambda s: s.model_provider or "", reverse=reverse)
            elif sort_by == "model_name":
                summaries.sort(key=lambda s: s.model_name or "", reverse=reverse)
            elif sort_by == "source":
                summaries.sort(key=lambda s: s.source or "", reverse=reverse)
            elif sort_by == "n_trials":
                summaries.sort(key=lambda s: s.n_trials, reverse=reverse)
            elif sort_by == "n_errors":
                summaries.sort(key=lambda s: s.n_errors, reverse=reverse)
            elif sort_by == "avg_duration_ms":
                # Put None values at the end
                summaries.sort(
                    key=lambda s: (
                        s.avg_duration_ms is None,
                        s.avg_duration_ms or 0,
                    ),
                    reverse=reverse,
                )
            elif sort_by == "avg_reward":
                summaries.sort(key=lambda s: s.avg_reward or 0, reverse=reverse)
            elif sort_by == "exception_types":
                summaries.sort(
                    key=lambda s: s.exception_types[0] if s.exception_types else "",
                    reverse=reverse,
                )
            elif sort_by == "avg_input_tokens":
                summaries.sort(
                    key=lambda s: (s.avg_input_tokens is None, s.avg_input_tokens or 0),
                    reverse=reverse,
                )
            elif sort_by == "avg_cached_input_tokens":
                summaries.sort(
                    key=lambda s: (
                        s.avg_cached_input_tokens is None,
                        s.avg_cached_input_tokens or 0,
                    ),
                    reverse=reverse,
                )
            elif sort_by == "avg_output_tokens":
                summaries.sort(
                    key=lambda s: (
                        s.avg_output_tokens is None,
                        s.avg_output_tokens or 0,
                    ),
                    reverse=reverse,
                )
            elif sort_by == "avg_cost_usd":
                summaries.sort(
                    key=lambda s: (s.avg_cost_usd is None, s.avg_cost_usd or 0),
                    reverse=reverse,
                )
            elif sort_by == "avg_peak_context_tokens":
                summaries.sort(
                    key=lambda s: (
                        s.avg_peak_context_tokens is None,
                        s.avg_peak_context_tokens or 0,
                    ),
                    reverse=reverse,
                )
            elif sort_by == "avg_agent_steps":
                summaries.sort(
                    key=lambda s: (s.avg_agent_steps is None, s.avg_agent_steps or 0),
                    reverse=reverse,
                )

        # Paginate
        total = len(summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_summaries = summaries[start_idx:end_idx]

        return PaginatedResponse(
            items=page_summaries,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def _build_heatmap(
        job_names: list[str],
        *,
        row_by: str,
        column_by: str,
        q: str | None,
        agent: list[str],
        provider: list[str],
        model: list[str],
        source: list[str],
        task: list[str],
        exclude_errored: bool,
        only_successful: bool,
        include_job_in_rows: bool,
    ) -> JobHeatmapData:
        """Build a heatmap for one or more jobs from raw trial results."""
        existing_jobs = scanner.list_jobs()
        for job_name in job_names:
            if job_name not in existing_jobs:
                raise HTTPException(
                    status_code=404, detail=f"Job '{job_name}' not found"
                )

        rows: dict[str, JobHeatmapRow] = {}
        columns: dict[str, JobHeatmapColumn] = {}
        cell_stats: dict[tuple[str, str], HeatmapGroupStats] = {}
        route_params: dict[tuple[str, str], JobHeatmapRouteParams] = {}

        def new_stats() -> HeatmapGroupStats:
            return {
                "n_trials": 0,
                "n_completed": 0,
                "n_errors": 0,
                "exception_counts": {},
                "total_reward": 0.0,
                "reward_count": 0,
                "total_duration_ms": 0.0,
                "duration_count": 0,
                "total_input_tokens": 0,
                "input_tokens_count": 0,
                "total_cached_input_tokens": 0,
                "cached_input_tokens_count": 0,
                "total_output_tokens": 0,
                "output_tokens_count": 0,
                "total_cost_usd": 0.0,
                "cost_usd_count": 0,
                "total_peak_context_tokens": 0,
                "peak_context_tokens_count": 0,
                "total_agent_steps": 0,
                "agent_steps_count": 0,
            }

        def full_model_name(
            model_provider: str | None, model_name: str | None
        ) -> str | None:
            if model_provider and model_name:
                return f"{model_provider}/{model_name}"
            return model_name

        def row_for(
            current_job_name: str,
            agent_name: str | None,
            model_provider: str | None,
            model_name: str | None,
            reasoning_effort: str | None,
        ) -> JobHeatmapRow:
            full_model = full_model_name(model_provider, model_name)
            key_prefix = f"job::{current_job_name}::" if include_job_in_rows else ""
            label_prefix = f"{current_job_name} / " if include_job_in_rows else ""
            if row_by == "agent":
                key = f"{key_prefix}agent::{agent_name or ''}"
                label = agent_name or "Unknown agent"
                return JobHeatmapRow(
                    key=key,
                    label=f"{label_prefix}{label}",
                    job_name=current_job_name if include_job_in_rows else None,
                    agent_name=agent_name,
                )
            # Effort meaningfully changes results, so it must be part of the row
            # identity (otherwise different effort levels of the same model would
            # be averaged together). The agent view stays effort-agnostic.
            effort_suffix = f" [{reasoning_effort}]" if reasoning_effort else ""
            effort_key = f"::effort::{reasoning_effort or ''}"
            if row_by == "model":
                key = (
                    f"{key_prefix}model::{model_provider or ''}::"
                    f"{model_name or ''}{effort_key}"
                )
                label = full_model or "Unknown model"
                return JobHeatmapRow(
                    key=key,
                    label=f"{label_prefix}{label}{effort_suffix}",
                    job_name=current_job_name if include_job_in_rows else None,
                    model_provider=model_provider,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                )
            key = (
                f"{key_prefix}config::{agent_name or ''}::{model_provider or ''}::"
                f"{model_name or ''}{effort_key}"
            )
            parts = [p for p in [agent_name, full_model] if p]
            return JobHeatmapRow(
                key=key,
                label=f"{label_prefix}{' / '.join(parts) or 'Unknown config'}{effort_suffix}",
                job_name=current_job_name if include_job_in_rows else None,
                agent_name=agent_name,
                model_provider=model_provider,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
            )

        def column_for(source_name: str | None, task_name: str) -> JobHeatmapColumn:
            if column_by == "dataset":
                key = f"dataset::{source_name or ''}"
                return JobHeatmapColumn(
                    key=key,
                    label=source_name or "No dataset",
                    source=source_name,
                )
            key = f"task::{source_name or ''}::{task_name}"
            label = f"{source_name} / {task_name}" if source_name else task_name
            return JobHeatmapColumn(
                key=key,
                label=label,
                source=source_name,
                task_name=task_name,
            )

        for job_name in job_names:
            for trial_name in scanner.list_trials(job_name):
                result = scanner.get_trial_result(job_name, trial_name)
                if not result:
                    continue

                agent_name = result.agent_info.name
                model_info = result.agent_info.model_info
                model_provider = model_info.provider if model_info else None
                model_name = model_info.name if model_info else None
                result_full_model_name = full_model_name(model_provider, model_name)
                source_name = result.source
                task_name = result.task_name

                if q:
                    query = q.lower()
                    searchable = [
                        job_name,
                        task_name,
                        source_name,
                        agent_name,
                        model_provider,
                        model_name,
                        result_full_model_name,
                    ]
                    if not any(value and query in value.lower() for value in searchable):
                        continue
                if agent and agent_name not in agent:
                    continue
                if provider and model_provider not in provider:
                    continue
                if model and model_name not in model:
                    continue
                if source and source_name not in source:
                    continue
                if task and task_name not in task:
                    continue
                if (exclude_errored or only_successful) and result.exception_info:
                    continue
                if only_successful:
                    trial_reward = (
                        result.verifier_result.rewards.get("reward")
                        if result.verifier_result and result.verifier_result.rewards
                        else None
                    )
                    if trial_reward is None or trial_reward < 1:
                        continue

                reasoning_effort = _extract_reasoning_effort(result)
                row = row_for(
                    job_name,
                    agent_name,
                    model_provider,
                    model_name,
                    reasoning_effort,
                )
                column = column_for(source_name, task_name)
                rows[row.key] = row
                columns[column.key] = column

                key = (row.key, column.key)
                stats = cell_stats.setdefault(key, new_stats())
                stats["n_trials"] += 1

                if result.finished_at:
                    stats["n_completed"] += 1
                    if result.started_at:
                        duration_ms = (
                            result.finished_at - result.started_at
                        ).total_seconds() * 1000
                        stats["total_duration_ms"] += duration_ms
                        stats["duration_count"] += 1

                if result.exception_info:
                    stats["n_errors"] += 1
                    exception_type = result.exception_info.exception_type
                    stats["exception_counts"][exception_type] = (
                        stats["exception_counts"].get(exception_type, 0) + 1
                    )

                reward = (
                    result.verifier_result.rewards.get("reward", 0)
                    if result.verifier_result and result.verifier_result.rewards
                    else 0
                )
                stats["total_reward"] += reward
                stats["reward_count"] += 1

                n_input, n_cache, n_output, cost = result.compute_token_cost_totals()
                uncached = _uncached_input(n_input, n_cache)
                if uncached is not None:
                    stats["total_input_tokens"] += uncached
                    stats["input_tokens_count"] += 1
                if n_cache is not None:
                    stats["total_cached_input_tokens"] += n_cache
                    stats["cached_input_tokens_count"] += 1
                if n_output is not None:
                    stats["total_output_tokens"] += n_output
                    stats["output_tokens_count"] += 1
                if cost is not None:
                    stats["total_cost_usd"] += cost
                    stats["cost_usd_count"] += 1

                if (
                    result.agent_result is not None
                    and result.agent_result.peak_context_tokens is not None
                ):
                    stats["total_peak_context_tokens"] += (
                        result.agent_result.peak_context_tokens
                    )
                    stats["peak_context_tokens_count"] += 1

                agent_steps = _agent_step_count_from_result(result)
                if agent_steps is not None:
                    stats["total_agent_steps"] += agent_steps
                    stats["agent_steps_count"] += 1

                if row_by == "config" and column_by == "task":
                    route_params[key] = JobHeatmapRouteParams(
                        job_name=job_name,
                        source=source_name,
                        agent_name=agent_name,
                        model_provider=model_provider,
                        model_name=model_name,
                        task_name=task_name,
                    )

        cells: dict[str, dict[str, JobHeatmapCell]] = {}
        row_reward_totals: dict[str, list[float]] = {}
        column_reward_totals: dict[str, list[float]] = {}

        def average(total_key: str, count_key: str, stats: HeatmapGroupStats):
            count = stats[count_key]  # type: ignore[literal-required]
            if count <= 0:
                return None
            total = stats[total_key]  # type: ignore[literal-required]
            return total / count

        for (row_key, column_key), stats in cell_stats.items():
            dominant_exception = None
            if stats["exception_counts"]:
                dominant_exception = max(
                    stats["exception_counts"].items(),
                    key=lambda item: (item[1], item[0]),
                )[0]

            avg_reward = average("total_reward", "reward_count", stats)
            if avg_reward is not None:
                row_reward_totals.setdefault(row_key, []).append(avg_reward)
                column_reward_totals.setdefault(column_key, []).append(avg_reward)

            cell = JobHeatmapCell(
                row_key=row_key,
                column_key=column_key,
                n_trials=stats["n_trials"],
                n_completed=stats["n_completed"],
                n_errors=stats["n_errors"],
                avg_reward=avg_reward,
                avg_duration_ms=average("total_duration_ms", "duration_count", stats),
                avg_input_tokens=average(
                    "total_input_tokens", "input_tokens_count", stats
                ),
                avg_cached_input_tokens=average(
                    "total_cached_input_tokens", "cached_input_tokens_count", stats
                ),
                avg_output_tokens=average(
                    "total_output_tokens", "output_tokens_count", stats
                ),
                avg_cost_usd=average("total_cost_usd", "cost_usd_count", stats),
                total_cost_usd=(
                    stats["total_cost_usd"] if stats["cost_usd_count"] > 0 else None
                ),
                avg_peak_context_tokens=average(
                    "total_peak_context_tokens", "peak_context_tokens_count", stats
                ),
                avg_agent_steps=average(
                    "total_agent_steps", "agent_steps_count", stats
                ),
                exception_counts=stats["exception_counts"],
                dominant_exception=dominant_exception,
                route_params=route_params.get((row_key, column_key)),
            )
            cells.setdefault(row_key, {})[column_key] = cell

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        sorted_rows = sorted(
            rows.values(),
            key=lambda row: (-mean(row_reward_totals.get(row.key, [])), row.label),
        )
        if column_by == "dataset":
            sorted_columns = sorted(
                columns.values(),
                key=lambda column: (column.label.lower(), column.key),
            )
        else:
            sorted_columns = sorted(
                columns.values(),
                key=lambda column: (
                    -mean(column_reward_totals.get(column.key, [])),
                    column.label,
                ),
            )

        return JobHeatmapData(
            rows=sorted_rows,
            columns=sorted_columns,
            cells=cells,
        )

    @app.get(
        "/api/jobs/{job_name}/heatmap",
        response_model=JobHeatmapData,
    )
    def get_job_heatmap(
        job_name: str,
        row_by: str = Query(
            default="config",
            pattern="^(config|agent|model)$",
            description="Row grouping: config, agent, or model",
        ),
        column_by: str = Query(
            default="task",
            pattern="^(task|dataset)$",
            description="Column grouping: task or dataset",
        ),
        q: str | None = Query(default=None, description="Search query"),
        agent: list[str] = Query(default=[], description="Filter by agent names"),
        provider: list[str] = Query(default=[], description="Filter by provider names"),
        model: list[str] = Query(default=[], description="Filter by model names"),
        source: list[str] = Query(default=[], description="Filter by datasets/sources"),
        task: list[str] = Query(default=[], description="Filter by task names"),
        exclude_errored: bool = Query(
            default=False,
            description="If true, drop errored trials from aggregation entirely.",
        ),
        only_successful: bool = Query(
            default=False,
            description=(
                "If true, only include trials with reward >= 1. Implies "
                "exclude_errored."
            ),
        ),
    ) -> JobHeatmapData:
        """Build a heatmap for one job from raw trial results."""
        return _build_heatmap(
            [job_name],
            row_by=row_by,
            column_by=column_by,
            q=q,
            agent=agent,
            provider=provider,
            model=model,
            source=source,
            task=task,
            exclude_errored=exclude_errored,
            only_successful=only_successful,
            include_job_in_rows=False,
        )

    @app.get(
        "/api/compare/heatmap",
        response_model=JobHeatmapData,
    )
    def get_comparison_heatmap(
        job: list[str] = Query(..., description="Job names to compare"),
        row_by: str = Query(
            default="config",
            pattern="^(config|agent|model)$",
            description="Row grouping: config, agent, or model",
        ),
        column_by: str = Query(
            default="task",
            pattern="^(task|dataset)$",
            description="Column grouping: task or dataset",
        ),
        q: str | None = Query(default=None, description="Search query"),
        agent: list[str] = Query(default=[], description="Filter by agent names"),
        provider: list[str] = Query(default=[], description="Filter by provider names"),
        model: list[str] = Query(default=[], description="Filter by model names"),
        source: list[str] = Query(default=[], description="Filter by datasets/sources"),
        task: list[str] = Query(default=[], description="Filter by task names"),
        exclude_errored: bool = Query(
            default=False,
            description="If true, drop errored trials from aggregation entirely.",
        ),
        only_successful: bool = Query(
            default=False,
            description=(
                "If true, only include trials with reward >= 1. Implies "
                "exclude_errored."
            ),
        ),
    ) -> JobHeatmapData:
        """Build the shared heatmap view for cross-job comparison."""
        return _build_heatmap(
            job,
            row_by=row_by,
            column_by=column_by,
            q=q,
            agent=agent,
            provider=provider,
            model=model,
            source=source,
            task=task,
            exclude_errored=exclude_errored,
            only_successful=only_successful,
            include_job_in_rows=True,
        )

    @app.get(
        "/api/jobs/{job_name}/trials",
        response_model=PaginatedResponse[TrialSummary],
    )
    def list_trials(
        job_name: str,
        task_name: str | None = Query(default=None, description="Filter by task name"),
        source: str | None = Query(
            default=None, description="Filter by source/dataset"
        ),
        agent_name: str | None = Query(
            default=None, description="Filter by agent name"
        ),
        model_name: str | None = Query(
            default=None, description="Filter by model name"
        ),
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=100, ge=1, le=100, description="Number of items per page"
        ),
    ) -> PaginatedResponse[TrialSummary]:
        """List trials in a job with pagination and optional filtering."""
        trial_names = scanner.list_trials(job_name)
        if not trial_names:
            if job_name not in scanner.list_jobs():
                raise HTTPException(
                    status_code=404, detail=f"Job '{job_name}' not found"
                )
            return PaginatedResponse(
                items=[], total=0, page=page, page_size=page_size, total_pages=0
            )

        # Build list of trial summaries with filtering
        all_summaries = []
        for name in trial_names:
            result = scanner.get_trial_result(job_name, name)
            if not result:
                continue

            # Apply filters
            if task_name is not None and result.task_name != task_name:
                continue
            if source is not None and result.source != source:
                continue
            if agent_name is not None and result.agent_info.name != agent_name:
                continue
            model_info = result.agent_info.model_info
            # Build full model name (provider/name) to match frontend format
            if model_info and model_info.provider:
                result_full_model_name = f"{model_info.provider}/{model_info.name}"
            elif model_info:
                result_full_model_name = model_info.name
            else:
                result_full_model_name = None
            if model_name is not None and result_full_model_name != model_name:
                continue

            # Extract primary reward if available
            reward = None
            if result.verifier_result and result.verifier_result.rewards:
                reward = result.verifier_result.rewards.get("reward")

            result_model_provider = model_info.provider if model_info else None
            result_model_name = model_info.name if model_info else None

            n_input, n_cache, n_output, cost = result.compute_token_cost_totals()

            peak_context_tokens = (
                result.agent_result.peak_context_tokens
                if result.agent_result is not None
                else None
            )
            agent_steps = _agent_step_count_from_result(result)

            all_summaries.append(
                TrialSummary(
                    name=name,
                    task_name=result.task_name,
                    id=result.id,
                    source=result.source,
                    agent_name=result.agent_info.name,
                    model_provider=result_model_provider,
                    model_name=result_model_name,
                    reward=reward,
                    error_type=(
                        result.exception_info.exception_type
                        if result.exception_info
                        else None
                    ),
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                    input_tokens=_uncached_input(n_input, n_cache),
                    cached_input_tokens=n_cache,
                    output_tokens=n_output,
                    cost_usd=cost,
                    peak_context_tokens=peak_context_tokens,
                    agent_steps=agent_steps,
                )
            )

        # Paginate
        total = len(all_summaries)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_summaries = all_summaries[start_idx:end_idx]

        return PaginatedResponse(
            items=page_summaries,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @app.get("/api/jobs/{job_name}/trials/{trial_name}", response_model=TrialResult)
    def get_trial(job_name: str, trial_name: str) -> TrialResult:
        """Get full trial result details."""
        result = scanner.get_trial_result(job_name, trial_name)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )
        return result

    @app.get(
        "/api/jobs/{job_name}/trials/{trial_name}/critiques",
        response_model=list[TrialCritiqueDetail],
    )
    def list_trial_critiques(
        job_name: str, trial_name: str
    ) -> list[TrialCritiqueDetail]:
        """List all critique outputs applicable to a source trial."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        details: list[TrialCritiqueDetail] = []
        for run_dir in _critique_run_dirs(job_name):
            item_dir = run_dir / trial_name
            if not item_dir.exists():
                continue
            details.append(_trial_critique_detail(job_name, trial_name, run_dir.name))
        return details

    @app.get(
        "/api/jobs/{job_name}/trials/{trial_name}/critiques/{critique_run_name}",
        response_model=TrialCritiqueDetail,
    )
    def get_trial_critique(
        job_name: str, trial_name: str, critique_run_name: str
    ) -> TrialCritiqueDetail:
        """Get critique output for one source trial and critique run."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        item_dir = _validate_critique_item_path(
            job_name, critique_run_name, trial_name
        )
        if not item_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Critique run '{critique_run_name}' has no item "
                    f"for trial '{trial_name}'"
                ),
            )
        return _trial_critique_detail(job_name, trial_name, critique_run_name)

    @app.get(
        "/api/jobs/{job_name}/trials/{trial_name}/critiques/{critique_run_name}/trajectory"
    )
    def get_trial_critique_trajectory(
        job_name: str, trial_name: str, critique_run_name: str
    ) -> dict[str, Any] | None:
        """Get the critique agent ATIF trajectory for one critique item."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        item_dir = _validate_critique_item_path(
            job_name, critique_run_name, trial_name
        )
        if not item_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Critique run '{critique_run_name}' has no item "
                    f"for trial '{trial_name}'"
                ),
            )

        trajectory_path = item_dir / "agent" / "trajectory.json"
        if not trajectory_path.exists():
            return None
        try:
            return json.loads(trajectory_path.read_text())
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="Failed to parse critique trajectory.json"
            )

    @app.post("/api/jobs/{job_name}/trials/{trial_name}/summarize")
    async def summarize_trial(
        job_name: str, trial_name: str, request: TrialSummarizeRequest
    ) -> dict[str, str | None]:
        """Generate an analysis for a single trial using pier analyze."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        from pier.analyze.analyzer import Analyzer
        from pier.analyze.models import format_analysis_plain_text

        analyzer = Analyzer(model=request.model)
        result = await analyzer.analyze_trial(trial_dir)

        return {"summary": format_analysis_plain_text(result)}

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/trajectory")
    def get_trajectory(
        job_name: str,
        trial_name: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> dict[str, Any] | None:
        """Get trajectory.json content for a trial (optionally a specific step)."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        root = _resolve_step_root(trial_dir, step)
        trajectory_path = root / "agent" / "trajectory.json"
        if not trajectory_path.exists():
            return None

        try:
            return json.loads(trajectory_path.read_text())
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="Failed to parse trajectory.json"
            )

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/verifier-output")
    def get_verifier_output(
        job_name: str,
        trial_name: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> dict[str, str | dict[str, Any] | None]:
        """Get verifier output files from the trial's verifier directory.

        Returns test-stdout.txt, test-stderr.txt, ctrf.json as text, plus reward.json
        and reward-details.json (rewardkit) parsed as JSON.
        """
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        verifier_dir = _resolve_step_root(trial_dir, step) / "verifier"

        def _read_text(path: Path) -> str | None:
            if not path.exists():
                return None
            try:
                return path.read_text()
            except Exception:
                return "[Error reading file]"

        def _read_json(path: Path) -> dict[str, Any] | None:
            if not path.exists():
                return None
            try:
                parsed = json.loads(path.read_text())
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        return {
            "stdout": _read_text(verifier_dir / "test-stdout.txt"),
            "stderr": _read_text(verifier_dir / "test-stderr.txt"),
            "ctrf": _read_text(verifier_dir / "ctrf.json"),
            "reward": _read_json(verifier_dir / "reward.json"),
            "reward_details": _read_json(verifier_dir / "reward-details.json"),
        }

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/files")
    def list_trial_files(
        job_name: str,
        trial_name: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> list[FileInfo]:
        """List all files in a trial directory (optionally a specific step)."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        root = _resolve_step_root(trial_dir, step)
        files: list[FileInfo] = []

        def scan_dir(dir_path: Path, relative_base: str = "") -> None:
            try:
                for item in sorted(dir_path.iterdir()):
                    relative_path = (
                        f"{relative_base}/{item.name}" if relative_base else item.name
                    )
                    if item.is_dir():
                        files.append(
                            FileInfo(
                                path=relative_path,
                                name=item.name,
                                is_dir=True,
                                size=None,
                            )
                        )
                        scan_dir(item, relative_path)
                    else:
                        files.append(
                            FileInfo(
                                path=relative_path,
                                name=item.name,
                                is_dir=False,
                                size=item.stat().st_size,
                            )
                        )
            except PermissionError:
                pass

        scan_dir(root)
        return files

    @app.get(
        "/api/jobs/{job_name}/trials/{trial_name}/files/{file_path:path}",
        response_model=None,
    )
    def get_trial_file(
        job_name: str,
        trial_name: str,
        file_path: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> PlainTextResponse | FileResponse:
        """Get content of a file in a trial directory.

        For text files, returns PlainTextResponse with the content.
        For image files (png, jpg, gif, webp), returns FileResponse with appropriate media type.
        """
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        root = _resolve_step_root(trial_dir, step)

        # Resolve the path and ensure it's within the trial directory (prevent traversal)
        try:
            full_path = (root / file_path).resolve()
            if trial_dir.resolve() not in full_path.parents:
                raise HTTPException(status_code=403, detail="Access denied")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if full_path.is_dir():
            raise HTTPException(status_code=400, detail="Cannot read directory")

        def _format_size(size_bytes: int) -> str:
            """Format bytes as human-readable string."""
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"

        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {_format_size(file_size)} (max {_format_size(MAX_FILE_SIZE)})",
            )

        # Handle image files - serve as binary with correct media type
        image_extensions = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }
        suffix = full_path.suffix.lower()
        if suffix in image_extensions:
            return FileResponse(
                path=full_path,
                media_type=image_extensions[suffix],
                filename=full_path.name,
            )

        # For text files, read and return as plain text
        try:
            content = full_path.read_text()
            return PlainTextResponse(content)
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=415, detail="File is binary and cannot be displayed"
            )

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/artifacts")
    def get_artifacts(
        job_name: str,
        trial_name: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> dict[str, Any]:
        """Get artifacts collected from the trial sandbox."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        artifacts_dir = _resolve_step_root(trial_dir, step) / "artifacts"
        if not artifacts_dir.exists():
            return {"files": [], "manifest": None}

        # Parse manifest.json if present
        manifest = None
        manifest_path = artifacts_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                manifest = None

        # Scan artifacts directory for files, excluding manifest.json
        files: list[FileInfo] = []

        def scan_dir(dir_path: Path, relative_base: str = "") -> None:
            try:
                for item in sorted(dir_path.iterdir()):
                    relative_path = (
                        f"{relative_base}/{item.name}" if relative_base else item.name
                    )
                    if item.name == "manifest.json" and not relative_base:
                        continue
                    if item.is_dir():
                        scan_dir(item, relative_path)
                    else:
                        files.append(
                            FileInfo(
                                path=relative_path,
                                name=item.name,
                                is_dir=False,
                                size=item.stat().st_size,
                            )
                        )
            except PermissionError:
                pass

        scan_dir(artifacts_dir)
        return {"files": files, "manifest": manifest}

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/agent-logs")
    def get_agent_logs(
        job_name: str,
        trial_name: str,
        step: str | None = Query(default=None, description="Step name to scope to"),
    ) -> dict[str, Any]:
        """Get agent log files (oracle.txt, setup/stdout.txt, command-*/stdout.txt)."""
        trial_dir = _validate_trial_path(job_name, trial_name)
        if not trial_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Trial '{trial_name}' not found in job '{job_name}'",
            )

        root = _resolve_step_root(trial_dir, step)
        agent_dir = root / "agent"
        logs: dict[str, Any] = {
            "oracle": None,
            "setup": None,
            "commands": [],
            "summary": None,
        }

        # Read analysis.md if it exists (always trial-level)
        analysis_path_md = trial_dir / "analysis.md"
        if analysis_path_md.exists():
            try:
                logs["summary"] = analysis_path_md.read_text()
            except Exception:
                logs["summary"] = "[Error reading file]"

        # Read analysis.json if it exists (structured analysis from pier analyze)
        analysis_path = trial_dir / "analysis.json"
        if analysis_path.exists():
            try:
                logs["analysis"] = json.loads(analysis_path.read_text())
            except Exception:
                logs["analysis"] = None

        # Read oracle.txt if it exists
        oracle_path = agent_dir / "oracle.txt"
        if oracle_path.exists():
            try:
                logs["oracle"] = oracle_path.read_text()
            except Exception:
                logs["oracle"] = "[Error reading file]"

        # Read setup/stdout.txt if it exists
        setup_stdout_path = agent_dir / "setup" / "stdout.txt"
        if setup_stdout_path.exists():
            try:
                logs["setup"] = setup_stdout_path.read_text()
            except Exception:
                logs["setup"] = "[Error reading file]"

        # Read command-*/stdout.txt files
        i = 0
        while True:
            command_dir = agent_dir / f"command-{i}"
            if not command_dir.exists():
                break
            stdout_path = command_dir / "stdout.txt"
            if stdout_path.exists():
                try:
                    logs["commands"].append(
                        {"index": i, "content": stdout_path.read_text()}
                    )
                except Exception:
                    logs["commands"].append(
                        {"index": i, "content": "[Error reading file]"}
                    )
            i += 1

        return logs

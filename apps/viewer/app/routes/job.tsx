import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import type {
  Column,
  ColumnDef,
  SortingState,
  VisibilityState,
} from "@tanstack/react-table";
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowUp,
  ArrowUpDown,
  ArrowUpFromLine,
  Database,
  FileText,
  Layers,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { parseAsArrayOf, parseAsString, useQueryState } from "nuqs";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { Link, useNavigate, useParams } from "react-router";
import { toast } from "sonner";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { Button } from "~/components/ui/button";
import { CodeBlock } from "~/components/ui/code-block";
import { CopyButton } from "~/components/ui/copy-button";
import { Markdown } from "~/components/ui/markdown";
import { Combobox, type ComboboxOption } from "~/components/ui/combobox";
import { DataTable, SortableHeader } from "~/components/ui/data-table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Checkbox } from "~/components/ui/checkbox";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "~/components/ui/hover-card";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "~/components/ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { LoadingDots } from "~/components/ui/loading-dots";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { Kbd } from "~/components/ui/kbd";
import {
  deleteJob,
  fetchJob,
  fetchJobHeatmap,
  fetchJobSummary,
  fetchTaskFilters,
  fetchTasks,
  summarizeJob,
} from "~/lib/api";
import {
  ChartToolbar,
  ChartToolbarAction,
  ChartToolbarSelect,
} from "~/components/ui/chart-toolbar";
import { JobEfficiencyChart } from "~/components/job-efficiency-chart";
import { JobScalingChart } from "~/components/job-scaling-chart";
import { JobScatterChart } from "~/components/job-scatter-chart";
import { JobSlopeChart } from "~/components/job-slope-chart";
import { ResizableHeatmapGrid } from "~/components/resizable-heatmap-grid";
import { useDebouncedValue, useKeyboardTableNavigation } from "~/lib/hooks";
import type { JobHeatmapTrialsFilter } from "~/lib/api";
import type {
  JobHeatmapCell,
  JobHeatmapColumnBy,
  JobHeatmapData,
  JobHeatmapRowBy,
  TaskSummary,
} from "~/lib/types";
import { cn } from "~/lib/utils";

function CopyableValue({ value }: { value: string }) {
  const handleClick = async () => {
    await navigator.clipboard.writeText(value);
    toast("Copied to clipboard");
  };

  return (
    <span
      onClick={handleClick}
      className="cursor-default hover:text-foreground transition-colors"
    >
      {value}
    </span>
  );
}

function AnalyzeDialog({ jobName }: { jobName: string }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState("haiku");
  const [nConcurrent, setNConcurrent] = useState(32);
  const [onlyFailed, setOnlyFailed] = useState(true);

  const mutation = useMutation({
    mutationFn: () => summarizeJob(jobName, model, nConcurrent, onlyFailed),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["job-summary", jobName] });
      setOpen(false);

      // Show appropriate toast based on what was done
      if (data.n_trials_summarized > 0 && data.job_summary_created) {
        toast.success(
          `Analyzed ${data.n_trials_summarized} trial${data.n_trials_summarized === 1 ? "" : "s"}`
        );
      } else if (data.job_summary_created) {
        toast.success("Job analysis updated");
      } else {
        toast.info("No trials to analyze");
      }
    },
    onError: (error) => {
      toast.error("Failed to generate analysis", { description: error.message });
    },
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Generate Analysis</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generate Analysis</DialogTitle>
          <DialogDescription>
            Use Claude to analyze all failing trials and generate an analysis.
            This can take a couple minutes.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 pt-4">
          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger id="model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="haiku">Haiku (Recommended)</SelectItem>
                <SelectItem value="sonnet">Sonnet</SelectItem>
                <SelectItem value="opus">Opus</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="n-concurrent">Concurrent Claude Codes</Label>
            <Input
              id="n-concurrent"
              type="number"
              min={1}
              max={100}
              value={nConcurrent}
              onChange={(e) => setNConcurrent(parseInt(e.target.value) || 1)}
            />
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="only-failed"
              checked={onlyFailed}
              onCheckedChange={(checked) => setOnlyFailed(checked === true)}
            />
            <Label htmlFor="only-failed" className="font-normal">
              Only analyze failed trials
            </Label>
          </div>
          <Button
            className="w-full"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            {mutation.isPending
              ? <LoadingDots text="Generating" />
              : "Generate"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function formatTokens(n: number | null): string {
  if (n === null) return "-";
  return Math.round(n).toLocaleString();
}

function formatCount(n: number | null): string {
  if (n === null) return "-";
  const value = Number.isInteger(n) ? n.toString() : n.toFixed(1);
  return value.replace(/\.0$/, "");
}

function formatCostUSD(cost: number | null): string {
  if (cost === null) return "-";
  return `$${cost.toFixed(2)}`;
}

function formatDurationMs(durationMs: number): string {
  const seconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

function RewardText({ reward }: { reward: number }) {
  return (
    <span className="font-mono tabular-nums text-foreground">
      {reward.toFixed(2)}
    </span>
  );
}

function getTaskUrl(task: TaskSummary, jobName: string): string {
  const source = task.source || "_";
  const agent = task.agent_name || "_";
  const modelProvider = task.model_provider || "_";
  const modelName = task.model_name || "_";
  return `/jobs/${encodeURIComponent(jobName)}/tasks/${encodeURIComponent(source)}/${encodeURIComponent(agent)}/${encodeURIComponent(modelProvider)}/${encodeURIComponent(modelName)}/${encodeURIComponent(task.task_name)}`;
}

function getHeatmapTaskUrl(jobName: string, cell: JobHeatmapCell): string | null {
  const params = cell.route_params;
  if (!params) return null;
  const targetJobName = params.job_name ?? jobName;
  const source = params.source || "_";
  const agent = params.agent_name || "_";
  const modelProvider = params.model_provider || "_";
  const modelName = params.model_name || "_";
  return `/jobs/${encodeURIComponent(targetJobName)}/tasks/${encodeURIComponent(source)}/${encodeURIComponent(agent)}/${encodeURIComponent(modelProvider)}/${encodeURIComponent(modelName)}/${encodeURIComponent(params.task_name)}`;
}

function NumericHeader({
  children,
  column,
}: {
  children: React.ReactNode;
  column: Column<TaskSummary, unknown>;
}) {
  return (
    <div className="flex items-center justify-end">
      <SortableHeader column={column}>{children}</SortableHeader>
    </div>
  );
}

function TextHeader({
  children,
  column,
}: {
  children: React.ReactNode;
  column: Column<TaskSummary, unknown>;
}) {
  return (
    <div className="flex items-center">
      <SortableHeader column={column}>{children}</SortableHeader>
    </div>
  );
}

/**
 * Icon-only header for compact columns (e.g. token counts). The button
 * itself is the sortable trigger; hovering or focusing it reveals a tooltip
 * with the full label and a one-line description.
 *
 * We don't reuse `SortableHeader` here because it wraps its children in a
 * `truncate` span, which clips the SVG icon when the column is narrow.
 */
function IconHeader({
  icon: Icon,
  label,
  description,
  column,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
  description?: React.ReactNode;
  column: Column<TaskSummary, unknown>;
}) {
  const sorted = column.getIsSorted();
  return (
    <div className="flex items-center justify-end">
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex">
            <Button
              variant="ghost"
              size="sm"
              className="-ml-3 h-8 gap-1 px-2"
              onClick={() => column.toggleSorting(sorted === "asc")}
              aria-label={`Sort by ${label}`}
            >
              <Icon className="size-4" aria-hidden />
              {sorted === "asc" ? (
                <ArrowUp className="size-3.5" />
              ) : sorted === "desc" ? (
                <ArrowDown className="size-3.5" />
              ) : (
                <ArrowUpDown className="size-3.5 opacity-50" />
              )}
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="font-medium">{label}</div>
          {description ? (
            <div className="mt-0.5 opacity-80">{description}</div>
          ) : null}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

// ---- Task column width: a tiny external store so the Task header can
// drive the cell width without re-creating the columns array on every
// render (or moving the column defs inside the component).
const TASK_WIDTH_STORAGE_KEY = "pier.job.taskColWidth";
const TASK_WIDTH_DEFAULT = 280;
const TASK_WIDTH_MIN = 120;
const TASK_WIDTH_MAX = 1200;

const taskWidthStore = (() => {
  let value = TASK_WIDTH_DEFAULT;
  if (typeof window !== "undefined") {
    const raw = window.localStorage.getItem(TASK_WIDTH_STORAGE_KEY);
    const parsed = raw ? Number.parseInt(raw, 10) : NaN;
    if (Number.isFinite(parsed)) {
      value = Math.min(Math.max(parsed, TASK_WIDTH_MIN), TASK_WIDTH_MAX);
    }
  }
  const subs = new Set<() => void>();
  return {
    get: () => value,
    set: (next: number) => {
      const clamped = Math.min(
        Math.max(Math.round(next), TASK_WIDTH_MIN),
        TASK_WIDTH_MAX
      );
      if (clamped === value) return;
      value = clamped;
      try {
        window.localStorage.setItem(TASK_WIDTH_STORAGE_KEY, String(value));
      } catch {}
      subs.forEach((cb) => cb());
    },
    subscribe: (cb: () => void) => {
      subs.add(cb);
      return () => {
        subs.delete(cb);
      };
    },
  };
})();

function useTaskWidth(): number {
  return useSyncExternalStore(
    taskWidthStore.subscribe,
    taskWidthStore.get,
    () => TASK_WIDTH_DEFAULT
  );
}

function TaskHeaderCell({ column }: { column: Column<TaskSummary, unknown> }) {
  const width = useTaskWidth();
  // The resize handle lives in <TaskColResizeOverlay /> rendered as a
  // sibling of the table, so it can extend the full table height (the table
  // container's overflow-x: auto would otherwise clip a handle rendered
  // inside a <th>). We just mark this header as the anchor.
  return (
    <div
      data-resize-anchor="task-end"
      className="flex items-center"
      style={{ width }}
    >
      <SortableHeader column={column}>Task</SortableHeader>
    </div>
  );
}

function TaskNameCell({ name }: { name: string }) {
  const width = useTaskWidth();
  return (
    <div className="truncate" style={{ width }} title={name}>
      {name}
    </div>
  );
}

/**
 * Full-height overlay that draws the Task column's resize handle. Rendered
 * as a sibling of the data table so it can extend past the header row and
 * span every row in the table, regardless of the table container's overflow
 * settings.
 */
function TaskColResizeOverlay({
  containerRef,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
}) {
  const width = useTaskWidth();
  const [pos, setPos] = useState<{ left: number; height: number } | null>(
    null
  );
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Re-measure the Task column's right edge. We need to handle four sources
  // of position change:
  //   1. The persisted Task width changes (drag).
  //   2. The container's own size changes (window resize, sidebar collapse).
  //   3. Other columns reflow because their content changed (filters,
  //      sort, page change). The wrapper height/width may stay constant
  //      here — only the inner cell widths shift.
  //   4. Tab switches that remount the wrapper.
  // ResizeObserver handles (2). useLayoutEffect after every render covers
  // (1), (3), (4). The early-return on equal values avoids render loops.
  const measure = () => {
    const container = containerRef.current;
    if (!container) return;
    const anchor = container.querySelector<HTMLElement>(
      '[data-resize-anchor="task-end"]'
    );
    const th = anchor?.closest("th");
    if (!th) {
      setPos((prev) => (prev === null ? prev : null));
      return;
    }
    const cRect = container.getBoundingClientRect();
    const thRect = th.getBoundingClientRect();
    const left = thRect.right - cRect.left;
    const height = container.clientHeight;
    setPos((prev) => {
      if (prev && prev.left === left && prev.height === height) return prev;
      return { left, height };
    });
  };

  useLayoutEffect(measure);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(measure);
    ro.observe(container);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
    // measure is intentionally referenced via closure; it always reads the
    // latest containerRef.current.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef]);

  // Drag handling: shared with the rest of the page, persists on mouseup.
  useEffect(() => {
    function onMove(e: MouseEvent) {
      const drag = dragRef.current;
      if (!drag) return;
      taskWidthStore.set(drag.startWidth + (e.clientX - drag.startX));
    }
    function onUp() {
      if (!dragRef.current) return;
      dragRef.current = null;
      setIsDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape" || !dragRef.current) return;
      taskWidthStore.set(dragRef.current.startWidth);
      dragRef.current = null;
      setIsDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  if (!pos) return null;

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize Task column (double-click to reset)"
      onMouseDown={(e) => {
        e.preventDefault();
        e.stopPropagation();
        dragRef.current = { startX: e.clientX, startWidth: width };
        setIsDragging(true);
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
      }}
      onDoubleClick={() => taskWidthStore.set(TASK_WIDTH_DEFAULT)}
      className="group/resize absolute z-30 flex w-2.5 -translate-x-1/2 cursor-col-resize select-none touch-none items-stretch justify-center"
      style={{ left: pos.left, top: 0, height: pos.height }}
    >
      <div
        className={cn(
          "pointer-events-none transition-all duration-150",
          isDragging
            ? "w-[2px] bg-ring"
            : "w-px bg-transparent group-hover/resize:w-[2px] group-hover/resize:bg-ring/80"
        )}
      />
    </div>
  );
}

const columns: ColumnDef<TaskSummary>[] = [
  {
    accessorKey: "avg_reward",
    header: ({ column }) => (
      <NumericHeader column={column}>Reward</NumericHeader>
    ),
    cell: ({ row }) => {
      const avgReward = row.original.avg_reward;
      if (avgReward === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return (
        <div className="text-right">
          <RewardText reward={avgReward} />
        </div>
      );
    },
  },
  {
    accessorKey: "task_name",
    header: ({ column }) => <TaskHeaderCell column={column} />,
    cell: ({ row }) => <TaskNameCell name={row.original.task_name} />,
  },
  {
    accessorKey: "agent_name",
    header: ({ column }) => (
      <TextHeader column={column}>Agent</TextHeader>
    ),
    cell: ({ row }) => row.original.agent_name || "-",
  },
  {
    accessorKey: "model_provider",
    header: ({ column }) => (
      <TextHeader column={column}>Provider</TextHeader>
    ),
    cell: ({ row }) => row.original.model_provider || "-",
  },
  {
    accessorKey: "model_name",
    header: ({ column }) => (
      <TextHeader column={column}>Model</TextHeader>
    ),
    cell: ({ row }) => row.original.model_name || "-",
  },
  {
    accessorKey: "source",
    header: ({ column }) => (
      <TextHeader column={column}>Dataset</TextHeader>
    ),
    cell: ({ row }) => row.original.source || "-",
  },
  {
    accessorKey: "n_trials",
    header: ({ column }) => (
      <NumericHeader column={column}>Trials</NumericHeader>
    ),
    cell: ({ row }) => {
      const { n_trials, n_completed } = row.original;
      if (n_completed < n_trials) {
        return (
          <div className="text-right">
            {n_completed}/{n_trials}
          </div>
        );
      }
      return <div className="text-right">{n_trials}</div>;
    },
  },
  {
    accessorKey: "n_errors",
    header: ({ column }) => (
      <NumericHeader column={column}>Errors</NumericHeader>
    ),
    cell: ({ row }) => {
      const errors = row.original.n_errors;
      return <div className="text-right">{errors}</div>;
    },
  },
  {
    accessorKey: "avg_cost_usd",
    header: ({ column }) => (
      <NumericHeader column={column}>Cost</NumericHeader>
    ),
    cell: ({ row }) => {
      const cost = row.original.avg_cost_usd;
      if (cost === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatCostUSD(cost)}</div>;
    },
  },
  {
    accessorKey: "avg_agent_steps",
    header: ({ column }) => (
      <NumericHeader column={column}>Steps</NumericHeader>
    ),
    cell: ({ row }) => {
      const value = row.original.avg_agent_steps;
      if (value === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatCount(value)}</div>;
    },
  },
  {
    accessorKey: "avg_duration_ms",
    header: ({ column }) => (
      <NumericHeader column={column}>Duration</NumericHeader>
    ),
    cell: ({ row }) => {
      const avgDurationMs = row.original.avg_duration_ms;
      if (avgDurationMs === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right">{formatDurationMs(avgDurationMs)}</div>;
    },
  },
  {
    accessorKey: "exception_types",
    header: ({ column }) => (
      <TextHeader column={column}>Exceptions</TextHeader>
    ),
    sortingFn: (a, b) => {
      const aVal = a.original.exception_types[0] ?? "";
      const bVal = b.original.exception_types[0] ?? "";
      return aVal.localeCompare(bVal);
    },
    cell: ({ row }) => {
      const exceptionTypes = row.original.exception_types;
      if (exceptionTypes.length === 0)
        return <span className="text-muted-foreground">-</span>;
      if (exceptionTypes.length === 1) {
        return <span className="text-sm">{exceptionTypes[0]}</span>;
      }
      return (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="text-sm cursor-default">
              {exceptionTypes[0]}{" "}
              <span className="text-muted-foreground">
                +{exceptionTypes.length - 1} more
              </span>
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <div className="space-y-1">
              {exceptionTypes.map((exceptionType) => (
                <div key={exceptionType}>{exceptionType}</div>
              ))}
            </div>
          </TooltipContent>
        </Tooltip>
      );
    },
  },
  {
    accessorKey: "avg_input_tokens",
    header: ({ column }) => (
      <IconHeader
        column={column}
        icon={ArrowDownToLine}
        label="Uncached Input Tokens"
      />
    ),
    cell: ({ row }) => {
      const value = row.original.avg_input_tokens;
      if (value === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatTokens(value)}</div>;
    },
  },
  {
    accessorKey: "avg_cached_input_tokens",
    header: ({ column }) => (
      <IconHeader
        column={column}
        icon={Database}
        label="Cached Input Tokens"
      />
    ),
    cell: ({ row }) => {
      const value = row.original.avg_cached_input_tokens;
      if (value === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatTokens(value)}</div>;
    },
  },
  {
    accessorKey: "avg_output_tokens",
    header: ({ column }) => (
      <IconHeader
        column={column}
        icon={ArrowUpFromLine}
        label="Output Tokens"
      />
    ),
    cell: ({ row }) => {
      const value = row.original.avg_output_tokens;
      if (value === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatTokens(value)}</div>;
    },
  },
  {
    accessorKey: "avg_peak_context_tokens",
    header: ({ column }) => (
      <IconHeader
        column={column}
        icon={Layers}
        label="Peak Context"
        description="Trajectory Length"
      />
    ),
    cell: ({ row }) => {
      const value = row.original.avg_peak_context_tokens;
      if (value === null) {
        return <div className="text-right text-muted-foreground">-</div>;
      }
      return <div className="text-right tabular-nums">{formatTokens(value)}</div>;
    },
  },
];

export type HeatmapStatKey =
  | "avg_reward"
  | "avg_duration_ms"
  | "avg_cost_usd"
  | "total_cost_usd"
  | "avg_agent_steps"
  | "avg_input_tokens"
  | "avg_cached_input_tokens"
  | "avg_output_tokens"
  | "avg_peak_context_tokens"
  | "n_trials"
  | "n_errors"
  | "exceptions";

export const HEATMAP_STATS: { value: HeatmapStatKey; label: string }[] = [
  { value: "avg_reward", label: "Avg Reward" },
  { value: "avg_duration_ms", label: "Avg Duration" },
  { value: "avg_cost_usd", label: "Avg Cost" },
  { value: "total_cost_usd", label: "Total Cost" },
  { value: "avg_agent_steps", label: "Avg Agent Steps" },
  { value: "avg_input_tokens", label: "Avg Uncached Input" },
  { value: "avg_cached_input_tokens", label: "Avg Cached Input" },
  { value: "avg_output_tokens", label: "Avg Output" },
  { value: "avg_peak_context_tokens", label: "Avg Peak Context" },
  { value: "n_trials", label: "Trials" },
  { value: "n_errors", label: "Errors" },
  { value: "exceptions", label: "Exceptions" },
];

function getHeatmapNumericValue(
  cell: JobHeatmapCell,
  stat: HeatmapStatKey
): number | null {
  if (stat === "exceptions") return null;
  return cell[stat];
}

function formatHeatmapValue(
  cell: JobHeatmapCell,
  stat: HeatmapStatKey
): string {
  if (stat === "exceptions") {
    if (!cell.dominant_exception) return "OK";
    return cell.n_errors > 1 ? `${cell.dominant_exception} (${cell.n_errors})` : cell.dominant_exception;
  }
  const value = getHeatmapNumericValue(cell, stat);
  if (value === null) return "-";
  if (stat === "avg_reward") return value.toFixed(2);
  if (stat === "avg_cost_usd" || stat === "total_cost_usd") return formatCostUSD(value);
  if (stat === "avg_duration_ms") return formatDurationMs(value);
  if (stat.includes("tokens") || stat === "avg_peak_context_tokens") {
    return formatTokens(value);
  }
  return formatCount(value);
}

function exceptionColor(exceptionType: string): string {
  let hash = 0;
  for (let i = 0; i < exceptionType.length; i += 1) {
    hash = (hash * 31 + exceptionType.charCodeAt(i)) % 360;
  }
  return `oklch(0.72 0.14 ${hash})`;
}

function HeatmapControls({
  searchQuery,
  setSearchQuery,
  agentOptions,
  agentFilter,
  setAgentFilter,
  providerOptions,
  providerFilter,
  setProviderFilter,
  modelOptions,
  modelFilter,
  setModelFilter,
  sourceOptions,
  sourceFilter,
  setSourceFilter,
  taskOptions,
  taskFilter,
  setTaskFilter,
}: {
  searchQuery: string;
  setSearchQuery: (value: string | null) => void;
  agentOptions: ComboboxOption[];
  agentFilter: string[];
  setAgentFilter: (value: string[] | null) => void;
  providerOptions: ComboboxOption[];
  providerFilter: string[];
  setProviderFilter: (value: string[] | null) => void;
  modelOptions: ComboboxOption[];
  modelFilter: string[];
  setModelFilter: (value: string[] | null) => void;
  sourceOptions: ComboboxOption[];
  sourceFilter: string[];
  setSourceFilter: (value: string[] | null) => void;
  taskOptions: ComboboxOption[];
  taskFilter: string[];
  setTaskFilter: (value: string[] | null) => void;
}) {
  return (
    <>
      <div className="col-span-2 relative">
        <Input
          placeholder="Filter tasks..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value || null)}
          size="lg"
          variant="card"
          className="peer pl-9 pr-10 shadow-none"
        />
        <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-border transition-colors peer-focus-visible:text-ring" />
        {searchQuery && (
          <button
            type="button"
            onClick={() => setSearchQuery(null)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
      <Combobox
        options={agentOptions}
        value={agentFilter}
        onValueChange={setAgentFilter}
        placeholder="All agents"
        searchPlaceholder="Search agents..."
        emptyText="No agents found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={providerOptions}
        value={providerFilter}
        onValueChange={setProviderFilter}
        placeholder="All providers"
        searchPlaceholder="Search providers..."
        emptyText="No providers found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={modelOptions}
        value={modelFilter}
        onValueChange={setModelFilter}
        placeholder="All models"
        searchPlaceholder="Search models..."
        emptyText="No models found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={sourceOptions}
        value={sourceFilter}
        onValueChange={setSourceFilter}
        placeholder="All datasets"
        searchPlaceholder="Search datasets..."
        emptyText="No datasets found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
      <Combobox
        options={taskOptions}
        value={taskFilter}
        onValueChange={setTaskFilter}
        placeholder="All tasks"
        searchPlaceholder="Search tasks..."
        emptyText="No tasks found."
        variant="card"
        className="w-full border-l-0 shadow-none"
      />
    </>
  );
}

function HeatmapAxisBar({
  rowBy,
  setRowBy,
  columnBy,
  setColumnBy,
  stat,
  setStat,
  trialsFilter,
  setTrialsFilter,
  isColumnOrderCustom,
  resetColumnOrder,
}: {
  rowBy: JobHeatmapRowBy;
  setRowBy: (value: string | null) => void;
  columnBy: JobHeatmapColumnBy;
  setColumnBy: (value: string | null) => void;
  stat: HeatmapStatKey;
  setStat: (value: string | null) => void;
  trialsFilter: JobHeatmapTrialsFilter;
  setTrialsFilter: (value: JobHeatmapTrialsFilter) => void;
  isColumnOrderCustom: boolean;
  resetColumnOrder: () => void;
}) {
  const rowAxisLabel =
    rowBy === "agent"
      ? "agents"
      : rowBy === "model"
        ? "models"
        : "agent + model configs";
  const colAxisLabel = columnBy === "dataset" ? "datasets" : "tasks";
  const colSortDescription =
    isColumnOrderCustom
      ? `custom ${colAxisLabel} order`
      : columnBy === "dataset"
        ? `${colAxisLabel} sorted alphabetically`
        : `${colAxisLabel} sorted by avg reward across ${rowAxisLabel}`;
  return (
    <ChartToolbar
      description={
        <>
          {rowAxisLabel} sorted by avg reward across {colAxisLabel};{" "}
          {colSortDescription}
        </>
      }
    >
      <ChartToolbarSelect
        label="Rows"
        value={rowBy}
        onValueChange={(value) => setRowBy(value)}
        options={[
          { value: "config", label: "Agent + Model" },
          { value: "agent", label: "Agent" },
          { value: "model", label: "Model" },
        ]}
      />
      <ChartToolbarSelect
        label="Columns"
        value={columnBy}
        onValueChange={(value) => setColumnBy(value)}
        options={[
          { value: "task", label: "Task" },
          { value: "dataset", label: "Dataset" },
        ]}
      />
      <ChartToolbarSelect
        label="Color by"
        value={stat}
        onValueChange={(value) => setStat(value)}
        options={HEATMAP_STATS.map((option) => ({
          value: option.value,
          label: option.label,
        }))}
      />
      <ChartToolbarSelect
        label="Show"
        value={trialsFilter}
        onValueChange={(value) =>
          setTrialsFilter(value as JobHeatmapTrialsFilter)
        }
        options={[
          { value: "all", label: "All trials" },
          { value: "non_errored", label: "Exclude errored" },
          { value: "successful", label: "Only successful (reward = 1)" },
        ]}
      />
      {isColumnOrderCustom && (
        <ChartToolbarAction onClick={resetColumnOrder}>
          Reset order
        </ChartToolbarAction>
      )}
    </ChartToolbar>
  );
}

export function JobHeatmap({
  jobName,
  data,
  isLoading,
  isFetching = false,
  rowBy,
  setRowBy,
  columnBy,
  setColumnBy,
  stat,
  setStat,
  trialsFilter,
  setTrialsFilter,
}: {
  jobName: string;
  data: JobHeatmapData | undefined;
  isLoading: boolean;
  isFetching?: boolean;
  rowBy: JobHeatmapRowBy;
  setRowBy: (value: string | null) => void;
  columnBy: JobHeatmapColumnBy;
  setColumnBy: (value: string | null) => void;
  stat: HeatmapStatKey;
  setStat: (value: string | null) => void;
  trialsFilter: JobHeatmapTrialsFilter;
  setTrialsFilter: (value: JobHeatmapTrialsFilter) => void;
}) {
  const navigate = useNavigate();

  const numericValues = useMemo(() => {
    const values: number[] = [];
    if (!data || stat === "exceptions") return values;
    for (const rowCells of Object.values(data.cells)) {
      for (const cell of Object.values(rowCells)) {
        const value = getHeatmapNumericValue(cell, stat);
        if (value !== null) values.push(value);
      }
    }
    return values;
  }, [data, stat]);

  const minValue = numericValues.length > 0 ? Math.min(...numericValues) : 0;
  const maxValue = numericValues.length > 0 ? Math.max(...numericValues) : 1;
  const valueRange = Math.max(maxValue - minValue, 1);

  const autoRowLabelWidth = useMemo(() => {
    if (!data || data.rows.length === 0) return 220;
    const maxChars = Math.max(...data.rows.map((row) => row.label.length));
    return Math.min(Math.max(220, maxChars * 7 + 32), 480);
  }, [data]);

  const autoColHeaderHeight = useMemo(() => {
    if (!data || data.columns.length === 0) return 144;
    const maxChars = Math.max(...data.columns.map((col) => col.label.length));
    return Math.min(Math.max(120, maxChars * 6 + 32), 400);
  }, [data]);

  return (
    <ResizableHeatmapGrid
      rows={data?.rows}
      columns={data?.columns}
      getCell={(row, column) => data?.cells[row.key]?.[column.key]}
      renderControls={({ isColumnOrderCustom, resetColumnOrder }) => (
        <HeatmapAxisBar
          rowBy={rowBy}
          setRowBy={setRowBy}
          columnBy={columnBy}
          setColumnBy={setColumnBy}
          stat={stat}
          setStat={setStat}
          trialsFilter={trialsFilter}
          setTrialsFilter={setTrialsFilter}
          isColumnOrderCustom={isColumnOrderCustom}
          resetColumnOrder={resetColumnOrder}
        />
      )}
      renderRowHeader={(row) => (
        <span className="text-xs whitespace-nowrap text-right">
          {row.label}
        </span>
      )}
      renderCell={(row, column, cell) => {
        if (!cell) {
          return (
            <div
              key={`${row.key}-${column.key}`}
              className="h-16 border-r border-b bg-muted/20"
            />
          );
        }

        const numericValue = getHeatmapNumericValue(cell, stat);
        const intensity =
          numericValue !== null ? (numericValue - minValue) / valueRange : 0;
        const background =
          stat === "exceptions"
            ? cell.dominant_exception
              ? exceptionColor(cell.dominant_exception)
              : "transparent"
            : `color-mix(in oklch, var(--foreground) ${(intensity * 90 + 8).toFixed(1)}%, transparent)`;
        const url = getHeatmapTaskUrl(jobName, cell);

        return (
          <HoverCard key={`${row.key}-${column.key}`} openDelay={150}>
            <HoverCardTrigger asChild>
              <button
                type="button"
                onClick={() => {
                  if (url) navigate(url);
                }}
                className={cn(
                  "group relative isolate flex h-16 items-center justify-center border-r border-b text-xs transition-opacity",
                  url && "hover:opacity-85",
                  !url && "cursor-default"
                )}
              >
                <div className="absolute inset-0" style={{ background }} />
                <span
                  className={cn(
                    "relative z-10 max-w-24 truncate px-2 font-mono tabular-nums",
                    stat !== "exceptions" && intensity > 0.55
                      ? "text-background"
                      : "text-foreground"
                  )}
                >
                  {formatHeatmapValue(cell, stat)}
                </span>
              </button>
            </HoverCardTrigger>
            <HoverCardContent className="w-72 text-xs">
              <div className="space-y-3">
                <div>
                  <div className="text-muted-foreground">Row</div>
                  <div>{row.label}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Column</div>
                  <div>{column.label}</div>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div>
                    <div className="text-muted-foreground">Avg Reward</div>
                    <div className="font-mono">
                      {cell.avg_reward?.toFixed(4) ?? "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Trials</div>
                    <div className="font-mono">
                      {cell.n_completed}/{cell.n_trials}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Avg Duration</div>
                    <div className="font-mono">
                      {cell.avg_duration_ms != null
                        ? formatDurationMs(cell.avg_duration_ms)
                        : "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Avg Cost</div>
                    <div className="font-mono">
                      {formatCostUSD(cell.avg_cost_usd)}
                    </div>
                  </div>
                </div>
                {Object.keys(cell.exception_counts).length > 0 && (
                  <div>
                    <div className="mb-1 text-muted-foreground">Exceptions</div>
                    <div className="space-y-1">
                      {Object.entries(cell.exception_counts).map(
                        ([name, count]) => (
                          <div key={name} className="flex justify-between gap-4">
                            <span className="truncate">{name}</span>
                            <span className="font-mono">{count}</span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}
              </div>
            </HoverCardContent>
          </HoverCard>
        );
      }}
      isLoading={isLoading}
      isFetching={isFetching}
      emptyTitle="No heat map cells"
      emptyDescription="No trials match the current filters."
      storageKeyPrefix="pier.heatmap"
      autoRowLabelWidth={autoRowLabelWidth}
      autoColumnHeaderHeight={autoColHeaderHeight}
    />
  );
}

const PAGE_SIZE = 100;

export default function Job() {
  const { jobName } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [isDeleting, setIsDeleting] = useState(false);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useQueryState(
    "q",
    parseAsString.withDefault("")
  );
  const [agentFilter, setAgentFilter] = useQueryState(
    "agent",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [providerFilter, setProviderFilter] = useQueryState(
    "provider",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [modelFilter, setModelFilter] = useQueryState(
    "model",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [sourceFilter, setSourceFilter] = useQueryState(
    "source",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [taskFilter, setTaskFilter] = useQueryState(
    "task",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [heatmapRowBy, setHeatmapRowBy] = useQueryState(
    "heatmap_row",
    parseAsString.withDefault("config")
  );
  const [heatmapColumnBy, setHeatmapColumnBy] = useQueryState(
    "heatmap_col",
    parseAsString.withDefault("task")
  );
  const [heatmapStat, setHeatmapStat] = useQueryState(
    "heatmap_stat",
    parseAsString.withDefault("avg_reward")
  );
  const [heatmapTrialsRaw, setHeatmapTrialsRaw] = useQueryState(
    "heatmap_trials",
    parseAsString.withDefault("all")
  );
  const heatmapTrialsFilter: JobHeatmapTrialsFilter =
    heatmapTrialsRaw === "non_errored" || heatmapTrialsRaw === "successful"
      ? heatmapTrialsRaw
      : "all";
  const setHeatmapTrialsFilter = (value: JobHeatmapTrialsFilter) =>
    setHeatmapTrialsRaw(value === "all" ? null : value);
  const [hiddenColumns, setHiddenColumns] = useQueryState(
    "hide",
    parseAsArrayOf(parseAsString).withDefault([])
  );
  const [sortBy, setSortBy] = useQueryState("sort_by", parseAsString);
  const [sortOrder, setSortOrder] = useQueryState(
    "sort_order",
    parseAsString.withDefault("asc")
  );
  const searchInputRef = useRef<HTMLInputElement>(null);
  const tasksTableRef = useRef<HTMLDivElement>(null);

  // Convert URL params to SortingState for DataTable
  const sorting: SortingState = sortBy
    ? [{ id: sortBy, desc: sortOrder === "desc" }]
    : [];

  // Handle sorting changes from DataTable
  const handleSortingChange = (newSorting: SortingState) => {
    if (newSorting.length === 0) {
      setSortBy(null);
      setSortOrder(null);
    } else {
      setSortBy(newSorting[0].id);
      setSortOrder(newSorting[0].desc ? "desc" : "asc");
    }
  };

  // Column options for the visibility toggle. Labels match the (shortened)
  // header labels so the combobox stays consistent with the table.
  const columnOptions: ComboboxOption[] = useMemo(() => [
    { value: "avg_reward", label: "Reward" },
    { value: "task_name", label: "Task" },
    { value: "agent_name", label: "Agent" },
    { value: "model_provider", label: "Provider" },
    { value: "model_name", label: "Model" },
    { value: "source", label: "Dataset" },
    { value: "n_trials", label: "Trials" },
    { value: "n_errors", label: "Errors" },
    { value: "avg_cost_usd", label: "Cost" },
    { value: "avg_agent_steps", label: "Steps" },
    { value: "avg_duration_ms", label: "Duration" },
    { value: "exception_types", label: "Exceptions" },
    { value: "avg_input_tokens", label: "Input (uncached)" },
    { value: "avg_cached_input_tokens", label: "Cached input" },
    { value: "avg_output_tokens", label: "Output" },
    { value: "avg_peak_context_tokens", label: "Peak context" },
  ], []);

  // Derive column visibility state from hidden columns
  const columnVisibility = useMemo(() => {
    const visibility: VisibilityState = {};
    for (const col of hiddenColumns) {
      visibility[col] = false;
    }
    return visibility;
  }, [hiddenColumns]);

  // Get the list of visible columns (those not in hiddenColumns)
  const visibleColumns = useMemo(() => {
    return columnOptions
      .filter((col) => !hiddenColumns.includes(col.value))
      .map((col) => col.value);
  }, [columnOptions, hiddenColumns]);

  // Handle column visibility changes from the combobox
  const handleColumnVisibilityChange = (selectedValues: string[]) => {
    const newHidden = columnOptions
      .filter((col) => !selectedValues.includes(col.value))
      .map((col) => col.value);
    setHiddenColumns(newHidden.length > 0 ? newHidden : null);
  };

  useHotkeys(
    "mod+k",
    (e) => {
      e.preventDefault();
      searchInputRef.current?.focus();
    },
    { enableOnFormTags: true }
  );

  // Debounce search to avoid excessive API calls while typing
  const debouncedSearch = useDebouncedValue(searchQuery, 300);

  // Reset to page 1 when any filter or sort changes
  useEffect(() => {
    setPage(1);
  }, [debouncedSearch, agentFilter, providerFilter, modelFilter, sourceFilter, taskFilter, sortBy, sortOrder]);

  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ["job", jobName],
    queryFn: () => fetchJob(jobName!),
    enabled: !!jobName,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.finished_at ? false : 2000;
    },
  });

  // Fetch filter options
  const { data: filtersData } = useQuery({
    queryKey: ["task-filters", jobName],
    queryFn: () => fetchTaskFilters(jobName!),
    enabled: !!jobName,
    refetchInterval: job?.finished_at ? false : 2000,
    staleTime: 60000, // Cache for 1 minute
  });

  const agentOptions: ComboboxOption[] = useMemo(() => {
    return (filtersData?.agents ?? []).map((opt) => ({
      value: opt.value,
      label: opt.value,
      count: opt.count,
    }));
  }, [filtersData?.agents]);

  const providerOptions: ComboboxOption[] = useMemo(() => {
    return (filtersData?.providers ?? []).map((opt) => ({
      value: opt.value,
      label: opt.value,
      count: opt.count,
    }));
  }, [filtersData?.providers]);

  const modelOptions: ComboboxOption[] = useMemo(() => {
    return (filtersData?.models ?? []).map((opt) => ({
      value: opt.value,
      label: opt.value,
      count: opt.count,
    }));
  }, [filtersData?.models]);

  const sourceOptions: ComboboxOption[] = useMemo(() => {
    return (filtersData?.sources ?? []).map((opt) => ({
      value: opt.value,
      label: opt.value,
      count: opt.count,
    }));
  }, [filtersData?.sources]);

  const taskOptions: ComboboxOption[] = useMemo(() => {
    return (filtersData?.tasks ?? []).map((opt) => ({
      value: opt.value,
      label: opt.value,
      count: opt.count,
    }));
  }, [filtersData?.tasks]);

  const {
    data: tasksData,
    isLoading: tasksLoading,
    isPlaceholderData: tasksIsPlaceholder,
  } = useQuery({
    queryKey: [
      "tasks",
      jobName,
      page,
      debouncedSearch,
      agentFilter,
      providerFilter,
      modelFilter,
      sourceFilter,
      taskFilter,
      sortBy,
      sortOrder,
    ],
    queryFn: () =>
      fetchTasks(jobName!, page, PAGE_SIZE, {
        search: debouncedSearch || undefined,
        agents: agentFilter.length > 0 ? agentFilter : undefined,
        providers: providerFilter.length > 0 ? providerFilter : undefined,
        models: modelFilter.length > 0 ? modelFilter : undefined,
        sources: sourceFilter.length > 0 ? sourceFilter : undefined,
        tasks: taskFilter.length > 0 ? taskFilter : undefined,
        sortBy: sortBy || undefined,
        sortOrder: sortOrder as "asc" | "desc" | undefined,
      }),
    enabled: !!jobName,
    refetchInterval: job?.finished_at ? false : 2000,
    placeholderData: keepPreviousData,
  });

  const tasks = tasksData?.items ?? [];
  const totalPages = tasksData?.total_pages ?? 0;
  const total = tasksData?.total ?? 0;

  const [activeTab, setActiveTab] = useQueryState(
    "tab",
    parseAsString.withDefault("results")
  );

  const heatmapRowValue: JobHeatmapRowBy =
    heatmapRowBy === "agent" || heatmapRowBy === "model" ? heatmapRowBy : "config";
  const heatmapColumnValue: JobHeatmapColumnBy =
    heatmapColumnBy === "dataset" ? "dataset" : "task";
  const heatmapStatValue: HeatmapStatKey = HEATMAP_STATS.some(
    (option) => option.value === heatmapStat
  )
    ? (heatmapStat as HeatmapStatKey)
    : "avg_reward";

  const {
    data: heatmapData,
    isLoading: heatmapLoading,
    isPlaceholderData: heatmapIsPlaceholder,
  } = useQuery({
    queryKey: [
      "job-heatmap",
      jobName,
      debouncedSearch,
      agentFilter,
      providerFilter,
      modelFilter,
      sourceFilter,
      taskFilter,
      heatmapRowValue,
      heatmapColumnValue,
      heatmapTrialsFilter,
    ],
    queryFn: () =>
      fetchJobHeatmap(jobName!, {
        search: debouncedSearch || undefined,
        agents: agentFilter.length > 0 ? agentFilter : undefined,
        providers: providerFilter.length > 0 ? providerFilter : undefined,
        models: modelFilter.length > 0 ? modelFilter : undefined,
        sources: sourceFilter.length > 0 ? sourceFilter : undefined,
        tasks: taskFilter.length > 0 ? taskFilter : undefined,
        rowBy: heatmapRowValue,
        columnBy: heatmapColumnValue,
        trialsFilter:
          heatmapTrialsFilter === "all" ? undefined : heatmapTrialsFilter,
      }),
    enabled: !!jobName && activeTab === "heatmap",
    refetchInterval: job?.finished_at ? false : 2000,
    placeholderData: keepPreviousData,
  });

  // Cross-Bench + Scatter tabs share the same query: rows = agent+model
  // configs, columns = datasets, errored trials excluded, unfinished
  // cells skipped client-side.
  const {
    data: slopeData,
    isLoading: slopeLoading,
    isPlaceholderData: slopeIsPlaceholder,
  } = useQuery({
    queryKey: [
      "job-slope",
      jobName,
      debouncedSearch,
      agentFilter,
      providerFilter,
      modelFilter,
      sourceFilter,
      taskFilter,
    ],
    queryFn: () =>
      fetchJobHeatmap(jobName!, {
        search: debouncedSearch || undefined,
        agents: agentFilter.length > 0 ? agentFilter : undefined,
        providers: providerFilter.length > 0 ? providerFilter : undefined,
        models: modelFilter.length > 0 ? modelFilter : undefined,
        sources: sourceFilter.length > 0 ? sourceFilter : undefined,
        tasks: taskFilter.length > 0 ? taskFilter : undefined,
        rowBy: "config",
        columnBy: "dataset",
        trialsFilter: "non_errored",
      }),
    enabled:
      !!jobName &&
      (activeTab === "cross-bench" ||
        activeTab === "scatter" ||
        activeTab === "efficiency"),
    refetchInterval: job?.finished_at ? false : 5000,
    placeholderData: keepPreviousData,
  });

  // Scaling tab needs the same per-config rows, but columns are individual
  // tasks so the chart can place each task on the X axis by average task scale.
  const {
    data: scalingData,
    isLoading: scalingLoading,
    isPlaceholderData: scalingIsPlaceholder,
  } = useQuery({
    queryKey: [
      "job-scaling",
      jobName,
      debouncedSearch,
      agentFilter,
      providerFilter,
      modelFilter,
      sourceFilter,
      taskFilter,
    ],
    queryFn: () =>
      fetchJobHeatmap(jobName!, {
        search: debouncedSearch || undefined,
        agents: agentFilter.length > 0 ? agentFilter : undefined,
        providers: providerFilter.length > 0 ? providerFilter : undefined,
        models: modelFilter.length > 0 ? modelFilter : undefined,
        sources: sourceFilter.length > 0 ? sourceFilter : undefined,
        tasks: taskFilter.length > 0 ? taskFilter : undefined,
        rowBy: "config",
        columnBy: "task",
        trialsFilter: "non_errored",
      }),
    enabled: !!jobName && activeTab === "scaling",
    refetchInterval: job?.finished_at ? false : 5000,
    placeholderData: keepPreviousData,
  });

  // Handle Escape to navigate back when not on Results tab
  // (Results tab handles Escape via useKeyboardTableNavigation)
  useHotkeys("escape", () => navigate("/"), {
    enabled: activeTab !== "results",
  });

  const { highlightedIndex } = useKeyboardTableNavigation({
    rows: tasks,
    onNavigate: (task) => navigate(getTaskUrl(task, jobName!)),
    onEscapeUnhighlighted: () => navigate("/"),
    enabled: activeTab === "results",
  });

  const { data: summaryData } = useQuery({
    queryKey: ["job-summary", jobName],
    queryFn: () => fetchJobSummary(jobName!),
    enabled: !!jobName,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteJob(jobName!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      toast("Job deleted", { description: jobName });
      navigate("/");
    },
    onError: (error) => {
      toast.error("Failed to delete job", { description: error.message });
      setIsDeleting(false);
    },
  });

  const handleDelete = () => {
    if (isDeleting) {
      deleteMutation.mutate();
    } else {
      setIsDeleting(true);
    }
  };

  if (!jobLoading && !job) {
    return (
      <div className="px-4 py-10">
        <div className="text-destructive">Failed to load job</div>
      </div>
    );
  }

  const completedTrials = job?.stats.n_completed_trials ?? 0;
  const totalTrials = job?.n_total_trials ?? 0;
  const errors = job?.stats.n_errored_trials ?? 0;
  const runningTrials = job?.stats.n_running_trials ?? 0;
  const pendingTrials = job?.stats.n_pending_trials ?? 0;
  const cancelledTrials = job?.stats.n_cancelled_trials ?? 0;
  const retries = job?.stats.n_retries ?? 0;
  const evals = job?.stats.evals ?? {};
  const evalEntries = Object.entries(evals);

  return (
    <div className="px-4 py-10">
      <div className="mb-8">
        <Breadcrumb className="mb-4">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to="/">Jobs</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{jobName}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <div className="flex flex-col xl:flex-row xl:justify-between gap-4">
          <div className="flex flex-col gap-4 justify-between min-w-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <h1 className="text-4xl font-normal tracking-tighter font-mono truncate">
                  {jobName}
                </h1>
              </TooltipTrigger>
              <TooltipContent>{jobName}</TooltipContent>
            </Tooltip>
            <div className="flex gap-2 text-sm text-muted-foreground min-w-0">
              <span className="truncate min-w-0">
                {completedTrials}/{totalTrials} trials completed
              </span>
              <span className="text-border shrink-0">|</span>
              <span className="truncate min-w-0">{errors} errors</span>
              {runningTrials > 0 && (
                <>
                  <span className="text-border shrink-0">|</span>
                  <span className="truncate min-w-0">{runningTrials} running</span>
                </>
              )}
              {pendingTrials > 0 && completedTrials < totalTrials && (
                <>
                  <span className="text-border shrink-0">|</span>
                  <span className="truncate min-w-0">{pendingTrials} pending</span>
                </>
              )}
              {cancelledTrials > 0 && (
                <>
                  <span className="text-border shrink-0">|</span>
                  <span className="truncate min-w-0">{cancelledTrials} cancelled</span>
                </>
              )}
              {retries > 0 && (
                <>
                  <span className="text-border shrink-0">|</span>
                  <span className="truncate min-w-0">{retries} retries</span>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-col justify-between items-start xl:items-end gap-6">
            <div className="flex items-center gap-2">
              <Button variant="secondary" asChild>
                <Link to={`/jobs/${encodeURIComponent(jobName!)}/critiques`}>
                  Critiques
                </Link>
              </Button>
              <Button
                variant={isDeleting ? "destructive" : "secondary"}
                onClick={handleDelete}
                onBlur={() => setIsDeleting(false)}
                disabled={deleteMutation.isPending}
              >
                <Trash2 className="h-4 w-4" />
                {isDeleting ? "Confirm delete" : "Delete"}
              </Button>
            </div>
          </div>
        </div>
        {evalEntries.length > 0 && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            {evalEntries.map(([key, evalItem]) => {
              const firstMetric = evalItem.metrics[0];
              if (!firstMetric) return null;
              const [metricName, metricValue] = Object.entries(firstMetric)[0];
              const formatted =
                typeof metricValue === "number"
                  ? metricValue.toFixed(2)
                  : String(metricValue);
              const keyDisplay = key.split("__").join(", ");
              return (
                <Tooltip key={key}>
                  <TooltipTrigger asChild>
                    <span className="text-sm text-muted-foreground cursor-default">
                      <span className="font-mono tabular-nums text-foreground">
                        {formatted}
                      </span>{" "}
                      {metricName}{" "}
                      <span>({keyDisplay})</span>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    <ul className="space-y-0.5">
                      {evalItem.metrics.map((metric, i) => {
                        const [name, val] = Object.entries(metric)[0];
                        const valStr =
                          typeof val === "number" ? val.toFixed(2) : val;
                        return (
                          <li key={i}>
                            {name}={valStr}
                          </li>
                        );
                      })}
                    </ul>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        )}
        {job?.job_uri && (
          <div className="mt-4 space-y-1 text-xs text-muted-foreground">
            {(() => {
              const localPath = job.job_uri.startsWith("file://")
                ? job.job_uri.slice(7)
                : job.job_uri;
              return (
                <div className="flex items-baseline gap-3 min-w-0">
                  <span className="shrink-0 w-14 text-[10px] font-medium uppercase tracking-wide">
                    Local
                  </span>
                  <CopyableValue value={localPath} />
                  <CopyButton
                    value={localPath}
                    iconClassName="size-3"
                    className="shrink-0 self-center text-muted-foreground hover:text-foreground"
                  />
                </div>
              );
            })()}
          </div>
        )}
      </div>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-6">
        <div className="flex items-center justify-between bg-card border border-b-0">
          <TabsList className="border-0">
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="heatmap">Heat Map</TabsTrigger>
            <TabsTrigger value="cross-bench">Cross-Bench</TabsTrigger>
            <TabsTrigger value="scatter">Scatter</TabsTrigger>
            <TabsTrigger value="efficiency">Efficiency</TabsTrigger>
            <TabsTrigger value="scaling">Scaling</TabsTrigger>
            <TabsTrigger value="summary">Analysis</TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-3 px-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Kbd>j</Kbd>
              <Kbd>k</Kbd>
              <span>navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Enter</Kbd>
              <span>open</span>
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Esc</Kbd>
              <span>{highlightedIndex >= 0 ? "deselect" : "go back"}</span>
            </span>
          </div>
        </div>
        <TabsContent value="results">
          <div className="grid grid-cols-8 -mb-px">
            <div className="col-span-2 relative">
              <Input
                ref={searchInputRef}
                placeholder="Search for tasks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value || null)}
                size="lg"
                variant="card"
                className="peer pl-9 pr-16 shadow-none"
              />
              <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-border transition-colors peer-focus-visible:text-ring" />
              {searchQuery ? (
                <button
                  type="button"
                  onClick={() => setSearchQuery(null)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              ) : (
                <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
                  <Kbd>⌘</Kbd>
                  <Kbd>K</Kbd>
                </div>
              )}
            </div>
            <Combobox
              options={agentOptions}
              value={agentFilter}
              onValueChange={setAgentFilter}
              placeholder="All agents"
              searchPlaceholder="Search agents..."
              emptyText="No agents found."
              variant="card"
              className="w-full border-l-0 shadow-none"
            />
            <Combobox
              options={providerOptions}
              value={providerFilter}
              onValueChange={setProviderFilter}
              placeholder="All providers"
              searchPlaceholder="Search providers..."
              emptyText="No providers found."
              variant="card"
              className="w-full border-l-0 shadow-none"
            />
            <Combobox
              options={modelOptions}
              value={modelFilter}
              onValueChange={setModelFilter}
              placeholder="All models"
              searchPlaceholder="Search models..."
              emptyText="No models found."
              variant="card"
              className="w-full border-l-0 shadow-none"
            />
            <Combobox
              options={sourceOptions}
              value={sourceFilter}
              onValueChange={setSourceFilter}
              placeholder="All datasets"
              searchPlaceholder="Search datasets..."
              emptyText="No datasets found."
              variant="card"
              className="w-full border-l-0 shadow-none"
            />
            <Combobox
              options={taskOptions}
              value={taskFilter}
              onValueChange={setTaskFilter}
              placeholder="All tasks"
              searchPlaceholder="Search tasks..."
              emptyText="No tasks found."
              variant="card"
              className="w-full border-l-0 shadow-none"
            />
            <Combobox
              options={columnOptions}
              value={visibleColumns}
              onValueChange={handleColumnVisibilityChange}
              placeholder="Columns"
              searchPlaceholder="Search columns..."
              emptyText="No columns."
              variant="card"
              className="w-full border-l-0 shadow-none"
              multiSelectLabel="columns"
            />
          </div>
          <div className="relative" ref={tasksTableRef}>
            <DataTable
              columns={columns}
              data={tasks}
              onRowClick={(task) => navigate(getTaskUrl(task, jobName!))}
              isLoading={tasksLoading}
              isFetching={tasksIsPlaceholder}
              className="border-t-0"
              highlightedIndex={highlightedIndex}
              columnVisibility={columnVisibility}
              sorting={sorting}
              onSortingChange={handleSortingChange}
              manualSorting
            />
            <TaskColResizeOverlay containerRef={tasksTableRef} />
          </div>
          {totalPages > 1 && (
            <div className="grid grid-cols-3 items-center mt-4">
              <div className="text-sm text-muted-foreground">
                Showing {(page - 1) * PAGE_SIZE + 1}-
                {Math.min(page * PAGE_SIZE, total)} of {total} tasks
              </div>
              <Pagination>
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      className={
                        page === 1
                          ? "pointer-events-none opacity-50"
                          : "cursor-pointer"
                      }
                    />
                  </PaginationItem>
                  {/* First page */}
                  {page > 2 && (
                    <PaginationItem>
                      <PaginationLink
                        onClick={() => setPage(1)}
                        className="cursor-pointer"
                      >
                        1
                      </PaginationLink>
                    </PaginationItem>
                  )}
                  {/* Ellipsis before current */}
                  {page > 3 && (
                    <PaginationItem>
                      <PaginationEllipsis />
                    </PaginationItem>
                  )}
                  {/* Previous page */}
                  {page > 1 && (
                    <PaginationItem>
                      <PaginationLink
                        onClick={() => setPage(page - 1)}
                        className="cursor-pointer"
                      >
                        {page - 1}
                      </PaginationLink>
                    </PaginationItem>
                  )}
                  {/* Current page */}
                  <PaginationItem>
                    <PaginationLink isActive>{page}</PaginationLink>
                  </PaginationItem>
                  {/* Next page */}
                  {page < totalPages && (
                    <PaginationItem>
                      <PaginationLink
                        onClick={() => setPage(page + 1)}
                        className="cursor-pointer"
                      >
                        {page + 1}
                      </PaginationLink>
                    </PaginationItem>
                  )}
                  {/* Ellipsis after current */}
                  {page < totalPages - 2 && (
                    <PaginationItem>
                      <PaginationEllipsis />
                    </PaginationItem>
                  )}
                  {/* Last page */}
                  {page < totalPages - 1 && (
                    <PaginationItem>
                      <PaginationLink
                        onClick={() => setPage(totalPages)}
                        className="cursor-pointer"
                      >
                        {totalPages}
                      </PaginationLink>
                    </PaginationItem>
                  )}
                  <PaginationItem>
                    <PaginationNext
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      className={
                        page === totalPages
                          ? "pointer-events-none opacity-50"
                          : "cursor-pointer"
                      }
                    />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
              <div />
            </div>
          )}
        </TabsContent>
        <TabsContent value="heatmap">
          <div className="grid grid-cols-7 -mb-px">
            <HeatmapControls
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              agentOptions={agentOptions}
              agentFilter={agentFilter}
              setAgentFilter={setAgentFilter}
              providerOptions={providerOptions}
              providerFilter={providerFilter}
              setProviderFilter={setProviderFilter}
              modelOptions={modelOptions}
              modelFilter={modelFilter}
              setModelFilter={setModelFilter}
              sourceOptions={sourceOptions}
              sourceFilter={sourceFilter}
              setSourceFilter={setSourceFilter}
              taskOptions={taskOptions}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
            />
          </div>
          <JobHeatmap
            jobName={jobName!}
            data={heatmapData}
            isLoading={heatmapLoading}
            isFetching={heatmapIsPlaceholder}
            rowBy={heatmapRowValue}
            setRowBy={setHeatmapRowBy}
            columnBy={heatmapColumnValue}
            setColumnBy={setHeatmapColumnBy}
            stat={heatmapStatValue}
            setStat={setHeatmapStat}
            trialsFilter={heatmapTrialsFilter}
            setTrialsFilter={setHeatmapTrialsFilter}
          />
        </TabsContent>
        <TabsContent value="cross-bench">
          <div className="grid grid-cols-7 -mb-px">
            <HeatmapControls
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              agentOptions={agentOptions}
              agentFilter={agentFilter}
              setAgentFilter={setAgentFilter}
              providerOptions={providerOptions}
              providerFilter={providerFilter}
              setProviderFilter={setProviderFilter}
              modelOptions={modelOptions}
              modelFilter={modelFilter}
              setModelFilter={setModelFilter}
              sourceOptions={sourceOptions}
              sourceFilter={sourceFilter}
              setSourceFilter={setSourceFilter}
              taskOptions={taskOptions}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
            />
          </div>
          <JobSlopeChart
            data={slopeData}
            isLoading={slopeLoading}
            isFetching={slopeIsPlaceholder}
          />
        </TabsContent>
        <TabsContent value="scatter">
          <div className="grid grid-cols-7 -mb-px">
            <HeatmapControls
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              agentOptions={agentOptions}
              agentFilter={agentFilter}
              setAgentFilter={setAgentFilter}
              providerOptions={providerOptions}
              providerFilter={providerFilter}
              setProviderFilter={setProviderFilter}
              modelOptions={modelOptions}
              modelFilter={modelFilter}
              setModelFilter={setModelFilter}
              sourceOptions={sourceOptions}
              sourceFilter={sourceFilter}
              setSourceFilter={setSourceFilter}
              taskOptions={taskOptions}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
            />
          </div>
          <JobScatterChart
            data={slopeData}
            isLoading={slopeLoading}
            isFetching={slopeIsPlaceholder}
          />
        </TabsContent>
        <TabsContent value="efficiency">
          <div className="grid grid-cols-7 -mb-px">
            <HeatmapControls
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              agentOptions={agentOptions}
              agentFilter={agentFilter}
              setAgentFilter={setAgentFilter}
              providerOptions={providerOptions}
              providerFilter={providerFilter}
              setProviderFilter={setProviderFilter}
              modelOptions={modelOptions}
              modelFilter={modelFilter}
              setModelFilter={setModelFilter}
              sourceOptions={sourceOptions}
              sourceFilter={sourceFilter}
              setSourceFilter={setSourceFilter}
              taskOptions={taskOptions}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
            />
          </div>
          <JobEfficiencyChart
            data={slopeData}
            isLoading={slopeLoading}
            isFetching={slopeIsPlaceholder}
          />
        </TabsContent>
        <TabsContent value="scaling">
          <div className="grid grid-cols-7 -mb-px">
            <HeatmapControls
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              agentOptions={agentOptions}
              agentFilter={agentFilter}
              setAgentFilter={setAgentFilter}
              providerOptions={providerOptions}
              providerFilter={providerFilter}
              setProviderFilter={setProviderFilter}
              modelOptions={modelOptions}
              modelFilter={modelFilter}
              setModelFilter={setModelFilter}
              sourceOptions={sourceOptions}
              sourceFilter={sourceFilter}
              setSourceFilter={setSourceFilter}
              taskOptions={taskOptions}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
            />
          </div>
          <JobScalingChart
            jobName={jobName!}
            data={scalingData}
            isLoading={scalingLoading}
            isFetching={scalingIsPlaceholder}
          />
        </TabsContent>
        <TabsContent value="summary">
          {summaryData?.summary ? (
            <Markdown>{summaryData.summary}</Markdown>
          ) : (
            <Empty className="bg-card border">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <FileText />
                </EmptyMedia>
                <EmptyTitle>No analysis</EmptyTitle>
                <EmptyDescription>
                  Generate an analysis of all trials in this job using Claude.
                </EmptyDescription>
              </EmptyHeader>
              <AnalyzeDialog jobName={jobName!} />
            </Empty>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

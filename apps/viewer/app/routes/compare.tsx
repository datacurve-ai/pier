import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { parseAsArrayOf, parseAsString, useQueryState } from "nuqs";
import { useMemo } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { Link, useNavigate, useSearchParams } from "react-router";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "~/components/ui/breadcrumb";
import { Button } from "~/components/ui/button";
import { Combobox, type ComboboxOption } from "~/components/ui/combobox";
import { Kbd } from "~/components/ui/kbd";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { JobEfficiencyChart } from "~/components/job-efficiency-chart";
import { JobScatterChart } from "~/components/job-scatter-chart";
import { JobSlopeChart } from "~/components/job-slope-chart";
import { fetchComparisonHeatmap, type JobHeatmapTrialsFilter } from "~/lib/api";
import type {
  JobHeatmapColumnBy,
  JobHeatmapData,
  JobHeatmapRowBy,
} from "~/lib/types";
import { HEATMAP_STATS, JobHeatmap, type HeatmapStatKey } from "./job";

interface CompareFilters {
  agents: string[];
  models: string[];
  sources: string[];
  tasks: string[];
}

function modelKey(provider: string | null, name: string | null): string | null {
  if (!name) return null;
  return provider ? `${provider}/${name}` : name;
}

/**
 * Hide rows / columns the user has filtered out without re-fetching, then
 * re-rank the remaining axes by mean reward over the *visible* cells so
 * the ranking reflects only what's on screen. Without the re-rank, the
 * server's "sorted by avg reward across all columns" order would be stale
 * (e.g. a model still ranked first because of strong performance on a
 * dataset the user just filtered away).
 *
 * Filters only apply when the row/column carries the relevant field. If
 * `rowBy=agent`, rows have no model_name, so a model filter is a no-op for
 * those rows (switch to rowBy=config or rowBy=model to filter by model).
 *
 * Pass `sortColumnsByReward` to mirror the server's column sort: true for
 * task-columns (sorted by avg reward), false for dataset-columns (kept in
 * the alphabetical order the server returned).
 */
function applyCompareFilters(
  data: JobHeatmapData | undefined,
  filters: CompareFilters,
  sortColumnsByReward: boolean
): JobHeatmapData | undefined {
  if (!data) return data;
  const { agents, models, sources, tasks } = filters;
  const hasFilters =
    agents.length > 0 ||
    models.length > 0 ||
    sources.length > 0 ||
    tasks.length > 0;
  if (!hasFilters) return data;

  const filteredRows = data.rows.filter((row) => {
    if (
      agents.length > 0 &&
      row.agent_name !== null &&
      !agents.includes(row.agent_name)
    ) {
      return false;
    }
    if (models.length > 0 && row.model_name !== null) {
      const key = modelKey(row.model_provider, row.model_name);
      if (!key || !models.includes(key)) return false;
    }
    return true;
  });
  const filteredCols = data.columns.filter((col) => {
    if (
      sources.length > 0 &&
      col.source !== null &&
      !sources.includes(col.source)
    ) {
      return false;
    }
    if (
      tasks.length > 0 &&
      col.task_name !== null &&
      !tasks.includes(col.task_name)
    ) {
      return false;
    }
    return true;
  });
  const rowKeys = new Set(filteredRows.map((r) => r.key));
  const colKeys = new Set(filteredCols.map((c) => c.key));
  const filteredCells: typeof data.cells = {};
  for (const [rowKey, rowCells] of Object.entries(data.cells)) {
    if (!rowKeys.has(rowKey)) continue;
    const next: Record<string, (typeof rowCells)[string]> = {};
    for (const [colKey, cell] of Object.entries(rowCells)) {
      if (colKeys.has(colKey)) next[colKey] = cell;
    }
    filteredCells[rowKey] = next;
  }

  // Mean of non-null avg_reward over a list of cells. Mirrors the
  // server's mean() helper (lists without any rewards score 0, matching
  // server.py:2630).
  const meanReward = (cells: Iterable<{ avg_reward: number | null }>): number => {
    let total = 0;
    let count = 0;
    for (const cell of cells) {
      if (cell.avg_reward !== null) {
        total += cell.avg_reward;
        count += 1;
      }
    }
    return count > 0 ? total / count : 0;
  };

  const sortedRows = [...filteredRows].sort((a, b) => {
    const aCells = filteredCols
      .map((c) => filteredCells[a.key]?.[c.key])
      .filter((c): c is NonNullable<typeof c> => c !== undefined);
    const bCells = filteredCols
      .map((c) => filteredCells[b.key]?.[c.key])
      .filter((c): c is NonNullable<typeof c> => c !== undefined);
    const diff = meanReward(bCells) - meanReward(aCells);
    if (Math.abs(diff) > 1e-9) return diff;
    return a.label.localeCompare(b.label);
  });

  const sortedCols = sortColumnsByReward
    ? [...filteredCols].sort((a, b) => {
        const aCells = sortedRows
          .map((r) => filteredCells[r.key]?.[a.key])
          .filter((c): c is NonNullable<typeof c> => c !== undefined);
        const bCells = sortedRows
          .map((r) => filteredCells[r.key]?.[b.key])
          .filter((c): c is NonNullable<typeof c> => c !== undefined);
        const diff = meanReward(bCells) - meanReward(aCells);
        if (Math.abs(diff) > 1e-9) return diff;
        return a.label.localeCompare(b.label);
      })
    : filteredCols;

  return { rows: sortedRows, columns: sortedCols, cells: filteredCells };
}

interface FilterOptionSet {
  agents: ComboboxOption[];
  models: ComboboxOption[];
  sources: ComboboxOption[];
  tasks: ComboboxOption[];
}

function buildFilterOptions(
  ...sources: (JobHeatmapData | undefined)[]
): FilterOptionSet {
  const agentCounts = new Map<string, number>();
  const modelCounts = new Map<string, number>();
  const sourceCounts = new Map<string, number>();
  const taskCounts = new Map<string, number>();
  for (const data of sources) {
    if (!data) continue;
    for (const row of data.rows) {
      if (row.agent_name) {
        agentCounts.set(
          row.agent_name,
          (agentCounts.get(row.agent_name) ?? 0) + 1
        );
      }
      const key = modelKey(row.model_provider, row.model_name);
      if (key) {
        modelCounts.set(key, (modelCounts.get(key) ?? 0) + 1);
      }
    }
    for (const col of data.columns) {
      if (col.source) {
        sourceCounts.set(col.source, (sourceCounts.get(col.source) ?? 0) + 1);
      }
      if (col.task_name) {
        taskCounts.set(col.task_name, (taskCounts.get(col.task_name) ?? 0) + 1);
      }
    }
  }
  const toOptions = (m: Map<string, number>): ComboboxOption[] =>
    [...m.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([value, count]) => ({ value, label: value, count }));
  return {
    agents: toOptions(agentCounts),
    models: toOptions(modelCounts),
    sources: toOptions(sourceCounts),
    tasks: toOptions(taskCounts),
  };
}

export default function ComparePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const jobNames = searchParams.getAll("job");

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
  const [tabParam, setTabParam] = useQueryState(
    "tab",
    parseAsString.withDefault("heatmap")
  );

  const [agentFilter, setAgentFilter] = useQueryState(
    "agent",
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

  useHotkeys("escape", () => navigate("/"));

  const activeTab =
    tabParam === "cross-bench" ||
    tabParam === "scatter" ||
    tabParam === "efficiency"
      ? tabParam
      : "heatmap";

  const heatmapRowValue: JobHeatmapRowBy =
    heatmapRowBy === "agent" || heatmapRowBy === "model" ? heatmapRowBy : "config";
  const heatmapColumnValue: JobHeatmapColumnBy =
    heatmapColumnBy === "dataset" ? "dataset" : "task";
  const heatmapStatValue: HeatmapStatKey = HEATMAP_STATS.some(
    (option) => option.value === heatmapStat
  )
    ? (heatmapStat as HeatmapStatKey)
    : "avg_reward";
  const heatmapTrialsFilter: JobHeatmapTrialsFilter =
    heatmapTrialsRaw === "non_errored" || heatmapTrialsRaw === "successful"
      ? heatmapTrialsRaw
      : "all";
  const setHeatmapTrialsFilter = (value: JobHeatmapTrialsFilter) =>
    setHeatmapTrialsRaw(value === "all" ? null : value);

  const {
    data: heatmapData,
    isLoading: heatmapLoading,
    error: heatmapError,
    isPlaceholderData: heatmapIsPlaceholder,
  } = useQuery({
    queryKey: [
      "comparison-heatmap",
      jobNames,
      heatmapRowValue,
      heatmapColumnValue,
      heatmapTrialsFilter,
    ],
    queryFn: () =>
      fetchComparisonHeatmap(jobNames, {
        rowBy: heatmapRowValue,
        columnBy: heatmapColumnValue,
        trialsFilter:
          heatmapTrialsFilter === "all" ? undefined : heatmapTrialsFilter,
      }),
    enabled: jobNames.length >= 1 && activeTab === "heatmap",
    placeholderData: keepPreviousData,
  });

  const {
    data: slopeData,
    isLoading: slopeLoading,
    error: slopeError,
    isPlaceholderData: slopeIsPlaceholder,
  } = useQuery({
    queryKey: ["comparison-cross-bench", jobNames],
    queryFn: () =>
      fetchComparisonHeatmap(jobNames, {
        rowBy: "config",
        columnBy: "dataset",
        trialsFilter: "non_errored",
      }),
    enabled:
      jobNames.length >= 1 &&
      (activeTab === "cross-bench" ||
        activeTab === "scatter" ||
        activeTab === "efficiency"),
    placeholderData: keepPreviousData,
  });

  const filters = useMemo<CompareFilters>(
    () => ({
      agents: agentFilter,
      models: modelFilter,
      sources: sourceFilter,
      tasks: taskFilter,
    }),
    [agentFilter, modelFilter, sourceFilter, taskFilter]
  );

  const filterOptions = useMemo(
    () => buildFilterOptions(heatmapData, slopeData),
    [heatmapData, slopeData]
  );

  const filteredHeatmapData = useMemo(
    () =>
      applyCompareFilters(
        heatmapData,
        filters,
        heatmapColumnValue === "task"
      ),
    [heatmapData, filters, heatmapColumnValue]
  );
  // Slope/scatter charts do their own client-side column ordering, and
  // both visualisations group rows by model family rather than by reward
  // rank, so we leave both axes in the server's returned order.
  const filteredSlopeData = useMemo(
    () => applyCompareFilters(slopeData, filters, false),
    [slopeData, filters]
  );

  const activeFilterCount =
    agentFilter.length +
    modelFilter.length +
    sourceFilter.length +
    taskFilter.length;

  const clearFilters = () => {
    setAgentFilter(null);
    setModelFilter(null);
    setSourceFilter(null);
    setTaskFilter(null);
  };

  if (jobNames.length < 1) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <p className="text-muted-foreground">
          Select at least 1 job to compare.
        </p>
        <Button asChild>
          <Link to="/">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Jobs
          </Link>
        </Button>
      </div>
    );
  }

  // Hide task filter unless the loaded data actually has task-level columns,
  // since slope/scatter views aggregate to datasets.
  const showTaskFilter =
    filterOptions.tasks.length > 0 || taskFilter.length > 0;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between py-3 px-4">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link to="/">Jobs</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>Compare ({jobNames.length} jobs)</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Kbd>Esc</Kbd>
          <span>go back</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t px-4 py-2">
        <span className="text-xs text-muted-foreground mr-1">Filter</span>
        <Combobox
          options={filterOptions.agents}
          value={agentFilter}
          onValueChange={(v) => setAgentFilter(v.length > 0 ? v : null)}
          placeholder="All agents"
          searchPlaceholder="Search agents..."
          emptyText="No agents found."
          multiSelectLabel="agents"
          className="h-8 w-44 rounded-md text-xs"
        />
        <Combobox
          options={filterOptions.models}
          value={modelFilter}
          onValueChange={(v) => setModelFilter(v.length > 0 ? v : null)}
          placeholder="All models"
          searchPlaceholder="Search models..."
          emptyText="No models found."
          multiSelectLabel="models"
          className="h-8 w-52 rounded-md text-xs"
        />
        <Combobox
          options={filterOptions.sources}
          value={sourceFilter}
          onValueChange={(v) => setSourceFilter(v.length > 0 ? v : null)}
          placeholder="All datasets"
          searchPlaceholder="Search datasets..."
          emptyText="No datasets found."
          multiSelectLabel="datasets"
          className="h-8 w-48 rounded-md text-xs"
        />
        {showTaskFilter && (
          <Combobox
            options={filterOptions.tasks}
            value={taskFilter}
            onValueChange={(v) => setTaskFilter(v.length > 0 ? v : null)}
            placeholder="All tasks"
            searchPlaceholder="Search tasks..."
            emptyText="No tasks found."
            multiSelectLabel="tasks"
            className="h-8 w-44 rounded-md text-xs"
          />
        )}
        {activeFilterCount > 0 && (
          <button
            type="button"
            onClick={clearFilters}
            className="ml-1 inline-flex h-8 items-center px-2 text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          >
            Clear {activeFilterCount}
          </button>
        )}
      </div>

      <div className="flex-1 border-t">
        <Tabs
          value={activeTab}
          onValueChange={(value) =>
            setTabParam(value === "heatmap" ? null : value)
          }
          className="h-full"
        >
          <div className="px-4">
            <TabsList className="border-0">
              <TabsTrigger value="heatmap">Heat Map</TabsTrigger>
              <TabsTrigger value="cross-bench">Cross-Bench</TabsTrigger>
              <TabsTrigger value="scatter">Scatter</TabsTrigger>
              <TabsTrigger value="efficiency">Efficiency</TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="heatmap" className="mt-0 p-4">
            {heatmapError ? (
              <CompareError message={`Error loading comparison heat map: ${heatmapError.message}`} />
            ) : (
              <JobHeatmap
                jobName={jobNames[0]}
                data={filteredHeatmapData}
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
            )}
          </TabsContent>
          <TabsContent value="cross-bench" className="mt-0 p-4">
            {slopeError ? (
              <CompareError message={`Error loading cross-bench comparison: ${slopeError.message}`} />
            ) : (
              <JobSlopeChart
                data={filteredSlopeData}
                isLoading={slopeLoading}
                isFetching={slopeIsPlaceholder}
                defaultConnectionMode="model"
              />
            )}
          </TabsContent>
          <TabsContent value="scatter" className="mt-0 p-4">
            {slopeError ? (
              <CompareError message={`Error loading scatter comparison: ${slopeError.message}`} />
            ) : (
              <JobScatterChart
                data={filteredSlopeData}
                isLoading={slopeLoading}
                isFetching={slopeIsPlaceholder}
              />
            )}
          </TabsContent>
          <TabsContent value="efficiency" className="mt-0 p-4">
            {slopeError ? (
              <CompareError message={`Error loading efficiency comparison: ${slopeError.message}`} />
            ) : (
              <JobEfficiencyChart
                data={filteredSlopeData}
                isLoading={slopeLoading}
                isFetching={slopeIsPlaceholder}
              />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function CompareError({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4">
      <p className="text-destructive">{message}</p>
      <Button asChild>
        <Link to="/">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </Link>
      </Button>
    </div>
  );
}

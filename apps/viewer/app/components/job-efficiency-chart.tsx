import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

import {
  buildAgentShapeMap,
  MarkerShape,
  placeLabels,
  type ShapeKey,
} from "~/components/job-scatter-chart";
import {
  ChartToolbar,
  ChartToolbarSelect,
} from "~/components/ui/chart-toolbar";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty";
import { IndeterminateBar } from "~/components/ui/indeterminate-bar";
import {
  bareModelName,
  FAMILY_CONFIG,
  FAMILY_ORDER,
  familyColor,
  getFamily,
  sortByFamilyRank,
} from "~/lib/model-family";
import type { JobHeatmapCell, JobHeatmapData } from "~/lib/types";
import { cn } from "~/lib/utils";

// ---------------------------------------------------------------------------
// Efficiency metrics (X axis). Each is a per-task average pulled from the
// heatmap cells and aggregated (trial-weighted) across every column.
// ---------------------------------------------------------------------------

type EfficiencyMetricKey =
  | "avg_cost_usd"
  | "avg_output_tokens"
  | "avg_duration_ms"
  | "avg_input_tokens"
  | "avg_cached_input_tokens"
  | "avg_peak_context_tokens"
  | "avg_agent_steps";

interface EfficiencyMetric {
  key: EfficiencyMetricKey;
  label: string;
  axisLabel: string;
  prefersLog: boolean;
  format: (value: number) => string;
}

function formatCompact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${Math.round(value / 1_000)}k`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function formatDurationMs(ms: number): string {
  const seconds = ms / 1000;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
}

function formatCost(value: number): string {
  if (value >= 1) return `$${value.toFixed(2)}`;
  if (value >= 0.01) return `$${value.toFixed(3)}`;
  return `$${value.toPrecision(2)}`;
}

const EFFICIENCY_METRICS: EfficiencyMetric[] = [
  {
    key: "avg_cost_usd",
    label: "Cost",
    axisLabel: "Avg cost per task",
    prefersLog: true,
    format: formatCost,
  },
  {
    key: "avg_output_tokens",
    label: "Output tokens",
    axisLabel: "Avg output tokens per task",
    prefersLog: true,
    format: formatCompact,
  },
  {
    key: "avg_duration_ms",
    label: "Wall clock time",
    axisLabel: "Avg wall clock per task",
    prefersLog: true,
    format: formatDurationMs,
  },
  {
    key: "avg_input_tokens",
    label: "Input tokens",
    axisLabel: "Avg uncached input tokens per task",
    prefersLog: true,
    format: formatCompact,
  },
  {
    key: "avg_cached_input_tokens",
    label: "Cached input tokens",
    axisLabel: "Avg cached input tokens per task",
    prefersLog: true,
    format: formatCompact,
  },
  {
    key: "avg_peak_context_tokens",
    label: "Peak context tokens",
    axisLabel: "Avg peak context tokens per task",
    prefersLog: true,
    format: formatCompact,
  },
  {
    key: "avg_agent_steps",
    label: "Agent steps",
    axisLabel: "Avg agent steps per task",
    prefersLog: false,
    format: formatCompact,
  },
];

function metricValue(
  cell: JobHeatmapCell,
  metricKey: EfficiencyMetricKey
): number | null {
  return cell[metricKey];
}

// ---------------------------------------------------------------------------
// Series construction (one point per model, or per agent + model config)
// ---------------------------------------------------------------------------

/**
 * How points are collapsed: "model" merges every agent running a given
 * model into a single point (all circles); "config" keeps one point per
 * agent + model and distinguishes agents by marker shape.
 */
type GroupMode = "model" | "config";

interface EfficiencyPoint {
  rowKey: string;
  /** Friendly label, e.g. "gpt-5.5". */
  label: string;
  /** Long label for tooltip, e.g. "claude-code / claude-opus-4-7". */
  fullLabel: string;
  agent: string;
  model: string;
  family: string;
  rankIndex: number;
  rankCount: number;
  /** Efficiency metric (cost / time / tokens) — the X axis. */
  x: number;
  /** Success rate (mean reward) — the Y axis. */
  y: number;
  /** Completed trials backing the X aggregate. */
  nX: number;
  /** Completed trials backing the Y aggregate. */
  nY: number;
}

interface BuiltEfficiency {
  points: EfficiencyPoint[];
  agentShapes: Map<string, ShapeKey>;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function buildEfficiency(
  data: JobHeatmapData,
  metricKey: EfficiencyMetricKey,
  groupMode: GroupMode
): BuiltEfficiency | null {
  type Acc = {
    rowKey: string;
    label: string;
    fullLabel: string;
    agent: string;
    agents: Set<string>;
    model: string;
    family: string;
    xTotal: number;
    xN: number;
    yTotal: number;
    yN: number;
  };
  const byKey = new Map<string, Acc>();

  for (const row of data.rows) {
    const model = bareModelName(row.model_name ?? "(unknown)");
    const agent = row.agent_name ?? "(unknown)";
    const family = getFamily(row.model_provider, row.model_name);
    const seriesKey =
      groupMode === "model" ? `model::${model}` : `${agent}::${model}`;

    let xTotal = 0;
    let xN = 0;
    let yTotal = 0;
    let yN = 0;
    for (const col of data.columns) {
      const cell = data.cells[row.key]?.[col.key];
      if (!cell || cell.n_completed <= 0) continue;
      if (cell.avg_reward !== null && Number.isFinite(cell.avg_reward)) {
        yTotal += cell.avg_reward * cell.n_completed;
        yN += cell.n_completed;
      }
      const x = metricValue(cell, metricKey);
      if (x !== null && Number.isFinite(x)) {
        xTotal += x * cell.n_completed;
        xN += cell.n_completed;
      }
    }
    if (xN <= 0 || yN <= 0) continue;

    const existing =
      byKey.get(seriesKey) ?? {
        rowKey: seriesKey,
        label: model,
        fullLabel: groupMode === "model" ? model : `${agent} / ${model}`,
        agent: groupMode === "model" ? "" : agent,
        agents: new Set<string>(),
        model,
        family,
        xTotal: 0,
        xN: 0,
        yTotal: 0,
        yN: 0,
      };
    existing.agents.add(agent);
    existing.xTotal += xTotal;
    existing.xN += xN;
    existing.yTotal += yTotal;
    existing.yN += yN;
    byKey.set(seriesKey, existing);
  }

  if (byKey.size === 0) return null;

  // In model mode, note when a point spans several agents so the legend
  // tooltip stays informative.
  if (groupMode === "model") {
    for (const acc of byKey.values()) {
      if (acc.agents.size > 1) {
        acc.fullLabel = `${acc.model} (${acc.agents.size} agents)`;
      }
    }
  }

  // Group by family for ranking + colour assignment.
  const byFamily = new Map<string, Acc[]>();
  for (const acc of byKey.values()) {
    const list = byFamily.get(acc.family) ?? [];
    list.push(acc);
    byFamily.set(acc.family, list);
  }

  const points: EfficiencyPoint[] = [];
  for (const [family, list] of byFamily.entries()) {
    const sorted = sortByFamilyRank(list, family, (a) => a.label);
    sorted.forEach((acc, i) => {
      points.push({
        rowKey: acc.rowKey,
        label: acc.label,
        fullLabel: acc.fullLabel,
        agent: acc.agent,
        model: acc.model,
        family,
        rankIndex: i,
        rankCount: sorted.length,
        x: acc.xTotal / acc.xN,
        y: acc.yTotal / acc.yN,
        nX: acc.xN,
        nY: acc.yN,
      });
    });
  }

  let xMin = Number.POSITIVE_INFINITY;
  let xMax = Number.NEGATIVE_INFINITY;
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;
  for (const p of points) {
    if (p.x < xMin) xMin = p.x;
    if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y;
    if (p.y > yMax) yMax = p.y;
  }
  if (!Number.isFinite(xMin)) xMin = 0;
  if (!Number.isFinite(xMax)) xMax = 1;
  if (!Number.isFinite(yMin)) yMin = 0;
  if (!Number.isFinite(yMax)) yMax = 1;

  const agentShapes = buildAgentShapeMap(points.map((p) => p.agent));

  return { points, agentShapes, xMin, xMax, yMin, yMax };
}

// ---------------------------------------------------------------------------
// Axis helpers
// ---------------------------------------------------------------------------

type XScaleMode = "linear" | "log";

function linearTicks(min: number, max: number, count = 6): number[] {
  if (Math.abs(max - min) < 1e-9) return [min];
  return Array.from(
    { length: count },
    (_, i) => min + ((max - min) * i) / (count - 1)
  );
}

function logTicks(min: number, max: number): number[] {
  const lo = Math.log10(Math.max(min, Number.MIN_VALUE));
  const hi = Math.log10(Math.max(max, Number.MIN_VALUE));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const ticks: number[] = [];
  for (let p = Math.floor(lo); p <= Math.ceil(hi); p += 1) {
    for (const m of [1, 2, 5]) {
      const tick = m * 10 ** p;
      if (tick >= min * 0.999 && tick <= max * 1.001) ticks.push(tick);
    }
  }
  if (ticks.length > 8) {
    return linearTicks(lo, hi, 6).map((v) => 10 ** v);
  }
  return ticks.length > 0 ? ticks : [min, max];
}

function padDomain(min: number, max: number): [number, number] {
  if (Math.abs(max - min) < 1e-9) {
    const pad = Math.max(Math.abs(max) * 0.1, 1);
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function buildYDomain(min: number, max: number, usePercent: boolean): [number, number] {
  if (usePercent) {
    const lo = Math.max(0, Math.floor(min * 10) / 10 - 0.05);
    const hi = Math.min(1, Math.ceil(max * 10) / 10 + 0.05);
    if (hi - lo < 0.2) {
      return [Math.max(0, lo - 0.1), Math.min(1, hi + 0.1)];
    }
    return [lo, hi];
  }
  return padDomain(min, max);
}

function formatPercent(v: number): string {
  return `${(v * 100).toFixed(0)}`;
}

function formatScore(v: number, usePercent: boolean): string {
  if (usePercent) return `${(v * 100).toFixed(1)}%`;
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const WIDTH = 900;
const MARGIN_TOP = 48;
const MARGIN_BOTTOM = 76;
const MARGIN_LEFT = 76;
const MARGIN_RIGHT = 52;
const PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const PLOT_H = 520;
const HEIGHT = MARGIN_TOP + PLOT_H + MARGIN_BOTTOM;

export interface JobEfficiencyChartProps {
  data: JobHeatmapData | undefined;
  isLoading?: boolean;
  isFetching?: boolean;
}

export function JobEfficiencyChart({
  data,
  isLoading,
  isFetching,
}: JobEfficiencyChartProps) {
  const [metricKey, setMetricKey] =
    useState<EfficiencyMetricKey>("avg_cost_usd");
  const [xScaleMode, setXScaleMode] = useState<XScaleMode>("log");
  const [groupMode, setGroupMode] = useState<GroupMode>("model");
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  const metric =
    EFFICIENCY_METRICS.find((m) => m.key === metricKey) ?? EFFICIENCY_METRICS[0];

  const chart = useMemo(
    () => (data ? buildEfficiency(data, metricKey, groupMode) : null),
    [data, metricKey, groupMode]
  );

  // When switching metric, default the scale to the metric's preference.
  useEffect(() => {
    setXScaleMode(metric.prefersLog ? "log" : "linear");
  }, [metric.prefersLog]);

  const controls = (
    <ChartToolbar
      description={
        <>
          Success rate (mean reward over non-errored trials) per{" "}
          {groupMode === "model" ? "model" : "agent + model"} vs{" "}
          <span className="text-foreground">{metric.label.toLowerCase()}</span>{" "}
          per task. The X axis is inverted, so the{" "}
          <span className="text-foreground">top-right is most efficient</span>{" "}
          (high success, low {metric.label.toLowerCase()}). Hue = model family
          {groupMode === "config" ? "; shape = agent" : ""}.
        </>
      }
    >
      <ChartToolbarSelect
        label="Group by"
        value={groupMode}
        onValueChange={(v) => setGroupMode(v as GroupMode)}
        options={[
          { value: "model", label: "Model" },
          { value: "config", label: "Agent + model" },
        ]}
      />
      <ChartToolbarSelect
        label="X metric"
        value={metricKey}
        onValueChange={(v) => setMetricKey(v as EfficiencyMetricKey)}
        options={EFFICIENCY_METRICS.map((m) => ({
          value: m.key,
          label: m.label,
        }))}
      />
      <ChartToolbarSelect
        label="X scale"
        value={xScaleMode}
        onValueChange={(v) => setXScaleMode(v as XScaleMode)}
        options={[
          { value: "log", label: "Log scale" },
          { value: "linear", label: "Linear" },
        ]}
      />
    </ChartToolbar>
  );

  if (isLoading || (!chart && isFetching)) {
    return (
      <div className="border bg-card relative min-h-80">
        {(isLoading || isFetching) && <IndeterminateBar className="-top-px" />}
      </div>
    );
  }

  if (!chart) {
    return (
      <div className="border bg-card relative">
        {controls}
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Search />
            </EmptyMedia>
            <EmptyTitle>No efficiency data</EmptyTitle>
            <EmptyDescription>
              No completed trials report both reward and the selected metric.
              Try another metric or widen the active filters.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const { points, agentShapes, xMin, xMax, yMin, yMax } = chart;

  // Y axis: show as percent when all rewards fall in the 0–1 range.
  const usePercent = yMin >= 0 && yMax <= 1;
  const [yLo, yHi] = buildYDomain(yMin, yMax, usePercent);
  const yRange = Math.max(yHi - yLo, 1e-6);

  // X axis: optionally log-scaled and always inverted (smaller = righter).
  const positiveXMin = Math.min(...points.map((p) => p.x).filter((x) => x > 0));
  const canUseLog = Number.isFinite(positiveXMin) && xMax > 0;
  const effectiveScale: XScaleMode =
    xScaleMode === "log" && canUseLog ? "log" : "linear";
  const [linXLo, linXHi] = padDomain(xMin, xMax);
  const rawXLo = effectiveScale === "log" ? Math.max(positiveXMin, Number.MIN_VALUE) : linXLo;
  const rawXHi = effectiveScale === "log" ? xMax : linXHi;
  const xDomainLo = effectiveScale === "log" ? Math.log10(rawXLo) : rawXLo;
  const xDomainHi =
    effectiveScale === "log"
      ? Math.log10(Math.max(rawXHi, rawXLo * 1.001))
      : rawXHi;
  const xRange = Math.max(xDomainHi - xDomainLo, 1e-9);

  // Inverted: fraction 0 (smallest value) maps to the right edge.
  const xForValue = (v: number) => {
    const scaled =
      effectiveScale === "log" ? Math.log10(Math.max(v, rawXLo)) : v;
    const frac = (scaled - xDomainLo) / xRange;
    return MARGIN_LEFT + (1 - frac) * PLOT_W;
  };
  const yForValue = (v: number) =>
    MARGIN_TOP + (1 - (v - yLo) / yRange) * PLOT_H;

  const xTicks =
    effectiveScale === "log"
      ? logTicks(rawXLo, rawXHi)
      : linearTicks(rawXLo, rawXHi, 6);
  const yTicks = usePercent
    ? [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].filter(
        (t) => t >= yLo - 1e-9 && t <= yHi + 1e-9
      )
    : linearTicks(yLo, yHi, 6);

  // Labels: prioritise efficient points (high reward, then low cost).
  const placements = placeLabels(
    points,
    xForValue,
    yForValue,
    (label) => Math.max(label.length * 6.6 + 6, 18),
    14,
    {
      left: MARGIN_LEFT - 4,
      right: MARGIN_LEFT + PLOT_W + 4,
      top: MARGIN_TOP - 4,
      bottom: MARGIN_TOP + PLOT_H + 4,
    },
    (p) => p.y - (p.x - xMin) / Math.max(xMax - xMin, 1e-9)
  );

  const familyGroups = (() => {
    const map = new Map<string, EfficiencyPoint[]>();
    for (const p of points) {
      const list = map.get(p.family) ?? [];
      list.push(p);
      map.set(p.family, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.rankIndex - b.rankIndex);
    }
    return FAMILY_ORDER.filter((f) => map.has(f)).map((f) => ({
      family: f,
      members: map.get(f)!,
    }));
  })();

  const sortedAgents = [...agentShapes.entries()].sort((a, b) =>
    a[0].localeCompare(b[0])
  );

  const showAgentShapes = groupMode === "config";

  // The hovered point drives the axis crosshair (no popup card).
  const hovered = hoveredKey
    ? (points.find((p) => p.rowKey === hoveredKey) ?? null)
    : null;
  const hoveredColor = hovered
    ? familyColor(hovered.family, hovered.rankIndex, hovered.rankCount)
    : null;
  const hcx = hovered ? xForValue(hovered.x) : 0;
  const hcy = hovered ? yForValue(hovered.y) : 0;

  return (
    <div className="border bg-card relative">
      {isFetching && <IndeterminateBar className="-top-px" />}
      {controls}
      <div className="overflow-x-auto">
        <div className="relative mx-auto" style={{ width: WIDTH }}>
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            width={WIDTH}
            style={{ display: "block" }}
            role="img"
            aria-label={`Efficiency: success rate vs ${metric.axisLabel}`}
          >
            <rect
              x={MARGIN_LEFT}
              y={MARGIN_TOP}
              width={PLOT_W}
              height={PLOT_H}
              fill="transparent"
              stroke="var(--border)"
              strokeWidth={1}
            />
            {xTicks.map((t) => {
              const x = xForValue(t);
              return (
                <g key={`xt-${t}`}>
                  <line
                    x1={x}
                    x2={x}
                    y1={MARGIN_TOP}
                    y2={MARGIN_TOP + PLOT_H}
                    stroke="var(--border)"
                    strokeWidth={1}
                    strokeDasharray="2 4"
                    opacity={0.35}
                  />
                  <line
                    x1={x}
                    x2={x}
                    y1={MARGIN_TOP + PLOT_H}
                    y2={MARGIN_TOP + PLOT_H + 5}
                    stroke="var(--border)"
                  />
                  <text
                    x={x}
                    y={MARGIN_TOP + PLOT_H + 18}
                    textAnchor="middle"
                    fontSize={11}
                    className="fill-muted-foreground"
                  >
                    {metric.format(t)}
                  </text>
                </g>
              );
            })}
            {yTicks.map((t) => {
              const y = yForValue(t);
              return (
                <g key={`yt-${t}`}>
                  <line
                    x1={MARGIN_LEFT}
                    x2={MARGIN_LEFT + PLOT_W}
                    y1={y}
                    y2={y}
                    stroke="var(--border)"
                    strokeWidth={1}
                    strokeDasharray="2 4"
                    opacity={0.35}
                  />
                  <line
                    x1={MARGIN_LEFT - 5}
                    x2={MARGIN_LEFT}
                    y1={y}
                    y2={y}
                    stroke="var(--border)"
                  />
                  <text
                    x={MARGIN_LEFT - 8}
                    y={y + 3.5}
                    textAnchor="end"
                    fontSize={11}
                    className="fill-muted-foreground"
                  >
                    {usePercent ? formatPercent(t) : formatScore(t, false)}
                  </text>
                </g>
              );
            })}

            {/* Axis labels */}
            <text
              x={MARGIN_LEFT + PLOT_W / 2}
              y={HEIGHT - 30}
              textAnchor="middle"
              fontSize={12}
              className="fill-foreground"
            >
              {metric.axisLabel}
              {effectiveScale === "log" ? " (log)" : ""} — inverted
            </text>
            <text
              x={MARGIN_LEFT + PLOT_W / 2}
              y={HEIGHT - 14}
              textAnchor="middle"
              fontSize={10}
              className="fill-muted-foreground"
            >
              ← less efficient · more efficient →
            </text>
            <text
              x={22}
              y={MARGIN_TOP + PLOT_H / 2}
              textAnchor="middle"
              fontSize={12}
              transform={`rotate(-90 22 ${MARGIN_TOP + PLOT_H / 2})`}
              className="fill-foreground"
            >
              Success rate{usePercent ? " (%)" : ""}
            </text>

            {/* "Most efficient" hint in the top-right corner */}
            <text
              x={MARGIN_LEFT + PLOT_W - 6}
              y={MARGIN_TOP + 14}
              textAnchor="end"
              fontSize={10}
              className="fill-muted-foreground"
              fontStyle="italic"
            >
              most efficient ↗
            </text>

            {/* Hover crosshair: dashed guides from the point to each axis,
                with the value highlighted where they meet the axis. */}
            {hovered && hoveredColor && (
              <g pointerEvents="none">
                <line
                  x1={hcx}
                  y1={hcy}
                  x2={hcx}
                  y2={MARGIN_TOP + PLOT_H}
                  stroke={hoveredColor}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.85}
                />
                <line
                  x1={MARGIN_LEFT}
                  y1={hcy}
                  x2={hcx}
                  y2={hcy}
                  stroke={hoveredColor}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.85}
                />
                <text
                  x={hcx}
                  y={MARGIN_TOP + PLOT_H + 18}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={600}
                  fill={hoveredColor}
                  style={{
                    paintOrder: "stroke",
                    stroke: "var(--card)",
                    strokeWidth: 4,
                    strokeLinejoin: "round",
                  }}
                >
                  {metric.format(hovered.x)}
                </text>
                <text
                  x={MARGIN_LEFT - 8}
                  y={hcy + 3.5}
                  textAnchor="end"
                  fontSize={11}
                  fontWeight={600}
                  fill={hoveredColor}
                  style={{
                    paintOrder: "stroke",
                    stroke: "var(--card)",
                    strokeWidth: 4,
                    strokeLinejoin: "round",
                  }}
                >
                  {usePercent
                    ? `${formatPercent(hovered.y)}%`
                    : formatScore(hovered.y, false)}
                </text>
              </g>
            )}

            {/* Markers — render hovered last so it always wins z-order */}
            {[...points]
              .sort((a, b) => {
                if (hoveredKey === a.rowKey) return 1;
                if (hoveredKey === b.rowKey) return -1;
                return 0;
              })
              .map((p) => {
                const cx = xForValue(p.x);
                const cy = yForValue(p.y);
                const color = familyColor(p.family, p.rankIndex, p.rankCount);
                const isHovered = hoveredKey === p.rowKey;
                const shape = showAgentShapes
                  ? (agentShapes.get(p.agent) ?? "circle")
                  : "circle";
                return (
                  <g
                    key={`pt-${p.rowKey}`}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoveredKey(p.rowKey)}
                    onMouseLeave={() => setHoveredKey(null)}
                  >
                    <circle
                      cx={cx}
                      cy={cy}
                      r={14}
                      fill="transparent"
                      pointerEvents="all"
                    />
                    <MarkerShape
                      shape={shape}
                      cx={cx}
                      cy={cy}
                      r={isHovered ? 8.5 : 6}
                      fill={color}
                      strokeWidth={isHovered ? 2 : 1.5}
                    />
                  </g>
                );
              })}

            {/* Inline labels for as many points as fit */}
            {placements.map((pl) => {
              const color = familyColor(
                pl.point.family,
                pl.point.rankIndex,
                pl.point.rankCount
              );
              return (
                <g key={`lbl-${pl.point.rowKey}`} pointerEvents="none">
                  {Math.abs(pl.labelY - pl.cy) > 1.5 && (
                    <line
                      x1={pl.cx}
                      y1={pl.cy}
                      x2={pl.labelX + (pl.side === "right" ? -2 : 2)}
                      y2={pl.labelY}
                      stroke={color}
                      strokeWidth={1}
                      opacity={0.4}
                    />
                  )}
                  <text
                    x={pl.labelX}
                    y={pl.labelY + 4}
                    textAnchor={pl.side === "right" ? "start" : "end"}
                    fontSize={11}
                    fill={color}
                    style={{
                      paintOrder: "stroke",
                      stroke: "var(--background)",
                      strokeWidth: 3,
                      strokeLinejoin: "round",
                    }}
                  >
                    {pl.point.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <EfficiencyLegend
        familyGroups={familyGroups}
        agents={showAgentShapes ? sortedAgents : []}
        hoveredKey={hoveredKey}
        setHovered={setHoveredKey}
      />
    </div>
  );
}

function EfficiencyLegend({
  familyGroups,
  agents,
  hoveredKey,
  setHovered,
}: {
  familyGroups: { family: string; members: EfficiencyPoint[] }[];
  agents: [string, ShapeKey][];
  hoveredKey: string | null;
  setHovered: (k: string | null) => void;
}) {
  return (
    <div className="border-t">
      <div className="flex flex-wrap gap-x-6 gap-y-2 px-4 py-3 text-xs">
        {familyGroups.map(({ family, members }) => (
          <div key={family} className="flex items-center gap-2">
            <span className="text-muted-foreground">
              {FAMILY_CONFIG[family]?.label ?? family}
            </span>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              {members.map((m) => {
                const color = familyColor(m.family, m.rankIndex, m.rankCount);
                const isDimmed = hoveredKey !== null && hoveredKey !== m.rowKey;
                return (
                  <button
                    key={m.rowKey}
                    type="button"
                    className={cn(
                      "inline-flex items-center gap-1.5 transition-opacity",
                      isDimmed && "opacity-30"
                    )}
                    onMouseEnter={() => setHovered(m.rowKey)}
                    onMouseLeave={() => setHovered(null)}
                    title={m.fullLabel}
                  >
                    <span
                      className="inline-block size-2.5 rounded-full"
                      style={{ background: color }}
                    />
                    <span className="font-mono">{m.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {agents.length > 1 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 pb-3 text-xs border-t pt-2">
          <span className="text-muted-foreground">Agents</span>
          {agents.map(([agent, shape]) => (
            <span key={agent} className="inline-flex items-center gap-1.5">
              <svg width={16} height={16} viewBox="0 0 16 16">
                <MarkerShape
                  shape={shape}
                  cx={8}
                  cy={8}
                  r={5}
                  fill="var(--muted-foreground)"
                  stroke="var(--background)"
                  strokeWidth={1}
                />
              </svg>
              <span className="font-mono">{agent}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

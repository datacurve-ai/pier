import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

import {
  ChartToolbar,
  ChartToolbarAction,
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
// Marker shapes
// ---------------------------------------------------------------------------

export const SHAPE_KEYS = [
  "circle",
  "square",
  "triangle",
  "diamond",
  "plus",
  "cross",
] as const;
export type ShapeKey = (typeof SHAPE_KEYS)[number];

interface ShapeProps {
  cx: number;
  cy: number;
  r: number;
  fill: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
}

export function MarkerShape({
  shape,
  cx,
  cy,
  r,
  fill,
  stroke = "var(--background)",
  strokeWidth = 1.5,
  opacity = 1,
}: ShapeProps & { shape: ShapeKey }) {
  switch (shape) {
    case "circle":
      return (
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    case "square": {
      const s = r * 1.85;
      return (
        <rect
          x={cx - s / 2}
          y={cy - s / 2}
          width={s}
          height={s}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    }
    case "triangle": {
      const s = r * 2.1;
      const h = (Math.sqrt(3) / 2) * s;
      const points = `${cx},${cy - h * 0.62} ${cx - s / 2},${cy + h * 0.38} ${cx + s / 2},${cy + h * 0.38}`;
      return (
        <polygon
          points={points}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    }
    case "diamond": {
      const s = r * 1.45;
      const points = `${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`;
      return (
        <polygon
          points={points}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    }
    case "plus": {
      const a = r * 0.5;
      const b = r * 1.4;
      const points = `${cx - a},${cy - b} ${cx + a},${cy - b} ${cx + a},${cy - a} ${cx + b},${cy - a} ${cx + b},${cy + a} ${cx + a},${cy + a} ${cx + a},${cy + b} ${cx - a},${cy + b} ${cx - a},${cy + a} ${cx - b},${cy + a} ${cx - b},${cy - a} ${cx - a},${cy - a}`;
      return (
        <polygon
          points={points}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    }
    case "cross": {
      // 45-degree rotated plus.
      const a = r * 0.42;
      const b = r * 1.25;
      const pts: [number, number][] = [
        [cx - a, cy - b],
        [cx + a, cy - b],
        [cx + a, cy - a],
        [cx + b, cy - a],
        [cx + b, cy + a],
        [cx + a, cy + a],
        [cx + a, cy + b],
        [cx - a, cy + b],
        [cx - a, cy + a],
        [cx - b, cy + a],
        [cx - b, cy - a],
        [cx - a, cy - a],
      ];
      const sin = Math.sin(Math.PI / 4);
      const cos = Math.cos(Math.PI / 4);
      const rotated = pts.map(([x, y]) => {
        const dx = x - cx;
        const dy = y - cy;
        return `${cx + dx * cos - dy * sin},${cy + dx * sin + dy * cos}`;
      });
      return (
        <polygon
          points={rotated.join(" ")}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeWidth}
          opacity={opacity}
        />
      );
    }
  }
}

/** Stable mapping from agent name to a shape. */
export function buildAgentShapeMap(agents: string[]): Map<string, ShapeKey> {
  const sorted = [...new Set(agents)].sort((a, b) => a.localeCompare(b));
  const map = new Map<string, ShapeKey>();
  sorted.forEach((agent, i) => {
    map.set(agent, SHAPE_KEYS[i % SHAPE_KEYS.length]);
  });
  return map;
}

// ---------------------------------------------------------------------------
// Series construction
// ---------------------------------------------------------------------------

interface ScatterPoint {
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
  x: number;
  y: number;
  nX: number;
  nY: number;
}

interface BuiltScatter {
  points: ScatterPoint[];
  agentShapes: Map<string, ShapeKey>;
  /** Min/max x and y across visible points. */
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function cellLookup(
  data: JobHeatmapData,
  rowKey: string,
  colKey: string
): JobHeatmapCell | undefined {
  return data.cells[rowKey]?.[colKey];
}

function buildScatter(
  data: JobHeatmapData,
  xColKey: string,
  yColKey: string
): BuiltScatter | null {
  // Group rows by (agent + model) so multiple jobs collapse into one
  // visual point — same convention as the slope chart's "config" mode.
  type Acc = {
    rowKey: string;
    label: string;
    fullLabel: string;
    agent: string;
    model: string;
    family: string;
    xTotal: number;
    xN: number;
    yTotal: number;
    yN: number;
  };
  const byKey = new Map<string, Acc>();
  for (const row of data.rows) {
    const xCell = cellLookup(data, row.key, xColKey);
    const yCell = cellLookup(data, row.key, yColKey);
    if (
      !xCell ||
      xCell.avg_reward === null ||
      xCell.n_completed <= 0 ||
      !yCell ||
      yCell.avg_reward === null ||
      yCell.n_completed <= 0
    ) {
      continue;
    }
    const model = bareModelName(row.model_name ?? "(unknown)");
    const agent = row.agent_name ?? "(unknown)";
    const seriesKey = `${agent}::${model}`;
    const family = getFamily(row.model_provider, row.model_name);
    const existing =
      byKey.get(seriesKey) ?? {
        rowKey: seriesKey,
        label: model,
        fullLabel: `${agent} / ${model}`,
        agent,
        model,
        family,
        xTotal: 0,
        xN: 0,
        yTotal: 0,
        yN: 0,
      };
    existing.xTotal += xCell.avg_reward * xCell.n_completed;
    existing.xN += xCell.n_completed;
    existing.yTotal += yCell.avg_reward * yCell.n_completed;
    existing.yN += yCell.n_completed;
    byKey.set(seriesKey, existing);
  }

  if (byKey.size === 0) return null;

  // Group by family for ranking + colour assignment.
  const byFamily = new Map<string, Acc[]>();
  for (const acc of byKey.values()) {
    const list = byFamily.get(acc.family) ?? [];
    list.push(acc);
    byFamily.set(acc.family, list);
  }

  const points: ScatterPoint[] = [];
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
// Label collision avoidance (inline labels next to points)
// ---------------------------------------------------------------------------

export interface LabelPlacement<T> {
  point: T;
  cx: number;
  cy: number;
  /** Final label position; may be nudged off the anchor. */
  labelX: number;
  labelY: number;
  /** True if the label sits to the right of the point. */
  side: "left" | "right";
}

/** Minimal shape a point needs to be labelled. */
interface LabelablePoint {
  x: number;
  y: number;
  label: string;
  rowKey: string;
}

/**
 * Place text labels next to chart points. Greedy: sort by score desc,
 * try to place each label without overlapping an existing one's bounding
 * box. If no slot fits within `maxNudge` px of the anchor, drop it.
 *
 * `scoreOf` controls placement priority when labels compete for space;
 * it defaults to x + y (favours the top-right) but charts with an
 * inverted axis can pass a custom ranking.
 */
export function placeLabels<T extends LabelablePoint>(
  points: T[],
  xForValue: (v: number) => number,
  yForValue: (v: number) => number,
  estimateWidth: (label: string) => number,
  textHeight: number,
  plot: { left: number; right: number; top: number; bottom: number },
  scoreOf: (p: T) => number = (p) => p.x + p.y
): LabelPlacement<T>[] {
  const sorted = [...points].sort((a, b) => scoreOf(b) - scoreOf(a));
  const placements: LabelPlacement<T>[] = [];
  const occupied: { x1: number; y1: number; x2: number; y2: number }[] = [];

  const overlaps = (a: typeof occupied[number], b: typeof occupied[number]) =>
    !(a.x2 < b.x1 || a.x1 > b.x2 || a.y2 < b.y1 || a.y1 > b.y2);

  for (const p of sorted) {
    const cx = xForValue(p.x);
    const cy = yForValue(p.y);
    const w = estimateWidth(p.label);
    const h = textHeight;

    // Try a sequence of candidate offsets. (dx, dy) is the offset from the
    // marker; `side` picks which edge the label aligns to. Order favours
    // right > top > left > bottom, and small nudges before large ones.
    const baseOffset = 12;
    const candidates: { dx: number; dy: number; side: "left" | "right" }[] = [];
    for (const nudge of [0, -10, 10, -20, 20, -30, 30]) {
      candidates.push({ dx: baseOffset, dy: nudge, side: "right" });
      candidates.push({ dx: -baseOffset, dy: nudge, side: "left" });
    }

    let placed: LabelPlacement<T> | null = null;
    for (const c of candidates) {
      const lx =
        c.side === "right" ? cx + c.dx : cx + c.dx; // anchor x
      const ly = cy + c.dy;
      const x1 = c.side === "right" ? lx : lx - w;
      const x2 = c.side === "right" ? lx + w : lx;
      const y1 = ly - h / 2;
      const y2 = ly + h / 2;
      if (x1 < plot.left || x2 > plot.right || y1 < plot.top || y2 > plot.bottom) {
        continue;
      }
      const box = { x1, y1, x2, y2 };
      if (occupied.some((o) => overlaps(o, box))) continue;
      placed = {
        point: p,
        cx,
        cy,
        labelX: lx,
        labelY: ly,
        side: c.side,
      };
      occupied.push(box);
      break;
    }
    if (placed) placements.push(placed);
  }
  return placements;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

// Square plot: PLOT_W == PLOT_H so visual distance from y=x corresponds
// to the same magnitude on either axis (and points read as "above /
// below the diagonal" intuitively).
const WIDTH = 900;
const MARGIN_TOP = 56;
const MARGIN_BOTTOM = 72;
const MARGIN_LEFT = 80;
const MARGIN_RIGHT = 64;
const PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const PLOT_H = PLOT_W;
const HEIGHT = MARGIN_TOP + PLOT_H + MARGIN_BOTTOM;

function formatPercent(v: number): string {
  return `${(v * 100).toFixed(1)}`;
}

function formatRaw(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function buildAxisDomain(
  min: number,
  max: number,
  usePercent: boolean
): [number, number] {
  if (usePercent) {
    const lo = Math.max(0, Math.floor(min * 10) / 10 - 0.05);
    const hi = Math.min(1, Math.ceil(max * 10) / 10 + 0.05);
    if (hi - lo < 0.2) {
      // Avoid degenerate scales when all points cluster.
      return [Math.max(0, lo - 0.1), Math.min(1, hi + 0.1)];
    }
    return [lo, hi];
  }
  const span = Math.max(max - min, 1e-6);
  const pad = span * 0.1;
  return [min - pad, max + pad];
}

function buildTicks(min: number, max: number, usePercent: boolean): number[] {
  const span = max - min;
  if (usePercent) {
    const all = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
    const ticks = all.filter((t) => t >= min - 1e-9 && t <= max + 1e-9);
    return ticks.length > 0 ? ticks : [min, max];
  }
  return Array.from({ length: 5 }, (_, i) => min + (span * i) / 4);
}

interface HoveredPoint {
  point: ScatterPoint;
  cx: number;
  cy: number;
}

export interface JobScatterChartProps {
  data: JobHeatmapData | undefined;
  isLoading?: boolean;
  isFetching?: boolean;
  /** Initial dataset selection. Falls back to first two columns. */
  defaultXColumn?: string;
  defaultYColumn?: string;
}

export function JobScatterChart({
  data,
  isLoading,
  isFetching,
  defaultXColumn,
  defaultYColumn,
}: JobScatterChartProps) {
  // Dataset (column) options.
  const columns = useMemo(() => data?.columns ?? [], [data]);
  const columnOptions = useMemo(
    () => columns.map((c) => ({ value: c.key, label: c.label })),
    [columns]
  );

  // Pick sensible default selections. We prefer the two columns with the
  // most rows reporting both values (so the scatter actually shows
  // something interesting); ties broken by stable input order.
  const defaultPair = useMemo(() => {
    if (columns.length < 2) return null;
    if (defaultXColumn && defaultYColumn) {
      return { x: defaultXColumn, y: defaultYColumn };
    }
    let best: { x: string; y: string; count: number } | null = null;
    for (let i = 0; i < columns.length; i += 1) {
      for (let j = i + 1; j < columns.length; j += 1) {
        let count = 0;
        for (const row of data?.rows ?? []) {
          const a = cellLookup(data!, row.key, columns[i].key);
          const b = cellLookup(data!, row.key, columns[j].key);
          if (
            a &&
            b &&
            a.avg_reward !== null &&
            a.n_completed > 0 &&
            b.avg_reward !== null &&
            b.n_completed > 0
          ) {
            count += 1;
          }
        }
        if (!best || count > best.count) {
          best = { x: columns[i].key, y: columns[j].key, count };
        }
      }
    }
    return best ? { x: best.x, y: best.y } : null;
  }, [columns, data, defaultXColumn, defaultYColumn]);

  const [xColumn, setXColumn] = useState<string | null>(null);
  const [yColumn, setYColumn] = useState<string | null>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null);

  // Reset selection when the dataset list changes (e.g. filters change
  // and the previously-selected column is no longer present).
  useEffect(() => {
    if (!defaultPair) {
      setXColumn(null);
      setYColumn(null);
      return;
    }
    setXColumn((prev) =>
      prev && columns.some((c) => c.key === prev) ? prev : defaultPair.x
    );
    setYColumn((prev) =>
      prev && columns.some((c) => c.key === prev) ? prev : defaultPair.y
    );
  }, [defaultPair, columns]);

  const effectiveX = xColumn ?? defaultPair?.x ?? null;
  const effectiveY = yColumn ?? defaultPair?.y ?? null;

  const chart = useMemo(() => {
    if (!data || !effectiveX || !effectiveY || effectiveX === effectiveY) {
      return null;
    }
    return buildScatter(data, effectiveX, effectiveY);
  }, [data, effectiveX, effectiveY]);

  if (isLoading || (!chart && isFetching)) {
    return (
      <div className="border bg-card relative min-h-80">
        {(isLoading || isFetching) && <IndeterminateBar className="-top-px" />}
      </div>
    );
  }

  // Need at least 2 datasets to plot anything.
  if (columns.length < 2) {
    return (
      <div className="border bg-card relative">
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Search />
            </EmptyMedia>
            <EmptyTitle>Need at least 2 datasets</EmptyTitle>
            <EmptyDescription>
              The scatter chart compares accuracy across two datasets. Add
              another dataset (or widen the active filters) to use it.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const xColumnObj = columns.find((c) => c.key === effectiveX) ?? null;
  const yColumnObj = columns.find((c) => c.key === effectiveY) ?? null;
  const sameColumn = effectiveX !== null && effectiveX === effectiveY;

  if (!chart || sameColumn || !xColumnObj || !yColumnObj) {
    return (
      <div className="border bg-card relative">
        <ScatterControls
          columnOptions={columnOptions}
          xColumn={effectiveX}
          yColumn={effectiveY}
          setXColumn={setXColumn}
          setYColumn={setYColumn}
          xColumnLabel={xColumnObj?.label}
          yColumnLabel={yColumnObj?.label}
        />
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <Search />
            </EmptyMedia>
            <EmptyTitle>
              {sameColumn ? "Pick two different datasets" : "No overlap"}
            </EmptyTitle>
            <EmptyDescription>
              {sameColumn
                ? "X and Y must be different datasets to draw a scatter."
                : "No agent + model configuration has completed runs on both datasets."}
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    );
  }

  const { points, agentShapes, xMin, xMax, yMin, yMax } = chart;

  // Decide whether to display as percentages. If both axes are in the
  // 0–1 range (all reward signals are normalised) we show percents;
  // otherwise use raw values with adaptive precision.
  const usePercent = xMin >= 0 && xMax <= 1 && yMin >= 0 && yMax <= 1;
  const [xLo, xHi] = buildAxisDomain(xMin, xMax, usePercent);
  const [yLo, yHi] = buildAxisDomain(yMin, yMax, usePercent);
  const xRange = Math.max(xHi - xLo, 1e-6);
  const yRange = Math.max(yHi - yLo, 1e-6);

  const xForValue = (v: number) => MARGIN_LEFT + ((v - xLo) / xRange) * PLOT_W;
  const yForValue = (v: number) =>
    MARGIN_TOP + (1 - (v - yLo) / yRange) * PLOT_H;

  const xTicks = buildTicks(xLo, xHi, usePercent);
  const yTicks = buildTicks(yLo, yHi, usePercent);

  // y = x diagonal. Clip to the visible domain intersection.
  const diagLo = Math.max(xLo, yLo);
  const diagHi = Math.min(xHi, yHi);
  const drawDiagonal = diagHi > diagLo;

  // Place labels for the top-N points (by total score) so the chart isn't
  // overwhelmed; remaining points still appear as markers.
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
    }
  );

  // Group points by family for the legend.
  const familyGroups = (() => {
    const map = new Map<string, ScatterPoint[]>();
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

  return (
    <div className="border bg-card relative">
      {isFetching && <IndeterminateBar className="-top-px" />}
      <ScatterControls
        columnOptions={columnOptions}
        xColumn={effectiveX}
        yColumn={effectiveY}
        setXColumn={setXColumn}
        setYColumn={setYColumn}
        xColumnLabel={xColumnObj.label}
        yColumnLabel={yColumnObj.label}
      />
      <div className="overflow-x-auto">
        <div className="relative mx-auto" style={{ width: WIDTH }}>
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            width={WIDTH}
            style={{ display: "block" }}
            role="img"
            aria-label={`Scatter: ${xColumnObj.label} vs ${yColumnObj.label}`}
          >
            {/* Plot border + gridlines */}
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
                    {usePercent ? formatPercent(t) : formatRaw(t)}
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
                    {usePercent ? formatPercent(t) : formatRaw(t)}
                  </text>
                </g>
              );
            })}

            {/* Axis labels */}
            <text
              x={MARGIN_LEFT + PLOT_W / 2}
              y={HEIGHT - 18}
              textAnchor="middle"
              fontSize={12}
              className="fill-foreground"
            >
              {xColumnObj.label}
              {usePercent ? "  (%)" : ""}
            </text>
            <text
              x={20}
              y={MARGIN_TOP + PLOT_H / 2}
              textAnchor="middle"
              fontSize={12}
              transform={`rotate(-90 20 ${MARGIN_TOP + PLOT_H / 2})`}
              className="fill-foreground"
            >
              {yColumnObj.label}
              {usePercent ? "  (%)" : ""}
            </text>

            {/* y = x diagonal */}
            {drawDiagonal && (
              <>
                <line
                  x1={xForValue(diagLo)}
                  y1={yForValue(diagLo)}
                  x2={xForValue(diagHi)}
                  y2={yForValue(diagHi)}
                  stroke="var(--muted-foreground)"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.5}
                />
                <text
                  x={xForValue(diagHi) - 4}
                  y={yForValue(diagHi) + 12}
                  textAnchor="end"
                  fontSize={10}
                  className="fill-muted-foreground"
                  fontStyle="italic"
                >
                  y = x
                </text>
              </>
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
                const isDimmed = hoveredKey !== null && hoveredKey !== p.rowKey;
                const isHovered = hoveredKey === p.rowKey;
                const shape = agentShapes.get(p.agent) ?? "circle";
                return (
                  <g
                    key={`pt-${p.rowKey}`}
                    style={{ opacity: isDimmed ? 0.18 : 1, cursor: "pointer" }}
                    onMouseEnter={() => {
                      setHoveredKey(p.rowKey);
                      setHoveredPoint({ point: p, cx, cy });
                    }}
                    onMouseLeave={() => {
                      setHoveredKey(null);
                      setHoveredPoint(null);
                    }}
                  >
                    {/* Big invisible hit area for easier hover */}
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
                      r={isHovered ? 9 : 7}
                      fill={color}
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
              const isDimmed =
                hoveredKey !== null && hoveredKey !== pl.point.rowKey;
              return (
                <g
                  key={`lbl-${pl.point.rowKey}`}
                  style={{ opacity: isDimmed ? 0.18 : 1 }}
                  pointerEvents="none"
                >
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
          {hoveredPoint && (
            <div
              className="pointer-events-none absolute z-10 rounded-md border bg-popover text-popover-foreground shadow-md px-3 py-2 text-xs"
              style={{
                left: `${(hoveredPoint.cx / WIDTH) * 100}%`,
                top: `${(hoveredPoint.cy / HEIGHT) * 100}%`,
                transform: "translate(14px, -50%)",
                minWidth: 220,
              }}
            >
              <div className="font-medium mb-1.5">
                {hoveredPoint.point.fullLabel}
              </div>
              <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
                <span className="text-muted-foreground">{xColumnObj.label}</span>
                <span className="font-mono">
                  {usePercent
                    ? `${formatPercent(hoveredPoint.point.x)}%`
                    : formatRaw(hoveredPoint.point.x)}{" "}
                  <span className="text-muted-foreground">
                    (n = {hoveredPoint.point.nX})
                  </span>
                </span>
                <span className="text-muted-foreground">{yColumnObj.label}</span>
                <span className="font-mono">
                  {usePercent
                    ? `${formatPercent(hoveredPoint.point.y)}%`
                    : formatRaw(hoveredPoint.point.y)}{" "}
                  <span className="text-muted-foreground">
                    (n = {hoveredPoint.point.nY})
                  </span>
                </span>
                <span className="text-muted-foreground">Agent</span>
                <span>{hoveredPoint.point.agent}</span>
                <span className="text-muted-foreground">Family</span>
                <span>
                  {FAMILY_CONFIG[hoveredPoint.point.family]?.label ??
                    hoveredPoint.point.family}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
      <ScatterLegend
        familyGroups={familyGroups}
        agents={sortedAgents}
        hoveredKey={hoveredKey}
        setHovered={setHoveredKey}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ScatterControls({
  columnOptions,
  xColumn,
  yColumn,
  setXColumn,
  setYColumn,
  xColumnLabel,
  yColumnLabel,
}: {
  columnOptions: { value: string; label: string }[];
  xColumn: string | null;
  yColumn: string | null;
  setXColumn: (v: string | null) => void;
  setYColumn: (v: string | null) => void;
  xColumnLabel?: string;
  yColumnLabel?: string;
}) {
  const swap = () => {
    const xv = xColumn;
    const yv = yColumn;
    setXColumn(yv);
    setYColumn(xv);
  };

  const xOptions = columnOptions.map((option) => ({
    ...option,
    disabled: option.value === yColumn,
  }));
  const yOptions = columnOptions.map((option) => ({
    ...option,
    disabled: option.value === xColumn,
  }));

  return (
    <ChartToolbar
      description={
        <>
          Avg reward per (agent + model) on{" "}
          <span className="text-foreground">{xColumnLabel ?? "?"}</span> vs{" "}
          <span className="text-foreground">{yColumnLabel ?? "?"}</span>.
          Points above the y = x line do better on the y-axis dataset; below,
          better on x. Hue = model family; shape = agent.
        </>
      }
    >
      <ChartToolbarSelect
        label="X"
        value={xColumn ?? undefined}
        onValueChange={(v) => setXColumn(v)}
        options={xOptions}
        placeholder="Select dataset"
        className="max-w-64"
      />
      <ChartToolbarSelect
        label="Y"
        value={yColumn ?? undefined}
        onValueChange={(v) => setYColumn(v)}
        options={yOptions}
        placeholder="Select dataset"
        className="max-w-64"
      />
      <ChartToolbarAction onClick={swap}>Swap</ChartToolbarAction>
    </ChartToolbar>
  );
}

function ScatterLegend({
  familyGroups,
  agents,
  hoveredKey,
  setHovered,
}: {
  familyGroups: { family: string; members: ScatterPoint[] }[];
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

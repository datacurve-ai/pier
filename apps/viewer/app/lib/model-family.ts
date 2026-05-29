// Shared model-family + color helpers used by the slope and scatter
// charts. Keeping these in one place ensures both views agree on which
// model belongs to which family (and therefore which colour) so cross-
// referencing the two charts is intuitive.

export interface FamilyConfig {
  /** OKLCH hue for the family. */
  hue: number;
  /** Display name for the family. */
  label: string;
}

export const FAMILY_CONFIG: Record<string, FamilyConfig> = {
  openai: { hue: 155, label: "OpenAI" },
  anthropic: { hue: 35, label: "Anthropic" },
  gemini: { hue: 250, label: "Gemini" },
  glm: { hue: 195, label: "GLM" },
  kimi: { hue: 5, label: "Kimi" },
  deepseek: { hue: 280, label: "DeepSeek" },
  qwen: { hue: 220, label: "Qwen" },
  xiaomi: { hue: 60, label: "Xiaomi" },
  minimax: { hue: 325, label: "MiniMax" },
  other: { hue: 0, label: "Other" },
};

/**
 * Within-family rank (best → worst by our subjective expectation). Lines
 * for ranks not listed here fall through to alphabetical sort.
 */
export const FAMILY_RANK: Record<string, string[]> = {
  openai: ["gpt-5.5", "gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"],
  anthropic: [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
  ],
  gemini: ["gemini-3.1-pro-preview", "gemini-3-flash-preview"],
};

export const FAMILY_ORDER = [
  "openai",
  "anthropic",
  "gemini",
  "glm",
  "kimi",
  "deepseek",
  "qwen",
  "xiaomi",
  "minimax",
  "other",
];

/** Strip provider prefix from model strings like "openrouter/qwen/qwen3.6-plus". */
export function bareModelName(model: string): string {
  const lower = model.toLowerCase();
  const slash = lower.lastIndexOf("/");
  return slash >= 0 ? lower.slice(slash + 1) : lower;
}

export function getFamily(
  provider: string | null,
  model: string | null,
): string {
  if (!model) return "other";
  const m = model.toLowerCase();
  const p = provider?.toLowerCase() ?? "";
  if (p === "openai" || m.includes("gpt-")) return "openai";
  if (p === "anthropic" || m.includes("claude-")) return "anthropic";
  if (p === "gemini" || m.includes("gemini-")) return "gemini";
  if (m.includes("glm-")) return "glm";
  if (m.includes("kimi-")) return "kimi";
  if (m.includes("deepseek-")) return "deepseek";
  if (m.includes("qwen")) return "qwen";
  if (m.includes("mimo-")) return "xiaomi";
  if (m.includes("minimax")) return "minimax";
  return "other";
}

/**
 * Compute an OKLCH colour for a member of a family. `rankIndex` is the
 * member's position within the family (best = 0); `rankCount` is the
 * total number of members. Best is brightest/most saturated; worst stays
 * legible.
 */
export function familyColor(
  family: string,
  rankIndex: number,
  rankCount: number,
): string {
  const cfg = FAMILY_CONFIG[family] ?? FAMILY_CONFIG.other;
  const t = rankCount > 1 ? rankIndex / (rankCount - 1) : 0;
  const L = 0.74 - t * 0.16;
  const C = 0.18 - t * 0.06;
  return `oklch(${L.toFixed(3)} ${C.toFixed(3)} ${cfg.hue})`;
}

/**
 * Sort a list of items in family rank order. Items not found in the
 * family ranking sort alphabetically at the bottom.
 */
export function sortByFamilyRank<T>(
  items: T[],
  family: string,
  labelOf: (item: T) => string,
): T[] {
  const ranking = FAMILY_RANK[family];
  return [...items].sort((a, b) => {
    if (ranking) {
      const al = labelOf(a);
      const bl = labelOf(b);
      const aIdx = ranking.findIndex((m) => al === m);
      const bIdx = ranking.findIndex((m) => bl === m);
      const aRank = aIdx === -1 ? Number.POSITIVE_INFINITY : aIdx;
      const bRank = bIdx === -1 ? Number.POSITIVE_INFINITY : bIdx;
      if (aRank !== bRank) return aRank - bRank;
    }
    return labelOf(a).localeCompare(labelOf(b));
  });
}

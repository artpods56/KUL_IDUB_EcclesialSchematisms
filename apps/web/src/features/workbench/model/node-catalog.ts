import type { NodeRegistry, NodeSpec } from "@/lib/api";

export type NodeGoalCategoryId =
  | "suggested"
  | "start"
  | "transform"
  | "analyze"
  | "present"
  | "reuse"
  | "all";

export interface NodeGoalCategory {
  id: NodeGoalCategoryId;
  title: string;
}

export const NODE_GOAL_CATEGORIES: readonly NodeGoalCategory[] = [
  { id: "suggested", title: "Suggested" },
  { id: "start", title: "Add data" },
  { id: "transform", title: "Transform" },
  { id: "analyze", title: "Analyze" },
  { id: "present", title: "Present" },
  { id: "reuse", title: "Workspace library" },
  { id: "all", title: "All" },
];

const ANALYSIS_TERMS = [
  "analyze",
  "classify",
  "completion",
  "count",
  "extract",
  "fuzzy",
  "llm",
  "match",
  "ocr",
  "query",
  "sum",
];

const PRESENTATION_TERMS = [
  "compose",
  "display",
  "export",
  "layer",
  "map",
  "markdown",
  "render",
  "report",
  "visualize",
];

function goalSearchWords(spec: NodeSpec): ReadonlySet<string> {
  return new Set(
    `${spec.title} ${spec.description} ${spec.operator_id}`
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean),
  );
}

export function nodeGoalCategory(spec: NodeSpec): Exclude<
  NodeGoalCategoryId,
  "suggested" | "all"
> {
  if (spec.module_graph_id || spec.plugin_slug === "graph.module") {
    return "reuse";
  }
  if (spec.inputs.length === 0) return "start";

  const searchWords = goalSearchWords(spec);
  if (ANALYSIS_TERMS.some((term) => searchWords.has(term))) {
    return "analyze";
  }
  if (PRESENTATION_TERMS.some((term) => searchWords.has(term))) {
    return "present";
  }
  if (spec.outputs.length === 0) return "present";
  return "transform";
}

export function catalogNodesForGoal(
  nodes: readonly NodeSpec[],
  category: NodeGoalCategoryId,
): readonly NodeSpec[] {
  if (category === "all") return nodes;
  if (category !== "suggested") {
    return nodes.filter((spec) => nodeGoalCategory(spec) === category);
  }

  const suggested: NodeSpec[] = [];
  for (const goal of ["start", "transform", "analyze", "present", "reuse"] as const) {
    const match = nodes.find((spec) => nodeGoalCategory(spec) === goal);
    if (match) suggested.push(match);
  }
  for (const spec of nodes) {
    if (suggested.length >= 6) break;
    if (
      !suggested.some(
        (candidate) =>
          candidate.operator_id === spec.operator_id &&
          candidate.operator_version === spec.operator_version,
      )
    ) {
      suggested.push(spec);
    }
  }
  return suggested;
}

export function catalogNodeSpecs(
  registry: NodeRegistry,
  activeGraphId: string | null,
): readonly NodeSpec[] {
  return registry.nodes.filter(
    (spec) =>
      spec.catalog_visible !== false &&
      (activeGraphId === null || spec.module_graph_id !== activeGraphId),
  );
}

/** All published releases for a module (including non-current). */
export function moduleReleaseSpecs(
  registry: NodeRegistry,
  moduleId: string | null | undefined,
  moduleGraphId: string | null | undefined,
): readonly NodeSpec[] {
  return registry.nodes
    .filter((spec) => {
      if (moduleId && spec.module_id === moduleId) return true;
      if (!moduleId && moduleGraphId && spec.module_graph_id === moduleGraphId) {
        return true;
      }
      return false;
    })
    .slice()
    .sort(
      (left, right) =>
        (right.module_graph_revision ?? 0) - (left.module_graph_revision ?? 0),
    );
}

/** Current library release when the pinned call is behind it; otherwise null. */
export function moduleCallUpgradeTarget(
  registry: NodeRegistry,
  pinned: NodeSpec,
): NodeSpec | null {
  if (!pinned.module_graph_id) return null;
  const current = moduleReleaseSpecs(
    registry,
    pinned.module_id,
    pinned.module_graph_id,
  ).find((release) => release.is_current_library_release);
  if (!current) return null;
  if (
    current.operator_id === pinned.operator_id &&
    current.operator_version === pinned.operator_version
  ) {
    return null;
  }
  return current;
}

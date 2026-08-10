import type { NodeRegistry, NodeSpec } from "@/lib/api";

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

export function catalogPluginSections(registry: NodeRegistry) {
  return [
    {
      origin: "builtin" as const,
      title: "Built-in",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "builtin"),
    },
    {
      origin: "module" as const,
      title: "Workspace library",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "module"),
    },
    {
      origin: "external" as const,
      title: "External",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "external"),
    },
  ];
}

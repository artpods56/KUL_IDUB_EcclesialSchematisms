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

export function catalogPluginSections(registry: NodeRegistry) {
  return [
    {
      origin: "builtin" as const,
      title: "Built-in",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "builtin"),
    },
    {
      origin: "module" as const,
      title: "Modules",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "module"),
    },
    {
      origin: "external" as const,
      title: "External",
      plugins: registry.plugins.filter((plugin) => plugin.origin === "external"),
    },
  ];
}

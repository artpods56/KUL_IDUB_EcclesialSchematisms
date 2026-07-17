import { describe, expect, it } from "vitest";

import type { NodeRegistry, NodeSpec } from "@/lib/api";
import { catalogNodeSpecs, catalogPluginSections } from "./node-catalog";

const FIRST_GRAPH_ID = "00000000-0000-4000-8000-000000000001";
const SECOND_GRAPH_ID = "00000000-0000-4000-8000-000000000002";

function nodeSpec(
  operatorId: string,
  pluginSlug: string,
  operatorVersion = 1,
  moduleGraphId: string | null = null,
  catalogVisible = true,
): NodeSpec {
  return {
    operator_id: operatorId,
    operator_version: operatorVersion,
    plugin_slug: pluginSlug,
    title: operatorId,
    description: operatorId,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
    module_graph_id: moduleGraphId,
    module_graph_revision: moduleGraphId ? operatorVersion : null,
    catalog_visible: catalogVisible,
  };
}

function registry(): NodeRegistry {
  return {
    plugins: [
      { slug: "builtin", title: "Built-in", origin: "builtin" },
      { slug: "saved-graph-modules", title: "Modules", origin: "module" },
      { slug: "external", title: "External", origin: "external" },
    ],
    artifact_types: [],
    artifact_conversions: [],
    nodes: [
      nodeSpec("text.input", "builtin"),
      nodeSpec(
        `module.graph.${FIRST_GRAPH_ID}`,
        "saved-graph-modules",
        1,
        FIRST_GRAPH_ID,
        false,
      ),
      nodeSpec(
        `module.graph.${FIRST_GRAPH_ID}`,
        "saved-graph-modules",
        2,
        FIRST_GRAPH_ID,
      ),
      nodeSpec(
        `module.graph.${SECOND_GRAPH_ID}`,
        "saved-graph-modules",
        1,
        SECOND_GRAPH_ID,
      ),
      nodeSpec("ocr.external", "external"),
    ],
  };
}

describe("node catalog modules", () => {
  it("groups saved graphs separately from built-in and external plugins", () => {
    const sections = catalogPluginSections(registry());

    expect(sections.map((section) => section.title)).toEqual([
      "Built-in",
      "Modules",
      "External",
    ]);
    expect(sections[1]?.plugins).toEqual([
      { slug: "saved-graph-modules", title: "Modules", origin: "module" },
    ]);
  });

  it("offers only visible revisions and excludes the graph being edited", () => {
    const available = catalogNodeSpecs(registry(), FIRST_GRAPH_ID);

    expect(
      available.map((spec) => [spec.operator_id, spec.operator_version]),
    ).toEqual([
      ["text.input", 1],
      [`module.graph.${SECOND_GRAPH_ID}`, 1],
      ["ocr.external", 1],
    ]);
  });

  it("keeps the latest visible revision available outside its own graph", () => {
    const available = catalogNodeSpecs(registry(), null);

    expect(
      available
        .filter((spec) => spec.module_graph_id === FIRST_GRAPH_ID)
        .map((spec) => spec.operator_version),
    ).toEqual([2]);
  });
});

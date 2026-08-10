import { describe, expect, it } from "vitest";

import type { NodeRegistry, NodeSpec } from "@/lib/api";
import {
  catalogNodeSpecs,
  catalogPluginSections,
  moduleCallUpgradeTarget,
} from "./node-catalog";

const FIRST_GRAPH_ID = "00000000-0000-4000-8000-000000000001";
const SECOND_GRAPH_ID = "00000000-0000-4000-8000-000000000002";
const FIRST_MODULE_ID = "10000000-0000-4000-8000-000000000001";

function nodeSpec(
  operatorId: string,
  pluginSlug: string,
  operatorVersion = 1,
  moduleGraphId: string | null = null,
  catalogVisible = true,
  options: {
    moduleId?: string | null;
    isCurrentLibraryRelease?: boolean | null;
  } = {},
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
    module_id: options.moduleId ?? null,
    is_current_library_release: options.isCurrentLibraryRelease ?? null,
    catalog_visible: catalogVisible,
  };
}

function registry(): NodeRegistry {
  return {
    plugins: [
      { slug: "builtin", title: "Built-in", origin: "builtin" },
      { slug: "graph.module", title: "Workspace library", origin: "module" },
      { slug: "external", title: "External", origin: "external" },
    ],
    artifact_types: [],
    artifact_conversions: [],
    nodes: [
      nodeSpec("text.input", "builtin"),
      nodeSpec(
        `module.graph.${FIRST_GRAPH_ID}`,
        "graph.module",
        1,
        FIRST_GRAPH_ID,
        false,
        {
          moduleId: FIRST_MODULE_ID,
          isCurrentLibraryRelease: false,
        },
      ),
      nodeSpec(
        `module.graph.${FIRST_GRAPH_ID}`,
        "graph.module",
        2,
        FIRST_GRAPH_ID,
        true,
        {
          moduleId: FIRST_MODULE_ID,
          isCurrentLibraryRelease: true,
        },
      ),
      nodeSpec(
        `module.graph.${SECOND_GRAPH_ID}`,
        "graph.module",
        1,
        SECOND_GRAPH_ID,
        true,
        { isCurrentLibraryRelease: true },
      ),
      nodeSpec("ocr.external", "external"),
    ],
  };
}

describe("node catalog modules", () => {
  it("groups workspace library separately from built-in and external plugins", () => {
    const sections = catalogPluginSections(registry());

    expect(sections.map((section) => section.title)).toEqual([
      "Built-in",
      "Workspace library",
      "External",
    ]);
    expect(sections[1]?.plugins).toEqual([
      { slug: "graph.module", title: "Workspace library", origin: "module" },
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

  it("offers an upgrade target when a newer library release exists", () => {
    const pinned = registry().nodes.find(
      (spec) =>
        spec.module_graph_id === FIRST_GRAPH_ID && spec.operator_version === 1,
    );
    expect(pinned).toBeDefined();
    const target = moduleCallUpgradeTarget(registry(), pinned!);
    expect(target?.operator_version).toBe(2);
    expect(
      moduleCallUpgradeTarget(registry(), target!),
    ).toBeNull();
  });
});

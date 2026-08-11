import { describe, expect, it } from "vitest";

import type { NodeRegistry, NodeSpec, Port } from "@/lib/api";
import {
  catalogNodesForGoal,
  catalogNodeSpecs,
  moduleCallUpgradeTarget,
  nodeGoalCategory,
} from "./node-catalog";

const FIRST_GRAPH_ID = "00000000-0000-4000-8000-000000000001";
const SECOND_GRAPH_ID = "00000000-0000-4000-8000-000000000002";
const FIRST_MODULE_ID = "10000000-0000-4000-8000-000000000001";

function port(name: string, direction: Port["direction"]): Port {
  return {
    name,
    title: name,
    description: null,
    direction,
    artifact_type: { id: "scalar.text", schema_version: 1 },
    artifact_type_variable: null,
    shape: "one",
    accepted_shapes: ["one"],
    instance_plugs: false,
    variadic: false,
    required: true,
  };
}

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

describe("node catalog goals", () => {
  it("derives user-goal categories from node contracts and searchable metadata", () => {
    const start = nodeSpec("table.file.import", "builtin");
    const transform = {
      ...nodeSpec("text.replace", "builtin"),
      inputs: [port("text", "input")],
      outputs: [port("text", "output")],
    };
    const analyze = {
      ...transform,
      operator_id: "table.markdown.extract",
      title: "Extract tables",
    };
    const present = {
      ...transform,
      operator_id: "gis.map.compose",
      title: "Compose map",
    };
    const reuse = registry().nodes[2]!;

    expect([
      nodeGoalCategory(start),
      nodeGoalCategory(transform),
      nodeGoalCategory(analyze),
      nodeGoalCategory(present),
      nodeGoalCategory(reuse),
    ]).toEqual(["start", "transform", "analyze", "present", "reuse"]);
  });

  it("builds a small deterministic suggested set across available goals", () => {
    const nodes = [
      nodeSpec("text.input", "builtin"),
      {
        ...nodeSpec("text.replace", "builtin"),
        inputs: [port("text", "input")],
        outputs: [port("text", "output")],
      },
      {
        ...nodeSpec("ocr.extract", "external"),
        inputs: [port("image", "input")],
        outputs: [port("text", "output")],
      },
      ...registry().nodes.filter((spec) => spec.catalog_visible !== false),
    ];

    const suggested = catalogNodesForGoal(nodes, "suggested");

    expect(suggested.length).toBeLessThanOrEqual(6);
    expect(suggested.slice(0, 4).map(nodeGoalCategory)).toEqual([
      "start",
      "transform",
      "analyze",
      "reuse",
    ]);
    expect(catalogNodesForGoal(nodes, "reuse")).toEqual([
      registry().nodes[2],
      registry().nodes[3],
    ]);
  });
});

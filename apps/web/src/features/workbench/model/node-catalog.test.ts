import { describe, expect, it } from "vitest";

import type { NodeRegistry, NodeSpec, Port } from "@/lib/api";
import { encodeHandleId } from "../canvas/handles";
import { portMetaForPort } from "../canvas/types";
import {
  artifactFilterId,
  buildCatalogFilters,
  catalogNodePortSummary,
  catalogNodeSpecs,
  catalogNodesForFilter,
  downstreamCandidatesFromOutput,
  filterAndSearchCatalogNodes,
  moduleCallUpgradeTarget,
  sortCatalogNodes,
  upstreamCandidatesFromInput,
} from "./node-catalog";

const FIRST_GRAPH_ID = "00000000-0000-4000-8000-000000000001";
const SECOND_GRAPH_ID = "00000000-0000-4000-8000-000000000002";
const FIRST_MODULE_ID = "10000000-0000-4000-8000-000000000001";

function port(
  name: string,
  direction: Port["direction"],
  options: Partial<Port> = {},
): Port {
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
    ...options,
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
    title?: string;
    description?: string;
    inputs?: Port[];
    outputs?: Port[];
  } = {},
): NodeSpec {
  return {
    operator_id: operatorId,
    operator_version: operatorVersion,
    plugin_slug: pluginSlug,
    title: options.title ?? operatorId,
    description: options.description ?? operatorId,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: options.inputs ?? [],
    outputs: options.outputs ?? [],
    module_graph_id: moduleGraphId,
    module_graph_revision: moduleGraphId ? operatorVersion : null,
    module_id: options.moduleId ?? null,
    is_current_library_release: options.isCurrentLibraryRelease ?? null,
    catalog_visible: catalogVisible,
    runnable: true,
  };
}

function registry(): NodeRegistry {
  return {
    plugins: [
      { slug: "builtin", title: "Built-in", origin: "builtin", runnable: true },
      {
        slug: "graph.module",
        title: "Workspace library",
        origin: "module",
        runnable: true,
      },
      { slug: "external", title: "External", origin: "external", runnable: true },
    ],
    artifact_types: [
      {
        key: { id: "scalar.text", schema_version: 1 },
        title: "Text",
        payload_schema: {},
        field_projections: [],
      },
      {
        key: { id: "scalar.text", schema_version: 2 },
        title: "Text",
        payload_schema: {},
        field_projections: [],
      },
      {
        key: { id: "table.data", schema_version: 1 },
        title: "Table",
        payload_schema: {},
        field_projections: [],
      },
    ],
    artifact_conversions: [],
    nodes: [
      nodeSpec("text.input", "builtin", 1, null, true, {
        title: "Enter text",
        outputs: [port("text", "output")],
      }),
      nodeSpec("text.replace", "builtin", 1, null, true, {
        title: "Replace text",
        inputs: [port("text", "input")],
        outputs: [port("text", "output")],
      }),
      nodeSpec("table.batch", "builtin", 1, null, true, {
        title: "Batch table",
        inputs: [
          port("rows", "input", {
            artifact_type: { id: "table.data", schema_version: 1 },
            shape: "many",
            accepted_shapes: ["many"],
          }),
        ],
        outputs: [
          port("rows", "output", {
            artifact_type: { id: "table.data", schema_version: 1 },
            shape: "many",
            accepted_shapes: ["many"],
          }),
        ],
      }),
      nodeSpec("generic.pass", "builtin", 1, null, true, {
        title: "Pass any",
        inputs: [
          port("value", "input", {
            artifact_type: null,
            artifact_type_variable: "T",
          }),
        ],
        outputs: [
          port("value", "output", {
            artifact_type: null,
            artifact_type_variable: "T",
          }),
        ],
      }),
      nodeSpec(
        `module.graph.${FIRST_GRAPH_ID}`,
        "graph.module",
        1,
        FIRST_GRAPH_ID,
        false,
        {
          moduleId: FIRST_MODULE_ID,
          isCurrentLibraryRelease: false,
          title: "Normalize invoices",
          inputs: [port("text", "input")],
          outputs: [
            port("table", "output", {
              artifact_type: { id: "table.data", schema_version: 1 },
            }),
          ],
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
          title: "Normalize invoices",
          inputs: [port("text", "input")],
          outputs: [
            port("table", "output", {
              artifact_type: { id: "table.data", schema_version: 1 },
            }),
          ],
        },
      ),
      nodeSpec(
        `module.graph.${SECOND_GRAPH_ID}`,
        "graph.module",
        1,
        SECOND_GRAPH_ID,
        true,
        {
          title: "Other module",
          isCurrentLibraryRelease: true,
          inputs: [port("text", "input")],
          outputs: [port("text", "output")],
        },
      ),
      nodeSpec("ocr.external", "external", 1, null, true, {
        title: "OCR",
      }),
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
      ["text.replace", 1],
      ["table.batch", 1],
      ["generic.pass", 1],
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
    expect(moduleCallUpgradeTarget(registry(), target!)).toBeNull();
  });
});

describe("artifact catalog filters", () => {
  it("builds artifact filters with version labels only for duplicate titles", () => {
    const filters = buildCatalogFilters(registry());
    expect(filters.map((filter) => filter.title)).toEqual([
      "All nodes",
      "Table",
      "Text · v1",
      "Text · v2",
      "Single value",
      "Sequence",
      "Any artifact",
      "Workspace library",
    ]);
  });

  it("matches exact artifact, shape, any-artifact, and library filters", () => {
    const nodes = catalogNodeSpecs(registry(), null);
    const filters = buildCatalogFilters(registry());
    const byId = Object.fromEntries(filters.map((filter) => [filter.id, filter]));

    expect(
      catalogNodesForFilter(
        nodes,
        byId[artifactFilterId({ id: "scalar.text", schema_version: 1 })]!,
      ).map((spec) => spec.operator_id),
    ).toEqual([
      "text.input",
      "text.replace",
      `module.graph.${FIRST_GRAPH_ID}`,
      `module.graph.${SECOND_GRAPH_ID}`,
    ]);

    expect(
      catalogNodesForFilter(nodes, byId.sequence!).map((spec) => spec.operator_id),
    ).toEqual(["table.batch"]);

    expect(
      catalogNodesForFilter(nodes, byId["any-artifact"]!).map(
        (spec) => spec.operator_id,
      ),
    ).toEqual(["generic.pass"]);

    expect(
      catalogNodesForFilter(nodes, byId["workspace-library"]!).map(
        (spec) => spec.operator_id,
      ),
    ).toEqual([
      `module.graph.${FIRST_GRAPH_ID}`,
      `module.graph.${SECOND_GRAPH_ID}`,
    ]);
  });

  it("sorts by title then operator identity and intersects search with filters", () => {
    const nodes = catalogNodeSpecs(registry(), null);
    const filter = buildCatalogFilters(registry()).find(
      (candidate) => candidate.id === "all",
    )!;

    expect(sortCatalogNodes(nodes).map((spec) => spec.title)).toEqual([
      "Batch table",
      "Enter text",
      "Normalize invoices",
      "OCR",
      "Other module",
      "Pass any",
      "Replace text",
    ]);

    expect(
      filterAndSearchCatalogNodes(nodes, filter, "replace", registry()).map(
        (spec) => spec.operator_id,
      ),
    ).toEqual(["text.replace"]);

    expect(
      filterAndSearchCatalogNodes(
        nodes,
        buildCatalogFilters(registry()).find(
          (candidate) => candidate.id === "workspace-library",
        )!,
        "normalize",
        registry(),
      ).map((spec) => spec.operator_id),
    ).toEqual([`module.graph.${FIRST_GRAPH_ID}`]);
  });

  it("summarizes ports with artifact titles", () => {
    const replace = catalogNodeSpecs(registry(), null).find(
      (spec) => spec.operator_id === "text.replace",
    )!;
    expect(catalogNodePortSummary(replace, registry())).toBe("Text → Text");
  });
});

describe("downstream candidates", () => {
  it("lists compatible nodes once and excludes optional instance-plug-only targets", () => {
    const source = port("text", "output");
    const sourceHandle = encodeHandleId(portMetaForPort(source));
    const nodes = [
      nodeSpec("text.replace", "builtin", 1, null, true, {
        title: "Replace text",
        inputs: [port("text", "input")],
        outputs: [port("text", "output")],
      }),
      nodeSpec("collect.optional", "builtin", 1, null, true, {
        title: "Optional collect",
        inputs: [
          port("items", "input", {
            instance_plugs: true,
            required: false,
          }),
        ],
        outputs: [port("text", "output")],
      }),
      nodeSpec("collect.required", "builtin", 1, null, true, {
        title: "Required collect",
        inputs: [
          port("items", "input", {
            instance_plugs: true,
            required: true,
          }),
        ],
        outputs: [port("text", "output")],
      }),
    ];

    const candidates = downstreamCandidatesFromOutput({
      sourcePort: source as Port & { readonly direction: "output" },
      sourceHandle,
      registry: { ...registry(), nodes },
      nodes,
    });

    expect(candidates.map((candidate) => candidate.spec.operator_id)).toEqual([
      "text.replace",
      "collect.required",
    ]);
    expect(candidates[0]?.choices).toHaveLength(1);
  });
});

describe("upstream candidates", () => {
  it("lists nodes whose outputs can feed the input", () => {
    const target = port("text", "input");
    const targetHandle = encodeHandleId(portMetaForPort(target));
    const nodes = [
      nodeSpec("text.replace", "builtin", 1, null, true, {
        title: "Replace text",
        inputs: [port("text", "input")],
        outputs: [port("text", "output")],
      }),
      nodeSpec("collect.optional", "builtin", 1, null, true, {
        title: "Optional collect",
        inputs: [
          port("items", "input", {
            instance_plugs: true,
            required: false,
          }),
        ],
        outputs: [port("text", "output")],
      }),
      nodeSpec("source.broken", "builtin", 1, null, true, {
        title: "Broken source",
        inputs: [],
        outputs: [
          port("rows", "output", {
            artifact_type: { id: "table.data", schema_version: 1 },
          }),
        ],
      }),
    ];

    const candidates = upstreamCandidatesFromInput({
      targetPort: target as Port & { readonly direction: "input" },
      targetHandle,
      registry: { ...registry(), nodes },
      nodes,
    });

    expect(candidates.map((candidate) => candidate.spec.operator_id)).toEqual([
      "collect.optional",
      "text.replace",
    ]);
    expect(candidates[0]?.choices[0]?.candidatePort.direction).toBe("output");
  });
});

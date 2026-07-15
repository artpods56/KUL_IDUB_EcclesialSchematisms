import { describe, expect, it } from "vitest";

import type {
  ArtifactConversionSpec,
  NodeRegistry,
  NodeSpec,
  SavedGraph,
} from "@/lib/api";
import { decodeHandleId } from "./handles";
import { hydrateSavedGraph, savedGraphDraft } from "./saved-graph";
import { serializeWorkflowEdgeTransport } from "./types";

function conversion(
  id: string,
  source: string,
  target: string,
): ArtifactConversionSpec {
  return {
    key: { id, version: 1 },
    source_artifact_type: { id: source, schema_version: 1 },
    target_artifact_type: { id: target, schema_version: 1 },
    title: id,
  };
}

function nodeSpec(
  operatorId: string,
  direction: "input" | "output",
  artifactTypeId: string,
  shape: "one" | "many" = "one",
  acceptedShapes: readonly ("one" | "many")[] = [shape],
): NodeSpec {
  const port = {
    name: direction,
    title: direction,
    description: null,
    direction,
    artifact_type: { id: artifactTypeId, schema_version: 1 },
    shape,
    accepted_shapes: acceptedShapes,
    instance_plugs: false,
    variadic: false,
    required: true,
  };
  return {
    operator_id: operatorId,
    operator_version: 1,
    plugin_slug: "test",
    title: operatorId,
    description: operatorId,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: direction === "input" ? [port] : [],
    outputs: direction === "output" ? [port] : [],
  };
}

const conversionPath = [
  { id: "x-to-y", version: 1 },
  { id: "y-to-z", version: 1 },
] as const;

function registry(
  sourceShape: "one" | "many" = "one",
  targetShape: "one" | "many" = "one",
  targetAcceptedShapes: readonly ("one" | "many")[] = [targetShape],
): NodeRegistry {
  return {
    plugins: [],
    artifact_types: ["x", "y", "z"].map((id) => ({
      key: { id, schema_version: 1 },
      title: id,
      payload_schema: {},
      field_projections: [],
    })),
    artifact_conversions: [
      conversion("x-to-z", "x", "z"),
      conversion("x-to-y", "x", "y"),
      conversion("y-to-z", "y", "z"),
    ],
    nodes: [
      nodeSpec("source", "output", "x", sourceShape),
      nodeSpec(
        "target",
        "input",
        "z",
        targetShape,
        targetAcceptedShapes,
      ),
    ],
  };
}

function graphWithEdge(
  edgeConversion: Pick<
    NonNullable<SavedGraph["edges"]>[number],
    "conversion_path"
  >,
): SavedGraph {
  return {
    id: "00000000-0000-4000-8000-000000000001",
    revision: 1,
    name: "Conversion path",
    created_at: "2026-07-15T12:00:00Z",
    updated_at: "2026-07-15T12:00:00Z",
    nodes: [
      {
        id: "source-node",
        operator_id: "source",
        operator_version: 1,
        config: {},
        input_plugs: [],
        position: { x: 0, y: 0 },
      },
      {
        id: "target-node",
        operator_id: "target",
        operator_version: 1,
        config: {},
        input_plugs: [],
        position: { x: 300, y: 0 },
      },
    ],
    edges: [
      {
        id: "edge",
        from_node: "source-node",
        from_port: "output",
        to_node: "target-node",
        to_port: "input",
        to_plug: null,
        collection_mode: "direct",
        projection: null,
        route_offset: null,
        ...edgeConversion,
      },
    ],
  };
}

function graphWithCollectionMode(
  collectionMode: "direct" | "map",
): SavedGraph {
  const graph = graphWithEdge({ conversion_path: conversionPath });
  return {
    ...graph,
    edges: (graph.edges ?? []).map((edge) => ({
      ...edge,
      collection_mode: collectionMode,
    })),
  };
}

describe("saved conversion paths", () => {
  it("hydrates, serializes for a run, and resaves the exact ordered path", () => {
    const hydrated = hydrateSavedGraph(
      graphWithEdge({ conversion_path: conversionPath }),
      registry(),
    );
    const edge = hydrated.edges[0];

    expect(edge?.data?.conversionPath).toEqual(conversionPath);
    expect(serializeWorkflowEdgeTransport(edge?.data).conversion_path).toEqual(
      conversionPath,
    );
    expect(
      savedGraphDraft("Conversion path", hydrated.nodes, hydrated.edges).edges?.[0]
        ?.conversion_path,
    ).toEqual(conversionPath);
  });
});

describe("saved collection modes", () => {
  it("hydrates a direct edge when the target accepts the source shape", () => {
    const hydrated = hydrateSavedGraph(
      graphWithCollectionMode("direct"),
      registry("many", "one", ["one", "many"]),
    );

    expect(hydrated.edges[0]?.data?.collectionMode).toBe("direct");
    expect(decodeHandleId(hydrated.edges[0]?.sourceHandle)?.shape).toBe(
      "many",
    );
  });

  it("hydrates a map edge from a many source into a one target", () => {
    const hydrated = hydrateSavedGraph(
      graphWithCollectionMode("map"),
      registry("many", "one"),
    );

    expect(hydrated.edges[0]?.data?.collectionMode).toBe("map");
    expect(decodeHandleId(hydrated.edges[0]?.sourceHandle)?.shape).toBe(
      "many",
    );
    expect(decodeHandleId(hydrated.edges[0]?.targetHandle)?.shape).toBe("one");
  });

  it("rejects map edges targeting different inputs on the same node", () => {
    const sourceSpec = nodeSpec("source", "output", "x", "many");
    const otherSourceSpec = nodeSpec(
      "other-source",
      "output",
      "x",
      "many",
    );
    const targetSpec = nodeSpec("target", "input", "z");
    const targetInput = targetSpec.inputs[0]!;
    const testRegistry: NodeRegistry = {
      ...registry("many", "one"),
      nodes: [
        sourceSpec,
        otherSourceSpec,
        {
          ...targetSpec,
          inputs: [
            { ...targetInput, name: "left", title: "Left" },
            { ...targetInput, name: "right", title: "Right" },
          ],
        },
      ],
    };
    const graph = graphWithCollectionMode("map");
    const sourceNode = graph.nodes?.[0];
    const targetNode = graph.nodes?.[1];
    const edge = graph.edges?.[0];
    if (!sourceNode || !targetNode || !edge) {
      throw new Error("map-edge test fixture is incomplete");
    }
    const invalidGraph: SavedGraph = {
      ...graph,
      nodes: [
        sourceNode,
        {
          ...sourceNode,
          id: "other-source-node",
          operator_id: "other-source",
        },
        targetNode,
      ],
      edges: [
        { ...edge, id: "map-left", to_port: "left" },
        {
          ...edge,
          id: "map-right",
          from_node: "other-source-node",
          to_port: "right",
        },
      ],
    };

    expect(() => hydrateSavedGraph(invalidGraph, testRegistry)).toThrow(
      "node target-node has more than one map edge: map-left targets input left and map-right targets input right; exactly one edge may drive mapped execution",
    );
  });

  it.each([
    {
      collectionMode: "direct" as const,
      sourceShape: "many" as const,
      targetShape: "one" as const,
      targetAcceptedShapes: ["one"] as const,
      expectedMode: "map",
    },
    {
      collectionMode: "direct" as const,
      sourceShape: "one" as const,
      targetShape: "many" as const,
      targetAcceptedShapes: ["many"] as const,
      expectedMode: null,
    },
    {
      collectionMode: "map" as const,
      sourceShape: "one" as const,
      targetShape: "one" as const,
      targetAcceptedShapes: ["one"] as const,
      expectedMode: "direct",
    },
    {
      collectionMode: "map" as const,
      sourceShape: "many" as const,
      targetShape: "many" as const,
      targetAcceptedShapes: ["many"] as const,
      expectedMode: "direct",
    },
    {
      collectionMode: "map" as const,
      sourceShape: "many" as const,
      targetShape: "one" as const,
      targetAcceptedShapes: ["one", "many"] as const,
      expectedMode: "direct",
    },
  ])(
    "rejects $collectionMode for $sourceShape → $targetShape when new connections require $expectedMode",
    ({
      collectionMode,
      sourceShape,
      targetShape,
      targetAcceptedShapes,
      expectedMode,
    }) => {
      const expected = expectedMode
        ? `'${expectedMode}'`
        : "no supported collection mode";
      expect(() =>
        hydrateSavedGraph(
          graphWithCollectionMode(collectionMode),
          registry(sourceShape, targetShape, targetAcceptedShapes),
        ),
      ).toThrow(
        `uses collection mode '${collectionMode}' for source shape '${sourceShape}' and target shape '${targetShape}', expected ${expected}`,
      );
    },
  );
});

function collectRegistry(): NodeRegistry {
  const source = nodeSpec("source", "output", "text");
  const collector: NodeSpec = {
    operator_id: "text.collect",
    operator_version: 1,
    plugin_slug: "test",
    title: "Collect text",
    description: "Collect ordered text inputs.",
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [
      {
        name: "items",
        title: "Items",
        description: null,
        direction: "input",
        artifact_type: { id: "text", schema_version: 1 },
        shape: "many",
        accepted_shapes: ["one", "many"],
        instance_plugs: true,
        variadic: false,
        required: true,
      },
    ],
    outputs: [],
  };
  return {
    plugins: [],
    artifact_types: [
      {
        key: { id: "text", schema_version: 1 },
        title: "Text",
        payload_schema: {},
        field_projections: [],
      },
    ],
    artifact_conversions: [],
    nodes: [source, collector],
  };
}

function graphWithCollectPlugs(): SavedGraph {
  return {
    id: "00000000-0000-4000-8000-000000000002",
    revision: 1,
    name: "Collect inputs",
    created_at: "2026-07-15T12:00:00Z",
    updated_at: "2026-07-15T12:00:00Z",
    nodes: [
      {
        id: "source-node",
        operator_id: "source",
        operator_version: 1,
        config: {},
        input_plugs: [],
        position: { x: 0, y: 0 },
      },
      {
        id: "collect-node",
        operator_id: "text.collect",
        operator_version: 1,
        config: {},
        input_plugs: [
          { id: "plug-first", port: "items" },
          { id: "plug-second", port: "items" },
        ],
        position: { x: 300, y: 0 },
      },
    ],
    edges: [
      {
        id: "edge",
        from_node: "source-node",
        from_port: "output",
        to_node: "collect-node",
        to_port: "items",
        to_plug: "plug-second",
        collection_mode: "direct",
        projection: null,
        conversion_path: [],
        route_offset: { x: 12, y: -4 },
      },
    ],
  };
}

describe("saved instance plugs", () => {
  it("hydrates and resaves plug order and edge targeting", () => {
    const hydrated = hydrateSavedGraph(
      graphWithCollectPlugs(),
      collectRegistry(),
    );
    const collectNode = hydrated.nodes.find((node) => node.id === "collect-node");
    const edge = hydrated.edges[0];

    expect(collectNode?.data.inputPlugs).toEqual([
      { id: "plug-first", portName: "items" },
      { id: "plug-second", portName: "items" },
    ]);
    expect(decodeHandleId(edge?.targetHandle)?.plugId).toBe("plug-second");

    const draft = savedGraphDraft(
      "Collect inputs",
      hydrated.nodes,
      hydrated.edges,
    );
    expect(draft.nodes?.[1]?.input_plugs).toEqual([
      { id: "plug-first", port: "items" },
      { id: "plug-second", port: "items" },
    ]);
    expect(draft.edges?.[0]?.to_plug).toBe("plug-second");
    expect(draft.edges?.[0]?.route_offset).toEqual({ x: 12, y: -4 });
  });

  it("rejects an edge that targets a missing plug", () => {
    const graph = graphWithCollectPlugs();
    const edge = graph.edges?.[0];
    const invalidGraph: SavedGraph = {
      ...graph,
      edges: edge ? [{ ...edge, to_plug: "plug-missing" }] : [],
    };

    expect(() => hydrateSavedGraph(invalidGraph, collectRegistry())).toThrow(
      "references missing input plug plug-missing",
    );
  });
});

function genericCollectRegistry(): NodeRegistry {
  const collector: NodeSpec = {
    operator_id: "sequence.collect",
    operator_version: 1,
    plugin_slug: "test",
    title: "Collect",
    description: "Collect ordered artifacts.",
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [
      {
        name: "items",
        title: "Items",
        description: null,
        direction: "input",
        artifact_type: null,
        artifact_type_variable: "T",
        shape: "one",
        accepted_shapes: ["one", "many"],
        instance_plugs: true,
        variadic: false,
        required: true,
      },
    ],
    outputs: [
      {
        name: "items",
        title: "Items",
        description: null,
        direction: "output",
        artifact_type: null,
        artifact_type_variable: "T",
        shape: "many",
        accepted_shapes: ["many"],
        instance_plugs: false,
        variadic: false,
        required: true,
      },
    ],
  };
  return {
    plugins: [],
    artifact_types: [
      {
        key: { id: "scalar.integer", schema_version: 1 },
        title: "Integer",
        payload_schema: {},
        field_projections: [],
      },
    ],
    artifact_conversions: [],
    nodes: [nodeSpec("source", "output", "scalar.integer"), collector],
  };
}

function graphWithGenericCollectBinding(): SavedGraph {
  return {
    id: "00000000-0000-4000-8000-000000000003",
    revision: 1,
    name: "Generic collect",
    created_at: "2026-07-15T12:00:00Z",
    updated_at: "2026-07-15T12:00:00Z",
    nodes: [
      {
        id: "source-node",
        operator_id: "source",
        operator_version: 1,
        config: {},
        input_plugs: [],
        artifact_type_bindings: [],
        position: { x: 0, y: 0 },
      },
      {
        id: "collect-node",
        operator_id: "sequence.collect",
        operator_version: 1,
        config: {},
        input_plugs: [{ id: "plug-1", port: "items" }],
        artifact_type_bindings: [
          {
            variable: "T",
            artifact_type: { id: "scalar.integer", schema_version: 1 },
          },
        ],
        position: { x: 300, y: 0 },
      },
    ],
    edges: [
      {
        id: "edge",
        from_node: "source-node",
        from_port: "output",
        to_node: "collect-node",
        to_port: "items",
        to_plug: "plug-1",
        collection_mode: "direct",
        projection: null,
        conversion_path: [],
        route_offset: null,
      },
    ],
  };
}

describe("saved generic artifact type bindings", () => {
  it("hydrates bound handles and resaves the binding", () => {
    const hydrated = hydrateSavedGraph(
      graphWithGenericCollectBinding(),
      genericCollectRegistry(),
    );
    const collectNode = hydrated.nodes.find((node) => node.id === "collect-node");
    const edge = hydrated.edges[0];

    expect(collectNode?.data.artifactTypeBindings).toEqual({
      T: { id: "scalar.integer", schema_version: 1 },
    });
    expect(decodeHandleId(edge?.targetHandle)).toMatchObject({
      artifactTypeId: "scalar.integer",
      schemaVersion: 1,
      plugId: "plug-1",
    });
    expect(
      savedGraphDraft("Generic collect", hydrated.nodes, hydrated.edges)
        .nodes?.[1]?.artifact_type_bindings,
    ).toEqual([
      {
        variable: "T",
        artifact_type: { id: "scalar.integer", schema_version: 1 },
      },
    ]);
  });

  it("rejects a binding for a variable not declared by the node ports", () => {
    const graph = graphWithGenericCollectBinding();
    const invalid: SavedGraph = {
      ...graph,
      nodes: (graph.nodes ?? []).map((node) =>
        node.id === "collect-node"
          ? {
              ...node,
              artifact_type_bindings: [
                {
                  variable: "Missing",
                  artifact_type: {
                    id: "scalar.integer",
                    schema_version: 1,
                  },
                },
              ],
            }
          : node,
      ),
    };

    expect(() => hydrateSavedGraph(invalid, genericCollectRegistry())).toThrow(
      "binds undeclared artifact type variable Missing",
    );
  });

  it("rejects an unavailable artifact type on an isolated generic node", () => {
    const graph = graphWithGenericCollectBinding();
    const collectNode = graph.nodes?.find((node) => node.id === "collect-node");
    const invalid: SavedGraph = {
      ...graph,
      nodes: collectNode
        ? [
            {
              ...collectNode,
              artifact_type_bindings: [
                {
                  variable: "T",
                  artifact_type: {
                    id: "artifact.missing",
                    schema_version: 9,
                  },
                },
              ],
            },
          ]
        : [],
      edges: [],
    };

    expect(() => hydrateSavedGraph(invalid, genericCollectRegistry())).toThrow(
      "binds unavailable artifact type artifact.missing@9",
    );
  });
});

import { describe, expect, it } from "vitest";

import type {
  ArtifactConversionSpec,
  NodeRegistry,
  NodeSpec,
  SavedGraph,
} from "@/lib/api";
import { decodeHandleId } from "./handles";
import {
  hydrateSavedGraph,
  savedGraphDraft,
  savedGraphFingerprint,
} from "./saved-graph";
import {
  IMAGE_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  effectivePortShape,
  serializeWorkflowEdgeTransport,
} from "./types";

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
    catalog_visible: true,
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
      ({
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
      }) as unknown as NonNullable<SavedGraph["edges"]>[number],
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

describe("unavailable saved operators", () => {
  it("hydrates a placeholder and losslessly resaves its incident connection", () => {
    const base = graphWithEdge({ conversion_path: conversionPath });
    const graph: SavedGraph = {
      ...base,
      nodes: (base.nodes ?? []).map((node) =>
        node.id === "source-node"
          ? {
              ...node,
              operator_id: "gis.map.compose",
              config: { nested: { preserved: true } },
              input_plugs: [
                { id: "historical-plug", port: "historical-input" },
              ],
              artifact_type_bindings: [
                {
                  variable: "Z",
                  artifact_type: { id: "z", schema_version: 1 },
                },
                {
                  variable: "A",
                  artifact_type: { id: "x", schema_version: 1 },
                },
              ],
              layout: null,
            }
          : {
              ...node,
              artifact_type_bindings: [],
              layout: null,
            },
      ),
    };
    const liveRegistry = registry();
    const hydrated = hydrateSavedGraph(graph, {
      ...liveRegistry,
      nodes: liveRegistry.nodes.filter(
        (spec) => spec.operator_id !== "gis.map.compose",
      ),
    });
    const sourceNode = hydrated.nodes.find(
      (node) => node.id === "source-node",
    );
    const edge = hydrated.edges[0];

    expect(sourceNode?.data.compatibility).toMatchObject({
      status: "unsupported",
      outputs: [{ portName: "output" }],
    });
    expect(sourceNode?.data.compatibility).toHaveProperty(
      "issues.0",
      "Operator gis.map.compose@1 is unavailable. This saved node is preserved but cannot run.",
    );
    expect(sourceNode?.data.layout).toBeNull();
    expect(edge?.sourceHandle).toContain("$compatibility::output");
    expect(decodeHandleId(edge?.sourceHandle)).toBeNull();
    expect(decodeHandleId(edge?.targetHandle)).toMatchObject({
      portName: "input",
      direction: "input",
    });
    expect(edge?.data?.compatibilityIssues).toHaveLength(1);

    const draft = savedGraphDraft(
      graph.name,
      hydrated.nodes,
      hydrated.edges,
    );
    expect(draft.nodes?.[0]).toEqual(graph.nodes?.[0]);
    expect(draft.edges?.[0]).toMatchObject({
      from_node: "source-node",
      from_port: "output",
      to_node: "target-node",
      to_port: "input",
      to_plug: null,
      collection_mode: "direct",
      conversion_path: conversionPath,
    });
    expect(savedGraphFingerprint(draft)).toBe(
      savedGraphFingerprint({
        name: graph.name,
        nodes: graph.nodes,
        edges: graph.edges,
      }),
    );
  });
});

describe("saved node layout", () => {
  it("hydrates and persists node chrome sizes", () => {
    const base = graphWithEdge({ conversion_path: conversionPath });
    const withLayout: SavedGraph = {
      ...base,
      nodes: (base.nodes ?? []).map((node, index) =>
        index === 0
          ? {
              ...node,
              layout: {
                width: 420,
                body_height: 180,
                appendix_height: 320,
              },
            }
          : node,
      ),
    };
    const hydrated = hydrateSavedGraph(withLayout, registry());
    expect(hydrated.nodes[0]?.data.layout).toEqual({
      width: 420,
      bodyHeight: 180,
      appendixHeight: 320,
    });
    expect(hydrated.nodes[1]?.data.layout).toBeNull();
    expect(
      savedGraphDraft("Layout", hydrated.nodes, hydrated.edges).nodes?.[0]
        ?.layout,
    ).toEqual({
      width: 420,
      body_height: 180,
      appendix_height: 320,
    });
  });

  it("keeps an API partial-layout response fingerprint stable after hydration", () => {
    const base = graphWithEdge({ conversion_path: conversionPath });
    const hydratedBase = hydrateSavedGraph(base, registry());
    const responseDraft = savedGraphDraft(
      "Partial layout",
      hydratedBase.nodes,
      hydratedBase.edges,
    );
    const responseNodes = (responseDraft.nodes ?? []).map((node, index) =>
      index === 0
        ? {
            ...node,
            layout: {
              width: 420,
              body_height: null,
              appendix_height: null,
            },
          }
        : node,
    );
    const apiResponse: SavedGraph = {
      ...base,
      name: responseDraft.name,
      nodes: responseNodes,
      edges: responseDraft.edges ?? [],
    };

    const hydrated = hydrateSavedGraph(apiResponse, registry());
    const reserialized = savedGraphDraft(
      apiResponse.name,
      hydrated.nodes,
      hydrated.edges,
    );

    expect(savedGraphFingerprint(reserialized)).toBe(
      savedGraphFingerprint({ ...responseDraft, nodes: responseNodes }),
    );
  });
});

describe("ephemeral execution progress", () => {
  it("does not persist user-authored progress messages with the graph", () => {
    const hydrated = hydrateSavedGraph(
      graphWithEdge({ conversion_path: conversionPath }),
      registry(),
    );
    const firstNode = hydrated.nodes[0];
    if (!firstNode) throw new Error("Expected a hydrated workflow node");
    firstNode.data.progress = {
      omittedCount: 2,
      entries: [{
        sequence: 3,
        message: "payload contains tenant-private detail",
        current: 1,
        total: 4,
        sourceNodePath: [],
        invocationIndex: null,
        invocationPath: [],
      }],
    };

    const draft = savedGraphDraft(
      "Ephemeral state",
      hydrated.nodes,
      hydrated.edges,
    );
    const serialized = JSON.stringify(draft);

    expect(draft.nodes?.every((node) => !("progress" in node))).toBe(true);
    expect(serialized).not.toContain("tenant-private detail");
  });

  it("excludes selection, callbacks, viewport, drafts, presence, secrets, history, and runtime overlays", () => {
    const hydrated = hydrateSavedGraph(
      graphWithEdge({ conversion_path: conversionPath }),
      registry(),
    );
    const firstNode = hydrated.nodes[0];
    if (!firstNode) throw new Error("Expected a hydrated workflow node");
    firstNode.selected = true;
    firstNode.data.onConfigChange = () => undefined;
    firstNode.data.secretStatuses = {};
    firstNode.data.historyContext = { graphId: "private", isDirty: true };
    Object.assign(firstNode.data, {
      viewport: { x: 100, y: 200, zoom: 1.1 },
      privateFieldDraft: "draft-only value",
      presence: { userId: "other-user" },
    });
    firstNode.data.execution = {
      status: "failed",
      error: "runtime-only failure",
    };
    firstNode.data.progress = {
      omittedCount: 0,
      entries: [],
    };

    const draft = savedGraphDraft(
      "Ephemeral state",
      hydrated.nodes,
      hydrated.edges,
    );
    const serialized = JSON.stringify(draft);

    expect(draft.nodes?.[0]).not.toHaveProperty("selected");
    expect(serialized).not.toContain("onConfigChange");
    expect(serialized).not.toContain("secretStatuses");
    expect(serialized).not.toContain("historyContext");
    expect(serialized).not.toContain("viewport");
    expect(serialized).not.toContain("privateFieldDraft");
    expect(serialized).not.toContain("presence");
    expect(serialized).not.toContain("runtime-only failure");
    expect(serialized).not.toContain("progress");
  });
});

describe("saved edge enablement", () => {
  it("hydrates a legacy edge as enabled and keeps enablement out of run transport", () => {
    const hydrated = hydrateSavedGraph(
      graphWithEdge({ conversion_path: conversionPath }),
      registry(),
    );
    const edge = hydrated.edges[0];

    expect(edge?.data?.enabled).toBe(true);
    expect(serializeWorkflowEdgeTransport(edge?.data)).not.toHaveProperty(
      "enabled",
    );
    expect(
      savedGraphDraft("Legacy edge", hydrated.nodes, hydrated.edges).edges?.[0],
    ).toMatchObject({ enabled: true });
  });

  it("persists disabled state and includes it in the dirty fingerprint", () => {
    const legacyGraph = graphWithEdge({ conversion_path: conversionPath });
    const disabledGraph: SavedGraph = {
      ...legacyGraph,
      edges: (legacyGraph.edges ?? []).map((edge) => ({
        ...edge,
        enabled: false,
      })),
    };
    const enabled = hydrateSavedGraph(legacyGraph, registry());
    const disabled = hydrateSavedGraph(disabledGraph, registry());
    const enabledDraft = savedGraphDraft(
      "Edge state",
      enabled.nodes,
      enabled.edges,
    );
    const disabledDraft = savedGraphDraft(
      "Edge state",
      disabled.nodes,
      disabled.edges,
    );

    expect(disabled.edges[0]?.data?.enabled).toBe(false);
    expect(disabledDraft.edges?.[0]).toMatchObject({ enabled: false });
    expect(savedGraphFingerprint(disabledDraft)).not.toBe(
      savedGraphFingerprint(enabledDraft),
    );
  });

  it("normalizes a missing legacy flag to enabled in the fingerprint", () => {
    const hydrated = hydrateSavedGraph(
      graphWithEdge({ conversion_path: conversionPath }),
      registry(),
    );
    const enabledDraft = savedGraphDraft(
      "Legacy fingerprint",
      hydrated.nodes,
      hydrated.edges,
    );
    const enabledEdge = enabledDraft.edges?.[0];
    if (!enabledEdge) throw new Error("enabled-edge fixture is incomplete");
    const legacyEdge = { ...enabledEdge } as
      Omit<typeof enabledEdge, "enabled"> & { enabled?: boolean };
    delete legacyEdge.enabled;
    const legacyDraft = {
      ...enabledDraft,
      edges: [legacyEdge],
    } as unknown as typeof enabledDraft;

    expect(
      savedGraphFingerprint(legacyDraft),
    ).toBe(savedGraphFingerprint(enabledDraft));
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
    operator_id: "test.collect",
    operator_version: 1,
    plugin_slug: "test",
    title: "Collect text",
    description: "Collect ordered text inputs.",
    catalog_visible: true,
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
        operator_id: "test.collect",
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
        enabled: true,
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

describe("saved image upload config", () => {
  it("preserves the exact ordered uploads payload used by run requests", () => {
    const imageUploadSpec = nodeSpec(
      IMAGE_UPLOAD_OPERATOR_ID,
      "output",
      "image.raster",
      "many",
    );
    const data = createWorkflowNodeData(imageUploadSpec);
    data.config.uploads = [
      {
        upload_key: "second.png",
        filename: "second.png",
        byte_size: 22,
      },
      {
        upload_key: "first.png",
        filename: "first.png",
        byte_size: 11,
      },
    ];
    const draft = savedGraphDraft(
      "Upload order",
      [
        {
          id: "image-upload-node",
          type: WORKFLOW_NODE_TYPE,
          position: { x: 0, y: 0 },
          data,
        },
      ],
      [],
    );
    const savedGraph: SavedGraph = {
      id: "00000000-0000-4000-8000-000000000004",
      revision: 1,
      name: draft.name,
      created_at: "2026-07-15T12:00:00Z",
      updated_at: "2026-07-15T12:00:00Z",
      nodes: draft.nodes,
      edges: draft.edges,
    };
    const hydrated = hydrateSavedGraph(savedGraph, {
      plugins: [],
      artifact_types: [],
      artifact_conversions: [],
      nodes: [imageUploadSpec],
    });
    const savedConfig = draft.nodes?.[0]?.config;

    expect(savedConfig).toEqual({
      uploads: [
        {
          upload_key: "second.png",
          filename: "second.png",
          byte_size: 22,
        },
        {
          upload_key: "first.png",
          filename: "first.png",
          byte_size: 11,
        },
      ],
    });
    expect(hydrated.nodes[0]!.data.config).toEqual(savedConfig);
  });
});

describe("saved write-only node state", () => {
  it("persists ordinary config without secret status or callbacks", () => {
    const spec = nodeSpec(
      "llm.openai.completion",
      "output",
      "llm.completion",
    );
    const data = createWorkflowNodeData(spec);
    data.config = {
      base_url: "https://api.openai.com/v1",
      model: "gpt-5-mini",
    };
    data.secretStatuses = { api_key: { state: "configured" } };
    data.secretInputReadiness = { api_key: true };
    data.secretInputScope = "graph-1:2";
    data.onApplyNodeSecret = async () => true;

    const draft = savedGraphDraft(
      "Write-only node",
      [{
        id: "llm-node",
        type: WORKFLOW_NODE_TYPE,
        position: { x: 0, y: 0 },
        data,
      }],
      [],
    );

    expect(draft.nodes?.[0]?.config).toEqual(data.config);
    expect(draft.nodes?.[0]).not.toHaveProperty("secretStatuses");
    expect(draft.nodes?.[0]).not.toHaveProperty("secretInputReadiness");
    expect(draft.nodes?.[0]).not.toHaveProperty("secretInputScope");
    expect(draft.nodes?.[0]).not.toHaveProperty("onApplyNodeSecret");
    expect(JSON.stringify(draft)).not.toContain("api_key");
  });
});

function genericCollectRegistry(): NodeRegistry {
  const collector: NodeSpec = {
    operator_id: "sequence.collect",
    operator_version: 1,
    plugin_slug: "test",
    title: "Collect",
    description: "Collect ordered artifacts.",
    catalog_visible: true,
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
        enabled: true,
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

  it("marks a binding for an undeclared variable invalid without rejecting the graph", () => {
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

    const hydrated = hydrateSavedGraph(invalid, genericCollectRegistry());
    const collectNode = hydrated.nodes.find(
      (node) => node.id === "collect-node",
    );
    expect(collectNode?.data.compatibility).toMatchObject({
      status: "invalid",
      issues: [
        "node collect-node binds undeclared artifact type variable Missing",
      ],
    });
    expect(
      savedGraphDraft(invalid.name, hydrated.nodes, hydrated.edges)
        .nodes?.[1]?.artifact_type_bindings,
    ).toEqual([
      {
        variable: "Missing",
        artifact_type: { id: "scalar.integer", schema_version: 1 },
      },
    ]);
  });

  it("marks an unavailable artifact type invalid without rejecting the graph", () => {
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

    const hydrated = hydrateSavedGraph(invalid, genericCollectRegistry());
    expect(hydrated.nodes[0]?.data.compatibility).toMatchObject({
      status: "invalid",
      issues: [
        "node collect-node binds unavailable artifact type artifact.missing@9",
      ],
    });
    expect(
      savedGraphDraft(invalid.name, hydrated.nodes, hydrated.edges)
        .nodes?.[0]?.artifact_type_bindings,
    ).toEqual([
      {
        variable: "T",
        artifact_type: { id: "artifact.missing", schema_version: 9 },
      },
    ]);
  });
});

describe("saved graph module nodes", () => {
  const moduleGraphId = "00000000-0000-4000-8000-000000000008";
  const moduleOperatorId = `module.graph.${moduleGraphId}`;
  const imageType = { id: "image.raster", schema_version: 1 } as const;
  const completionType = { id: "llm.completion", schema_version: 1 } as const;
  const moduleInput = {
    name: "image",
    title: "Image",
    description: null,
    direction: "input" as const,
    artifact_type: imageType,
    shape: "one" as const,
    accepted_shapes: ["one" as const],
    instance_plugs: false,
    variadic: false,
    required: true,
  };
  const moduleOutput = {
    name: "completion",
    title: "Completion",
    description: null,
    direction: "output" as const,
    artifact_type: completionType,
    shape: "one" as const,
    accepted_shapes: ["one" as const],
    instance_plugs: false,
    variadic: false,
    required: true,
  };

  function moduleSpec(revision: number, catalogVisible: boolean): NodeSpec {
    return {
      operator_id: moduleOperatorId,
      operator_version: revision,
      plugin_slug: "saved-graph-modules",
      title: `Extract image r${revision}`,
      description: "Hidden structured extraction graph",
      config_schema: {},
      input_schema: {},
      output_schema: {},
      inputs: [moduleInput],
      outputs: [moduleOutput],
      module_graph_id: moduleGraphId,
      module_graph_revision: revision,
      catalog_visible: catalogVisible,
    };
  }

  it("hydrates the pinned historical revision and exposes mapped outputs as a sequence", () => {
    const sourceSpec = nodeSpec(
      IMAGE_UPLOAD_OPERATOR_ID,
      "output",
      "image.raster",
      "many",
    );
    const savedGraph: SavedGraph = {
      id: "00000000-0000-4000-8000-000000000009",
      revision: 1,
      name: "Map extraction module",
      created_at: "2026-07-16T12:00:00Z",
      updated_at: "2026-07-16T12:00:00Z",
      nodes: [
        {
          id: "images",
          operator_id: IMAGE_UPLOAD_OPERATOR_ID,
          operator_version: 1,
          config: {},
          input_plugs: [],
          position: { x: 0, y: 0 },
        },
        {
          id: "extract",
          operator_id: moduleOperatorId,
          operator_version: 1,
          config: {},
          input_plugs: [],
          position: { x: 300, y: 0 },
        },
      ],
      edges: [{
        id: "map-images",
        from_node: "images",
        from_port: "output",
        to_node: "extract",
        to_port: "image",
        to_plug: null,
        enabled: true,
        collection_mode: "map",
        projection: null,
        conversion_path: [],
        route_offset: null,
      }],
    };
    const moduleV1 = moduleSpec(1, false);
    const hydrated = hydrateSavedGraph(savedGraph, {
      plugins: [{
        slug: "saved-graph-modules",
        title: "Modules",
        origin: "module",
      }],
      artifact_types: [
        {
          key: imageType,
          title: "Image",
          payload_schema: {},
          field_projections: [],
        },
        {
          key: completionType,
          title: "Completion",
          payload_schema: {},
          field_projections: [],
        },
      ],
      artifact_conversions: [],
      nodes: [sourceSpec, moduleV1, moduleSpec(2, true)],
    });
    const moduleNode = hydrated.nodes.find((node) => node.id === "extract");

    expect(moduleNode?.data.spec).toBe(moduleV1);
    expect(moduleNode?.data.spec.module_graph_revision).toBe(1);
    expect(hydrated.edges[0]?.data?.collectionMode).toBe("map");
    expect(
      moduleNode
        ? effectivePortShape(
            { ...moduleNode.data, mappedInputPort: "image" },
            moduleNode.data.spec.outputs[0]!,
          )
        : null,
    ).toBe("many");
  });
});

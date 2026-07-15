import type { Node } from "@xyflow/react";

import {
  type CreateSavedGraphRequest,
  type NodeRegistry,
  type NodeSpec,
  type RunNodeResult,
  type SavedGraph,
  type SavedGraphEdge,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  connectionRouteForSelection,
  decodeHandleId,
  encodeHandleId,
} from "./handles";
import { ARTIFACT_TYPE_COLOR } from "./nodes.css";
import {
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  acceptedPortShapes,
  createWorkflowNodeData,
  declaredArtifactTypeVariables,
  portHasInstancePlugs,
  portMetaForPort,
  resolvedPortArtifactType,
  serializeArtifactTypeBindings,
  serializeInputPlugs,
  serializeWorkflowEdgeTransport,
  type WorkflowEdge,
  type WorkflowArtifactTypeBindingInput,
  type WorkflowArtifactTypeBindings,
  type WorkflowNodeData,
} from "./types";

export type SavedGraphWorkflowNode = Node<
  WorkflowNodeData,
  typeof WORKFLOW_NODE_TYPE
>;

export interface HydratedSavedGraph {
  nodes: SavedGraphWorkflowNode[];
  edges: WorkflowEdge[];
}

export function withMaterializedNodeRuns(
  nodes: readonly SavedGraphWorkflowNode[],
  nodeRuns: readonly RunNodeResult[],
): SavedGraphWorkflowNode[] {
  const runsByNodeId = new Map(
    nodeRuns
      .filter((run) => run.status === "succeeded")
      .map((run) => [run.node_id, run]),
  );

  return nodes.map((node) => {
    const run = runsByNodeId.get(node.id) ?? null;
    return {
      ...node,
      data: {
        ...node.data,
        run,
        execution: run ? { status: "succeeded" } : { status: "idle" },
      },
    };
  });
}

export class SavedGraphHydrationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SavedGraphHydrationError";
  }
}

function operatorKey(operatorId: string, operatorVersion: number): string {
  return `${operatorId}@${operatorVersion}`;
}

function sortedRecord(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => sortedRecord(item));
  }
  if (typeof value !== "object" || value === null) return value;

  const sorted: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value).sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    if (item !== undefined) sorted[key] = sortedRecord(item);
  }
  return sorted;
}

export function savedGraphDraft(
  name: string,
  nodes: readonly SavedGraphWorkflowNode[],
  edges: readonly WorkflowEdge[],
): CreateSavedGraphRequest {
  const savedNodes = nodes.map((node) => ({
    id: node.id,
    operator_id: node.data.spec.operator_id,
    operator_version: node.data.spec.operator_version,
    config: structuredClone(node.data.config),
    input_plugs: serializeInputPlugs(node.data),
    artifact_type_bindings: serializeArtifactTypeBindings(node.data),
    position: {
      x: node.position.x,
      y: node.position.y,
    },
  }));
  return {
    name: name.trim(),
    nodes: savedNodes,
    edges: edges.map((edge) => {
      const source = decodeHandleId(edge.sourceHandle);
      const target = decodeHandleId(edge.targetHandle);
      if (!source || !target) {
        throw new Error(`Cannot save edge ${edge.id}: its port handles are invalid`);
      }
      return {
        id: edge.id,
        from_node: edge.source,
        from_port: source.portName,
        to_node: edge.target,
        to_port: target.portName,
        to_plug: target.plugId ?? null,
        ...serializeWorkflowEdgeTransport(edge.data),
        route_offset: edge.data?.routeOffset
          ? {
              x: edge.data.routeOffset.x,
              y: edge.data.routeOffset.y,
            }
          : null,
      };
    }),
  };
}

function requireArtifactTypeBindings(
  savedGraph: SavedGraph,
  savedNode: NonNullable<SavedGraph["nodes"]>[number],
  spec: NodeSpec,
  registry: NodeRegistry,
): WorkflowArtifactTypeBindings {
  const declaredVariables = new Set(declaredArtifactTypeVariables(spec));
  const registryArtifactTypes = new Set(
    registry.artifact_types.map(
      (artifact) =>
        `${artifact.key.id}@${artifact.key.schema_version}`,
    ),
  );
  const bindings: Record<string, WorkflowArtifactTypeBindingInput["artifact_type"]> = {};
  for (const binding of savedNode.artifact_type_bindings ?? []) {
    if (!declaredVariables.has(binding.variable)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${savedNode.id} binds undeclared artifact type variable ${binding.variable}`,
      );
    }
    if (binding.variable in bindings) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${savedNode.id} binds artifact type variable ${binding.variable} more than once`,
      );
    }
    const artifactTypeKey =
      `${binding.artifact_type.id}@${binding.artifact_type.schema_version}`;
    if (!registryArtifactTypes.has(artifactTypeKey)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${savedNode.id} binds unavailable artifact type ${artifactTypeKey}`,
      );
    }
    bindings[binding.variable] = {
      id: binding.artifact_type.id,
      schema_version: binding.artifact_type.schema_version,
    };
  }
  return bindings;
}

export function savedGraphFingerprint(
  graph: CreateSavedGraphRequest,
): string {
  const normalized = {
    ...graph,
    nodes: [...(graph.nodes ?? [])].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
    edges: [...(graph.edges ?? [])].sort((left, right) =>
      left.id.localeCompare(right.id),
    ),
  };
  return JSON.stringify(sortedRecord(normalized));
}

function requireSpec(
  savedGraph: SavedGraph,
  savedNode: NonNullable<SavedGraph["nodes"]>[number],
  specs: ReadonlyMap<string, NodeSpec>,
): NodeSpec {
  const key = operatorKey(
    savedNode.operator_id,
    savedNode.operator_version,
  );
  const spec = specs.get(key);
  if (!spec) {
    throw new SavedGraphHydrationError(
      `Cannot open “${savedGraph.name}”: node ${savedNode.id} requires unavailable operator ${key}`,
    );
  }
  return spec;
}

function requireInputPlugs(
  savedGraph: SavedGraph,
  savedNode: NonNullable<SavedGraph["nodes"]>[number],
  spec: NodeSpec,
): NonNullable<typeof savedNode.input_plugs> {
  const inputPlugs = savedNode.input_plugs ?? [];
  const plugIds = new Set<string>();
  for (const plug of inputPlugs) {
    const port = spec.inputs.find((candidate) => candidate.name === plug.port);
    if (!port || !portHasInstancePlugs(port)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${savedNode.id} input plug ${plug.id} references non-instance input ${plug.port}`,
      );
    }
    if (plugIds.has(plug.id)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${savedNode.id} has duplicate input plug id ${plug.id}`,
      );
    }
    plugIds.add(plug.id);
  }
  return inputPlugs;
}

function requireEdgeEndpoint(
  savedGraph: SavedGraph,
  edge: SavedGraphEdge,
  nodeById: ReadonlyMap<string, SavedGraphWorkflowNode>,
): {
  sourceNode: SavedGraphWorkflowNode;
  targetNode: SavedGraphWorkflowNode;
} {
  const sourceNode = nodeById.get(edge.from_node);
  if (!sourceNode) {
    throw new SavedGraphHydrationError(
      `Cannot open “${savedGraph.name}”: edge ${edge.id} references missing source node ${edge.from_node}`,
    );
  }
  const targetNode = nodeById.get(edge.to_node);
  if (!targetNode) {
    throw new SavedGraphHydrationError(
      `Cannot open “${savedGraph.name}”: edge ${edge.id} references missing target node ${edge.to_node}`,
    );
  }
  return { sourceNode, targetNode };
}

function connectionRouteIsValid(
  edge: SavedGraphEdge,
  sourceNode: SavedGraphWorkflowNode,
  targetNode: SavedGraphWorkflowNode,
  registry: NodeRegistry,
): boolean {
  const sourcePort = sourceNode.data.spec.outputs.find(
    (port) => port.name === edge.from_port,
  );
  const targetPort = targetNode.data.spec.inputs.find(
    (port) => port.name === edge.to_port,
  );
  if (!sourcePort || !targetPort) return false;

  const connection = {
    sourceHandle: encodeHandleId(
      portMetaForPort(
        sourcePort,
        sourcePort.shape,
        undefined,
        sourceNode.data.artifactTypeBindings,
      ),
    ),
    targetHandle: encodeHandleId(
      portMetaForPort(
        targetPort,
        targetPort.shape,
        edge.to_plug ?? undefined,
        targetNode.data.artifactTypeBindings,
      ),
    ),
  };
  return connectionRouteForSelection(
    connection,
    registry.artifact_types,
    registry.artifact_conversions,
    {
      projection: edge.projection ?? undefined,
      conversionPath: edge.conversion_path ?? [],
    },
  ) !== null;
}

export function hydrateSavedGraph(
  savedGraph: SavedGraph,
  registry: NodeRegistry,
  nodeRuns: readonly RunNodeResult[] = [],
): HydratedSavedGraph {
  const specs = new Map(
    registry.nodes.map((spec) => [
      operatorKey(spec.operator_id, spec.operator_version),
      spec,
    ]),
  );
  const nodeIds = new Set<string>();
  const nodes = (savedGraph.nodes ?? []).map((savedNode) => {
    if (nodeIds.has(savedNode.id)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: duplicate node id ${savedNode.id}`,
      );
    }
    nodeIds.add(savedNode.id);
    const spec = requireSpec(savedGraph, savedNode, specs);
    const inputPlugs = requireInputPlugs(savedGraph, savedNode, spec);
    const data = createWorkflowNodeData(spec, inputPlugs);
    data.artifactTypeBindings = requireArtifactTypeBindings(
      savedGraph,
      savedNode,
      spec,
      registry,
    );
    data.config = structuredClone(savedNode.config ?? {});
    return {
      id: savedNode.id,
      type: WORKFLOW_NODE_TYPE,
      position: {
        x: savedNode.position.x,
        y: savedNode.position.y,
      },
      selected: false,
      data,
    } satisfies SavedGraphWorkflowNode;
  });
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const mapEdgeByTargetNode = new Map<string, SavedGraphEdge>();
  for (const edge of savedGraph.edges ?? []) {
    if (edge.collection_mode !== "map") continue;
    const existing = mapEdgeByTargetNode.get(edge.to_node);
    if (existing) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: node ${edge.to_node} has more than one map edge: ${existing.id} targets input ${existing.to_port} and ${edge.id} targets input ${edge.to_port}; exactly one edge may drive mapped execution`,
      );
    }
    mapEdgeByTargetNode.set(edge.to_node, edge);
  }
  const edgeIds = new Set<string>();
  const occupiedTargetPlugIds = new Set<string>();

  const edges = (savedGraph.edges ?? []).map((savedEdge) => {
    if (edgeIds.has(savedEdge.id)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: duplicate edge id ${savedEdge.id}`,
      );
    }
    edgeIds.add(savedEdge.id);
    const { sourceNode, targetNode } = requireEdgeEndpoint(
      savedGraph,
      savedEdge,
      nodeById,
    );
    const sourcePort = sourceNode.data.spec.outputs.find(
      (port) => port.name === savedEdge.from_port,
    );
    if (!sourcePort) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} references missing output ${sourceNode.data.spec.operator_id}.${savedEdge.from_port}`,
      );
    }
    const targetPort = targetNode.data.spec.inputs.find(
      (port) => port.name === savedEdge.to_port,
    );
    if (!targetPort) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} references missing input ${targetNode.data.spec.operator_id}.${savedEdge.to_port}`,
      );
    }
    if (portHasInstancePlugs(targetPort)) {
      const targetPlug = targetNode.data.inputPlugs.find(
        (plug) =>
          plug.id === savedEdge.to_plug && plug.portName === targetPort.name,
      );
      if (!targetPlug) {
        throw new SavedGraphHydrationError(
          `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} references missing input plug ${savedEdge.to_plug ?? "null"}`,
        );
      }
      const targetPlugKey = `${targetNode.id}::${targetPlug.id}`;
      if (occupiedTargetPlugIds.has(targetPlugKey)) {
        throw new SavedGraphHydrationError(
          `Cannot open “${savedGraph.name}”: multiple edges target input plug ${targetPlug.id} on node ${targetNode.id}`,
        );
      }
      occupiedTargetPlugIds.add(targetPlugKey);
    } else if (savedEdge.to_plug !== null && savedEdge.to_plug !== undefined) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} assigns input plug ${savedEdge.to_plug} to non-instance input ${targetPort.name}`,
      );
    }

    const sourceShape = mapEdgeByTargetNode.has(sourceNode.id)
      ? "many"
      : sourcePort.shape;
    const otherMapEdge = mapEdgeByTargetNode.get(targetNode.id);
    const targetShape =
      otherMapEdge !== savedEdge && otherMapEdge?.to_port === targetPort.name
        ? "many"
        : targetPort.shape;
    let expectedCollectionMode: SavedGraphEdge["collection_mode"] | null = null;
    if (acceptedPortShapes(targetPort).includes(sourceShape)) {
      expectedCollectionMode = "direct";
    } else if (!portHasInstancePlugs(targetPort)) {
      if (sourceShape === targetShape) {
        expectedCollectionMode = "direct";
      } else if (sourceShape === "many" && targetShape === "one") {
        expectedCollectionMode = "map";
      }
    }
    if (savedEdge.collection_mode !== expectedCollectionMode) {
      const expectedMode = expectedCollectionMode
        ? `'${expectedCollectionMode}'`
        : "no supported collection mode";
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} uses collection mode '${savedEdge.collection_mode}' for source shape '${sourceShape}' and target shape '${targetShape}', expected ${expectedMode}`,
      );
    }

    if (!connectionRouteIsValid(savedEdge, sourceNode, targetNode, registry)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} has an incompatible artifact route`,
      );
    }

    const sourceArtifactType = resolvedPortArtifactType(
      sourcePort,
      sourceNode.data.artifactTypeBindings,
    );
    const color = sourceArtifactType
      ? ARTIFACT_TYPE_COLOR[sourceArtifactType.id] ?? tokens.colorAccent
      : tokens.colorAccent;
    return {
      id: savedEdge.id,
      source: savedEdge.from_node,
      sourceHandle: encodeHandleId(
        portMetaForPort(
          sourcePort,
          sourceShape,
          undefined,
          sourceNode.data.artifactTypeBindings,
        ),
      ),
      target: savedEdge.to_node,
      targetHandle: encodeHandleId(
        portMetaForPort(
          targetPort,
          targetPort.shape,
          savedEdge.to_plug ?? undefined,
          targetNode.data.artifactTypeBindings,
        ),
      ),
      type: WORKFLOW_EDGE_TYPE,
      animated: false,
      data: {
        collectionMode: savedEdge.collection_mode,
        ...(savedEdge.projection
          ? { projection: { path: [...savedEdge.projection.path] } }
          : {}),
        conversionPath: [...(savedEdge.conversion_path ?? [])].map((conversion) => ({
          id: conversion.id,
          version: conversion.version,
        })),
        ...(savedEdge.route_offset
          ? {
              routeOffset: {
                x: savedEdge.route_offset.x,
                y: savedEdge.route_offset.y,
              },
            }
          : {}),
      },
      style: {
        stroke: color,
        strokeWidth: 2,
      },
    } satisfies WorkflowEdge;
  });

  return { nodes: withMaterializedNodeRuns(nodes, nodeRuns), edges };
}

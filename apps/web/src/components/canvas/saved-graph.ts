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
  connectionRouteMatchesSelection,
  connectionRoutesFor,
  decodeHandleId,
  encodeHandleId,
} from "./handles";
import { ARTIFACT_TYPE_COLOR } from "./nodes.css";
import {
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  portMetaForPort,
  type WorkflowEdge,
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
  return {
    name: name.trim(),
    nodes: nodes.map((node) => ({
      id: node.id,
      operator_id: node.data.spec.operator_id,
      operator_version: node.data.spec.operator_version,
      config: structuredClone(node.data.config),
      position: {
        x: node.position.x,
        y: node.position.y,
      },
    })),
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
        collection_mode: edge.data?.collectionMode ?? "direct",
        projection: edge.data?.projection
          ? { path: [...edge.data.projection.path] }
          : null,
        conversion: edge.data?.conversion
          ? {
              id: edge.data.conversion.id,
              version: edge.data.conversion.version,
            }
          : null,
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
    sourceHandle: encodeHandleId(portMetaForPort(sourcePort)),
    targetHandle: encodeHandleId(portMetaForPort(targetPort)),
  };
  return connectionRoutesFor(
    connection,
    registry.artifact_types,
    registry.artifact_conversions,
  ).some((route) =>
    connectionRouteMatchesSelection(route, {
      projection: edge.projection ?? undefined,
      conversion: edge.conversion ?? undefined,
    }),
  );
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
    const data = createWorkflowNodeData(spec);
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
  const mappedNodeIds = new Set(
    (savedGraph.edges ?? [])
      .filter((edge) => edge.collection_mode === "map")
      .map((edge) => edge.to_node),
  );
  const edgeIds = new Set<string>();

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
    if (!connectionRouteIsValid(savedEdge, sourceNode, targetNode, registry)) {
      throw new SavedGraphHydrationError(
        `Cannot open “${savedGraph.name}”: edge ${savedEdge.id} has an incompatible artifact route`,
      );
    }

    const sourceShape = mappedNodeIds.has(sourceNode.id)
      ? "many"
      : sourcePort.shape;
    const color =
      ARTIFACT_TYPE_COLOR[sourcePort.artifact_type.id] ?? tokens.colorAccent;
    return {
      id: savedEdge.id,
      source: savedEdge.from_node,
      sourceHandle: encodeHandleId(portMetaForPort(sourcePort, sourceShape)),
      target: savedEdge.to_node,
      targetHandle: encodeHandleId(portMetaForPort(targetPort)),
      type: WORKFLOW_EDGE_TYPE,
      animated: false,
      data: {
        collectionMode: savedEdge.collection_mode,
        ...(savedEdge.projection
          ? { projection: { path: [...savedEdge.projection.path] } }
          : {}),
        ...(savedEdge.conversion
          ? {
              conversion: {
                id: savedEdge.conversion.id,
                version: savedEdge.conversion.version,
              },
            }
          : {}),
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

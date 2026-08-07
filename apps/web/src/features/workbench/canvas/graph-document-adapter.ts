import type {
  Connection,
  EdgeChange,
  NodeChange,
} from "@xyflow/react";

import {
  applyGraphCommand,
  authoredGraphDocument,
  createSavedGraphRequest,
  executionInvalidatedNodeIds,
  type AuthoredGraphDocument,
  type GraphCommand,
} from "../model/graph-document";
import type { WorkflowNode } from "../model/execution-plan";
import type { WorkflowEdge, WorkflowNodeData } from "./types";
import { decodeHandleId } from "./handles";

export type NodeOverlays = Record<
  string,
  Pick<WorkflowNodeData, "run" | "execution" | "progress">
>;

export interface WorkbenchAuthoringState {
  document: AuthoredGraphDocument;
  nodeOverlays: NodeOverlays;
  error: string | null;
}

export type WorkbenchAuthoringAction =
  | {
      kind: "apply_commands";
      commands: readonly GraphCommand[];
    }
  | {
      kind: "replace_document";
      document: AuthoredGraphDocument;
      nodeOverlays: NodeOverlays;
    }
  | {
      kind: "update_overlays";
      update: NodeOverlays;
    }
  | {
      kind: "clear_error";
    };

export function nodeOverlaysFromNodes(
  nodes: readonly WorkflowNode[],
): NodeOverlays {
  return Object.fromEntries(
    nodes.map((node) => [node.id, {
      run: node.data.run,
      execution: node.data.execution,
      progress: node.data.progress,
    }]),
  );
}

export function reduceWorkbenchAuthoringState(
  state: WorkbenchAuthoringState,
  action: WorkbenchAuthoringAction,
): WorkbenchAuthoringState {
  if (action.kind === "replace_document") {
    const document = authoredGraphDocument(createSavedGraphRequest(action.document));
    return {
      document,
      nodeOverlays: action.nodeOverlays,
      error: null,
    };
  }
  if (action.kind === "update_overlays") {
    return { ...state, nodeOverlays: action.update };
  }
  if (action.kind === "clear_error") {
    return { ...state, error: null };
  }

  try {
    let nextDocument = state.document;
    const invalidatedNodeIds = new Set<string>();
    for (const command of action.commands) {
      for (const nodeId of executionInvalidatedNodeIds(nextDocument, command)) {
        invalidatedNodeIds.add(nodeId);
      }
      nextDocument = applyGraphCommand(nextDocument, command);
    }

    const nodeOverlays: NodeOverlays = {};
    for (const node of nextDocument.nodes) {
      const previous = state.nodeOverlays[node.id];
      if (!previous) continue;
      nodeOverlays[node.id] = invalidatedNodeIds.has(node.id)
        ? { run: null, execution: { status: "idle" }, progress: null }
        : previous;
    }
    return { document: nextDocument, nodeOverlays, error: null };
  } catch (error) {
    return {
      ...state,
      error: error instanceof Error ? error.message : "Graph edit failed.",
    };
  }
}

/** Translate renderer events into product-level authoring commands. */
export function graphCommandsFromNodeChanges(
  changes: readonly NodeChange<WorkflowNode>[],
): GraphCommand[] {
  const commands: GraphCommand[] = [];
  const positions = changes.flatMap((change) =>
    change.type === "position" && !change.dragging && change.position
      ? [{ node_id: change.id, x: change.position.x, y: change.position.y }]
      : [],
  );
  if (positions.length) commands.push({ kind: "move_nodes", positions });

  const node_ids = changes.flatMap((change) =>
    change.type === "remove" ? [change.id] : [],
  );
  if (node_ids.length) commands.push({ kind: "remove_nodes", node_ids });
  return commands;
}

export function graphCommandsFromEdgeChanges(
  changes: readonly EdgeChange<WorkflowEdge>[],
): GraphCommand[] {
  const edge_ids = changes.flatMap((change) =>
    change.type === "remove" ? [change.id] : [],
  );
  return edge_ids.length ? [{ kind: "remove_edges", edge_ids }] : [];
}

export function addEdgeCommand(
  connection: Connection,
  data: WorkflowEdge["data"],
  edgeId: string,
): GraphCommand {
  const source = decodeHandleId(connection.sourceHandle);
  const target = decodeHandleId(connection.targetHandle);
  if (!connection.source || !connection.target || !source?.portName || !target?.portName) {
    throw new Error("Cannot author an edge without typed endpoint handles");
  }
  return {
    kind: "add_edge",
    edge: {
      id: edgeId,
      from_node: connection.source,
      from_port: source.portName,
      to_node: connection.target,
      to_port: target.portName,
      to_plug: target.plugId ?? null,
      enabled: data?.enabled ?? true,
      collection_mode: data?.collectionMode ?? "direct",
      projection: data?.projection
        ? { path: [...data.projection.path] }
        : null,
      conversion_path: (data?.conversionPath ?? []).map((conversion) => ({
        id: conversion.id,
        version: conversion.version,
      })),
      route_offset: data?.routeOffset
        ? { x: data.routeOffset.x, y: data.routeOffset.y }
        : null,
    },
  };
}

import type { Node } from "@xyflow/react";

import { decodeHandleId } from "../canvas/handles";
import { inputPlugsForPort } from "../canvas/input-plugs";
import {
  IMAGE_UPLOAD_OPERATOR_ID,
  isFileUploadOperator,
  WORKFLOW_NODE_TYPE,
  imageUploads,
  portHasInstancePlugs,
  serializeRunNode,
  serializeWorkflowEdgeTransport,
  type WorkflowEdge,
  type WorkflowNodeData,
} from "../canvas/types";
import type {
  PinnedOutputInput,
  RunEdgeInput,
  RunRequest,
  RunScopeInput,
} from "@/lib/api";

export type WorkflowNode = Node<
  WorkflowNodeData,
  typeof WORKFLOW_NODE_TYPE
>;

export type RunScope = RunScopeInput;

export interface ExecutionSubgraph {
  nodeIds: ReadonlySet<string>;
  nodes: readonly WorkflowNode[];
  edges: readonly WorkflowEdge[];
}

export interface MissingRequiredInput {
  nodeId: string;
  nodeTitle: string;
  portName: string;
}

export interface ExecutionValidationIssue {
  nodeId: string | null;
  message: string;
}

export type ExecutionRequestPlanResult =
  | {
      status: "ready";
      request: RunRequest;
    }
  | {
      status: "invalid";
      message: string;
    };

export function selectedNodeAndAncestorIds(
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): Set<string> {
  const knownNodeIds = new Set(nodes.map((node) => node.id));
  const executionNodeIds = new Set(
    nodes.filter((node) => node.selected).map((node) => node.id),
  );
  const pendingNodeIds = [...executionNodeIds];

  while (pendingNodeIds.length) {
    const targetNodeId = pendingNodeIds.shift();
    if (targetNodeId === undefined) continue;

    for (const edge of edges) {
      if (
        edge.data?.enabled === false ||
        edge.target !== targetNodeId ||
        !knownNodeIds.has(edge.source) ||
        executionNodeIds.has(edge.source)
      ) {
        continue;
      }
      executionNodeIds.add(edge.source);
      pendingNodeIds.push(edge.source);
    }
  }

  return executionNodeIds;
}

export function executionSubgraphFor(
  scope: RunScope,
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): ExecutionSubgraph {
  const activeEdges = edges.filter((edge) => edge.data?.enabled !== false);
  let nodeIds: Set<string>;

  if (scope === "all") {
    nodeIds = new Set(nodes.map((node) => node.id));
  } else if (scope === "selected-with-dependencies") {
    nodeIds = selectedNodeAndAncestorIds(nodes, activeEdges);
  } else {
    nodeIds = new Set(
      nodes.filter((node) => node.selected).map((node) => node.id),
    );
  }

  const executionNodes = nodes.filter((node) => nodeIds.has(node.id));
  let executionEdges: WorkflowEdge[];

  if (scope === "all") {
    executionEdges = activeEdges;
  } else if (scope === "selected-with-dependencies") {
    executionEdges = activeEdges.filter(
      (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target),
    );
  } else {
    executionEdges = activeEdges.filter((edge) => nodeIds.has(edge.target));
  }

  return {
    nodeIds,
    nodes: executionNodes,
    edges: executionEdges,
  };
}

export function missingRequiredInputsFor(
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): MissingRequiredInput[] {
  return nodes.flatMap((node) =>
    node.data.spec.inputs.flatMap((port) => {
      if (portHasInstancePlugs(port)) {
        if (!port.required) return [];
        const plugs = inputPlugsForPort(node.data.inputPlugs, port.name);
        if (!plugs.length) {
          return [{
            nodeId: node.id,
            nodeTitle: node.data.spec.title,
            portName: port.name,
          }];
        }
        return plugs.flatMap((plug, index) =>
          edges.some(
            (edge) =>
              edge.data?.enabled !== false &&
              edge.target === node.id &&
              decodeHandleId(edge.targetHandle)?.plugId === plug.id,
          )
            ? []
            : [{
                nodeId: node.id,
                nodeTitle: node.data.spec.title,
                portName: `${port.name} input ${index + 1}`,
              }],
        );
      }
      if (!port.required) return [];
      return edges.some(
        (edge) =>
          edge.data?.enabled !== false &&
          edge.target === node.id &&
          decodeHandleId(edge.targetHandle)?.portName === port.name,
      )
        ? []
        : [{
            nodeId: node.id,
            nodeTitle: node.data.spec.title,
            portName: port.name,
          }];
    }),
  );
}

export function executionValidationIssue(
  scope: RunScope,
  executionNodes: readonly WorkflowNode[],
  executionEdges: readonly WorkflowEdge[],
): ExecutionValidationIssue | null {
  if (!executionNodes.length) {
    return {
      nodeId: null,
      message: scope !== "all"
        ? "Select at least one node before running a selection."
        : "Add at least one node before running the workflow.",
    };
  }

  const imageUploadWithoutImages = executionNodes.find(
    (node) =>
      isFileUploadOperator(node.data.spec.operator_id) &&
      !imageUploads(node.data).length,
  );
  if (imageUploadWithoutImages) {
    return {
      nodeId: imageUploadWithoutImages.id,
      message: imageUploadWithoutImages.data.spec.operator_id === IMAGE_UPLOAD_OPERATOR_ID
        ? `Choose images for ${imageUploadWithoutImages.data.spec.title} before running.`
        : `Choose a GeoJSON file for ${imageUploadWithoutImages.data.spec.title} before running.`,
    };
  }

  const missingInputs = missingRequiredInputsFor(
    executionNodes,
    executionEdges,
  );
  if (!missingInputs.length) return null;

  const first = missingInputs[0];
  return {
    nodeId: first.nodeId,
    message:
      `${first.nodeTitle}.${first.portName} is required but unconnected in this run.`,
  };
}

export function executionRequestPlan(
  scope: RunScope,
  planningNodes: readonly WorkflowNode[],
  execution: ExecutionSubgraph,
): ExecutionRequestPlanResult {
  const pinnedOutputs: PinnedOutputInput[] = [];

  if (scope === "selected") {
    const nodesById = new Map(planningNodes.map((node) => [node.id, node]));
    const pinnedSourcePorts = new Map<string, Set<string>>();
    const missingPinnedOutputs: string[] = [];

    for (const edge of execution.edges) {
      if (execution.nodeIds.has(edge.source)) continue;

      const source = decodeHandleId(edge.sourceHandle);
      const target = decodeHandleId(edge.targetHandle);
      if (!source || !target) {
        return {
          status: "invalid",
          message:
            `Cannot run the selection because edge ${edge.id} does not identify both source and target ports.`,
        };
      }

      const sourcePorts = pinnedSourcePorts.get(edge.source) ?? new Set<string>();
      if (sourcePorts.has(source.portName)) continue;
      sourcePorts.add(source.portName);
      pinnedSourcePorts.set(edge.source, sourcePorts);

      const sourceNode = nodesById.get(edge.source);
      const output = sourceNode?.data.run?.status === "succeeded"
        ? sourceNode.data.run.outputs.find(
            (candidate) => candidate.port === source.portName,
          )
        : undefined;
      if (!output) {
        const sourceName = sourceNode?.data.spec.title ?? edge.source;
        missingPinnedOutputs.push(`${sourceName}.${source.portName}`);
        continue;
      }

      pinnedOutputs.push({
        from_node: edge.source,
        from_port: source.portName,
        value: output.value,
      });
    }

    if (missingPinnedOutputs.length) {
      const endpoints = missingPinnedOutputs.join(", ");
      return {
        status: "invalid",
        message:
          `Cannot run the selection because no accessible materialized output is available for ${endpoints}. Select the missing upstream nodes too, or choose “Run with dependencies”.`,
      };
    }
  }

  const runEdges = execution.edges.flatMap<RunEdgeInput>((edge) => {
    const source = decodeHandleId(edge.sourceHandle);
    const target = decodeHandleId(edge.targetHandle);
    if (!source || !target) return [];
    return [{
      from_node: edge.source,
      from_port: source.portName,
      to_node: edge.target,
      to_port: target.portName,
      to_plug: target.plugId ?? null,
      ...serializeWorkflowEdgeTransport(edge.data),
    }];
  });

  const activeInputPlugIdsByNode = new Map<string, Set<string>>();
  for (const edge of execution.edges) {
    const target = decodeHandleId(edge.targetHandle);
    if (!target?.plugId) continue;
    const plugIds = activeInputPlugIdsByNode.get(edge.target) ?? new Set();
    plugIds.add(target.plugId);
    activeInputPlugIdsByNode.set(edge.target, plugIds);
  }

  const runNodes = execution.nodes.map((node) => {
    const activeInputPlugIds =
      activeInputPlugIdsByNode.get(node.id) ?? new Set<string>();
    return serializeRunNode(node.id, node.data, activeInputPlugIds);
  });

  const request: RunRequest = {
    nodes: runNodes,
    edges: runEdges,
    scope,
    ...(scope === "selected" ? { pinned_outputs: pinnedOutputs } : {}),
  };
  return { status: "ready", request };
}

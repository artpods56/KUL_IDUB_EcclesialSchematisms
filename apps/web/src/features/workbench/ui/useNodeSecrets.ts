"use client";

import * as React from "react";

import {
  applyNodeSecret,
  getGraphNodeSecrets,
  removeNodeSecret,
  type NodeSecretStatus,
  type SavedGraphNode,
} from "@/lib/api";
import {
  nodeSecretBindingReady,
  nodeSecretInputs,
  reconciledNodeSecretStatuses,
  type WorkflowNodeSecretStatus,
  type WorkflowNodeSecretStatuses,
} from "../canvas/node-secrets";
import type { WorkflowNode } from "../model/execution-plan";

export interface NodeSecretGraph {
  id: string;
  revision: number;
  nodes: readonly SavedGraphNode[];
}

export type NodeSecretStatusesByNode = Readonly<
  Record<string, WorkflowNodeSecretStatuses>
>;

export interface UseNodeSecretsResult {
  nodeSecretStatuses: NodeSecretStatusesByNode;
  refreshNodeSecretStatuses: (
    graph: NodeSecretGraph,
    graphNodes: readonly WorkflowNode[],
    signal?: AbortSignal,
  ) => Promise<boolean>;
  applyConfiguredNodeSecret: (
    nodeId: string,
    name: string,
    value: string,
  ) => Promise<boolean>;
  removeConfiguredNodeSecret: (
    nodeId: string,
    name: string,
  ) => Promise<boolean>;
  clearGraphSecretStatuses: () => void;
  forgetNodeSecretStatuses: (nodeId: string) => void;
}

function graphNodeSecretStatuses(
  nodes: readonly WorkflowNode[],
  remote: readonly NodeSecretStatus[],
): NodeSecretStatusesByNode {
  return Object.fromEntries(
    nodes
      .filter((node) => nodeSecretInputs(node.data.spec).length > 0)
      .map((node) => [
        node.id,
        reconciledNodeSecretStatuses(node.data.spec, node.id, remote),
      ]),
  );
}

function nodeSecretStatusesWithState(
  nodes: readonly WorkflowNode[],
  state: WorkflowNodeSecretStatus["state"],
  message?: string,
): NodeSecretStatusesByNode {
  return Object.fromEntries(
    nodes
      .filter((node) => nodeSecretInputs(node.data.spec).length > 0)
      .map((node) => [
        node.id,
        Object.fromEntries(
          nodeSecretInputs(node.data.spec).map((input) => [
            input.name,
            { state, message } satisfies WorkflowNodeSecretStatus,
          ]),
        ),
      ]),
  );
}

/** Keeps secret values write-only: only status metadata enters React state. */
export function useNodeSecrets(
  nodes: readonly WorkflowNode[],
): UseNodeSecretsResult {
  const [nodeSecretStatuses, setNodeSecretStatuses] =
    React.useState<NodeSecretStatusesByNode>({});
  const activeGraphRef = React.useRef<NodeSecretGraph | null>(null);
  const nodesByIdRef = React.useRef<ReadonlyMap<string, WorkflowNode>>(
    new Map(nodes.map((node) => [node.id, node])),
  );

  React.useEffect(() => {
    nodesByIdRef.current = new Map(
      nodes.map((node) => [node.id, node]),
    );
  }, [nodes]);

  const refreshNodeSecretStatuses = React.useCallback(async (
    graph: NodeSecretGraph,
    graphNodes: readonly WorkflowNode[],
    signal?: AbortSignal,
  ): Promise<boolean> => {
    activeGraphRef.current = graph;
    nodesByIdRef.current = new Map(
      graphNodes.map((node) => [node.id, node]),
    );

    if (!graphNodes.some(
      (node) => nodeSecretInputs(node.data.spec).length > 0,
    )) {
      setNodeSecretStatuses({});
      return true;
    }

    setNodeSecretStatuses(
      nodeSecretStatusesWithState(graphNodes, "loading"),
    );
    try {
      const response = await getGraphNodeSecrets(graph.id, signal);
      if (
        response.graph_id !== graph.id ||
        response.graph_revision !== graph.revision
      ) {
        throw new Error("Node secret status revision mismatch");
      }
      setNodeSecretStatuses(
        graphNodeSecretStatuses(graphNodes, response.secrets),
      );
      return true;
    } catch {
      if (signal?.aborted) return false;
      setNodeSecretStatuses(
        nodeSecretStatusesWithState(
          graphNodes,
          "error",
          "Secret status could not be loaded.",
        ),
      );
      return false;
    }
  }, []);

  const applyConfiguredNodeSecret = React.useCallback(async (
    nodeId: string,
    name: string,
    value: string,
  ): Promise<boolean> => {
    const graph = activeGraphRef.current;
    if (!graph) return false;
    const node = nodesByIdRef.current.get(nodeId);
    const input = node
      ? nodeSecretInputs(node.data.spec).find(
          (candidate) => candidate.name === name,
        )
      : undefined;
    const savedNode = graph.nodes.find((candidate) => candidate.id === nodeId);
    if (
      !node ||
      !input ||
      !nodeSecretBindingReady(
        input,
        {
          id: node.id,
          operator_id: node.data.spec.operator_id,
          operator_version: node.data.spec.operator_version,
          config: node.data.config,
        },
        savedNode,
      )
    ) {
      return false;
    }

    setNodeSecretStatuses((current) => ({
      ...current,
      [nodeId]: {
        ...(current[nodeId] ?? {}),
        [name]: { state: "applying" },
      },
    }));
    try {
      const response = await applyNodeSecret(graph.id, nodeId, name, {
        value,
        expected_graph_revision: graph.revision,
      });
      if (
        response.node_id !== nodeId ||
        response.name !== name ||
        response.configured !== true
      ) {
        throw new Error("Node secret response mismatch");
      }
      if (
        activeGraphRef.current?.id !== graph.id ||
        activeGraphRef.current.revision !== graph.revision
      ) {
        return true;
      }
      setNodeSecretStatuses((current) => ({
        ...current,
        [nodeId]: {
          ...(current[nodeId] ?? {}),
          [name]: { state: "configured" },
        },
      }));
      return true;
    } catch {
      if (
        activeGraphRef.current?.id === graph.id &&
        activeGraphRef.current.revision === graph.revision
      ) {
        setNodeSecretStatuses((current) => ({
          ...current,
          [nodeId]: {
            ...(current[nodeId] ?? {}),
            [name]: {
              state: "error",
              message: "The secret could not be applied.",
            },
          },
        }));
      }
      return false;
    }
  }, []);

  const removeConfiguredNodeSecret = React.useCallback(async (
    nodeId: string,
    name: string,
  ): Promise<boolean> => {
    const graph = activeGraphRef.current;
    if (!graph) return false;
    const node = nodesByIdRef.current.get(nodeId);
    const input = node
      ? nodeSecretInputs(node.data.spec).find(
          (candidate) => candidate.name === name,
        )
      : undefined;
    const savedNode = graph.nodes.find((candidate) => candidate.id === nodeId);
    if (
      !node ||
      !input ||
      !nodeSecretBindingReady(
        input,
        {
          id: node.id,
          operator_id: node.data.spec.operator_id,
          operator_version: node.data.spec.operator_version,
          config: node.data.config,
        },
        savedNode,
      )
    ) {
      return false;
    }

    setNodeSecretStatuses((current) => ({
      ...current,
      [nodeId]: {
        ...(current[nodeId] ?? {}),
        [name]: { state: "removing" },
      },
    }));
    try {
      await removeNodeSecret(graph.id, nodeId, name, graph.revision);
      if (
        activeGraphRef.current?.id !== graph.id ||
        activeGraphRef.current.revision !== graph.revision
      ) {
        return true;
      }
      setNodeSecretStatuses((current) => ({
        ...current,
        [nodeId]: {
          ...(current[nodeId] ?? {}),
          [name]: { state: "unconfigured" },
        },
      }));
      return true;
    } catch {
      if (
        activeGraphRef.current?.id === graph.id &&
        activeGraphRef.current.revision === graph.revision
      ) {
        setNodeSecretStatuses((current) => ({
          ...current,
          [nodeId]: {
            ...(current[nodeId] ?? {}),
            [name]: {
              state: "error",
              message: "The stored secret could not be removed.",
            },
          },
        }));
      }
      return false;
    }
  }, []);

  const clearGraphSecretStatuses = React.useCallback(() => {
    activeGraphRef.current = null;
    setNodeSecretStatuses({});
  }, []);

  const forgetNodeSecretStatuses = React.useCallback((nodeId: string) => {
    setNodeSecretStatuses((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== nodeId),
      ),
    );
  }, []);

  return {
    nodeSecretStatuses,
    refreshNodeSecretStatuses,
    applyConfiguredNodeSecret,
    removeConfiguredNodeSecret,
    clearGraphSecretStatuses,
    forgetNodeSecretStatuses,
  };
}

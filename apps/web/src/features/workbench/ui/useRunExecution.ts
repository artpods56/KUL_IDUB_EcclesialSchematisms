"use client";

import * as React from "react";

import {
  cancelRunExecution,
  getGraphMaterializations,
  getRunExecution,
  startRunExecution,
  type RunExecution,
} from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { withMaterializedNodeRuns } from "../canvas/saved-graph";
import {
  nodeSecretBindingReady,
  nodeSecretInputs,
  type WorkflowNodeSecretInput,
} from "../canvas/node-secrets";
import type { WorkflowEdge } from "../canvas/types";
import {
  executionRequestPlan,
  executionSubgraphFor,
  executionValidationIssue,
  type RunScope,
  type WorkflowNode,
} from "../model/execution-plan";
import type { NodeSecretStatusesByNode } from "./useNodeSecrets";
import type { ActiveSavedGraph } from "./useSavedGraphLifecycle";

interface VisibleRunExecution {
  generation: number;
  executionId: string | null;
  status: "preparing" | RunExecution["status"];
  activeNodeId: string | null;
  statusError: string | null;
}

interface RunExecutionGuard {
  generation: number;
  executionId: string | null;
  cancellationRequested: boolean;
  cancelInFlight: boolean;
  lastServerStatus: RunExecution["status"];
  activeNodeId: string | null;
}

interface UseRunExecutionOptions {
  registryAvailable: boolean;
  nodes: readonly WorkflowNode[];
  edges: readonly WorkflowEdge[];
  activeGraph: ActiveSavedGraph | null;
  currentFingerprint: string;
  isDirty: boolean;
  nodeSecretStatuses: NodeSecretStatusesByNode;
  setNodes: React.Dispatch<React.SetStateAction<WorkflowNode[]>>;
  setRunError: (message: string | null) => void;
  isGraphSnapshotCurrent: (
    graph: ActiveSavedGraph | null,
    fingerprint: string,
  ) => boolean;
  onMaterializationsLoaded: () => void;
}

export function useRunExecution({
  registryAvailable,
  nodes,
  edges,
  activeGraph,
  currentFingerprint,
  isDirty,
  nodeSecretStatuses,
  setNodes,
  setRunError,
  isGraphSnapshotCurrent,
  onMaterializationsLoaded,
}: UseRunExecutionOptions) {
  const [runningScope, setRunningScope] = React.useState<RunScope | null>(null);
  const running = runningScope !== null;
  const [visibleExecution, setVisibleExecution] =
    React.useState<VisibleRunExecution | null>(null);
  const [announcement, setAnnouncement] = React.useState("");
  const executionGenerationRef = React.useRef(0);
  const executionGuardRef = React.useRef<RunExecutionGuard | null>(null);
  const runRequestReservedRef = React.useRef(false);
  const mountedRef = React.useRef(true);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      executionGuardRef.current = null;
    };
  }, []);

  const performRunWorkflow = React.useCallback(async (scope: RunScope) => {
    if (!registryAvailable || running) return;
    const planningFingerprint = currentFingerprint;
    const planningActiveGraph = activeGraph;
    let planningNodes = nodes;

    if (scope === "selected" && activeGraph && !isDirty) {
      try {
        const materializations = await getGraphMaterializations(
          activeGraph.id,
          activeGraph.revision,
        );
        if (!mountedRef.current) return;
        if (!isGraphSnapshotCurrent(
          planningActiveGraph,
          planningFingerprint,
        )) {
          setRunError(
            "The active graph changed while latest materialized outputs were loading. Run the selection again.",
          );
          return;
        }
        planningNodes = withMaterializedNodeRuns(
          planningNodes,
          materializations.node_runs,
        );
        setNodes((current) =>
          withMaterializedNodeRuns(current, materializations.node_runs),
        );
        onMaterializationsLoaded();
      } catch (error) {
        if (!mountedRef.current) return;
        const message = error instanceof Error
          ? error.message
          : "Latest materialized outputs could not be loaded.";
        setRunError(
          `Cannot verify the latest upstream outputs for this saved graph: ${message}`,
        );
        return;
      }
    }

    const execution = executionSubgraphFor(scope, planningNodes, edges);
    const secretBackedNodes = execution.nodes
      .map((node) => ({ node, inputs: nodeSecretInputs(node.data.spec) }))
      .filter(({ inputs }) => inputs.length > 0);
    if (secretBackedNodes.length && !activeGraph) {
      setRunError(
        "Save the graph before running nodes that use stored secrets.",
      );
      return;
    }
    let changedSecretBinding: {
      node: WorkflowNode;
      input: WorkflowNodeSecretInput;
    } | undefined;
    if (activeGraph) {
      for (const { node, inputs } of secretBackedNodes) {
        const savedNode = activeGraph.nodes.find(
          (candidate) => candidate.id === node.id,
        );
        const changedInput = inputs.find((input) =>
          !nodeSecretBindingReady(input, {
            id: node.id,
            operator_id: node.data.spec.operator_id,
            operator_version: node.data.spec.operator_version,
            config: node.data.config,
          }, savedNode)
        );
        if (changedInput) {
          changedSecretBinding = { node, input: changedInput };
          break;
        }
      }
    }
    if (changedSecretBinding) {
      setRunError(
        `Save the graph before running ${changedSecretBinding.node.data.spec.title}: its ${changedSecretBinding.input.title} binding is new or changed.`,
      );
      return;
    }
    const unavailableSecret = secretBackedNodes.find(({ node, inputs }) =>
      inputs.some(
        (input) =>
          nodeSecretStatuses[node.id]?.[input.name]?.state !== "configured",
      ),
    );
    if (unavailableSecret) {
      setRunError(
        `Configure every required secret for ${unavailableSecret.node.data.spec.title} before running.`,
      );
      return;
    }
    const validationIssue = executionValidationIssue(
      scope,
      execution.nodes,
      execution.edges,
    );
    if (validationIssue) {
      if (validationIssue.nodeId) {
        setRunError(null);
        setNodes((current) => current.map((node) =>
          node.id === validationIssue.nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  run: null,
                  execution: {
                    status: "failed",
                    error: validationIssue.message,
                  },
                },
              }
            : node,
        ));
      } else {
        setRunError(validationIssue.message);
      }
      return;
    }

    const requestPlan = executionRequestPlan(scope, planningNodes, execution);
    if (requestPlan.status === "invalid") {
      setRunError(requestPlan.message);
      return;
    }

    executionGenerationRef.current += 1;
    const executionGeneration = executionGenerationRef.current;
    executionGuardRef.current = {
      generation: executionGeneration,
      executionId: null,
      cancellationRequested: false,
      cancelInFlight: false,
      lastServerStatus: "queued",
      activeNodeId: null,
    };
    setRunningScope(scope);
    setVisibleExecution({
      generation: executionGeneration,
      executionId: null,
      status: "preparing",
      activeNodeId: null,
      statusError: null,
    });
    setRunError(null);
    setNodes((current) => current.map((node) =>
      execution.nodeIds.has(node.id)
        ? {
            ...node,
            data: {
              ...node.data,
              run: null,
              execution: { status: "queued" },
            },
          }
        : node
    ));
    try {
      const materializesSavedGraph = Boolean(activeGraph && !isDirty);
      const graphContext = materializesSavedGraph && activeGraph
        ? {
            graph_id: activeGraph.id,
            graph_revision: activeGraph.revision,
          }
        : {};
      const secretGraphContext = secretBackedNodes.length && activeGraph
        ? {
            secret_graph_id: activeGraph.id,
            secret_graph_revision: activeGraph.revision,
          }
        : {};
      let response = await startRunExecution({
        ...requestPlan.request,
        ...graphContext,
        ...secretGraphContext,
      });
      let guard = executionGuardRef.current;
      if (
        !guard ||
        guard.generation !== executionGeneration
      ) {
        return;
      }
      guard.executionId = response.execution_id;
      let pollStatusError: string | null = null;

      while (true) {
        guard = executionGuardRef.current;
        if (
          !guard ||
          guard.generation !== executionGeneration ||
          guard.executionId !== response.execution_id
        ) {
          return;
        }

        const terminal =
          response.status === "cancelled" ||
          response.status === "succeeded" ||
          response.status === "failed";
        if (response.status === "cancelling") {
          guard.cancellationRequested = true;
        }
        const visibleStatus = guard.cancellationRequested && !terminal
          ? "cancelling"
          : response.status;
        const activeNodeId = response.active_node_id ??
          (terminal ? guard.activeNodeId : null);
        guard.lastServerStatus = response.status;
        guard.activeNodeId = activeNodeId;
        setVisibleExecution((current) =>
          current?.generation === executionGeneration
            ? {
                generation: executionGeneration,
                executionId: response.execution_id,
                status: visibleStatus,
                activeNodeId,
                statusError: pollStatusError,
              }
            : current
        );

        if (!terminal) {
          setNodes((current) => current.map((node) => {
            if (!execution.nodeIds.has(node.id)) return node;
            const active = node.id === activeNodeId;
            return {
              ...node,
              data: {
                ...node.data,
                run: null,
                execution: {
                  status: active
                    ? visibleStatus === "cancelling"
                      ? "cancelling"
                      : "running"
                    : "queued",
                },
              },
            };
          }));

          await new Promise<void>((resolve) => {
            window.setTimeout(resolve, 500);
          });
          guard = executionGuardRef.current;
          if (
            !guard ||
            guard.generation !== executionGeneration ||
            guard.executionId !== response.execution_id
          ) {
            return;
          }
          try {
            const polledResponse = await getRunExecution(response.execution_id);
            if (!mountedRef.current) return;
            if (polledResponse.execution_id !== guard.executionId) {
              pollStatusError =
                "Received status for another execution. Retrying…";
              setVisibleExecution((current) =>
                current?.generation === executionGeneration &&
                    current.executionId === guard?.executionId
                  ? {
                      ...current,
                      statusError: pollStatusError,
                    }
                  : current
              );
              continue;
            }
            pollStatusError = null;
            response = polledResponse;
          } catch (pollFailure) {
            guard = executionGuardRef.current;
            if (
              !guard ||
              guard.generation !== executionGeneration ||
              !guard.executionId
            ) {
              return;
            }
            if (
              pollFailure instanceof ApiError &&
              (pollFailure.status === 404 || pollFailure.status === 410)
            ) {
              const unavailableMessage =
                "Execution state is no longer available. The server may have restarted or expired this execution.";
              setNodes((current) => current.map((node) =>
                execution.nodeIds.has(node.id) &&
                    (node.data.execution.status === "queued" ||
                      node.data.execution.status === "running" ||
                      node.data.execution.status === "cancelling")
                  ? {
                      ...node,
                      data: {
                        ...node.data,
                        execution: { status: "idle" },
                      },
                    }
                  : node
              ));
              setRunError(unavailableMessage);
              setAnnouncement(
                "Execution failed because its status is no longer available.",
              );
              break;
            }
            const statusMessage = pollFailure instanceof Error
              ? pollFailure.message
              : "Execution status is unavailable.";
            pollStatusError = `${statusMessage} Retrying…`;
            setVisibleExecution((current) =>
              current?.generation === executionGeneration &&
                  current.executionId === guard?.executionId
                ? {
                    ...current,
                    statusError: pollStatusError,
                  }
                : current
            );
            continue;
          }
          continue;
        }

        if (!isGraphSnapshotCurrent(
          planningActiveGraph,
          planningFingerprint,
        )) {
          setNodes((current) => current.map((node) =>
            execution.nodeIds.has(node.id) &&
                (node.data.execution.status === "queued" ||
                  node.data.execution.status === "running" ||
                  node.data.execution.status === "cancelling")
              ? {
                  ...node,
                  data: {
                    ...node.data,
                    execution: { status: "idle" },
                  },
                }
              : node
          ));
          setRunError(
            response.status === "cancelled"
              ? "The graph changed while cancellation was in progress. Cancellation completed, but its node states were not applied to this canvas."
              : materializesSavedGraph
                ? "The graph changed while it was running. Results were recorded for the original saved revision and were not applied to this canvas."
                : "The graph changed while it was running. The completed run was not applied to this canvas.",
          );
          setAnnouncement(
            response.status === "cancelled"
              ? "Execution cancelled, but graph changes prevented its node states from being applied."
              : response.status === "failed" ||
                  response.result?.status === "failed"
                ? "Execution failed, but graph changes prevented its node states from being applied."
                : "Execution completed, but graph changes prevented its results from being applied.",
          );
          break;
        }

        if (response.status === "cancelled") {
          setNodes((current) => current.map((node) =>
            execution.nodeIds.has(node.id)
              ? {
                  ...node,
                  data: {
                    ...node.data,
                    run: null,
                    execution: { status: "cancelled" },
                  },
                }
              : node
          ));
          setAnnouncement("Execution cancelled.");
          break;
        }

        if (response.result) {
          const byNode = new Map(
            response.result.node_runs.map((run) => [run.node_id, run]),
          );
          setNodes((current) => current.map((node) => {
            if (!execution.nodeIds.has(node.id)) return node;
            const run = byNode.get(node.id);
            return {
              ...node,
              data: {
                ...node.data,
                run: run ?? null,
                execution: run
                  ? {
                      status: run.status,
                      error: run.error ?? (run.status === "failed"
                        ? "This node failed without error details."
                        : undefined),
                    }
                  : {
                      status: "skipped",
                      error:
                        "The server did not return a result for this node.",
                    },
              },
            };
          }));
          if (response.error) setRunError(response.error);
          setAnnouncement(
            response.result.status === "succeeded"
              ? "Execution completed successfully."
              : "Execution completed with errors.",
          );
          break;
        }

        const executionMessage = response.error ??
          "The execution ended without a workflow result.";
        const failedNodeId = guard.activeNodeId;
        setNodes((current) => current.map((node) => {
          if (!execution.nodeIds.has(node.id)) return node;
          const failed = failedNodeId === null || node.id === failedNodeId;
          return {
            ...node,
            data: {
              ...node.data,
              run: null,
              execution: failed
                ? { status: "failed", error: executionMessage }
                : { status: "idle" },
            },
          };
        }));
        setRunError(executionMessage);
        setAnnouncement("Execution failed.");
        break;
      }
    } catch (runFailure) {
      if (!mountedRef.current) return;
      setNodes((current) => current.map((node) =>
        execution.nodeIds.has(node.id) &&
          (node.data.execution.status === "queued" ||
            node.data.execution.status === "running" ||
            node.data.execution.status === "cancelling")
          ? {
              ...node,
              data: {
                ...node.data,
                execution: { status: "idle" },
              },
            }
          : node,
      ));
      if (!isGraphSnapshotCurrent(
        planningActiveGraph,
        planningFingerprint,
      )) {
        setRunError(
          "The graph changed while it was running. The completed run was not applied to this canvas.",
        );
        setAnnouncement(
          "Execution failed, and graph changes prevented any terminal state from being applied.",
        );
        return;
      }
      const missingPinnedArtifact =
        scope === "selected" &&
        runFailure instanceof ApiError &&
        runFailure.detail.includes("references missing artifact");
      const message = missingPinnedArtifact
        ? "A previously materialized upstream artifact is no longer accessible. Run the missing upstream nodes too, or choose “Run with dependencies”."
        : runFailure instanceof Error
          ? runFailure.message
          : "Workflow run failed";
      setRunError(message);
      setAnnouncement("Execution failed.");
    } finally {
      if (
        executionGuardRef.current?.generation === executionGeneration
      ) {
        executionGuardRef.current = null;
        setVisibleExecution((current) =>
          current?.generation === executionGeneration ? null : current
        );
        setRunningScope(null);
      }
    }
  }, [
    activeGraph,
    currentFingerprint,
    edges,
    isDirty,
    isGraphSnapshotCurrent,
    nodeSecretStatuses,
    nodes,
    onMaterializationsLoaded,
    registryAvailable,
    running,
    setNodes,
    setRunError,
  ]);

  const runWorkflow = React.useCallback(async (scope: RunScope) => {
    if (runRequestReservedRef.current) return;
    runRequestReservedRef.current = true;
    setAnnouncement("");
    try {
      await performRunWorkflow(scope);
    } finally {
      runRequestReservedRef.current = false;
    }
  }, [performRunWorkflow]);

  const cancelCurrentExecution = React.useCallback(async () => {
    const guard = executionGuardRef.current;
    if (
      !guard ||
      !guard.executionId ||
      guard.cancellationRequested ||
      guard.cancelInFlight
    ) {
      return;
    }

    const executionId = guard.executionId;
    const executionGeneration = guard.generation;
    guard.cancellationRequested = true;
    guard.cancelInFlight = true;
    setVisibleExecution((current) =>
      current?.generation === executionGeneration &&
          current.executionId === executionId
        ? { ...current, status: "cancelling", statusError: null }
        : current
    );
    setNodes((current) => current.map((node) =>
      node.id === guard.activeNodeId &&
          node.data.execution.status === "running"
        ? {
            ...node,
            data: {
              ...node.data,
              execution: { status: "cancelling" },
            },
          }
        : node
    ));

    try {
      const response = await cancelRunExecution(executionId);
      const currentGuard = executionGuardRef.current;
      if (
        !currentGuard ||
        currentGuard.generation !== executionGeneration ||
        currentGuard.executionId !== executionId
      ) {
        return;
      }
      if (response.execution_id !== executionId) {
        currentGuard.cancellationRequested = false;
        setVisibleExecution((current) =>
          current?.generation === executionGeneration &&
              current.executionId === executionId
            ? {
                ...current,
                status: currentGuard.lastServerStatus,
                statusError:
                  "Received cancellation status for another execution. You can try again.",
              }
            : current
        );
        setNodes((current) => current.map((node) =>
          node.id === currentGuard.activeNodeId &&
              node.data.execution.status === "cancelling"
            ? {
                ...node,
                data: {
                  ...node.data,
                  execution: {
                    status: currentGuard.lastServerStatus === "cancelling"
                      ? "cancelling"
                      : "running",
                  },
                },
              }
            : node
        ));
        return;
      }
      currentGuard.activeNodeId = response.active_node_id ??
        currentGuard.activeNodeId;
    } catch (cancelFailure) {
      const currentGuard = executionGuardRef.current;
      if (
        !currentGuard ||
        currentGuard.generation !== executionGeneration ||
        currentGuard.executionId !== executionId
      ) {
        return;
      }
      currentGuard.cancellationRequested =
        currentGuard.lastServerStatus === "cancelling";
      const message = cancelFailure instanceof Error
        ? cancelFailure.message
        : "The execution could not be cancelled.";
      setVisibleExecution((current) =>
        current?.generation === executionGeneration &&
            current.executionId === executionId
          ? {
              ...current,
              status: currentGuard.lastServerStatus,
              statusError: `${message} You can try again.`,
            }
          : current
      );
      setNodes((current) => current.map((node) =>
        node.id === currentGuard.activeNodeId &&
            node.data.execution.status === "cancelling"
          ? {
              ...node,
              data: {
                ...node.data,
                execution: {
                  status: currentGuard.lastServerStatus === "cancelling"
                    ? "cancelling"
                    : "running",
                },
              },
            }
          : node
      ));
    } finally {
      const currentGuard = executionGuardRef.current;
      if (
        currentGuard?.generation === executionGeneration &&
        currentGuard.executionId === executionId
      ) {
        currentGuard.cancelInFlight = false;
      }
    }
  }, [setNodes]);

  return {
    running,
    runningScope,
    visibleExecution,
    announcement,
    runWorkflow,
    cancelCurrentExecution,
  };
}

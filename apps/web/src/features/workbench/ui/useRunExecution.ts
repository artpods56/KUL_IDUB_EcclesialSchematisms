"use client";

import * as React from "react";

import {
  cancelRunExecution,
  getGraphMaterializations,
  getRunExecution,
  startRunExecution,
  subscribeRunExecutionEvents,
  type RunExecutionEventSubscription,
  type RunExecutionNodeProgressEvent,
  type RunExecution,
} from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { withMaterializedNodeRuns } from "../canvas/saved-graph";
import {
  nodeSecretBindingReady,
  nodeSecretInputs,
  type WorkflowNodeSecretInput,
} from "../canvas/node-secrets";
import type {
  NodeExecutionStatus,
  WorkflowEdge,
} from "../canvas/types";
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
  lastEventSequence: number;
  reconciliationRequested: boolean;
  terminalEventStatus: "cancelled" | "succeeded" | "failed" | null;
  planningActiveGraph: ActiveSavedGraph | null;
  planningFingerprint: string;
  finished: boolean;
}

interface PendingProgressBatch {
  generation: number;
  executionId: string;
  executionNodeIds: ReadonlySet<string>;
  progressByNode: Map<string, {
    events: RunExecutionNodeProgressEvent[];
    omittedCount: number;
  }>;
}

const MAX_PROGRESS_EVENTS_PER_NODE = 40;
const MAX_PROGRESS_MESSAGE_CHARACTERS = 500;

function nodeExecutionIsTerminal(status: NodeExecutionStatus): boolean {
  return status === "succeeded" ||
    status === "failed" ||
    status === "skipped" ||
    status === "cancelled";
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
  const executionEventStreamRef =
    React.useRef<RunExecutionEventSubscription | null>(null);
  const reconciliationWakeRef = React.useRef<(() => void) | null>(null);
  const pendingProgressBatchRef = React.useRef<PendingProgressBatch | null>(
    null,
  );
  const progressFrameRef = React.useRef<number | null>(null);
  const runRequestReservedRef = React.useRef(false);
  const mountedRef = React.useRef(true);

  const clearPendingProgress = React.useCallback(() => {
    if (progressFrameRef.current !== null) {
      window.cancelAnimationFrame(progressFrameRef.current);
      progressFrameRef.current = null;
    }
    pendingProgressBatchRef.current = null;
  }, []);

  const flushPendingProgress = React.useCallback(() => {
    if (progressFrameRef.current !== null) {
      window.cancelAnimationFrame(progressFrameRef.current);
      progressFrameRef.current = null;
    }
    const batch = pendingProgressBatchRef.current;
    pendingProgressBatchRef.current = null;
    if (!batch?.progressByNode.size) return;

    const guard = executionGuardRef.current;
    if (
      !guard ||
      guard.generation !== batch.generation ||
      guard.executionId !== batch.executionId ||
      !isGraphSnapshotCurrent(
        guard.planningActiveGraph,
        guard.planningFingerprint,
      )
    ) {
      return;
    }

    setNodes((current) => {
      const liveGuard = executionGuardRef.current;
      if (
        !liveGuard ||
        liveGuard.generation !== batch.generation ||
        liveGuard.executionId !== batch.executionId ||
        !isGraphSnapshotCurrent(
          liveGuard.planningActiveGraph,
          liveGuard.planningFingerprint,
        )
      ) {
        return current;
      }
      return current.map((node) => {
        const pending = batch.progressByNode.get(node.id);
        if (!pending || !batch.executionNodeIds.has(node.id)) return node;
        const previous = node.data.progress ?? {
          entries: [],
          omittedCount: 0,
        };
        const appended = [
          ...previous.entries,
          ...[...pending.events]
            .sort((left, right) => left.sequence - right.sequence)
            .map((event) => ({
              sequence: event.sequence,
              message: event.message.length > MAX_PROGRESS_MESSAGE_CHARACTERS
                ? `${event.message.slice(
                    0,
                    MAX_PROGRESS_MESSAGE_CHARACTERS - 1,
                  )}…`
                : event.message,
              current: event.current,
              total: event.total,
              sourceNodePath: event.node_path.slice(1),
              invocationIndex: event.invocation_index,
              invocationPath: [...event.invocation_path],
            })),
        ];
        const overflow = Math.max(
          0,
          appended.length - MAX_PROGRESS_EVENTS_PER_NODE,
        );
        return {
          ...node,
          data: {
            ...node.data,
            progress: {
              entries: overflow ? appended.slice(overflow) : appended,
              omittedCount:
                previous.omittedCount + pending.omittedCount + overflow,
            },
          },
        };
      });
    });
  }, [isGraphSnapshotCurrent, setNodes]);

  React.useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      executionEventStreamRef.current?.close();
      executionEventStreamRef.current = null;
      reconciliationWakeRef.current?.();
      reconciliationWakeRef.current = null;
      clearPendingProgress();
      executionGuardRef.current = null;
    };
  }, [clearPendingProgress]);

  const performRunWorkflow = React.useCallback(async (scope: RunScope) => {
    if (!registryAvailable || running) return;
    const planningFingerprint = currentFingerprint;
    const planningActiveGraph = activeGraph;
    let planningNodes = nodes;
    let execution = executionSubgraphFor(scope, planningNodes, edges);
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
        ).map((node) => ({
          ...node,
          data: { ...node.data, progress: null },
        }));
        setNodes((current) =>
          withMaterializedNodeRuns(current, materializations.node_runs).map(
            (node) => ({
              ...node,
              data: { ...node.data, progress: null },
            }),
          ),
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

    execution = executionSubgraphFor(scope, planningNodes, edges);
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
    const requestPlan = executionRequestPlan(scope, planningNodes, execution);
    if (requestPlan.status === "invalid") {
      setRunError(requestPlan.message);
      return;
    }

    executionEventStreamRef.current?.close();
    executionEventStreamRef.current = null;
    reconciliationWakeRef.current?.();
    reconciliationWakeRef.current = null;
    clearPendingProgress();
    executionGenerationRef.current += 1;
    const executionGeneration = executionGenerationRef.current;
    executionGuardRef.current = {
      generation: executionGeneration,
      executionId: null,
      cancellationRequested: false,
      cancelInFlight: false,
      lastServerStatus: "queued",
      activeNodeId: null,
      lastEventSequence: 0,
      reconciliationRequested: false,
      terminalEventStatus: null,
      planningActiveGraph,
      planningFingerprint,
      finished: false,
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
    setNodes((current) => current.map((node) => {
      if (execution.nodeIds.has(node.id)) {
        return {
          ...node,
          data: {
            ...node.data,
            run: null,
            execution: { status: "queued" },
            progress: null,
          },
        };
      }
      return node.data.progress
        ? { ...node, data: { ...node.data, progress: null } }
        : node;
    }));
    let eventSubscription: RunExecutionEventSubscription | null = null;
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
      let streamStatusError: string | null = null;

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
        const responseSupersededByTerminalEvent =
          guard.terminalEventStatus !== null && !terminal;
        const visibleStatus: RunExecution["status"] =
          responseSupersededByTerminalEvent
          ? guard.terminalEventStatus ?? response.status
          : guard.cancellationRequested && !terminal
            ? "cancelling"
            : response.status;
        const activeNodeId = responseSupersededByTerminalEvent
          ? guard.activeNodeId
          : response.active_node_id ?? (terminal ? guard.activeNodeId : null);
        if (!responseSupersededByTerminalEvent) {
          if (response.status === "cancelling") {
            guard.cancellationRequested = true;
          }
          guard.lastServerStatus = response.status;
          guard.activeNodeId = activeNodeId;
          setVisibleExecution((current) =>
            current?.generation === executionGeneration
              ? {
                  generation: executionGeneration,
                  executionId: response.execution_id,
                  status: visibleStatus,
                  activeNodeId,
                  statusError: pollStatusError ?? streamStatusError,
                }
              : current
          );
        }

        if (!eventSubscription) {
          eventSubscription = subscribeRunExecutionEvents(
            response.execution_id,
            {
              onOpen: () => {
                const currentGuard = executionGuardRef.current;
                if (
                  !currentGuard ||
                  currentGuard.generation !== executionGeneration ||
                  currentGuard.executionId !== response.execution_id ||
                  currentGuard.finished
                ) {
                  return;
                }
                const previousStreamError = streamStatusError;
                streamStatusError = null;
                setVisibleExecution((current) =>
                  current?.generation === executionGeneration &&
                      current.executionId === response.execution_id &&
                      current.statusError === previousStreamError
                    ? { ...current, statusError: pollStatusError }
                    : current
                );
              },
              onError: (error) => {
                const currentGuard = executionGuardRef.current;
                if (
                  !currentGuard ||
                  currentGuard.generation !== executionGeneration ||
                  currentGuard.executionId !== response.execution_id ||
                  currentGuard.finished
                ) {
                  return;
                }
                streamStatusError = error instanceof Error
                  ? "A live progress event could not be read. Status polling continues."
                  : "Live progress disconnected. Status polling continues.";
                setVisibleExecution((current) =>
                  current?.generation === executionGeneration &&
                      current.executionId === response.execution_id
                    ? {
                        ...current,
                        statusError: pollStatusError ?? streamStatusError,
                      }
                    : current
                );
              },
              onEvent: (event) => {
                const currentGuard = executionGuardRef.current;
                if (
                  !currentGuard ||
                  currentGuard.generation !== executionGeneration ||
                  currentGuard.executionId !== response.execution_id ||
                  currentGuard.finished ||
                  event.execution_id !== response.execution_id ||
                  event.sequence <= currentGuard.lastEventSequence ||
                  !isGraphSnapshotCurrent(
                    planningActiveGraph,
                    planningFingerprint,
                  )
                ) {
                  return;
                }
                currentGuard.lastEventSequence = event.sequence;

                if (event.kind === "node.progress") {
                  const outerNodeId = event.node_path[0];
                  if (!outerNodeId || !execution.nodeIds.has(outerNodeId)) {
                    return;
                  }
                  const pending = pendingProgressBatchRef.current;
                  let activeBatch: PendingProgressBatch;
                  if (
                    pending?.generation === executionGeneration &&
                    pending.executionId === event.execution_id
                  ) {
                    activeBatch = pending;
                  } else {
                    activeBatch = {
                      generation: executionGeneration,
                      executionId: event.execution_id,
                      executionNodeIds: execution.nodeIds,
                      progressByNode: new Map(),
                    };
                    pendingProgressBatchRef.current = activeBatch;
                  }
                  const nodeProgress = activeBatch.progressByNode.get(
                    outerNodeId,
                  ) ?? { events: [], omittedCount: 0 };
                  nodeProgress.events.push(event);
                  const overflow = Math.max(
                    0,
                    nodeProgress.events.length -
                      MAX_PROGRESS_EVENTS_PER_NODE,
                  );
                  if (overflow) {
                    nodeProgress.events.splice(0, overflow);
                    nodeProgress.omittedCount += overflow;
                  }
                  activeBatch.progressByNode.set(outerNodeId, nodeProgress);
                  if (progressFrameRef.current === null) {
                    progressFrameRef.current = window.requestAnimationFrame(
                      () => {
                        progressFrameRef.current = null;
                        flushPendingProgress();
                      },
                    );
                  }
                  return;
                }

                if (event.kind === "node.status") {
                  // Nested status is detail-only. A child completing must not
                  // make its outer module look terminal.
                  if (event.node_path.length !== 1) return;
                  const outerNodeId = event.node_path[0];
                  if (!outerNodeId || !execution.nodeIds.has(outerNodeId)) {
                    return;
                  }
                  const nodeStatus =
                    currentGuard.cancellationRequested &&
                      event.status === "running"
                      ? "cancelling"
                      : event.status;
                  if (nodeStatus === "running" || nodeStatus === "cancelling") {
                    currentGuard.activeNodeId = outerNodeId;
                    setVisibleExecution((current) =>
                      current?.generation === executionGeneration &&
                          current.executionId === event.execution_id
                        ? { ...current, activeNodeId: outerNodeId }
                        : current
                    );
                  }
                  setNodes((current) => {
                    const liveGuard = executionGuardRef.current;
                    if (
                      !liveGuard ||
                      liveGuard.generation !== executionGeneration ||
                      liveGuard.executionId !== event.execution_id ||
                      !isGraphSnapshotCurrent(
                        liveGuard.planningActiveGraph,
                        liveGuard.planningFingerprint,
                      )
                    ) {
                      return current;
                    }
                    return current.map((node) => {
                      if (node.id !== outerNodeId) return node;
                      if (
                        nodeExecutionIsTerminal(node.data.execution.status) &&
                        !nodeExecutionIsTerminal(nodeStatus)
                      ) {
                        return node;
                      }
                      return {
                        ...node,
                        data: {
                          ...node.data,
                          run: null,
                          execution: { status: nodeStatus },
                        },
                      };
                    });
                  });
                  return;
                }

                const eventTerminalStatus = event.status === "cancelled" ||
                    event.status === "succeeded" ||
                    event.status === "failed"
                  ? event.status
                  : null;
                const eventTerminal = eventTerminalStatus !== null;
                if (eventTerminalStatus) {
                  currentGuard.terminalEventStatus = eventTerminalStatus;
                }
                if (event.status === "cancelling") {
                  currentGuard.cancellationRequested = true;
                }
                const eventVisibleStatus =
                  currentGuard.cancellationRequested && !eventTerminal
                    ? "cancelling"
                    : event.status;
                const eventActiveNodeId = event.active_node_id ??
                  (eventTerminal ? currentGuard.activeNodeId : null);
                currentGuard.lastServerStatus = event.status;
                currentGuard.activeNodeId = eventActiveNodeId;
                setVisibleExecution((current) =>
                  current?.generation === executionGeneration &&
                      current.executionId === event.execution_id
                    ? {
                        ...current,
                        status: eventVisibleStatus,
                        activeNodeId: eventActiveNodeId,
                        statusError: pollStatusError ?? streamStatusError,
                      }
                    : current
                );
                if (!eventTerminal) {
                  setNodes((current) => {
                    const liveGuard = executionGuardRef.current;
                    if (
                      !liveGuard ||
                      liveGuard.generation !== executionGeneration ||
                      liveGuard.executionId !== event.execution_id ||
                      !isGraphSnapshotCurrent(
                        liveGuard.planningActiveGraph,
                        liveGuard.planningFingerprint,
                      )
                    ) {
                      return current;
                    }
                    return current.map((node) => {
                      if (
                        !execution.nodeIds.has(node.id) ||
                        nodeExecutionIsTerminal(node.data.execution.status)
                      ) {
                        return node;
                      }
                      const active = node.id === eventActiveNodeId;
                      return {
                        ...node,
                        data: {
                          ...node.data,
                          run: null,
                          execution: {
                            status: active
                              ? eventVisibleStatus === "cancelling"
                                ? "cancelling"
                                : "running"
                              : "queued",
                          },
                        },
                      };
                    });
                  });
                  return;
                }

                flushPendingProgress();
                currentGuard.reconciliationRequested = true;
                eventSubscription?.close();
                if (executionEventStreamRef.current === eventSubscription) {
                  executionEventStreamRef.current = null;
                }
                reconciliationWakeRef.current?.();
              },
            },
          );
          executionEventStreamRef.current = eventSubscription;
        }

        if (!terminal) {
          if (!responseSupersededByTerminalEvent) {
            setNodes((current) => {
              const liveGuard = executionGuardRef.current;
              if (
                !liveGuard ||
                liveGuard.generation !== executionGeneration ||
                liveGuard.executionId !== response.execution_id ||
                liveGuard.terminalEventStatus !== null ||
                !isGraphSnapshotCurrent(
                  liveGuard.planningActiveGraph,
                  liveGuard.planningFingerprint,
                )
              ) {
                return current;
              }
              return current.map((node) => {
                if (
                  !execution.nodeIds.has(node.id) ||
                  nodeExecutionIsTerminal(node.data.execution.status)
                ) {
                  return node;
                }
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
              });
            });
          }

          if (guard.reconciliationRequested) {
            guard.reconciliationRequested = false;
          } else {
            await new Promise<void>((resolve) => {
              const timeout = window.setTimeout(() => {
                if (reconciliationWakeRef.current === wake) {
                  reconciliationWakeRef.current = null;
                }
                resolve();
              }, 500);
              const wake = () => {
                window.clearTimeout(timeout);
                if (reconciliationWakeRef.current === wake) {
                  reconciliationWakeRef.current = null;
                }
                resolve();
              };
              reconciliationWakeRef.current = wake;
            });
          }
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
                      statusError: pollStatusError ?? streamStatusError,
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
              setNodes((current) => {
                const liveGuard = executionGuardRef.current;
                if (
                  !liveGuard ||
                  liveGuard.generation !== executionGeneration ||
                  liveGuard.executionId !== response.execution_id ||
                  !isGraphSnapshotCurrent(
                    liveGuard.planningActiveGraph,
                    liveGuard.planningFingerprint,
                  )
                ) {
                  return current;
                }
                return current.map((node) =>
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
                );
              });
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
                    statusError: pollStatusError ?? streamStatusError,
                  }
                : current
            );
            continue;
          }
          continue;
        }

        flushPendingProgress();
        eventSubscription?.close();
        if (executionEventStreamRef.current === eventSubscription) {
          executionEventStreamRef.current = null;
        }

        if (!isGraphSnapshotCurrent(
          planningActiveGraph,
          planningFingerprint,
        )) {
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
          setNodes((current) => {
            const liveGuard = executionGuardRef.current;
            if (
              !liveGuard ||
              liveGuard.generation !== executionGeneration ||
              liveGuard.executionId !== response.execution_id ||
              !isGraphSnapshotCurrent(
                liveGuard.planningActiveGraph,
                liveGuard.planningFingerprint,
              )
            ) {
              return current;
            }
            return current.map((node) => {
              if (
                !execution.nodeIds.has(node.id) ||
                nodeExecutionIsTerminal(node.data.execution.status)
              ) {
                return node;
              }
              return {
                ...node,
                data: {
                  ...node.data,
                  run: null,
                  execution: { status: "cancelled" },
                },
              };
            });
          });
          setAnnouncement("Execution cancelled.");
          break;
        }

        if (response.result) {
          const byNode = new Map(
            response.result.node_runs.map((run) => [run.node_id, run]),
          );
          setNodes((current) => {
            const liveGuard = executionGuardRef.current;
            if (
              !liveGuard ||
              liveGuard.generation !== executionGeneration ||
              liveGuard.executionId !== response.execution_id ||
              !isGraphSnapshotCurrent(
                liveGuard.planningActiveGraph,
                liveGuard.planningFingerprint,
              )
            ) {
              return current;
            }
            return current.map((node) => {
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
            });
          });
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
        setNodes((current) => {
          const liveGuard = executionGuardRef.current;
          if (
            !liveGuard ||
            liveGuard.generation !== executionGeneration ||
            liveGuard.executionId !== response.execution_id ||
            !isGraphSnapshotCurrent(
              liveGuard.planningActiveGraph,
              liveGuard.planningFingerprint,
            )
          ) {
            return current;
          }
          return current.map((node) => {
            if (
              !execution.nodeIds.has(node.id) ||
              nodeExecutionIsTerminal(node.data.execution.status)
            ) {
              return node;
            }
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
          });
        });
        setRunError(executionMessage);
        setAnnouncement("Execution failed.");
        break;
      }
    } catch (runFailure) {
      if (!mountedRef.current) return;
      setNodes((current) => {
        const liveGuard = executionGuardRef.current;
        if (
          !liveGuard ||
          liveGuard.generation !== executionGeneration ||
          !isGraphSnapshotCurrent(
            liveGuard.planningActiveGraph,
            liveGuard.planningFingerprint,
          )
        ) {
          return current;
        }
        return current.map((node) =>
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
        );
      });
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
      flushPendingProgress();
      clearPendingProgress();
      eventSubscription?.close();
      if (executionEventStreamRef.current === eventSubscription) {
        executionEventStreamRef.current = null;
      }
      if (
        executionGuardRef.current?.generation === executionGeneration
      ) {
        executionGuardRef.current.finished = true;
        setVisibleExecution((current) =>
          current?.generation === executionGeneration ? null : current
        );
        setRunningScope(null);
      }
    }
  }, [
    activeGraph,
    clearPendingProgress,
    currentFingerprint,
    edges,
    flushPendingProgress,
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
      guard.finished ||
      guard.terminalEventStatus !== null ||
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
    setNodes((current) => {
      const currentGuard = executionGuardRef.current;
      if (
        !currentGuard ||
        currentGuard.generation !== executionGeneration ||
        currentGuard.executionId !== executionId ||
        !isGraphSnapshotCurrent(
          currentGuard.planningActiveGraph,
          currentGuard.planningFingerprint,
        )
      ) {
        return current;
      }
      return current.map((node) =>
        node.id === currentGuard.activeNodeId &&
            node.data.execution.status === "running"
          ? {
              ...node,
              data: {
                ...node.data,
                execution: { status: "cancelling" },
              },
            }
          : node
      );
    });

    try {
      const response = await cancelRunExecution(executionId);
      const currentGuard = executionGuardRef.current;
      if (
        !currentGuard ||
        currentGuard.generation !== executionGeneration ||
        currentGuard.executionId !== executionId ||
        currentGuard.finished
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
        setNodes((current) => {
          const liveGuard = executionGuardRef.current;
          if (
            !liveGuard ||
            liveGuard.generation !== executionGeneration ||
            liveGuard.executionId !== executionId ||
            !isGraphSnapshotCurrent(
              liveGuard.planningActiveGraph,
              liveGuard.planningFingerprint,
            )
          ) {
            return current;
          }
          return current.map((node) =>
            node.id === liveGuard.activeNodeId &&
                node.data.execution.status === "cancelling"
              ? {
                  ...node,
                  data: {
                    ...node.data,
                    execution: {
                      status: liveGuard.lastServerStatus === "cancelling"
                        ? "cancelling"
                        : "running",
                    },
                  },
                }
              : node
          );
        });
        return;
      }
      currentGuard.activeNodeId = response.active_node_id ??
        currentGuard.activeNodeId;
    } catch (cancelFailure) {
      const currentGuard = executionGuardRef.current;
      if (
        !currentGuard ||
        currentGuard.generation !== executionGeneration ||
        currentGuard.executionId !== executionId ||
        currentGuard.finished
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
      setNodes((current) => {
        const liveGuard = executionGuardRef.current;
        if (
          !liveGuard ||
          liveGuard.generation !== executionGeneration ||
          liveGuard.executionId !== executionId ||
          !isGraphSnapshotCurrent(
            liveGuard.planningActiveGraph,
            liveGuard.planningFingerprint,
          )
        ) {
          return current;
        }
        return current.map((node) =>
          node.id === liveGuard.activeNodeId &&
              node.data.execution.status === "cancelling"
            ? {
                ...node,
                data: {
                  ...node.data,
                  execution: {
                    status: liveGuard.lastServerStatus === "cancelling"
                      ? "cancelling"
                      : "running",
                  },
                },
              }
            : node
        );
      });
    } finally {
      const currentGuard = executionGuardRef.current;
      if (
        currentGuard?.generation === executionGeneration &&
        currentGuard.executionId === executionId
      ) {
        currentGuard.cancelInFlight = false;
      }
    }
  }, [isGraphSnapshotCurrent, setNodes]);

  return {
    running,
    runningScope,
    visibleExecution,
    announcement,
    runWorkflow,
    cancelCurrentExecution,
  };
}

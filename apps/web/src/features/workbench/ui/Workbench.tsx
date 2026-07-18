"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Toast } from "@base-ui/react/toast";
import {
  NodeToolbar,
  Position,
  type Connection,
  type IsValidConnection,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Copy,
  History,
  LoaderCircle,
  Maximize2,
  Play,
  Plus,
  Square,
  Trash2,
  Workflow,
} from "lucide-react";

import { ExecutionHistoryDrawer } from "./ExecutionHistoryDrawer";
import {
  GlobalIssueToastList,
  type GlobalIssue,
} from "./GlobalIssueToastList";
import {
  ConnectionRouteDialog,
  type PendingConnectionRoute,
} from "./ConnectionRouteDialog";
import { workbenchStyles as s } from "./Workbench.styles";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { NodeSelector } from "./NodeSelector";
import { SavedGraphBrowser } from "./SavedGraphBrowser";
import { useNodeSecrets } from "./useNodeSecrets";
import {
  useSavedGraphLifecycle,
} from "./useSavedGraphLifecycle";
import { useRunExecution } from "./useRunExecution";
import {
  WorkflowCanvas,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "../canvas/WorkflowCanvas";
import {
  connectionRouteForSelection,
  connectionRouteMatchesSelection,
  connectionRouteSelection,
  connectionRoutesFor,
  decodedHandleArtifactType,
  decodeHandleId,
  encodeHandleId,
  type ConnectionRoute,
} from "../canvas/handles";
import {
  appendInputPlug,
  removeInputPlug as withoutInputPlug,
  reorderInputPlug as withReorderedInputPlug,
} from "../canvas/input-plugs";
import {
  nodeSecretBindingReady,
  nodeSecretInputs,
} from "../canvas/node-secrets";
import {
  ARTIFACT_TYPE_COLOR,
} from "../canvas/nodes.css";
import type { SchemaBuilderField } from "../canvas/schema-builder";
import { useTheme } from "@/components/theme";
import {
  isFileUploadOperator,
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  bindArtifactTypeVariable,
  createWorkflowNodeData,
  imageUploads,
  invalidateWorkflowNodeRuns,
  removeImageUpload,
  replaceImageUploads,
  resetArtifactTypeBinding,
  type WorkflowEdge,
  type WorkflowEdgeRouteOffset,
  type WorkflowEdgeUpdate,
  type WorkflowArtifactTypeBindings,
  type WorkflowNodeData,
  type WorkflowInputPlug,
} from "../canvas/types";
import {
  collectionModeForConnection,
  inputPlugBindingsForNode,
  isConnectionAccepted,
  mappedInputPortForNode,
  nodeAndDescendantIds,
  workflowEdgeRouteOption,
} from "../model/graph-authoring";
import {
  missingRequiredInputsFor,
  selectedNodeAndAncestorIds,
  type WorkflowNode,
} from "../model/execution-plan";
import { workbenchGraphPath } from "../routes";
import { useNodeRegistry } from "@/hooks/use-api";
import {
  fileToBase64,
  uploadImage,
  type ArtifactTypeKey,
  type NodeSpec,
  type RunEdgeCollectionMode,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

interface WorkbenchProps {
  workspaceSlug: string;
  initialGraphId: string | null;
}

const WORKBENCH_FIT_VIEW_OPTIONS = {
  padding: {
    top: "90px",
    right: "48px",
    bottom: "64px",
    left: "165px",
  },
  maxZoom: 0.88,
} as const;

interface PendingBoundEdge {
  nodeId: string;
  variable: string;
  artifactType: ArtifactTypeKey;
  edge: WorkflowEdge;
}


export function Workbench({
  workspaceSlug,
  initialGraphId,
}: WorkbenchProps) {
  const {
    data: registry,
    error: registryError,
    mutate: refreshNodeRegistry,
  } = useNodeRegistry();
  const { preference, cycleTheme } = useTheme();
  const [nodes, setNodes] = React.useState<WorkflowNode[]>([]);
  const [edges, setEdges] = React.useState<WorkflowEdge[]>([]);
  const {
    nodeSecretStatuses,
    refreshNodeSecretStatuses,
    applyConfiguredNodeSecret,
    removeConfiguredNodeSecret,
    clearGraphSecretStatuses,
    forgetNodeSecretStatuses,
  } = useNodeSecrets(nodes);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<WorkflowNode, WorkflowEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [executionHistoryTarget, setExecutionHistoryTarget] = React.useState<{
    nodeId: string | null;
  } | null>(null);
  const [runError, setRunError] = React.useState<string | null>(null);
  const clearRunError = React.useCallback(() => setRunError(null), []);
  const dismissRunError = React.useCallback((message: string) => {
    setRunError((current) => current === message ? null : current);
  }, []);
  const [pendingConnectionRoute, setPendingConnectionRoute] =
    React.useState<PendingConnectionRoute | null>(null);
  const [fitRevision, setFitRevision] = React.useState(0);
  const initializedRef = React.useRef(false);
  const executionRunningRef = React.useRef(false);
  const isExecutionRunning = React.useCallback(
    () => executionRunningRef.current,
    [],
  );
  const pendingBoundEdgesRef = React.useRef<PendingBoundEdge[]>([]);

  const handleNodeHandlesMeasured = React.useCallback((
    nodeId: string,
    artifactTypeBindings: WorkflowArtifactTypeBindings,
  ) => {
    const ready: PendingBoundEdge[] = [];
    const waiting: PendingBoundEdge[] = [];
    for (const pending of pendingBoundEdgesRef.current) {
      const measuredBinding = artifactTypeBindings[pending.variable];
      if (
        pending.nodeId === nodeId &&
        measuredBinding?.id === pending.artifactType.id &&
        measuredBinding.schema_version === pending.artifactType.schema_version
      ) {
        ready.push(pending);
      } else {
        waiting.push(pending);
      }
    }
    if (!ready.length) return;

    pendingBoundEdgesRef.current = waiting;
    setEdges((current) =>
      ready.reduce(
        (next, pending) => addEdge(pending.edge, next),
        current,
      ),
    );
  }, []);

  const updateConfig = React.useCallback(
    (nodeId: string, name: string, value: unknown) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              config:
                node.id === nodeId
                  ? { ...node.data.config, [name]: value }
                  : node.data.config,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const updateLayout = React.useCallback(
    (nodeId: string, layout: WorkflowNodeData["layout"]) => {
      setNodes((current) =>
        current.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  layout,
                },
              }
            : node,
        ),
      );
    },
    [],
  );

  const removeNode = React.useCallback((nodeId: string) => {
    const changedTargetNodeIds = edges
      .filter(
        (edge) =>
          edge.data?.enabled !== false && edge.source === nodeId,
      )
      .map((edge) => edge.target);
    setNodes((current) =>
      invalidateWorkflowNodeRuns(
        current.filter((node) => node.id !== nodeId),
        edges,
        changedTargetNodeIds,
      ),
    );
    setEdges((current) =>
      current.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
    );
    forgetNodeSecretStatuses(nodeId);
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [edges, forgetNodeSecretStatuses]);

  const handleRemoveImageUpload = React.useCallback(
    (nodeId: string, index: number) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...(node.id === nodeId
                ? removeImageUpload(node.data, index)
                : node.data),
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const addNodeInputPlug = React.useCallback(
    (nodeId: string, portName: string) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? appendInputPlug(node.data.inputPlugs, portName)
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const removeNodeInputPlug = React.useCallback(
    (nodeId: string, plugId: string) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? withoutInputPlug(node.data.inputPlugs, plugId)
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setEdges((current) =>
        current.filter(
          (edge) =>
            edge.target !== nodeId ||
            decodeHandleId(edge.targetHandle)?.plugId !== plugId,
        ),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const reorderNodeInputPlug = React.useCallback(
    (
      nodeId: string,
      portName: string,
      plugId: string,
      toIndex: number,
    ) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? withReorderedInputPlug(
                      node.data.inputPlugs,
                      portName,
                      plugId,
                      toIndex,
                    )
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const updateSchemaBuilderFields = React.useCallback(
    (
      nodeId: string,
      fields: readonly SchemaBuilderField[],
      inputPlugs: readonly WorkflowInputPlug[],
    ) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      const retainedPlugIds = new Set(inputPlugs.map((plug) => plug.id));
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              config:
                node.id === nodeId
                  ? { ...node.data.config, fields }
                  : node.data.config,
              inputPlugs:
                node.id === nodeId ? inputPlugs : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setEdges((current) =>
        current.filter((edge) => {
          if (edge.target !== nodeId) return true;
          const plugId = decodeHandleId(edge.targetHandle)?.plugId;
          return !plugId || retainedPlugIds.has(plugId);
        }),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const handleImagesSelected = React.useCallback(async (nodeId: string, files: File[]) => {
    const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
    setNodes((current) => current.map((node) => {
      if (!invalidatedNodeIds.has(node.id)) return node;
      return {
        ...node,
        data: {
          ...node.data,
          run: null,
          execution: node.id === nodeId
            ? { status: "uploading" }
            : { status: "idle" },
        },
      };
    }));
    setRunError(null);
    try {
      const uploads = await Promise.all(files.map(async (file) =>
        uploadImage(file.name, await fileToBase64(file)),
      ));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: invalidatedNodeIds.has(node.id)
          ? {
              ...(node.id === nodeId
                ? replaceImageUploads(node.data, uploads)
                : node.data),
              execution: { status: "idle" },
              run: null,
            }
          : node.data,
      })));
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "File upload failed";
      setNodes((current) => current.map((node) => node.id === nodeId ? {
        ...node,
        data: { ...node.data, execution: { status: "failed", error: message } },
      } : node));
    }
  }, [edges]);

  const resetNodeArtifactTypeBinding = React.useCallback(
    (nodeId: string, variable: string) => {
      const hasIncidentEdges = edges.some(
        (edge) => edge.source === nodeId || edge.target === nodeId,
      );
      if (hasIncidentEdges) return;

      setNodes((current) =>
        current.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: resetArtifactTypeBinding(node.data, variable, false),
              }
            : node,
        ),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const openGraphInNewTab = React.useCallback(
    (graphId: string) => {
      window.open(
        workbenchGraphPath(workspaceSlug, graphId),
        "_blank",
        "noopener,noreferrer",
      );
    },
    [workspaceSlug],
  );

  const openNodeExecutionHistory = React.useCallback((nodeId: string) => {
    setLibraryOpen(false);
    setExecutionHistoryTarget({ nodeId });
  }, []);

  const attachNodeCallbacks = React.useCallback(
    (data: WorkflowNodeData): WorkflowNodeData => ({
      ...data,
      onConfigChange: updateConfig,
      onLayoutChange: updateLayout,
      onRemoveNode: removeNode,
      onImagesSelected:
        isFileUploadOperator(data.spec.operator_id)
          ? handleImagesSelected
          : undefined,
      onRemoveImageUpload: handleRemoveImageUpload,
      onAddInputPlug: addNodeInputPlug,
      onRemoveInputPlug: removeNodeInputPlug,
      onReorderInputPlug: reorderNodeInputPlug,
      onSchemaBuilderFieldsChange: updateSchemaBuilderFields,
      onResetArtifactTypeBinding: resetNodeArtifactTypeBinding,
      onHandlesMeasured: handleNodeHandlesMeasured,
      onOpenModuleSource: data.spec.module_graph_id
        ? openGraphInNewTab
        : undefined,
      onOpenExecutionHistory: openNodeExecutionHistory,
    }),
    [
      addNodeInputPlug,
      handleImagesSelected,
      handleNodeHandlesMeasured,
      openGraphInNewTab,
      openNodeExecutionHistory,
      removeNode,
      removeNodeInputPlug,
      handleRemoveImageUpload,
      reorderNodeInputPlug,
      resetNodeArtifactTypeBinding,
      updateConfig,
      updateLayout,
      updateSchemaBuilderFields,
    ],
  );

  const replaceCanvas = React.useCallback((
    nextNodes: WorkflowNode[],
    nextEdges: WorkflowEdge[],
  ) => {
    pendingBoundEdgesRef.current = [];
    setNodes(nextNodes);
    setEdges(nextEdges);
  }, []);
  const clearPendingConnectionRoute = React.useCallback(() => {
    setPendingConnectionRoute(null);
  }, []);
  const closeNodeLibrary = React.useCallback(() => {
    setLibraryOpen(false);
  }, []);
  const requestCanvasRefit = React.useCallback(() => {
    setFitRevision((current) => current + 1);
  }, []);
  const requestNodeRegistryRefresh = React.useCallback(() => {
    void refreshNodeRegistry();
  }, [refreshNodeRegistry]);
  const uploading = nodes.some(
    (node) => node.data.execution.status === "uploading",
  );
  const {
    activeGraph,
    graphName,
    setGraphName,
    currentFingerprint,
    isDirty,
    saving,
    openingGraphId,
    deletingGraphId,
    persistenceError,
    clearPersistenceError,
    dismissPersistenceError,
    persistenceOperationBusy,
    graphBrowserOpen,
    toggleGraphBrowser,
    closeGraphBrowser,
    savedGraphs,
    savedGraphsLoading,
    savedGraphsRefreshing,
    savedGraphsError,
    refreshSavedGraphs,
    requestNewGraph,
    saveCurrentGraph,
    openSavedGraph,
    removeSavedGraph,
    isGraphSnapshotCurrent,
  } = useSavedGraphLifecycle({
    workspaceSlug,
    initialGraphId,
    registry,
    nodes,
    edges,
    isExecutionRunning,
    uploading,
    replaceCanvas,
    attachNodeCallbacks,
    refreshNodeSecretStatuses,
    clearGraphSecretStatuses,
    clearPendingConnectionRoute,
    clearRunError,
    closeNodeLibrary,
    requestCanvasRefit,
    refreshNodeRegistry: requestNodeRegistryRefresh,
  });
  const {
    running,
    runningScope,
    visibleExecution,
    announcement: executionAnnouncement,
    runWorkflow,
    cancelCurrentExecution,
  } = useRunExecution({
    registryAvailable: Boolean(registry),
    nodes,
    edges,
    activeGraph,
    currentFingerprint,
    isDirty,
    nodeSecretStatuses,
    setNodes,
    setRunError,
    isGraphSnapshotCurrent,
    onMaterializationsLoaded: clearPersistenceError,
  });
  React.useEffect(() => {
    executionRunningRef.current = running;
  }, [running]);
  const graphOperationBusy = persistenceOperationBusy || running;

  React.useEffect(() => {
    if (executionHistoryTarget) closeGraphBrowser();
  }, [closeGraphBrowser, executionHistoryTarget]);

  React.useEffect(() => {
    if (!registry) return;
    if (!initializedRef.current) {
      initializedRef.current = true;
      return;
    }
    const byOperator = new Map(
      registry.nodes.map((spec) => [
        `${spec.operator_id}@${spec.operator_version}`,
        spec,
      ]),
    );
    setNodes((current) => current.map((node) => {
      const spec = byOperator.get(
        `${node.data.spec.operator_id}@${node.data.spec.operator_version}`,
      );
      if (!spec) return { ...node, data: attachNodeCallbacks(node.data) };
      return {
        ...node,
        data: attachNodeCallbacks({ ...node.data, spec }),
      };
    }));
  }, [attachNodeCallbacks, registry]);

  React.useEffect(() => {
    if (!flow || !nodes.length) return;
    const frame = window.requestAnimationFrame(
      () => void flow.fitView(WORKBENCH_FIT_VIEW_OPTIONS),
    );
    return () => window.cancelAnimationFrame(frame);
  }, [fitRevision, flow, nodes.length]);

  const imageUploadWithoutImages = nodes.some(
    (node) =>
      isFileUploadOperator(node.data.spec.operator_id) &&
      !imageUploads(node.data).length,
  );
  const selectedNodeIds = React.useMemo(
    () => nodes.flatMap((node) => (node.selected ? [node.id] : [])),
    [nodes],
  );
  const selectedNodeCount = selectedNodeIds.length;
  const nodeTitles = React.useMemo(
    () => Object.fromEntries(
      nodes.map((node) => [node.id, node.data.spec.title]),
    ),
    [nodes],
  );
  const selectedWithDependenciesCount = selectedNodeAndAncestorIds(
    nodes,
    edges,
  ).size;
  const missingRequiredInputs = missingRequiredInputsFor(nodes, edges);
  const connectionInstruction = missingRequiredInputs.length
    ? `${missingRequiredInputs.length} required input${missingRequiredInputs.length === 1 ? "" : "s"} unconnected · drag between ports to connect them`
    : null;
  const runSelectedDisabled =
    !registry || running || selectedNodeCount === 0;
  const nodeErrorCount = nodes.filter(
    (node) => Boolean(node.data.execution.error),
  ).length;
  const globalIssues = React.useMemo<GlobalIssue[]>(() => {
    const issues: GlobalIssue[] = [];
    if (registryError) {
      issues.push({
        id: "registry",
        title: "Registry",
        message: registryError instanceof Error
          ? registryError.message
          : "The live node registry is unavailable.",
      });
    }
    if (persistenceError) {
      issues.push({
        id: "graph",
        title: "Graph",
        message: persistenceError,
      });
    }
    if (runError) {
      issues.push({
        id: "run",
        title: "Run",
        message: runError,
      });
    }
    return issues;
  }, [persistenceError, registryError, runError]);
  const dismissGlobalIssue = React.useCallback((issue: GlobalIssue) => {
    if (issue.id === "graph") {
      dismissPersistenceError(issue.message);
    }
    if (issue.id === "run") {
      dismissRunError(issue.message);
    }
  }, [dismissPersistenceError, dismissRunError]);
  const graphHasErrors = globalIssues.length > 0 || nodeErrorCount > 0;
  const graphNeedsAttention = imageUploadWithoutImages || missingRequiredInputs.length > 0;
  const canvasStatusMessage = runningScope === "selected"
    ? "running selected nodes · latest upstream outputs are pinned"
    : runningScope === "selected-with-dependencies"
      ? "running selected nodes and all upstream dependencies"
      : globalIssues.length
        ? `${globalIssues.length} workflow issue${globalIssues.length === 1 ? "" : "s"}`
        : nodeErrorCount
          ? `${nodeErrorCount} node issue${nodeErrorCount === 1 ? "" : "s"}`
          : !registry
            ? "loading live registry…"
            : imageUploadWithoutImages
              ? "choose images before running"
              : connectionInstruction ?? "all required inputs connected · ready to run";

  const onNodesChange: OnNodesChange<WorkflowNode> = React.useCallback(
    (changes) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );

  const invalidateWorkflowResults = React.useCallback(
    (
      changedTargetNodeIds: readonly string[],
      workflowEdges: readonly WorkflowEdge[],
    ) => {
      if (!changedTargetNodeIds.length) return;
      setNodes((current) =>
        invalidateWorkflowNodeRuns(
          current,
          workflowEdges,
          changedTargetNodeIds,
        ),
      );
      setRunError(null);
    },
    [],
  );

  const onEdgesChange: OnEdgesChange<WorkflowEdge> = React.useCallback(
    (changes) => {
      const changedTargetNodeIds = new Set<string>();
      for (const change of changes) {
        if (change.type === "remove" || change.type === "replace") {
          const previousEdge = edges.find((edge) => edge.id === change.id);
          if (previousEdge && previousEdge.data?.enabled !== false) {
            changedTargetNodeIds.add(previousEdge.target);
          }
        }
        if (change.type === "add" || change.type === "replace") {
          if (change.item.data?.enabled !== false) {
            changedTargetNodeIds.add(change.item.target);
          }
        }
      }
      setEdges((current) => applyEdgeChanges(changes, current));
      invalidateWorkflowResults([...changedTargetNodeIds], edges);
    },
    [edges, invalidateWorkflowResults],
  );

  const updateEdge = React.useCallback(
    (edgeId: string, update: WorkflowEdgeUpdate) => {
      const changedEdge = edges.find((edge) => edge.id === edgeId);
      if (!changedEdge) return;
      const updatedEdges = edges.map((edge) => {
        if (edge.id !== edgeId) return edge;
        const projection = update.route
          ? update.route.projection
          : edge.data?.projection;
        const conversionPath = update.route
          ? update.route.conversionPath.map((conversion) => ({
              id: conversion.id,
              version: conversion.version,
            }))
          : (edge.data?.conversionPath ?? []);
        return {
          ...edge,
          data: {
            ...edge.data,
            enabled: update.enabled ?? edge.data?.enabled ?? true,
            collectionMode:
              update.collectionMode ??
              edge.data?.collectionMode ??
              "direct",
            projection,
            conversionPath,
          },
        };
      });
      setEdges(updatedEdges);
      invalidateWorkflowResults([changedEdge.target], updatedEdges);
    },
    [edges, invalidateWorkflowResults],
  );

  const updateEdgeRoute = React.useCallback(
    (edgeId: string, routeOffset: WorkflowEdgeRouteOffset) => {
      setEdges((current) =>
        current.map((edge) =>
          edge.id === edgeId
            ? {
                ...edge,
                data: {
                  ...edge.data,
                  enabled: edge.data?.enabled ?? true,
                  collectionMode: edge.data?.collectionMode ?? "direct",
                  routeOffset,
                },
              }
            : edge,
        ),
      );
    },
    [],
  );

  const addWorkflowEdge = React.useCallback((
    connection: Connection,
    collectionMode: RunEdgeCollectionMode,
    route: ConnectionRoute,
  ) => {
    let committedConnection = connection;
    let newlyBoundNodeId: string | null = null;
    const binding = route.artifactTypeBinding;
    if (binding) {
      const handleId = binding.endpoint === "source"
        ? connection.sourceHandle
        : connection.targetHandle;
      const handle = decodeHandleId(handleId);
      const nodeId = binding.endpoint === "source"
        ? connection.source
        : connection.target;
      const node = nodes.find((candidate) => candidate.id === nodeId);
      const existingBinding = node?.data.artifactTypeBindings[binding.variable];
      if (
        !handle ||
        handle.artifactTypeVariable !== binding.variable ||
        !node ||
        (existingBinding &&
          (existingBinding.id !== binding.artifactType.id ||
            existingBinding.schema_version !==
              binding.artifactType.schema_version))
      ) {
        return;
      }

      const concreteHandleId = encodeHandleId({
        portName: handle.portName,
        artifactTypeId: binding.artifactType.id,
        schemaVersion: binding.artifactType.schema_version,
        shape: handle.shape,
        direction: handle.direction,
        ...(handle.plugId ? { plugId: handle.plugId } : {}),
      });
      committedConnection = binding.endpoint === "source"
        ? { ...connection, sourceHandle: concreteHandleId }
        : { ...connection, targetHandle: concreteHandleId };
      if (!existingBinding) newlyBoundNodeId = nodeId;
    }

    const source = decodeHandleId(committedConnection.sourceHandle);
    const sourceArtifactType = source
      ? decodedHandleArtifactType(source)
      : null;
    const color = sourceArtifactType
      ? ARTIFACT_TYPE_COLOR[sourceArtifactType.id] ?? tokens.colorAccent
      : tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 2,
    };
    const selection = connectionRouteSelection(route);
    const edge: WorkflowEdge = {
      ...committedConnection,
      id: `edge-${crypto.randomUUID()}`,
      type: WORKFLOW_EDGE_TYPE,
      animated: false,
      data: {
        enabled: true,
        collectionMode,
        projection: selection.projection
          ? { path: [...selection.projection.path] }
          : undefined,
        conversionPath: selection.conversionPath.map((conversion) => ({
          id: conversion.id,
          version: conversion.version,
        })),
      },
      style: edgeStyle,
    };
    if (binding && newlyBoundNodeId) {
      const bindingNodeId = newlyBoundNodeId;
      // Binding replaces the generic handle ID. Keep the concrete edge pending
      // until WorkflowNode confirms React Flow has measured the replacement.
      pendingBoundEdgesRef.current = [
        ...pendingBoundEdgesRef.current,
        {
          nodeId: bindingNodeId,
          variable: binding.variable,
          artifactType: binding.artifactType,
          edge,
        },
      ];
      setNodes((current) =>
        current.map((candidate) =>
          candidate.id === bindingNodeId
            ? {
                ...candidate,
                data: bindArtifactTypeVariable(
                  candidate.data,
                  binding.variable,
                  binding.artifactType,
                ),
              }
            : candidate,
        ),
      );
    } else {
      setEdges((current) => addEdge(edge, current));
    }
    invalidateWorkflowResults([edge.target], [...edges, edge]);
  }, [edges, invalidateWorkflowResults, nodes]);

  const isValidConnection = React.useCallback<
    IsValidConnection<WorkflowEdge>
  >((connection) => {
    const candidate: Connection = {
      source: connection.source,
      sourceHandle: connection.sourceHandle ?? null,
      target: connection.target,
      targetHandle: connection.targetHandle ?? null,
    };
    return isConnectionAccepted(
      candidate,
      nodes,
      edges,
      registry?.artifact_types ?? [],
      registry?.artifact_conversions ?? [],
      "id" in connection ? connection.id : null,
    );
  }, [
    edges,
    nodes,
    registry?.artifact_conversions,
    registry?.artifact_types,
  ]);

  const onConnect: OnConnect = React.useCallback((connection) => {
    if (!isValidConnection(connection)) return;
    const collectionMode = collectionModeForConnection(
      connection,
      nodes,
      edges,
    );
    if (!collectionMode) return;

    const candidates = connectionRoutesFor(
      connection,
      registry?.artifact_types ?? [],
      registry?.artifact_conversions ?? [],
    );
    const candidate = candidates[0];
    if (!candidate) return;
    if (candidates.length === 1) {
      addWorkflowEdge(connection, collectionMode, candidate);
      return;
    }

    const source = decodeHandleId(connection.sourceHandle);
    const sourceNode = nodes.find((node) => node.id === connection.source);

    const target = decodeHandleId(connection.targetHandle);
    const targetNode = nodes.find((node) => node.id === connection.target);
    if (!source || !target || !sourceNode || !targetNode) {
      return;
    }
    const sourceArtifactType = decodedHandleArtifactType(source);
    const targetArtifactType = decodedHandleArtifactType(target);

    setPendingConnectionRoute({
      connection,
      collectionMode,
      candidates,
      source: {
        nodeTitle: sourceNode.data.spec.title,
        portName: source.portName,
        artifactType: sourceArtifactType
          ? `${sourceArtifactType.id}@${sourceArtifactType.schema_version}`
          : `Any artifact · ${source.artifactTypeVariable}`,
      },
      target: {
        nodeTitle: targetNode.data.spec.title,
        portName: target.portName,
        artifactType: targetArtifactType
          ? `${targetArtifactType.id}@${targetArtifactType.schema_version}`
          : `Any artifact · ${target.artifactTypeVariable}`,
      },
    });
  }, [
    addWorkflowEdge,
    edges,
    isValidConnection,
    nodes,
    registry?.artifact_conversions,
    registry?.artifact_types,
  ]);

  const addCatalogNode = React.useCallback((spec: NodeSpec) => {
    const id = `node-${crypto.randomUUID()}`;
    const center = flow?.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }) ?? { x: 600, y: 280 };
    const data = attachNodeCallbacks(createWorkflowNodeData(spec));
    setNodes((current) => [
      ...current.map((node) => ({ ...node, selected: false })),
      { id, type: WORKFLOW_NODE_TYPE, position: { x: center.x - 140, y: center.y - 110 }, selected: true, data },
    ]);
    setLibraryOpen(false);
  }, [attachNodeCallbacks, flow]);

  const duplicateSelectedNodes = React.useCallback(() => {
    const selectedNodes = nodes.filter((node) => node.selected);
    if (!selectedNodes.length || running) return;

    const duplicates = selectedNodes.map((node) => ({
      node,
      id: `node-${crypto.randomUUID()}`,
    }));
    const duplicatedNodeIds = new Map(
      duplicates.map(({ node, id }) => [node.id, id]),
    );
    const duplicatedNodes: WorkflowNode[] = duplicates.map(({ node, id }) => ({
      ...node,
      id,
      position: { x: node.position.x + 36, y: node.position.y + 36 },
      selected: true,
      dragging: false,
      data: {
        ...node.data,
        inputPlugs: node.data.inputPlugs.map((plug) => ({ ...plug })),
        inputPlugBindings: {},
        artifactTypeBindings: structuredClone(
          node.data.artifactTypeBindings,
        ),
        mappedInputPort: null,
        config: structuredClone(node.data.config),
        run: null,
        execution: { status: "idle" },
      },
    }));
    const duplicatedEdges: WorkflowEdge[] = edges.flatMap((edge) => {
      const source = duplicatedNodeIds.get(edge.source);
      const target = duplicatedNodeIds.get(edge.target);
      if (!source || !target) return [];
      return [{
        ...edge,
        id: `edge-${crypto.randomUUID()}`,
        source,
        target,
        selected: false,
        data: edge.data ? structuredClone(edge.data) : undefined,
      }];
    });

    setNodes([
      ...nodes.map((node) => ({ ...node, selected: false })),
      ...duplicatedNodes,
    ]);
    setEdges([
      ...edges.map((edge) => ({ ...edge, selected: false })),
      ...duplicatedEdges,
    ]);
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [edges, nodes, running]);

  const deleteSelectedNodes = React.useCallback(() => {
    if (!flow || !selectedNodeIds.length || running) return;
    setPendingConnectionRoute(null);
    setRunError(null);
    void flow.deleteElements({
      nodes: selectedNodeIds.map((id) => ({ id })),
    });
  }, [flow, running, selectedNodeIds]);


  const canvasNodes = React.useMemo(
    () =>
      nodes.map((node) => {
        const savedNode = activeGraph?.nodes.find(
          (candidate) => candidate.id === node.id,
        );
        return {
          ...node,
          data: {
            ...node.data,
            secretStatuses: nodeSecretStatuses[node.id] ?? {},
            secretInputReadiness: Object.fromEntries(
              nodeSecretInputs(node.data.spec).map((input) => [
                input.name,
                nodeSecretBindingReady(input, {
                  id: node.id,
                  operator_id: node.data.spec.operator_id,
                  operator_version: node.data.spec.operator_version,
                  config: node.data.config,
                }, savedNode),
              ]),
            ),
            secretInputScope: `${activeGraph?.id ?? "unsaved"}:${activeGraph?.revision ?? "none"}`,
            onApplyNodeSecret: applyConfiguredNodeSecret,
            onRemoveNodeSecret: removeConfiguredNodeSecret,
            mappedInputPort: mappedInputPortForNode(node.id, edges),
            inputPlugBindings: inputPlugBindingsForNode(
              node,
              nodes,
              edges,
              registry?.artifact_conversions ?? [],
            ),
          },
        };
      }),
    [
      activeGraph,
      applyConfiguredNodeSecret,
      edges,
      nodeSecretStatuses,
      nodes,
      registry,
      removeConfiguredNodeSecret,
    ],
  );

  const canvasEdges = React.useMemo(
    () =>
      edges.map((edge) => {
        const connection: Connection = {
          source: edge.source,
          sourceHandle: edge.sourceHandle ?? null,
          target: edge.target,
          targetHandle: edge.targetHandle ?? null,
        };
        const source = decodeHandleId(edge.sourceHandle);
        const activeSelection = {
          projection: edge.data?.projection,
          conversionPath: edge.data?.conversionPath ?? [],
        };
        const routes = connectionRoutesFor(
          connection,
          registry?.artifact_types ?? [],
          registry?.artifact_conversions ?? [],
        );
        const activeRoute = connectionRouteForSelection(
          connection,
          registry?.artifact_types ?? [],
          registry?.artifact_conversions ?? [],
          activeSelection,
        );
        if (
          activeRoute &&
          !routes.some((route) =>
            connectionRouteMatchesSelection(route, activeSelection),
          )
        ) {
          routes.push(activeRoute);
        }
        const routeOptions = routes.map(workflowEdgeRouteOption);
        const conversionTitles = activeSelection.conversionPath.map(
          (requestedConversion) =>
            registry?.artifact_conversions.find(
              (conversion) =>
                conversion.key.id === requestedConversion.id &&
                conversion.key.version === requestedConversion.version,
            )?.title ?? `${requestedConversion.id}@${requestedConversion.version}`,
        );
        const otherEdges = edges.filter((candidate) => candidate.id !== edge.id);
        const validMode = collectionModeForConnection(
          connection,
          nodes,
          otherEdges,
        );
        return {
          ...edge,
          type: WORKFLOW_EDGE_TYPE,
          data: {
            ...edge.data,
            enabled: edge.data?.enabled ?? true,
            collectionMode: edge.data?.collectionMode ?? "direct",
            sourcePortName: source?.portName,
            conversionTitles,
            routeOptions,
            allowedCollectionModes: validMode ? [validMode] : [],
            onUpdate: updateEdge,
            onRouteOffsetChange: updateEdgeRoute,
          },
        };
      }),
    [
      edges,
      nodes,
      registry?.artifact_conversions,
      registry?.artifact_types,
      updateEdge,
      updateEdgeRoute,
    ],
  );


  // Firefox uses autocomplete to control restored dynamic button state, but
  // React's button typings omit that browser-specific attribute.
  const firefoxDynamicButtonProps: React.ButtonHTMLAttributes<HTMLButtonElement> & {
    autoComplete: "off";
  } = { autoComplete: "off" };
  const visibleExecutionNodeTitle = visibleExecution?.activeNodeId
    ? nodes.find((node) => node.id === visibleExecution.activeNodeId)?.data.spec.title
    : null;
  const executionCancelling = visibleExecution?.status === "cancelling";
  const visibleExecutionTitle = visibleExecutionNodeTitle ??
    (executionCancelling ? "Stopping execution…" : "Preparing…");
  const visibleExecutionStatus = visibleExecution?.statusError ??
    (executionCancelling
      ? "Waiting for the current node to stop"
      : visibleExecution?.status === "queued"
        ? "Waiting for a worker"
        : visibleExecution?.status === "running"
          ? "Processing node"
          : "Starting execution");

  return (
    <main {...stylex.props(s.shell)}>
      <span
        role="status"
        aria-live="polite"
        aria-atomic="true"
        {...stylex.props(s.visuallyHidden)}
      >
        {executionAnnouncement}
      </span>
      <section {...stylex.props(s.canvas)} aria-label="Workflow canvas">
        <WorkflowCanvas
          fitViewOptions={WORKBENCH_FIT_VIEW_OPTIONS}
          nodes={canvasNodes}
          edges={canvasEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onPaneReady={setFlow}
          onPaneClick={() => {
            setLibraryOpen(false);
            closeGraphBrowser();
          }}
          animateEdges={running}
        >
          {selectedNodeIds.length ? (
            <NodeToolbar
              nodeId={selectedNodeIds}
              isVisible
              position={Position.Bottom}
              offset={16}
              className={`ns-node-detail ${stylex.props(s.selectionToolbar).className}`}
            >
              <span {...stylex.props(s.selectionLabel)}>
                {selectedNodeCount} selected
              </span>
              <span {...stylex.props(s.selectionDivider)} />
              <button
                type="button"
                disabled={runSelectedDisabled}
                title="Run only the selected nodes; latest accessible upstream outputs are pinned"
                {...stylex.props(s.toolButton, s.primaryButton)}
                onClick={() => void runWorkflow("selected")}
              >
                {runningScope === "selected" ? (
                  <LoaderCircle size={13} {...stylex.props(s.spinner)} />
                ) : (
                  <Play size={13} />
                )}
                {runningScope === "selected" ? "Running…" : "Run"}
              </button>
              <button
                type="button"
                disabled={runSelectedDisabled}
                title={`Run the selection and every upstream dependency (${selectedWithDependenciesCount} total)`}
                {...stylex.props(s.toolButton)}
                onClick={() => void runWorkflow("selected-with-dependencies")}
              >
                {runningScope === "selected-with-dependencies" ? (
                  <LoaderCircle size={13} {...stylex.props(s.spinner)} />
                ) : (
                  <Workflow size={13} />
                )}
                {runningScope === "selected-with-dependencies"
                  ? "Running…"
                  : "With dependencies"}
              </button>
            </NodeToolbar>
          ) : null}
        </WorkflowCanvas>
      </section>

      {visibleExecution ? (
        <aside
          aria-label={`Execution: ${visibleExecutionTitle}`}
          {...stylex.props(s.executionBar)}
        >
          <span
            aria-hidden="true"
            {...stylex.props(
              s.executionIndicator,
              executionCancelling ? s.executionIndicatorCancelling : null,
            )}
          >
            <LoaderCircle size={15} {...stylex.props(s.spinner)} />
          </span>
          <span
            role="status"
            aria-live="polite"
            {...stylex.props(s.executionCopy)}
          >
            <span {...stylex.props(s.executionEyebrow)}>Execution</span>
            <span {...stylex.props(s.executionTitle)}>
              {visibleExecutionTitle}
            </span>
            <span
              {...stylex.props(
                s.executionStatus,
                visibleExecution.statusError ? s.executionStatusError : null,
              )}
            >
              {visibleExecutionStatus}
            </span>
          </span>
          <button
            type="button"
            disabled={!visibleExecution.executionId || executionCancelling}
            aria-label={executionCancelling ? "Cancelling execution" : "Cancel execution"}
            {...stylex.props(s.cancelExecutionButton)}
            onClick={() => void cancelCurrentExecution()}
          >
            {executionCancelling ? (
              <LoaderCircle size={12} {...stylex.props(s.spinner)} />
            ) : (
              <Square size={11} fill="currentColor" />
            )}
            {executionCancelling ? "Cancelling" : "Cancel"}
          </button>
        </aside>
      ) : null}

      <WorkbenchHeader
        graphName={graphName}
        activeGraphRevision={activeGraph?.revision ?? null}
        isDirty={isDirty}
        saving={saving}
        saveDisabled={
          saving ||
          running ||
          Boolean(openingGraphId) ||
          Boolean(deletingGraphId) ||
          !graphName.trim() ||
          Boolean(activeGraph && !isDirty)
        }
        nodeCount={nodes.length}
        edgeCount={edges.length}
        graphStatus={
          graphHasErrors
            ? "error"
            : running
              ? "running"
              : graphNeedsAttention
                ? "incomplete"
                : "ready"
        }
        canvasStatusMessage={canvasStatusMessage}
        themePreference={preference}
        onToggleGraphBrowser={() => {
          setExecutionHistoryTarget(null);
          toggleGraphBrowser();
        }}
        onGraphNameChange={setGraphName}
        onSaveGraph={() => void saveCurrentGraph()}
        onCycleTheme={cycleTheme}
      />

      <aside aria-label="Canvas actions" {...stylex.props(s.actionRail)}>
        <button
          type="button"
          {...firefoxDynamicButtonProps}
          aria-label="Add node"
          disabled={!registry || running}
          title="Add node"
          {...stylex.props(s.railButton, s.railPrimary)}
          onClick={() => {
            closeGraphBrowser();
            setLibraryOpen((open) => !open);
          }}
        >
          <Plus size={14} />
          <span {...stylex.props(s.railLabel)}>Node</span>
        </button>
        <button
          type="button"
          {...firefoxDynamicButtonProps}
          disabled={!flow}
          title="Fit workflow"
          {...stylex.props(s.railButton)}
          onClick={() => void flow?.fitView(WORKBENCH_FIT_VIEW_OPTIONS)}
        >
          <Maximize2 size={14} />
          <span {...stylex.props(s.railLabel)}>Fit</span>
        </button>
        <span {...stylex.props(s.railDivider)} />
        <button
          type="button"
          disabled={!activeGraph}
          title={activeGraph
            ? "Browse previous executions"
            : "Save the graph to browse executions"}
          {...stylex.props(s.railButton)}
          onClick={() => {
            closeGraphBrowser();
            setLibraryOpen(false);
            setExecutionHistoryTarget({ nodeId: null });
          }}
        >
          <History size={14} />
          <span {...stylex.props(s.railLabel)}>Runs</span>
        </button>
        <span {...stylex.props(s.railDivider)} />
        <button
          type="button"
          disabled={!selectedNodeCount || running}
          title={
            selectedNodeCount
              ? `Duplicate ${selectedNodeCount} selected node${selectedNodeCount === 1 ? "" : "s"}`
              : "Select one or more nodes to duplicate"
          }
          {...stylex.props(s.railButton)}
          onClick={duplicateSelectedNodes}
        >
          <Copy size={14} />
          <span {...stylex.props(s.railLabel)}>Duplicate</span>
        </button>
        <button
          type="button"
          disabled={!flow || !selectedNodeCount || running}
          title={
            selectedNodeCount
              ? `Delete ${selectedNodeCount} selected node${selectedNodeCount === 1 ? "" : "s"}`
              : "Select one or more nodes to delete"
          }
          {...stylex.props(s.railButton, s.railDanger)}
          onClick={deleteSelectedNodes}
        >
          <Trash2 size={14} />
          <span {...stylex.props(s.railLabel)}>Delete</span>
        </button>
      </aside>

      <Toast.Provider timeout={8000} limit={3}>
        <GlobalIssueToastList
          issues={globalIssues}
          onDismiss={dismissGlobalIssue}
        />
      </Toast.Provider>

      {graphBrowserOpen ? (
        <SavedGraphBrowser
          graphs={savedGraphs}
          activeGraphId={activeGraph?.id ?? null}
          openingGraphId={openingGraphId}
          deletingGraphId={deletingGraphId}
          busy={graphOperationBusy}
          loading={savedGraphsLoading}
          refreshing={savedGraphsRefreshing}
          error={savedGraphsError}
          onClose={closeGraphBrowser}
          onNew={requestNewGraph}
          onOpen={(graphId) => void openSavedGraph(graphId)}
          onDelete={(graph) => void removeSavedGraph(graph)}
          onRefresh={() => void refreshSavedGraphs()}
        />
      ) : null}

      {executionHistoryTarget ? (
        <ExecutionHistoryDrawer
          key={`${activeGraph?.id ?? "unsaved"}:${executionHistoryTarget.nodeId ?? "all"}`}
          graphId={activeGraph?.id ?? null}
          graphName={graphName}
          nodeId={executionHistoryTarget.nodeId}
          nodeTitles={nodeTitles}
          executionRunning={running}
          isDirty={isDirty}
          onClose={() => setExecutionHistoryTarget(null)}
        />
      ) : null}

      {registry ? (
        <NodeSelector
          open={libraryOpen}
          registry={registry}
          activeGraphId={activeGraph?.id ?? null}
          onOpenChange={setLibraryOpen}
          onAddNode={addCatalogNode}
          onOpenGraph={openGraphInNewTab}
        />
      ) : null}

      <ConnectionRouteDialog
        pendingRoute={pendingConnectionRoute}
        onSelect={(route) => {
          if (!pendingConnectionRoute) return;
          addWorkflowEdge(
            pendingConnectionRoute.connection,
            pendingConnectionRoute.collectionMode,
            route,
          );
        }}
        onClose={() => setPendingConnectionRoute(null)}
      />
    </main>
  );
}

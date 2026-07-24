"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Toast } from "@base-ui/react/toast";
import {
  NodeToolbar,
  Position,
  type Connection,
  type EdgeChange,
  type IsValidConnection,
  type NodeChange,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Copy,
  Eye,
  History,
  LoaderCircle,
  Maximize2,
  Play,
  Plus,
  Trash2,
  Workflow,
} from "lucide-react";

import { ExecutionHistoryDrawer } from "./ExecutionHistoryDrawer";
import {
  GlobalIssueToastList,
  type GlobalIssue,
} from "./GlobalIssueToastList";
import {
  WorkbenchActivityBar,
  type WorkbenchActivity,
} from "./WorkbenchActivityBar";
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
  ARTIFACT_VIEWER_EDGE_TYPE,
  ARTIFACT_VIEWER_INPUT_HANDLE,
  ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE,
  ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE,
  ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE,
  ARTIFACT_VIEWER_NODE_TYPE,
  artifactViewerStorageKey,
  hydrateArtifactViewerDocument,
  serializeArtifactViewerDocument,
  type ArtifactViewerCanvasState,
  type ArtifactViewerEdge,
  type ArtifactViewerInteractionEdge,
  type ArtifactViewerNode,
  type CanvasEdge,
  type CanvasNode,
} from "../canvas/artifact-viewer";
import {
  EMPTY_ARTIFACT_KEY_SELECTION,
  targetRowsForBinding,
  type ArtifactInteractionField,
  type ArtifactKeySelection,
  type ArtifactViewerActivity,
  type ArtifactViewerBinding,
} from "../canvas/artifact-interactions";
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
  artifactTypeColor,
} from "../canvas/nodes.css";
import type { ArtifactQueryRelation } from "../canvas/query-artifact-tables";
import type { SchemaBuilderField } from "../canvas/schema-builder";
import { useTheme } from "@/components/theme";
import {
  isFileUploadOperator,
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  bindArtifactTypeVariable,
  createWorkflowNodeData,
  effectivePortShape,
  imageUploads,
  invalidateWorkflowNodeRuns,
  removeImageUpload,
  replaceImageUploads,
  resetArtifactTypeBinding,
  resolvedPortArtifactType,
  type WorkflowEdge,
  type WorkflowEdgeRouteOffset,
  type WorkflowEdgeUpdate,
  type WorkflowArtifactTypeBindings,
  type WorkflowNodeData,
  type WorkflowInputPlug,
  portMetaForPort,
  workflowNodeIsSupported,
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
  uploadFile,
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

interface ActiveArtifactViewerActivity {
  activity: ArtifactViewerActivity;
  revision: number;
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
  const [artifactViewers, setArtifactViewers] =
    React.useState<ArtifactViewerCanvasState>({
      graphId: null,
      nodes: [],
      edges: [],
      bindings: [],
    });
  const [artifactViewerSelections, setArtifactViewerSelections] =
    React.useState<Record<string, ArtifactKeySelection>>({});
  const [artifactViewerFields, setArtifactViewerFields] =
    React.useState<Record<string, ArtifactInteractionField[]>>({});
  const [artifactViewerActivities, setArtifactViewerActivities] =
    React.useState<Record<string, ActiveArtifactViewerActivity>>({});
  const artifactViewerActivityRevisionRef = React.useRef(0);
  const artifactViewersInitializedRef = React.useRef(initialGraphId === null);
  const [artifactViewerPersistenceError, setArtifactViewerPersistenceError] =
    React.useState<string | null>(null);
  const {
    nodeSecretStatuses,
    refreshNodeSecretStatuses,
    applyConfiguredNodeSecret,
    removeConfiguredNodeSecret,
    clearGraphSecretStatuses,
    forgetNodeSecretStatuses,
  } = useNodeSecrets(nodes);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<CanvasNode, CanvasEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [executionHistoryTarget, setExecutionHistoryTarget] = React.useState<{
    nodeId: string | null;
    executionId: string | null;
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
              progress: null,
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

  const updateArtifactViewerLayout = React.useCallback((
    nodeId: string,
    layout: ArtifactViewerNode["data"]["layout"],
  ) => {
    setArtifactViewers((current) => ({
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, layout } }
          : node,
      ),
    }));
  }, []);

  const updateArtifactViewerMode = React.useCallback((
    nodeId: string,
    mode: string,
  ) => {
    setArtifactViewers((current) => ({
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, mode } }
          : node,
      ),
    }));
  }, []);

  const updateArtifactViewerSelection = React.useCallback((
    nodeId: string,
    selection: ArtifactKeySelection,
  ) => {
    setArtifactViewerSelections((current) => ({
      ...current,
      [nodeId]: selection,
    }));
  }, []);

  const updateArtifactViewerFields = React.useCallback((
    nodeId: string,
    fields: ArtifactInteractionField[],
  ) => {
    setArtifactViewerFields((current) => {
      if (JSON.stringify(current[nodeId] ?? []) === JSON.stringify(fields)) {
        return current;
      }
      return { ...current, [nodeId]: fields };
    });
  }, []);

  const updateArtifactViewerActivity = React.useCallback((
    nodeId: string,
    activity: ArtifactViewerActivity | null,
  ) => {
    if (!activity) {
      setArtifactViewerActivities((current) => {
        if (!current[nodeId]) return current;
        const next = { ...current };
        delete next[nodeId];
        return next;
      });
      return;
    }
    const revision = artifactViewerActivityRevisionRef.current + 1;
    artifactViewerActivityRevisionRef.current = revision;
    setArtifactViewerActivities((current) => {
      return {
        ...current,
        [nodeId]: {
          activity,
          revision,
        },
      };
    });
  }, []);

  const updateArtifactViewerBinding = React.useCallback((
    bindingId: string,
    binding: ArtifactViewerBinding,
  ) => {
    setArtifactViewers((current) => ({
      ...current,
      bindings: current.bindings.map((candidate) =>
        candidate.id === bindingId ? binding : candidate
      ),
    }));
  }, []);

  const removeArtifactViewer = React.useCallback((nodeId: string) => {
    setArtifactViewers((current) => ({
      ...current,
      nodes: current.nodes.filter((node) => node.id !== nodeId),
      edges: current.edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
      bindings: current.bindings.filter(
        (binding) =>
          binding.sourceViewerId !== nodeId &&
          binding.targetViewerId !== nodeId,
      ),
    }));
    setArtifactViewerSelections((current) => {
      const next = { ...current };
      delete next[nodeId];
      return next;
    });
    setArtifactViewerFields((current) => {
      const next = { ...current };
      delete next[nodeId];
      return next;
    });
    setArtifactViewerActivities((current) => {
      if (!current[nodeId]) return current;
      const next = { ...current };
      delete next[nodeId];
      return next;
    });
  }, []);

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
    setArtifactViewers((current) => ({
      ...current,
      edges: current.edges.filter((edge) => edge.source !== nodeId),
    }));
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
              progress: null,
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
              progress: null,
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
              progress: null,
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
              progress: null,
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
              progress: null,
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

  const updateArtifactQueryRelations = React.useCallback(
    (
      nodeId: string,
      relations: readonly ArtifactQueryRelation[],
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
                  ? { ...node.data.config, relations }
                  : node.data.config,
              inputPlugs:
                node.id === nodeId ? inputPlugs : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
              progress: null,
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
          progress: null,
          execution: node.id === nodeId
            ? { status: "uploading" }
            : { status: "idle" },
        },
      };
    }));
    setRunError(null);
    try {
      const uploads = await Promise.all(files.map((file) => uploadFile(file)));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: invalidatedNodeIds.has(node.id)
          ? {
              ...(node.id === nodeId
                ? replaceImageUploads(node.data, uploads)
                : node.data),
              execution: { status: "idle" },
              run: null,
              progress: null,
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

  const openNodeExecutionHistory = React.useCallback((
    nodeId: string,
    executionId?: string,
  ) => {
    setLibraryOpen(false);
    setExecutionHistoryTarget({ nodeId, executionId: executionId ?? null });
  }, []);

  const attachNodeCallbacks = React.useCallback(
    (data: WorkflowNodeData): WorkflowNodeData => {
      if (!workflowNodeIsSupported(data)) {
        return {
          ...data,
          onConfigChange: undefined,
          onLayoutChange: updateLayout,
          onRemoveNode: removeNode,
          onImagesSelected: undefined,
          onRemoveImageUpload: undefined,
          onAddInputPlug: undefined,
          onRemoveInputPlug: undefined,
          onReorderInputPlug: undefined,
          onSchemaBuilderFieldsChange: undefined,
          onArtifactQueryRelationsChange: undefined,
          onResetArtifactTypeBinding: undefined,
          onHandlesMeasured: undefined,
          onOpenModuleSource: undefined,
          onOpenExecutionHistory: openNodeExecutionHistory,
        };
      }
      return {
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
        onArtifactQueryRelationsChange: updateArtifactQueryRelations,
        onResetArtifactTypeBinding: resetNodeArtifactTypeBinding,
        onHandlesMeasured: handleNodeHandlesMeasured,
        onOpenModuleSource: data.spec.module_graph_id
          ? openGraphInNewTab
          : undefined,
        onOpenExecutionHistory: openNodeExecutionHistory,
      };
    },
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
      updateArtifactQueryRelations,
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
  const removeArtifactViewerDocument = React.useCallback((graphId: string) => {
    try {
      window.localStorage.removeItem(
        artifactViewerStorageKey(workspaceSlug, graphId),
      );
      setArtifactViewerPersistenceError(null);
    } catch {
      setArtifactViewerPersistenceError(
        "Artifact Viewer layout could not be removed from browser storage.",
      );
    }
  }, [workspaceSlug]);
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
    onGraphDeleted: removeArtifactViewerDocument,
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

  React.useEffect(() => {
    const graphId = activeGraph?.id ?? null;
    if (!artifactViewersInitializedRef.current && !graphId) return;
    if (artifactViewersInitializedRef.current && artifactViewers.graphId === graphId) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      if (!artifactViewersInitializedRef.current) {
        try {
          const serialized = window.localStorage.getItem(
            artifactViewerStorageKey(workspaceSlug, graphId!),
          );
          const hydrated = serialized
            ? hydrateArtifactViewerDocument(serialized, graphId!)
            : { graphId: graphId!, nodes: [], edges: [], bindings: [] };
          setArtifactViewers(
            hydrated ?? {
              graphId: graphId!,
              nodes: [],
              edges: [],
              bindings: [],
            },
          );
          setArtifactViewerPersistenceError(
            serialized && !hydrated
              ? "Saved Artifact Viewer layout was invalid and has been reset."
              : null,
          );
        } catch {
          setArtifactViewers({
            graphId: graphId!,
            nodes: [],
            edges: [],
            bindings: [],
          });
          setArtifactViewerPersistenceError(
            "Artifact Viewer layout could not be loaded from browser storage.",
          );
        }
        artifactViewersInitializedRef.current = true;
        return;
      }

      if (artifactViewers.graphId === null && graphId) {
        setArtifactViewers((current) => ({ ...current, graphId }));
        return;
      }
      if (!graphId) {
        setArtifactViewers({
          graphId: null,
          nodes: [],
          edges: [],
          bindings: [],
        });
        return;
      }

      try {
        const serialized = window.localStorage.getItem(
          artifactViewerStorageKey(workspaceSlug, graphId),
        );
        const hydrated = serialized
          ? hydrateArtifactViewerDocument(serialized, graphId)
          : { graphId, nodes: [], edges: [], bindings: [] };
        setArtifactViewers(
          hydrated ?? { graphId, nodes: [], edges: [], bindings: [] },
        );
        setArtifactViewerPersistenceError(
          serialized && !hydrated
            ? "Saved Artifact Viewer layout was invalid and has been reset."
            : null,
        );
      } catch {
        setArtifactViewers({
          graphId,
          nodes: [],
          edges: [],
          bindings: [],
        });
        setArtifactViewerPersistenceError(
          "Artifact Viewer layout could not be loaded from browser storage.",
        );
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeGraph?.id, artifactViewers.graphId, workspaceSlug]);

  React.useEffect(() => {
    const graphId = activeGraph?.id;
    if (
      !graphId ||
      !artifactViewersInitializedRef.current ||
      artifactViewers.graphId !== graphId
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(
          artifactViewerStorageKey(workspaceSlug, graphId),
          serializeArtifactViewerDocument(
            artifactViewers.nodes,
            artifactViewers.edges,
            artifactViewers.bindings,
          ),
        );
        setArtifactViewerPersistenceError(null);
      } catch {
        setArtifactViewerPersistenceError(
          "Artifact Viewer changes are available in this session but could not be saved in browser storage.",
        );
      }
    }, 160);
    return () => window.clearTimeout(timer);
  }, [
    activeGraph?.id,
    artifactViewers.edges,
    artifactViewers.bindings,
    artifactViewers.graphId,
    artifactViewers.nodes,
    workspaceSlug,
  ]);

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
      workflowNodeIsSupported(node.data) &&
      isFileUploadOperator(node.data.spec.operator_id) &&
      !imageUploads(node.data).length,
  );
  const selectedNodeIds = React.useMemo(
    () => nodes.flatMap((node) => (node.selected ? [node.id] : [])),
    [nodes],
  );
  const selectedNodeCount = selectedNodeIds.length;
  const selectedNodesAreRunnable = nodes.every(
    (node) => !node.selected || workflowNodeIsSupported(node.data),
  );
  const nodeTitles = React.useMemo(
    () => Object.fromEntries(
      nodes.map((node) => [node.id, node.data.spec.title]),
    ),
    [nodes],
  );
  const selectedWithDependencyIds = selectedNodeAndAncestorIds(
    nodes,
    edges,
  );
  const selectedWithDependenciesCount = selectedWithDependencyIds.size;
  const selectedWithDependenciesAreRunnable = nodes.every(
    (node) =>
      !selectedWithDependencyIds.has(node.id) ||
      workflowNodeIsSupported(node.data),
  );
  const missingRequiredInputs = missingRequiredInputsFor(nodes, edges);
  const connectionInstruction = missingRequiredInputs.length
    ? `${missingRequiredInputs.length} required input${missingRequiredInputs.length === 1 ? "" : "s"} unconnected · drag between ports to connect them`
    : null;
  const runSelectionBusy = !registry || running || selectedNodeCount === 0;
  const runSelectedDisabled = runSelectionBusy || !selectedNodesAreRunnable;
  const runSelectedWithDependenciesDisabled =
    runSelectionBusy || !selectedWithDependenciesAreRunnable;
  const nodeErrorCount = nodes.filter(
    (node) => Boolean(node.data.execution.error),
  ).length;
  const compatibilityIssueCount =
    nodes.filter((node) => !workflowNodeIsSupported(node.data)).length +
    edges.filter((edge) => edge.data?.compatibilityIssues?.length).length;
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
    if (artifactViewerPersistenceError) {
      issues.push({
        id: "presentation",
        title: "Artifact Viewer",
        message: artifactViewerPersistenceError,
      });
    }
    return issues;
  }, [
    artifactViewerPersistenceError,
    persistenceError,
    registryError,
    runError,
  ]);
  const dismissGlobalIssue = React.useCallback((issue: GlobalIssue) => {
    if (issue.id === "graph") {
      dismissPersistenceError(issue.message);
    }
    if (issue.id === "run") {
      dismissRunError(issue.message);
    }
    if (issue.id === "presentation") {
      setArtifactViewerPersistenceError((current) =>
        current === issue.message ? null : current,
      );
    }
  }, [dismissPersistenceError, dismissRunError]);
  const workflowGlobalIssueCount = globalIssues.filter(
    (issue) => issue.id !== "presentation",
  ).length;
  const graphHasErrors =
    workflowGlobalIssueCount > 0 ||
    nodeErrorCount > 0 ||
    compatibilityIssueCount > 0;
  const graphNeedsAttention =
    compatibilityIssueCount > 0 ||
    imageUploadWithoutImages ||
    missingRequiredInputs.length > 0;
  const canvasStatusMessage = runningScope === "selected"
    ? "running selected nodes · latest upstream outputs are pinned"
    : runningScope === "selected-with-dependencies"
      ? "running selected nodes and all upstream dependencies"
      : workflowGlobalIssueCount
        ? `${workflowGlobalIssueCount} workflow issue${workflowGlobalIssueCount === 1 ? "" : "s"}`
        : compatibilityIssueCount
          ? `${compatibilityIssueCount} compatibility issue${compatibilityIssueCount === 1 ? "" : "s"}`
        : nodeErrorCount
          ? `${nodeErrorCount} node issue${nodeErrorCount === 1 ? "" : "s"}`
          : !registry
            ? "loading live registry…"
            : imageUploadWithoutImages
              ? "choose images before running"
              : connectionInstruction ?? "all required inputs connected · ready to run";

  const activeArtifactViewers = artifactViewers.graphId ===
      (activeGraph?.id ?? null)
    ? artifactViewers
    : {
        graphId: activeGraph?.id ?? null,
        nodes: [],
        edges: [],
        bindings: [],
      };

  const onNodesChange: OnNodesChange<CanvasNode> = React.useCallback(
    (changes) => {
      const workflowNodeIds = new Set(nodes.map((node) => node.id));
      const artifactViewerIds = new Set(
        artifactViewers.nodes.map((node) => node.id),
      );
      const workflowChanges = changes.filter((change) =>
        change.type === "add" || change.type === "replace"
          ? change.item.type === WORKFLOW_NODE_TYPE
          : workflowNodeIds.has(change.id)
      ) as NodeChange<WorkflowNode>[];
      const artifactViewerChanges = changes.filter((change) =>
        change.type === "add" || change.type === "replace"
          ? change.item.type === ARTIFACT_VIEWER_NODE_TYPE
          : artifactViewerIds.has(change.id)
      ) as NodeChange<ArtifactViewerNode>[];
      if (workflowChanges.length) {
        setNodes((current) => applyNodeChanges(workflowChanges, current));
      }
      const removedArtifactViewerIds = new Set(
        artifactViewerChanges.flatMap((change) =>
          change.type === "remove" ? [change.id] : []
        ),
      );
      if (artifactViewerChanges.length) {
        setArtifactViewers((current) => ({
          ...current,
          nodes: applyNodeChanges(artifactViewerChanges, current.nodes),
          bindings: removedArtifactViewerIds.size
            ? current.bindings.filter(
                (binding) =>
                  !removedArtifactViewerIds.has(binding.sourceViewerId) &&
                  !removedArtifactViewerIds.has(binding.targetViewerId),
              )
            : current.bindings,
        }));
      }
      if (removedArtifactViewerIds.size) {
        setArtifactViewerSelections((current) => {
          const next = { ...current };
          for (const nodeId of removedArtifactViewerIds) delete next[nodeId];
          return next;
        });
        setArtifactViewerFields((current) => {
          const next = { ...current };
          for (const nodeId of removedArtifactViewerIds) delete next[nodeId];
          return next;
        });
        setArtifactViewerActivities((current) => {
          const next = { ...current };
          for (const nodeId of removedArtifactViewerIds) delete next[nodeId];
          return next;
        });
      }
      const removedWorkflowNodeIds = new Set(
        workflowChanges.flatMap((change) =>
          change.type === "remove" ? [change.id] : [],
        ),
      );
      if (removedWorkflowNodeIds.size) {
        setArtifactViewers((current) => ({
          ...current,
          edges: current.edges.filter(
            (edge) => !removedWorkflowNodeIds.has(edge.source),
          ),
        }));
      }
    },
    [artifactViewers.nodes, nodes],
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

  const onEdgesChange: OnEdgesChange<CanvasEdge> = React.useCallback(
    (changes) => {
      const workflowEdgeIds = new Set(edges.map((edge) => edge.id));
      const artifactViewerEdgeIds = new Set(
        artifactViewers.edges.map((edge) => edge.id),
      );
      const artifactViewerInteractionEdgeIds = new Set(
        artifactViewers.bindings.map((binding) => binding.id),
      );
      const workflowChanges = changes.filter((change) =>
        change.type === "add" || change.type === "replace"
          ? change.item.type !== ARTIFACT_VIEWER_EDGE_TYPE &&
            change.item.type !== ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE
          : workflowEdgeIds.has(change.id)
      ) as EdgeChange<WorkflowEdge>[];
      const artifactViewerChanges = changes.filter((change) =>
        change.type === "add" || change.type === "replace"
          ? change.item.type === ARTIFACT_VIEWER_EDGE_TYPE
          : artifactViewerEdgeIds.has(change.id)
      ) as EdgeChange<ArtifactViewerEdge>[];
      const artifactViewerInteractionChanges = changes.filter((change) =>
        change.type === "add" || change.type === "replace"
          ? change.item.type === ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE
          : artifactViewerInteractionEdgeIds.has(change.id)
      ) as EdgeChange<ArtifactViewerInteractionEdge>[];
      const changedTargetNodeIds = new Set<string>();
      for (const change of workflowChanges) {
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
      if (workflowChanges.length) {
        setEdges((current) => applyEdgeChanges(workflowChanges, current));
        invalidateWorkflowResults([...changedTargetNodeIds], edges);
      }
      if (artifactViewerChanges.length) {
        setArtifactViewers((current) => ({
          ...current,
          edges: applyEdgeChanges(
            artifactViewerChanges,
            current.edges,
          ),
        }));
      }
      if (artifactViewerInteractionChanges.length) {
        const removedBindingIds = new Set(
          artifactViewerInteractionChanges.flatMap((change) =>
            change.type === "remove" ? [change.id] : []
          ),
        );
        if (removedBindingIds.size) {
          setArtifactViewers((current) => ({
            ...current,
            bindings: current.bindings.filter(
              (binding) => !removedBindingIds.has(binding.id),
            ),
          }));
        }
      }
    },
    [
      artifactViewers.bindings,
      artifactViewers.edges,
      edges,
      invalidateWorkflowResults,
    ],
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
      ? artifactTypeColor(sourceArtifactType.id, tokens.colorAccent)
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
    const changedNodeIds = binding?.endpoint === "source" && newlyBoundNodeId
      ? [newlyBoundNodeId, edge.target]
      : [edge.target];
    invalidateWorkflowResults(changedNodeIds, [...edges, edge]);
  }, [edges, invalidateWorkflowResults, nodes]);

  const isValidConnection = React.useCallback<
    IsValidConnection<CanvasEdge>
  >((connection) => {
    const candidate: Connection = {
      source: connection.source,
      sourceHandle: connection.sourceHandle ?? null,
      target: connection.target,
      targetHandle: connection.targetHandle ?? null,
    };
    if (
      candidate.sourceHandle === ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE ||
      candidate.targetHandle === ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE
    ) {
      return (
        candidate.sourceHandle ===
          ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE &&
        candidate.targetHandle ===
          ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE &&
        candidate.source !== candidate.target &&
        activeArtifactViewers.nodes.some(
          (node) => node.id === candidate.source,
        ) &&
        activeArtifactViewers.nodes.some(
          (node) => node.id === candidate.target,
        ) &&
        !activeArtifactViewers.bindings.some(
          (binding) =>
            binding.sourceViewerId === candidate.source &&
            binding.targetViewerId === candidate.target,
        )
      );
    }
    if (
      candidate.targetHandle === ARTIFACT_VIEWER_INPUT_HANDLE &&
      activeArtifactViewers.nodes.some(
        (node) => node.id === candidate.target,
      )
    ) {
      const source = decodeHandleId(candidate.sourceHandle);
      const sourceNode = nodes.find(
        (node) => node.id === candidate.source,
      );
      return Boolean(
        source &&
        source.direction === "output" &&
        sourceNode?.data.spec.outputs.some(
          (port) => port.name === source.portName,
        ),
      );
    }
    return isConnectionAccepted(
      candidate,
      nodes,
      edges,
      registry?.artifact_types ?? [],
      registry?.artifact_conversions ?? [],
      "id" in connection ? connection.id : null,
    );
  }, [
    activeArtifactViewers.nodes,
    activeArtifactViewers.bindings,
    edges,
    nodes,
    registry?.artifact_conversions,
    registry?.artifact_types,
  ]);

  const onConnect: OnConnect = React.useCallback((connection) => {
    if (!isValidConnection(connection)) return;
    if (
      connection.sourceHandle ===
        ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE &&
      connection.targetHandle === ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE
    ) {
      const binding: ArtifactViewerBinding = {
        id: `artifact-viewer-binding-${crypto.randomUUID()}`,
        sourceViewerId: connection.source,
        targetViewerId: connection.target,
        mappings: [{ sourceField: "", targetField: "" }],
        effects: ["highlight", "focus"],
        emptySelection: "show_all",
      };
      setArtifactViewers((current) => ({
        ...current,
        bindings: [...current.bindings, binding],
      }));
      setPendingConnectionRoute(null);
      return;
    }
    if (connection.targetHandle === ARTIFACT_VIEWER_INPUT_HANDLE) {
      const source = decodeHandleId(connection.sourceHandle);
      if (!source || source.direction !== "output") return;
      const edge: ArtifactViewerEdge = {
        id: `artifact-viewer-edge-${crypto.randomUUID()}`,
        type: ARTIFACT_VIEWER_EDGE_TYPE,
        source: connection.source,
        target: connection.target,
        targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
        data: { sourcePortName: source.portName },
      };
      setArtifactViewers((current) => ({
        ...current,
        edges: [
          ...current.edges.filter(
            (candidate) => candidate.target !== connection.target,
          ),
          edge,
        ],
      }));
      setPendingConnectionRoute(null);
      return;
    }
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

  const addArtifactViewer = React.useCallback(() => {
    const id = `artifact-viewer-${crypto.randomUUID()}`;
    const center = flow?.screenToFlowPosition({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
    }) ?? { x: 600, y: 280 };
    const selectedSource = nodes.find((node) => node.selected);
    const position = selectedSource
      ? {
          x: selectedSource.position.x + 380,
          y: selectedSource.position.y - 20,
        }
      : { x: center.x - 260, y: center.y - 180 };
    setNodes((current) =>
      current.map((node) => ({ ...node, selected: false })),
    );
    setArtifactViewers((current) => ({
      ...current,
      nodes: [
        ...current.nodes.map((node) => ({ ...node, selected: false })),
        {
          id,
          type: ARTIFACT_VIEWER_NODE_TYPE,
          position,
          selected: true,
          data: {
            layout: { width: 520, appendixHeight: 300 },
            mode: null,
          },
        },
      ],
    }));
    setLibraryOpen(false);
    closeGraphBrowser();
    if (flow && selectedSource) {
      window.requestAnimationFrame(() => {
        void flow.fitView({
          nodes: [{ id: selectedSource.id }, { id }],
          padding: 0.22,
          maxZoom: 0.94,
          duration: 220,
        });
      });
    }
  }, [closeGraphBrowser, flow, nodes]);

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
        progress: null,
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
            historyContext: {
              graphId: activeGraph?.id ?? null,
              isDirty,
            },
            secretStatuses: nodeSecretStatuses[node.id] ?? {},
            secretInputReadiness: Object.fromEntries(
              (workflowNodeIsSupported(node.data)
                ? nodeSecretInputs(node.data.spec)
                : []).map((input) => [
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
      isDirty,
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
            sourcePortName:
              source?.portName ?? edge.data?.sourcePortName,
            conversionTitles,
            routeOptions,
            allowedCollectionModes:
              edge.data?.compatibilityIssues?.length || !validMode
                ? []
                : [validMode],
            onUpdate: edge.data?.compatibilityIssues?.length
              ? undefined
              : updateEdge,
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

  const artifactViewerCanvasNodes = React.useMemo<ArtifactViewerNode[]>(
    () => activeArtifactViewers.nodes.map((node) => {
      const sourceBindings = activeArtifactViewers.bindings.filter(
        (binding) => binding.sourceViewerId === node.id,
      );
      const incomingBindings = activeArtifactViewers.bindings
        .filter((binding) => binding.targetViewerId === node.id)
        .map((binding) => {
          const sourceSelection =
            artifactViewerSelections[binding.sourceViewerId] ??
              EMPTY_ARTIFACT_KEY_SELECTION;
          return {
            bindingId: binding.id,
            effects: binding.effects,
            sourceSelectionCount: sourceSelection.items.length,
            rows: targetRowsForBinding(binding, sourceSelection),
          };
        });
      return {
        ...node,
        data: {
          ...node.data,
          outgoingFields: [
            ...new Set(
              sourceBindings.flatMap((binding) =>
                binding.mappings.map((mapping) => mapping.sourceField)
              ),
            ),
          ].filter(Boolean),
          selection:
            artifactViewerSelections[node.id] ??
              EMPTY_ARTIFACT_KEY_SELECTION,
          incomingBindings,
          fields: artifactViewerFields[node.id] ?? [],
          onLayoutChange: updateArtifactViewerLayout,
          onModeChange: updateArtifactViewerMode,
          onSelectionChange: updateArtifactViewerSelection,
          onFieldsChange: updateArtifactViewerFields,
          onActivityChange: updateArtifactViewerActivity,
          onRemoveNode: removeArtifactViewer,
        },
      };
    }),
    [
      activeArtifactViewers.bindings,
      activeArtifactViewers.nodes,
      artifactViewerFields,
      artifactViewerSelections,
      removeArtifactViewer,
      updateArtifactViewerActivity,
      updateArtifactViewerLayout,
      updateArtifactViewerMode,
      updateArtifactViewerFields,
      updateArtifactViewerSelection,
    ],
  );

  const artifactViewerCanvasEdges = React.useMemo<ArtifactViewerEdge[]>(
    () => activeArtifactViewers.edges.map((edge) => {
      const sourceNode = nodes.find((node) => node.id === edge.source);
      const sourcePort = sourceNode?.data.spec.outputs.find(
        (port) => port.name === edge.data?.sourcePortName,
      );
      const sourceArtifactType = sourceNode && sourcePort
        ? resolvedPortArtifactType(
            sourcePort,
            sourceNode.data.artifactTypeBindings,
          )
        : null;
      const sourceHandle =
        sourceNode &&
          sourcePort &&
          workflowNodeIsSupported(sourceNode.data)
          ? encodeHandleId(
              portMetaForPort(
                sourcePort,
                effectivePortShape(sourceNode.data, sourcePort),
                undefined,
                sourceNode.data.artifactTypeBindings,
              ),
            )
          : null;
      return {
        ...edge,
        type: ARTIFACT_VIEWER_EDGE_TYPE,
        sourceHandle,
        targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
        style: {
          ...edge.style,
          stroke: sourceArtifactType
            ? artifactTypeColor(sourceArtifactType.id, tokens.colorAccent)
            : tokens.colorAccent,
          strokeWidth: 2,
        },
      };
    }),
    [activeArtifactViewers.edges, nodes],
  );

  const artifactViewerInteractionCanvasEdges =
    React.useMemo<ArtifactViewerInteractionEdge[]>(
      () => activeArtifactViewers.bindings.map((binding) => ({
        id: binding.id,
        type: ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE,
        source: binding.sourceViewerId,
        sourceHandle: ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE,
        target: binding.targetViewerId,
        targetHandle: ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE,
        data: {
          binding,
          sourceFields:
            artifactViewerFields[binding.sourceViewerId] ?? [],
          targetFields:
            artifactViewerFields[binding.targetViewerId] ?? [],
          onBindingChange: updateArtifactViewerBinding,
        },
        style: {
          stroke: tokens.colorInfo,
          strokeWidth: 2,
        },
      })),
      [
        activeArtifactViewers.bindings,
        artifactViewerFields,
        updateArtifactViewerBinding,
      ],
    );

  const allCanvasNodes = React.useMemo<CanvasNode[]>(
    () => [...canvasNodes, ...artifactViewerCanvasNodes],
    [artifactViewerCanvasNodes, canvasNodes],
  );
  const allCanvasEdges = React.useMemo<CanvasEdge[]>(
    () => [
      ...canvasEdges,
      ...artifactViewerCanvasEdges,
      ...artifactViewerInteractionCanvasEdges,
    ],
    [
      artifactViewerCanvasEdges,
      artifactViewerInteractionCanvasEdges,
      canvasEdges,
    ],
  );

  const latestArtifactViewerActivity = React.useMemo(() => {
    let latest: {
      nodeId: string;
      value: ActiveArtifactViewerActivity;
    } | null = null;
    for (const [nodeId, value] of Object.entries(artifactViewerActivities)) {
      if (!latest || value.revision > latest.value.revision) {
        latest = { nodeId, value };
      }
    }
    return latest;
  }, [artifactViewerActivities]);

  const dismissArtifactViewerActivity = React.useCallback((
    nodeId: string,
    revision: number,
  ) => {
    setArtifactViewerActivities((current) => {
      if (current[nodeId]?.revision !== revision) return current;
      const next = { ...current };
      delete next[nodeId];
      return next;
    });
  }, []);

  React.useEffect(() => {
    if (
      !latestArtifactViewerActivity ||
      latestArtifactViewerActivity.value.activity.state !== "success"
    ) {
      return;
    }
    const { nodeId, value } = latestArtifactViewerActivity;
    const timeout = window.setTimeout(
      () => dismissArtifactViewerActivity(nodeId, value.revision),
      4000,
    );
    return () => window.clearTimeout(timeout);
  }, [dismissArtifactViewerActivity, latestArtifactViewerActivity]);

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
  const viewerActivity = latestArtifactViewerActivity?.value.activity ?? null;
  let viewerActivityAction: WorkbenchActivity["action"];
  if (latestArtifactViewerActivity && viewerActivity?.retry) {
    viewerActivityAction = {
      kind: "retry",
      label: "Retry",
      ariaLabel: `Retry ${viewerActivity.title}`,
      onInvoke: viewerActivity.retry,
    };
  } else if (
    latestArtifactViewerActivity &&
    (
      viewerActivity?.state === "warning" ||
      viewerActivity?.state === "error"
    )
  ) {
    viewerActivityAction = {
      kind: "dismiss",
      label: "Dismiss",
      ariaLabel: `Dismiss ${viewerActivity.title}`,
      onInvoke: () =>
        dismissArtifactViewerActivity(
          latestArtifactViewerActivity.nodeId,
          latestArtifactViewerActivity.value.revision,
        ),
    };
  }
  const workbenchActivity: WorkbenchActivity | null = visibleExecution
    ? {
        eyebrow: "Execution",
        title: visibleExecutionTitle,
        message: visibleExecutionStatus,
        tone: executionCancelling
          ? "cancelling"
          : visibleExecution.statusError
            ? "error"
            : "working",
        action: {
          kind: "cancel",
          label: executionCancelling ? "Cancelling" : "Cancel",
          ariaLabel: executionCancelling
            ? "Cancelling execution"
            : "Cancel execution",
          disabled: !visibleExecution.executionId || executionCancelling,
          onInvoke: () => void cancelCurrentExecution(),
        },
      }
    : viewerActivity
      ? {
          eyebrow: "Linked view",
          title: viewerActivity.title,
          message: viewerActivity.message,
          tone: viewerActivity.state,
          action: viewerActivityAction,
        }
      : null;

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
          nodes={allCanvasNodes}
          edges={allCanvasEdges}
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
                title={selectedNodesAreRunnable
                  ? "Run only the selected nodes; latest accessible upstream outputs are pinned"
                  : "Unavailable or invalid selected nodes cannot run"}
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
                disabled={runSelectedWithDependenciesDisabled}
                title={selectedWithDependenciesAreRunnable
                  ? `Run the selection and every upstream dependency (${selectedWithDependenciesCount} total)`
                  : "Unavailable or invalid upstream dependencies cannot run"}
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

      {workbenchActivity ? (
        <WorkbenchActivityBar activity={workbenchActivity} />
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
          aria-label="Add Artifact Viewer"
          title="Add a presentation-only Artifact Viewer"
          {...stylex.props(s.railButton)}
          onClick={addArtifactViewer}
        >
          <Eye size={14} />
          <span {...stylex.props(s.railLabel)}>Viewer</span>
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
            setExecutionHistoryTarget({ nodeId: null, executionId: null });
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
          key={`${activeGraph?.id ?? "unsaved"}:${executionHistoryTarget.nodeId ?? "all"}:${executionHistoryTarget.executionId ?? "latest"}`}
          graphId={activeGraph?.id ?? null}
          graphName={graphName}
          nodeId={executionHistoryTarget.nodeId}
          initialExecutionId={executionHistoryTarget.executionId}
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

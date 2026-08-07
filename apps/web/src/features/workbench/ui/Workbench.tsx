"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
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
  type GraphRoomPersistenceAdapter,
} from "./useSavedGraphLifecycle";
import { useRunExecution } from "./useRunExecution";
import {
  PresenceOverlay,
  remoteSelectionColor,
  useGraphRoomSession,
  type RoomGraphCommand,
} from "../room";
import {
  WorkflowCanvas,
  applyEdgeChanges,
  applyNodeChanges,
} from "../canvas/WorkflowCanvas";
import {
  addEdgeCommand,
  graphCommandsFromEdgeChanges,
  graphCommandsFromNodeChanges,
  nodeOverlaysFromNodes,
  reduceWorkbenchAuthoringState,
} from "../canvas/graph-document-adapter";
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
  hydrateAuthoredGraphDocument,
} from "../canvas/saved-graph";
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
import { serializeNodeLayout } from "../canvas/node-layout";
import { useTheme } from "@/components/theme";
import {
  isFileUploadOperator,
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  effectivePortShape,
  imageUploads,
  invalidateWorkflowNodeRuns,
  removeImageUpload,
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
  type AuthoredGraphDocument,
  type GraphCommand,
} from "../model/graph-document";
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
  userId: string;
  workspaceId: string;
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
  userId,
  workspaceId,
  workspaceSlug,
  initialGraphId,
}: WorkbenchProps) {
  const {
    data: registry,
    error: registryError,
    mutate: refreshNodeRegistry,
  } = useNodeRegistry(workspaceId);
  const { preference, cycleTheme } = useTheme();
  const [authoringState, dispatchAuthoringState] = React.useReducer(
    reduceWorkbenchAuthoringState,
    {
      document: {
        name: "Untitled workflow",
        nodes: [],
        edges: [],
      },
      nodeOverlays: {},
      error: null,
    },
  );
  const authoredDocument = authoringState.document;
  const nodeOverlays = authoringState.nodeOverlays;
  const authoredDocumentRef = React.useRef(authoredDocument);
  const [selectedNodeIdSet, setSelectedNodeIdSet] =
    React.useState<ReadonlySet<string>>(new Set());
  const [selectedEdgeIdSet, setSelectedEdgeIdSet] =
    React.useState<ReadonlySet<string>>(new Set());
  const [positionOverrides, setPositionOverrides] =
    React.useState<Record<string, { x: number; y: number }>>({});
  const hydratedDocument = React.useMemo(
    () => registry
      ? hydrateAuthoredGraphDocument(authoredDocument, registry)
      : { nodes: [], edges: [] },
    [authoredDocument, registry],
  );
  const nodesRef = React.useRef<WorkflowNode[]>([]);
  const edgesRef = React.useRef<WorkflowEdge[]>([]);
  const nodes = React.useMemo<WorkflowNode[]>(
    () => hydratedDocument.nodes.map((node) => ({
      ...node,
      position: positionOverrides[node.id] ?? node.position,
      selected: selectedNodeIdSet.has(node.id),
      data: {
        ...node.data,
        ...(nodeOverlays[node.id] ?? {}),
      },
    })),
    [
      hydratedDocument.nodes,
      nodeOverlays,
      positionOverrides,
      selectedNodeIdSet,
    ],
  );
  const edges = React.useMemo<WorkflowEdge[]>(
    () => hydratedDocument.edges.map((edge) => ({
      ...edge,
      selected: selectedEdgeIdSet.has(edge.id),
    })),
    [hydratedDocument.edges, selectedEdgeIdSet],
  );
  React.useLayoutEffect(() => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
  }, [edges, nodes]);
  const setNodes = React.useCallback<
    React.Dispatch<React.SetStateAction<WorkflowNode[]>>
  >((action) => {
    const currentNodes = nodesRef.current;
    const nextNodes = typeof action === "function" ? action(currentNodes) : action;
    nodesRef.current = nextNodes;
    setSelectedNodeIdSet(
      new Set(nextNodes.filter((node) => node.selected).map((node) => node.id)),
    );
    setPositionOverrides(() => {
      const next: Record<string, { x: number; y: number }> = {};
      for (const node of nextNodes) {
        const authoredNode = authoredDocumentRef.current.nodes.find(
          (candidate) => candidate.id === node.id,
        );
        if (
          authoredNode &&
          (authoredNode.position.x !== node.position.x ||
            authoredNode.position.y !== node.position.y)
        ) {
          next[node.id] = { x: node.position.x, y: node.position.y };
        }
      }
      return next;
    });
    dispatchAuthoringState({
      kind: "update_overlays",
      update: nodeOverlaysFromNodes(nextNodes),
    });
  }, [dispatchAuthoringState]);
  const setEdges = React.useCallback<
    React.Dispatch<React.SetStateAction<WorkflowEdge[]>>
  >((action) => {
    const currentEdges = edgesRef.current;
    const nextEdges = typeof action === "function" ? action(currentEdges) : action;
    edgesRef.current = nextEdges;
    setSelectedEdgeIdSet(
      new Set(nextEdges.filter((edge) => edge.selected).map((edge) => edge.id)),
    );
  }, []);
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
  } = useNodeSecrets(workspaceId, nodes);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<CanvasNode, CanvasEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [executionHistoryTarget, setExecutionHistoryTarget] = React.useState<{
    nodeId: string | null;
    executionId: string | null;
  } | null>(null);
  const [transientRunError, setRunError] = React.useState<string | null>(null);
  const runError = authoringState.error ?? transientRunError;
  const clearRunError = React.useCallback(() => {
    setRunError(null);
    dispatchAuthoringState({ kind: "clear_error" });
  }, [dispatchAuthoringState]);
  const dismissRunError = React.useCallback((message: string) => {
    setRunError((current) => current === message ? null : current);
    if (authoringState.error === message) {
      dispatchAuthoringState({ kind: "clear_error" });
    }
  }, [authoringState.error, dispatchAuthoringState]);
  const [pendingConnectionRoute, setPendingConnectionRoute] =
    React.useState<PendingConnectionRoute | null>(null);
  const [fitRevision, setFitRevision] = React.useState(0);
  const executionRunningRef = React.useRef(false);
  const isExecutionRunning = React.useCallback(
    () => executionRunningRef.current,
    [],
  );
  const pendingBoundEdgesRef = React.useRef<PendingBoundEdge[]>([]);

  const applyAuthoringCommands = React.useCallback(
    (commands: readonly GraphCommand[]) => {
      if (!commands.length) return;
      dispatchAuthoringState({ kind: "apply_commands", commands });
      setPositionOverrides({});
      setSelectedNodeIdSet((current) => new Set(
        [...current].filter((nodeId) =>
          authoredDocument.nodes.some((node) => node.id === nodeId),
        ),
      ));
      setSelectedEdgeIdSet((current) => new Set(
        [...current].filter((edgeId) =>
          authoredDocument.edges.some((edge) => edge.id === edgeId),
        ),
      ));
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [authoredDocument.edges, authoredDocument.nodes, dispatchAuthoringState],
  );

  React.useLayoutEffect(() => {
    authoredDocumentRef.current = authoredDocument;
  }, [authoredDocument]);

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
    const commands = ready.map((pending) => {
      const connection: Connection = {
        source: pending.edge.source,
        sourceHandle: pending.edge.sourceHandle ?? null,
        target: pending.edge.target,
        targetHandle: pending.edge.targetHandle ?? null,
      };
      return addEdgeCommand(connection, pending.edge.data, pending.edge.id);
    });
    applyAuthoringCommands(commands);
  }, [applyAuthoringCommands]);

  const updateConfig = React.useCallback(
    (nodeId: string, name: string, value: unknown) => {
      applyAuthoringCommands([{
        kind: "update_node_configuration",
        node_id: nodeId,
        field: name,
        value,
      }]);
    },
    [applyAuthoringCommands],
  );

  const updateLayout = React.useCallback(
    (nodeId: string, layout: WorkflowNodeData["layout"]) => {
      applyAuthoringCommands([{
        kind: "update_node_layout",
        node_id: nodeId,
        layout,
      }]);
    },
    [applyAuthoringCommands],
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
    applyAuthoringCommands([{ kind: "remove_nodes", node_ids: [nodeId] }]);
    setArtifactViewers((current) => ({
      ...current,
      edges: current.edges.filter((edge) => edge.source !== nodeId),
    }));
    forgetNodeSecretStatuses(nodeId);
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [applyAuthoringCommands, forgetNodeSecretStatuses]);

  const handleRemoveImageUpload = React.useCallback(
    (nodeId: string, index: number) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return;
      applyAuthoringCommands([{
        kind: "update_node_configuration",
        node_id: nodeId,
        field: "uploads",
        value: imageUploads(removeImageUpload(node.data, index)),
      }]);
    },
    [applyAuthoringCommands, nodes],
  );

  const addNodeInputPlug = React.useCallback(
    (nodeId: string, portName: string) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return;
      const inputPlugs = appendInputPlug(node.data.inputPlugs, portName);
      const plug = inputPlugs[inputPlugs.length - 1];
      if (!plug) return;
      applyAuthoringCommands([{
        kind: "add_input_plug",
        node_id: nodeId,
        plug: { id: plug.id, port: plug.portName },
      }]);
    },
    [applyAuthoringCommands, nodes],
  );

  const removeNodeInputPlug = React.useCallback(
    (nodeId: string, plugId: string) => {
      applyAuthoringCommands([{
        kind: "remove_input_plug",
        node_id: nodeId,
        plug_id: plugId,
      }]);
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [applyAuthoringCommands],
  );

  const reorderNodeInputPlug = React.useCallback(
    (
      nodeId: string,
      portName: string,
      plugId: string,
      toIndex: number,
    ) => {
      applyAuthoringCommands([{
        kind: "reorder_input_plug",
        node_id: nodeId,
        port: portName,
        plug_id: plugId,
        to_index: toIndex,
      }]);
    },
    [applyAuthoringCommands],
  );

  const updateSchemaBuilderFields = React.useCallback(
    (
      nodeId: string,
      fields: readonly SchemaBuilderField[],
      inputPlugs: readonly WorkflowInputPlug[],
    ) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return;
      applyAuthoringCommands([{
        kind: "update_node_configuration_and_input_plugs",
        node_id: nodeId,
        config: { ...node.data.config, fields },
        input_plugs: inputPlugs.map((plug) => ({
          id: plug.id,
          port: plug.portName,
        })),
      }]);
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [applyAuthoringCommands, nodes],
  );

  const updateArtifactQueryRelations = React.useCallback(
    (
      nodeId: string,
      relations: readonly ArtifactQueryRelation[],
      inputPlugs: readonly WorkflowInputPlug[],
    ) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return;
      applyAuthoringCommands([{
        kind: "update_node_configuration_and_input_plugs",
        node_id: nodeId,
        config: { ...node.data.config, relations },
        input_plugs: inputPlugs.map((plug) => ({
          id: plug.id,
          port: plug.portName,
        })),
      }]);
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [applyAuthoringCommands, nodes],
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
      const uploads = await Promise.all(
        files.map((file) => uploadFile(workspaceId, file)),
      );
      applyAuthoringCommands([{
        kind: "update_node_configuration",
        node_id: nodeId,
        field: "uploads",
        value: uploads,
      }]);
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "File upload failed";
      setNodes((current) => current.map((node) => node.id === nodeId ? {
        ...node,
        data: { ...node.data, execution: { status: "failed", error: message } },
      } : node));
    }
  }, [applyAuthoringCommands, edges, setNodes, workspaceId]);

  const resetNodeArtifactTypeBinding = React.useCallback(
    (nodeId: string, variable: string) => {
      const hasIncidentEdges = edges.some(
        (edge) => edge.source === nodeId || edge.target === nodeId,
      );
      if (hasIncidentEdges) return;

      applyAuthoringCommands([{
        kind: "reset_artifact_type_binding",
        node_id: nodeId,
        variable,
      }]);
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [applyAuthoringCommands, edges],
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

  const replaceDocument = React.useCallback((
    nextDocument: AuthoredGraphDocument,
    overlayNodes: readonly WorkflowNode[] = [],
  ) => {
    pendingBoundEdgesRef.current = [];
    authoredDocumentRef.current = nextDocument;
    nodesRef.current = [...overlayNodes];
    edgesRef.current = [];
    dispatchAuthoringState({
      kind: "replace_document",
      document: nextDocument,
      nodeOverlays: nodeOverlaysFromNodes(overlayNodes),
    });
    setSelectedNodeIdSet(new Set());
    setSelectedEdgeIdSet(new Set());
    setPositionOverrides({});
  }, [dispatchAuthoringState]);
  const updateDocumentName = React.useCallback((name: string) => {
    applyAuthoringCommands([{ kind: "rename_graph", name }]);
  }, [applyAuthoringCommands]);
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
        artifactViewerStorageKey(userId, workspaceId, graphId),
      );
      setArtifactViewerPersistenceError(null);
    } catch {
      setArtifactViewerPersistenceError(
        "Artifact Viewer layout could not be removed from browser storage.",
      );
    }
  }, [userId, workspaceId]);
  const uploading = nodes.some(
    (node) => node.data.execution.status === "uploading",
  );
  const roomPersistenceRef = React.useRef<GraphRoomPersistenceAdapter>({
    canPersist: false,
    persistDocument: async () => {
      throw new Error("Graph room is not ready.");
    },
  });
  const roomPersistence = React.useMemo<GraphRoomPersistenceAdapter>(() => ({
    get canPersist() {
      return roomPersistenceRef.current.canPersist;
    },
    persistDocument: (draft) =>
      roomPersistenceRef.current.persistDocument(draft),
  }), []);
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
    syncFromCollaborativeHead,
    purgeLocalGraphState,
    isGraphSnapshotCurrent,
  } = useSavedGraphLifecycle({
    workspaceId,
    workspaceSlug,
    initialGraphId,
    registry,
    document: authoredDocument,
    nodes,
    isExecutionRunning,
    uploading,
    replaceDocument,
    updateDocumentName,
    attachNodeCallbacks,
    refreshNodeSecretStatuses,
    clearGraphSecretStatuses,
    clearPendingConnectionRoute,
    clearRunError,
    closeNodeLibrary,
    requestCanvasRefit,
    refreshNodeRegistry: requestNodeRegistryRefresh,
    onGraphDeleted: removeArtifactViewerDocument,
    roomPersistence,
  });
  const router = useRouter();
  const isDirtyRef = React.useRef(isDirty);
  const activeGraphIdRef = React.useRef(activeGraph?.id ?? null);
  React.useEffect(() => {
    isDirtyRef.current = isDirty;
  }, [isDirty]);
  React.useEffect(() => {
    activeGraphIdRef.current = activeGraph?.id ?? null;
  }, [activeGraph?.id]);
  const graphRoom = useGraphRoomSession({
    workspaceId,
    graphId: activeGraph?.id ?? null,
    onReady: (ready) => {
      // Avoid clobbering in-progress local edits before the first room save.
      if (!isDirtyRef.current) {
        syncFromCollaborativeHead(ready.head);
      }
    },
    onRehydrate: (head) => {
      syncFromCollaborativeHead(head);
    },
    onTerminalClose: (reason) => {
      const graphId = activeGraphIdRef.current;
      if (reason === "access_revoked" || reason === "graph_deleted") {
        if (graphId) {
          removeArtifactViewerDocument(graphId);
        }
        purgeLocalGraphState();
        router.replace(`/workspaces/${encodeURIComponent(workspaceSlug)}`);
        return;
      }
      if (reason === "permissions_changed") {
        // Stopped traffic; leave remount/reload to the operator.
        // Protected caches are cleared so stale authority is not reused.
        if (graphId) {
          removeArtifactViewerDocument(graphId);
        }
        purgeLocalGraphState();
      }
    },
  });
  React.useEffect(() => {
    const graphId = activeGraph?.id;
    roomPersistenceRef.current = {
      canPersist: graphRoom.canSubmitCommands && graphId !== undefined,
      persistDocument: async (draft) => {
        if (!graphId) {
          throw new Error("Graph room requires a saved graph id.");
        }
        const command = {
          kind: "replace_document",
          name: draft.name,
          document: {
            schema_version: 3,
            nodes: draft.nodes ?? [],
            edges: draft.edges ?? [],
          },
        } as RoomGraphCommand;
        const { head } = await graphRoom.submitCommand(command);
        return {
          id: graphId,
          revision: head.checkpoint_revision,
          name: head.name,
          nodes: head.nodes ?? [],
          edges: head.edges ?? [],
        };
      },
    };
  }, [
    activeGraph?.id,
    graphRoom.canSubmitCommands,
    graphRoom.submitCommand,
  ]);
  const {
    running,
    runningScope,
    visibleExecution,
    announcement: executionAnnouncement,
    runWorkflow,
    cancelCurrentExecution,
  } = useRunExecution({
    workspaceId,
    registryAvailable: Boolean(registry),
    nodes,
    edges,
    activeGraph,
    currentFingerprint,
    isDirty,
    nodeSecretStatuses,
    roomActiveExecution: graphRoom.activeExecution,
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
            artifactViewerStorageKey(userId, workspaceId, graphId!),
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
          artifactViewerStorageKey(userId, workspaceId, graphId),
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
  }, [activeGraph?.id, artifactViewers.graphId, userId, workspaceId]);

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
          artifactViewerStorageKey(userId, workspaceId, graphId),
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
    userId,
    workspaceId,
  ]);

  const graphOperationBusy = persistenceOperationBusy || running;

  React.useEffect(() => {
    if (executionHistoryTarget) closeGraphBrowser();
  }, [closeGraphBrowser, executionHistoryTarget]);

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
  const presenceSelectionKey = selectedNodeIds.join("\0");
  React.useEffect(() => {
    if (!graphRoom.canPublishPresence) return;
    graphRoom.publishPresence({
      selected_node_ids: selectedNodeIds,
      activity: null,
      activity_target_ids: [],
      transient_node_positions: [],
    });
  }, [
    graphRoom.canPublishPresence,
    graphRoom.publishPresence,
    presenceSelectionKey,
    selectedNodeIds,
  ]);
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
      const rendererChanges = workflowChanges.filter(
        (change) => change.type === "add" || change.type === "replace",
      );
      if (rendererChanges.length) {
        setNodes((current) => applyNodeChanges(rendererChanges, current));
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
      const semanticChanges = graphCommandsFromNodeChanges(workflowChanges);
      const transientChanges = workflowChanges.filter(
        (change) =>
          change.type === "select" ||
          (change.type === "position" && change.dragging === true),
      );
      if (transientChanges.length) {
        setNodes((current) => applyNodeChanges(transientChanges, current));
      }
      if (semanticChanges.length) applyAuthoringCommands(semanticChanges);
    },
    [applyAuthoringCommands, artifactViewers.nodes, nodes, setNodes],
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
    [setNodes],
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
        const semanticChanges = graphCommandsFromEdgeChanges(workflowChanges);
        const transientChanges = workflowChanges.filter(
          (change) => change.type !== "remove",
        );
        if (transientChanges.length) {
          setEdges((current) => applyEdgeChanges(transientChanges, current));
        }
        if (semanticChanges.length) applyAuthoringCommands(semanticChanges);
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
      applyAuthoringCommands,
      edges,
      invalidateWorkflowResults,
      setEdges,
    ],
  );

  const updateEdge = React.useCallback(
    (edgeId: string, update: WorkflowEdgeUpdate) => {
      const changedEdge = edges.find((edge) => edge.id === edgeId);
      if (!changedEdge) return;
      applyAuthoringCommands([{
        kind: "update_edge",
        edge_id: edgeId,
        update: {
          enabled: update.enabled ?? changedEdge.data?.enabled ?? true,
          collection_mode:
            update.collectionMode ??
            changedEdge.data?.collectionMode ??
            "direct",
          projection: update.route
            ? update.route.projection
              ? { path: [...update.route.projection.path] }
              : null
            : changedEdge.data?.projection
              ? { path: [...changedEdge.data.projection.path] }
              : null,
          conversion_path: update.route
            ? update.route.conversionPath.map((conversion) => ({
                id: conversion.id,
                version: conversion.version,
              }))
            : (changedEdge.data?.conversionPath ?? []).map((conversion) => ({
                id: conversion.id,
                version: conversion.version,
              })),
        },
      }]);
    },
    [applyAuthoringCommands, edges],
  );

  const updateEdgeRoute = React.useCallback(
    (edgeId: string, routeOffset: WorkflowEdgeRouteOffset) => {
      applyAuthoringCommands([{
        kind: "update_edge",
        edge_id: edgeId,
        update: { route_offset: routeOffset },
      }]);
    },
    [applyAuthoringCommands],
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
      applyAuthoringCommands([{
        kind: "bind_artifact_type",
        node_id: bindingNodeId,
        variable: binding.variable,
        artifact_type: binding.artifactType,
      }]);
    } else {
      applyAuthoringCommands([
        addEdgeCommand(committedConnection, edge.data, edge.id),
      ]);
    }
    const changedNodeIds = binding?.endpoint === "source" && newlyBoundNodeId
      ? [newlyBoundNodeId, edge.target]
      : [edge.target];
    invalidateWorkflowResults(changedNodeIds, [...edges, edge]);
  }, [
    applyAuthoringCommands,
    edges,
    invalidateWorkflowResults,
    nodes,
  ]);

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
    const authoredNode = {
      id,
      operator_id: data.spec.operator_id,
      operator_version: data.spec.operator_version,
      config: structuredClone(data.config),
      input_plugs: data.inputPlugs.map((plug) => ({
        id: plug.id,
        port: plug.portName,
      })),
      artifact_type_bindings: Object.entries(data.artifactTypeBindings).map(
        ([variable, artifactType]) => ({ variable, artifact_type: artifactType }),
      ),
      position: { x: center.x - 140, y: center.y - 110 },
      layout: serializeNodeLayout(data.layout),
    };
    applyAuthoringCommands([{ kind: "add_node", node: authoredNode }]);
    setSelectedNodeIdSet(new Set([id]));
    setSelectedEdgeIdSet(new Set());
    setLibraryOpen(false);
  }, [applyAuthoringCommands, attachNodeCallbacks, flow]);

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
  }, [closeGraphBrowser, flow, nodes, setNodes]);

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
    const duplicatedNodes = duplicates.flatMap(({ node, id }) => {
      const authoredNode = authoredDocument.nodes.find(
        (candidate) => candidate.id === node.id,
      );
      return authoredNode
        ? [{
            ...structuredClone(authoredNode),
            id,
            position: { x: node.position.x + 36, y: node.position.y + 36 },
          }]
        : [];
    });
    const duplicatedEdges = authoredDocument.edges.flatMap((edge) => {
      const source = duplicatedNodeIds.get(edge.from_node);
      const target = duplicatedNodeIds.get(edge.to_node);
      if (!source || !target) return [];
      return [{
        ...structuredClone(edge),
        id: `edge-${crypto.randomUUID()}`,
        from_node: source,
        to_node: target,
      }];
    });
    const commands: GraphCommand[] = [
      ...duplicatedNodes.map((node) => ({
        kind: "add_node" as const,
        node,
      })),
      ...duplicatedEdges.map((edge) => ({
        kind: "add_edge" as const,
        edge,
      })),
    ];
    applyAuthoringCommands(commands);
    const duplicatedNodeIdSet = new Set(duplicatedNodes.map((node) => node.id));
    setSelectedNodeIdSet(duplicatedNodeIdSet);
    setSelectedEdgeIdSet(new Set());
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [
    applyAuthoringCommands,
    authoredDocument,
    nodes,
    running,
  ]);

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
            ...attachNodeCallbacks(node.data),
            historyContext: {
              workspaceId,
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
      attachNodeCallbacks,
      edges,
      isDirty,
      nodeSecretStatuses,
      nodes,
      registry,
      removeConfiguredNodeSecret,
      workspaceId,
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
              : (edgeId: string, update: WorkflowEdgeUpdate) =>
                  updateEdge(edgeId, update),
            onRouteOffsetChange: (edgeId: string, routeOffset: WorkflowEdgeRouteOffset) =>
              updateEdgeRoute(edgeId, routeOffset),
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

  const allCanvasNodes = React.useMemo<CanvasNode[]>(() => {
    const combined = [...canvasNodes, ...artifactViewerCanvasNodes];
    return combined.map((node) => {
      const remoteColor = remoteSelectionColor(
        graphRoom.participants,
        graphRoom.localSessionId,
        node.id,
      );
      if (!remoteColor) return node;
      return {
        ...node,
        className: [node.className, "ns-remote-selected"]
          .filter(Boolean)
          .join(" "),
        style: {
          ...node.style,
          boxShadow: `0 0 0 2px ${remoteColor}`,
        },
      };
    });
  }, [
    artifactViewerCanvasNodes,
    canvasNodes,
    graphRoom.localSessionId,
    graphRoom.participants,
  ]);
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
      <section
        {...stylex.props(s.canvas)}
        aria-label="Workflow canvas"
        onPointerMove={(event) => {
          if (!graphRoom.canPublishPresence || !flow) return;
          const position = flow.screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
          });
          graphRoom.publishPresence({
            cursor: position,
            selected_node_ids: selectedNodeIds,
          });
        }}
        onPointerLeave={() => {
          if (!graphRoom.canPublishPresence) return;
          graphRoom.publishPresence({
            cursor: null,
            selected_node_ids: selectedNodeIds,
          });
        }}
      >
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
          <PresenceOverlay
            participants={graphRoom.participants}
            localSessionId={graphRoom.localSessionId}
          />
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
          workspaceId={workspaceId}
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

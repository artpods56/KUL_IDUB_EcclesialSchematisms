"use client";

import * as React from "react";
import { useRouter } from "next/navigation";

import { useSavedGraphs } from "@/hooks/use-api";
import {
  createSavedGraph,
  deleteSavedGraph,
  getGraphMaterializations,
  getSavedGraph,
  updateSavedGraph,
  type CreateSavedGraphRequest,
  type NodeRegistry,
  type RunNodeResult,
  type SavedGraphNode,
  type SavedGraphSummary,
} from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import {
  hydrateSavedGraph,
  savedGraphDraft,
  savedGraphFingerprint,
} from "../canvas/saved-graph";
import type {
  WorkflowEdge,
  WorkflowNodeData,
} from "../canvas/types";
import type { WorkflowNode } from "../model/execution-plan";
import {
  NEW_GRAPH_ROUTE_ID,
  workbenchGraphPath,
} from "../routes";

export interface ActiveSavedGraph {
  id: string;
  revision: number;
  nodes: readonly SavedGraphNode[];
}

interface UseSavedGraphLifecycleOptions {
  workspaceSlug: string;
  initialGraphId: string | null;
  registry: NodeRegistry | undefined;
  nodes: readonly WorkflowNode[];
  edges: readonly WorkflowEdge[];
  isExecutionRunning: () => boolean;
  uploading: boolean;
  replaceCanvas: (
    nodes: WorkflowNode[],
    edges: WorkflowEdge[],
  ) => void;
  attachNodeCallbacks: (data: WorkflowNodeData) => WorkflowNodeData;
  refreshNodeSecretStatuses: (
    graph: ActiveSavedGraph,
    nodes: readonly WorkflowNode[],
    signal?: AbortSignal,
  ) => Promise<boolean>;
  clearGraphSecretStatuses: () => void;
  clearPendingConnectionRoute: () => void;
  clearRunError: () => void;
  closeNodeLibrary: () => void;
  requestCanvasRefit: () => void;
  refreshNodeRegistry: () => void | Promise<unknown>;
}

export interface UseSavedGraphLifecycleResult {
  activeGraph: ActiveSavedGraph | null;
  graphName: string;
  setGraphName: (name: string) => void;
  currentFingerprint: string;
  isDirty: boolean;
  saving: boolean;
  openingGraphId: string | null;
  deletingGraphId: string | null;
  persistenceError: string | null;
  clearPersistenceError: () => void;
  dismissPersistenceError: (message: string) => void;
  persistenceOperationBusy: boolean;
  graphBrowserOpen: boolean;
  toggleGraphBrowser: () => void;
  closeGraphBrowser: () => void;
  savedGraphs: readonly SavedGraphSummary[];
  savedGraphsLoading: boolean;
  savedGraphsRefreshing: boolean;
  savedGraphsError: string | null;
  refreshSavedGraphs: () => void;
  requestNewGraph: () => void;
  saveCurrentGraph: () => Promise<void>;
  openSavedGraph: (graphId: string) => Promise<void>;
  removeSavedGraph: (graph: SavedGraphSummary) => Promise<void>;
  isGraphSnapshotCurrent: (
    graph: ActiveSavedGraph | null,
    fingerprint: string,
  ) => boolean;
}

const NEW_GRAPH_NAME = "Untitled workflow";

export function useSavedGraphLifecycle({
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
  refreshNodeRegistry,
}: UseSavedGraphLifecycleOptions): UseSavedGraphLifecycleResult {
  const router = useRouter();
  const {
    data: savedGraphList,
    error: savedGraphListError,
    isLoading: savedGraphsLoading,
    isValidating: savedGraphsRefreshing,
    mutate: mutateSavedGraphs,
  } = useSavedGraphs();
  const [graphName, setGraphNameState] = React.useState(NEW_GRAPH_NAME);
  const [activeGraph, setActiveGraph] =
    React.useState<ActiveSavedGraph | null>(null);
  const [savedFingerprint, setSavedFingerprint] =
    React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [openingGraphId, setOpeningGraphId] = React.useState<string | null>(null);
  const [deletingGraphId, setDeletingGraphId] = React.useState<string | null>(null);
  const [persistenceError, setPersistenceError] = React.useState<string | null>(null);
  const [graphBrowserOpen, setGraphBrowserOpen] = React.useState(false);
  const approvedRouteGraphIdRef = React.useRef<string | null>(null);
  const openRequestRef = React.useRef<AbortController | null>(null);
  const currentFingerprintRef = React.useRef("");
  const activeGraphRef = React.useRef<ActiveSavedGraph | null>(null);

  const currentDraft = React.useMemo(
    () => savedGraphDraft(graphName, nodes, edges),
    [edges, graphName, nodes],
  );
  const currentFingerprint = React.useMemo(
    () => savedGraphFingerprint(currentDraft),
    [currentDraft],
  );

  React.useEffect(() => {
    currentFingerprintRef.current = currentFingerprint;
  }, [currentFingerprint]);

  React.useEffect(() => {
    activeGraphRef.current = activeGraph;
  }, [activeGraph]);

  const hasUnsavedDraft =
    nodes.length > 0 ||
    edges.length > 0 ||
    graphName.trim() !== NEW_GRAPH_NAME;
  const isDirty = activeGraph
    ? savedFingerprint !== currentFingerprint
    : hasUnsavedDraft;
  const persistenceOperationBusy = Boolean(
    saving || openingGraphId || deletingGraphId || uploading,
  );

  React.useEffect(() => {
    if (!isDirty) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [isDirty]);

  React.useEffect(() => () => {
    openRequestRef.current?.abort();
  }, []);

  const confirmDiscard = React.useCallback(
    (action: string): boolean =>
      !isDirty ||
      window.confirm(
        `“${graphName.trim() || NEW_GRAPH_NAME}” has unsaved changes. Discard them and ${action}?`,
      ),
    [graphName, isDirty],
  );

  const showBlankGraph = React.useCallback(() => {
    openRequestRef.current?.abort();
    replaceCanvas([], []);
    clearGraphSecretStatuses();
    setGraphNameState(NEW_GRAPH_NAME);
    activeGraphRef.current = null;
    setActiveGraph(null);
    setSavedFingerprint(null);
    clearPendingConnectionRoute();
    clearRunError();
    setPersistenceError(null);
    closeNodeLibrary();
    requestCanvasRefit();
  }, [
    clearGraphSecretStatuses,
    clearPendingConnectionRoute,
    clearRunError,
    closeNodeLibrary,
    replaceCanvas,
    requestCanvasRefit,
  ]);

  const requestNewGraph = React.useCallback(() => {
    if (!confirmDiscard("start a new graph")) return;
    setGraphBrowserOpen(false);
    const path = workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID);
    if (window.location.pathname === path) {
      showBlankGraph();
      return;
    }
    approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
    router.push(path, { scroll: false });
  }, [confirmDiscard, router, showBlankGraph, workspaceSlug]);

  const saveCurrentGraph = React.useCallback(async () => {
    if (
      isExecutionRunning() ||
      saving ||
      openingGraphId ||
      deletingGraphId
    ) return;
    if (!currentDraft.name) {
      setPersistenceError("Enter a graph name before saving.");
      return;
    }

    const submittedDraft = currentDraft;
    setSaving(true);
    setPersistenceError(null);
    try {
      const savedGraph = activeGraph
        ? await updateSavedGraph(activeGraph.id, {
            ...submittedDraft,
            expected_revision: activeGraph.revision,
          })
        : await createSavedGraph(submittedDraft);
      const responseDraft = {
        name: savedGraph.name,
        nodes: savedGraph.nodes ?? [],
        edges: savedGraph.edges ?? [],
      };
      const nextActiveGraph = {
        id: savedGraph.id,
        revision: savedGraph.revision,
        nodes: savedGraph.nodes ?? [],
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      setGraphNameState((current) =>
        current.trim() === submittedDraft.name ? savedGraph.name : current,
      );
      await refreshNodeSecretStatuses(nextActiveGraph, nodes);
      if (!activeGraph) {
        approvedRouteGraphIdRef.current = savedGraph.id;
        router.replace(
          workbenchGraphPath(workspaceSlug, savedGraph.id),
          { scroll: false },
        );
      }
      void mutateSavedGraphs();
      void refreshNodeRegistry();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setPersistenceError(
          "This graph changed in another session. Your canvas is unchanged; refresh the list before deciding whether to reopen it.",
        );
      } else {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be saved.",
        );
      }
    } finally {
      setSaving(false);
    }
  }, [
    activeGraph,
    currentDraft,
    deletingGraphId,
    isExecutionRunning,
    mutateSavedGraphs,
    nodes,
    openingGraphId,
    refreshNodeRegistry,
    refreshNodeSecretStatuses,
    router,
    saving,
    workspaceSlug,
  ]);

  const openSavedGraph = React.useCallback(async (
    graphId: string,
    confirmBeforeOpen = true,
    updateAddress = true,
  ) => {
    if (!registry) {
      setPersistenceError("The live node registry must load before a graph can open.");
      return;
    }
    if (confirmBeforeOpen && !confirmDiscard("open another graph")) return;
    if (updateAddress) {
      setGraphBrowserOpen(false);
      if (activeGraph?.id !== graphId) {
        approvedRouteGraphIdRef.current = graphId;
        router.push(
          workbenchGraphPath(workspaceSlug, graphId),
          { scroll: false },
        );
      }
      return;
    }

    const openingFingerprint = currentFingerprint;
    openRequestRef.current?.abort();
    const controller = new AbortController();
    openRequestRef.current = controller;
    setOpeningGraphId(graphId);
    setPersistenceError(null);
    try {
      const savedGraph = await getSavedGraph(graphId, controller.signal);
      let materializationWarning: string | null = null;
      let materializedNodeRuns: RunNodeResult[] = [];
      try {
        const materializations = await getGraphMaterializations(
          graphId,
          savedGraph.revision,
          controller.signal,
        );
        materializedNodeRuns = [...materializations.node_runs];
      } catch (error) {
        if (controller.signal.aborted) return;
        const message = error instanceof Error
          ? error.message
          : "Latest materialized outputs could not be loaded.";
        materializationWarning =
          `Graph opened without its latest materialized outputs: ${message}`;
      }
      const hydrated = hydrateSavedGraph(
        savedGraph,
        registry,
        materializedNodeRuns,
      );
      if (controller.signal.aborted) return;
      if (currentFingerprintRef.current !== openingFingerprint) {
        setPersistenceError(
          "The canvas changed while the graph was loading. Your newer edits were kept; open the graph again when you are ready to replace them.",
        );
        return;
      }

      const responseDraft = {
        name: savedGraph.name,
        nodes: savedGraph.nodes ?? [],
        edges: savedGraph.edges ?? [],
      };
      const openedNodes = hydrated.nodes.map((node) => ({
        ...node,
        data: attachNodeCallbacks(node.data),
      }));
      replaceCanvas(openedNodes, hydrated.edges);
      setGraphNameState(savedGraph.name);
      const nextActiveGraph = {
        id: savedGraph.id,
        revision: savedGraph.revision,
        nodes: savedGraph.nodes ?? [],
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      await refreshNodeSecretStatuses(
        nextActiveGraph,
        openedNodes,
        controller.signal,
      );
      clearPendingConnectionRoute();
      clearRunError();
      setPersistenceError(materializationWarning);
      closeNodeLibrary();
      setGraphBrowserOpen(false);
      requestCanvasRefit();
    } catch (error) {
      if (!controller.signal.aborted) {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be opened.",
        );
      }
    } finally {
      if (openRequestRef.current === controller) {
        openRequestRef.current = null;
        setOpeningGraphId(null);
      }
    }
  }, [
    activeGraph?.id,
    attachNodeCallbacks,
    clearPendingConnectionRoute,
    clearRunError,
    closeNodeLibrary,
    confirmDiscard,
    currentFingerprint,
    refreshNodeSecretStatuses,
    registry,
    replaceCanvas,
    requestCanvasRefit,
    router,
    workspaceSlug,
  ]);

  React.useEffect(() => {
    if (!registry) return;

    const routeGraphId = initialGraphId ?? NEW_GRAPH_ROUTE_ID;
    const displayedGraphId = activeGraph?.id ?? NEW_GRAPH_ROUTE_ID;
    if (routeGraphId === displayedGraphId) {
      if (approvedRouteGraphIdRef.current === routeGraphId) {
        approvedRouteGraphIdRef.current = null;
      }
      return;
    }

    if (
      approvedRouteGraphIdRef.current !== null &&
      approvedRouteGraphIdRef.current !== routeGraphId
    ) {
      return;
    }

    const explicitlyApproved =
      approvedRouteGraphIdRef.current === routeGraphId;
    approvedRouteGraphIdRef.current = null;
    if (!explicitlyApproved && !confirmDiscard("navigate with browser history")) {
      approvedRouteGraphIdRef.current = displayedGraphId;
      router.push(
        workbenchGraphPath(workspaceSlug, displayedGraphId),
        { scroll: false },
      );
      return;
    }

    if (!initialGraphId) {
      // The App Router retains this workbench so history navigation can be confirmed first.
      // Once accepted, the route change is the boundary that replaces the canvas draft.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      showBlankGraph();
      return;
    }

    void openSavedGraph(initialGraphId, false, false);
  }, [
    activeGraph?.id,
    confirmDiscard,
    initialGraphId,
    openSavedGraph,
    registry,
    router,
    showBlankGraph,
    workspaceSlug,
  ]);

  const removeSavedGraph = React.useCallback(async (
    graph: SavedGraphSummary,
  ) => {
    const deletingActiveGraph = activeGraph?.id === graph.id;
    const warning = deletingActiveGraph && isDirty
      ? `Delete “${graph.name}”? Its unsaved canvas changes will also be discarded.`
      : `Delete “${graph.name}”? This cannot be undone.`;
    if (!window.confirm(warning)) return;

    const expectedRevision = deletingActiveGraph
      ? activeGraph.revision
      : graph.revision;
    const deletingFingerprint = currentFingerprint;
    setDeletingGraphId(graph.id);
    setPersistenceError(null);
    try {
      await deleteSavedGraph(graph.id, expectedRevision);
      if (deletingActiveGraph) {
        if (currentFingerprintRef.current === deletingFingerprint) {
          showBlankGraph();
          approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
          router.replace(
            workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID),
            { scroll: false },
          );
        } else {
          activeGraphRef.current = null;
          setActiveGraph(null);
          setSavedFingerprint(null);
          clearGraphSecretStatuses();
          setPersistenceError(
            "The saved graph was deleted. Changes made while deletion was in progress remain as an unsaved draft.",
          );
          approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
          router.replace(
            workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID),
            { scroll: false },
          );
        }
      }
      void mutateSavedGraphs();
      void refreshNodeRegistry();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setPersistenceError(
          "This graph changed before it could be deleted. Refresh the saved graph list and try again.",
        );
      } else {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be deleted.",
        );
      }
    } finally {
      setDeletingGraphId(null);
    }
  }, [
    activeGraph,
    clearGraphSecretStatuses,
    currentFingerprint,
    isDirty,
    mutateSavedGraphs,
    refreshNodeRegistry,
    router,
    showBlankGraph,
    workspaceSlug,
  ]);

  const setGraphName = React.useCallback((name: string) => {
    setGraphNameState(name);
    setPersistenceError(null);
  }, []);
  const clearPersistenceError = React.useCallback(() => {
    setPersistenceError(null);
  }, []);
  const dismissPersistenceError = React.useCallback((message: string) => {
    setPersistenceError((current) => current === message ? null : current);
  }, []);
  const toggleGraphBrowser = React.useCallback(() => {
    closeNodeLibrary();
    setGraphBrowserOpen((open) => !open);
  }, [closeNodeLibrary]);
  const closeGraphBrowser = React.useCallback(() => {
    setGraphBrowserOpen(false);
  }, []);
  const refreshSavedGraphs = React.useCallback(() => {
    void mutateSavedGraphs();
  }, [mutateSavedGraphs]);
  const isGraphSnapshotCurrent = React.useCallback((
    graph: ActiveSavedGraph | null,
    fingerprint: string,
  ): boolean => {
    const currentGraph = activeGraphRef.current;
    return (
      currentFingerprintRef.current === fingerprint &&
      currentGraph?.id === graph?.id &&
      currentGraph?.revision === graph?.revision
    );
  }, []);

  const savedGraphsError = savedGraphListError instanceof Error
    ? savedGraphListError.message
    : savedGraphListError
      ? "Saved graphs are unavailable."
      : null;

  return {
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
    savedGraphs: savedGraphList?.graphs ?? [],
    savedGraphsLoading,
    savedGraphsRefreshing,
    savedGraphsError,
    refreshSavedGraphs,
    requestNewGraph,
    saveCurrentGraph,
    openSavedGraph,
    removeSavedGraph,
    isGraphSnapshotCurrent,
  };
}

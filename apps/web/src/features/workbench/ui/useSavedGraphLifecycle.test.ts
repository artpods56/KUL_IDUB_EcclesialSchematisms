// @vitest-environment jsdom

import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  NodeRegistry,
  SavedGraph,
  SavedGraphSummary,
} from "@/lib/api";
import { deferred } from "./test/deferred";
import { renderHook } from "./test/renderHook";
import { useSavedGraphLifecycle } from "./useSavedGraphLifecycle";

const router = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
}));

const savedGraphQuery = vi.hoisted(() => ({
  mutate: vi.fn(),
}));

const api = vi.hoisted(() => ({
  createSavedGraph: vi.fn(),
  deleteSavedGraph: vi.fn(),
  getGraphMaterializations: vi.fn(),
  getSavedGraph: vi.fn(),
  updateSavedGraph: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
}));

vi.mock("@/hooks/use-api", () => ({
  useSavedGraphs: () => ({
    data: { graphs: [] },
    error: undefined,
    isLoading: false,
    isValidating: false,
    mutate: savedGraphQuery.mutate,
  }),
}));

vi.mock("@/lib/api", () => api);

const GRAPH_A_ID = "00000000-0000-4000-8000-00000000000a";
const GRAPH_B_ID = "00000000-0000-4000-8000-00000000000b";

const registry: NodeRegistry = {
  plugins: [],
  artifact_types: [],
  artifact_conversions: [],
  nodes: [],
};

function savedGraph(
  id: string,
  name: string,
  revision = 1,
): SavedGraph {
  return {
    id,
    revision,
    name,
    created_at: "2026-07-17T12:00:00Z",
    updated_at: "2026-07-17T12:00:00Z",
    nodes: [],
    edges: [],
  };
}

function savedGraphSummary(graph: SavedGraph): SavedGraphSummary {
  return {
    id: graph.id,
    revision: graph.revision,
    name: graph.name,
    node_count: graph.nodes?.length ?? 0,
    edge_count: graph.edges?.length ?? 0,
    updated_at: graph.updated_at,
  };
}

type LifecycleOptions = Parameters<typeof useSavedGraphLifecycle>[0];

interface LifecycleCallbacks {
  replaceDocument: ReturnType<typeof vi.fn>;
  refreshNodeSecretStatuses: LifecycleOptions["refreshNodeSecretStatuses"];
  clearGraphSecretStatuses: ReturnType<typeof vi.fn>;
  clearPendingConnectionRoute: ReturnType<typeof vi.fn>;
  clearRunError: ReturnType<typeof vi.fn>;
  closeNodeLibrary: ReturnType<typeof vi.fn>;
  requestCanvasRefit: ReturnType<typeof vi.fn>;
  refreshNodeRegistry: ReturnType<typeof vi.fn>;
}

function lifecycleOptions(
  initialGraphId: string | null,
  refreshNodeSecretStatuses: LifecycleOptions["refreshNodeSecretStatuses"] =
    vi.fn().mockResolvedValue(true),
  initialDocument: LifecycleOptions["document"] = {
    name: "Untitled workflow",
    nodes: [],
    edges: [],
  },
): { options: LifecycleOptions; callbacks: LifecycleCallbacks } {
  const document = {
    name: initialDocument.name,
    nodes: initialDocument.nodes,
    edges: initialDocument.edges,
  };
  const callbacks = {
    replaceDocument: vi.fn((nextDocument: LifecycleOptions["document"]) => {
      document.name = nextDocument.name;
      document.nodes = nextDocument.nodes;
      document.edges = nextDocument.edges;
    }),
    refreshNodeSecretStatuses,
    clearGraphSecretStatuses: vi.fn(),
    clearPendingConnectionRoute: vi.fn(),
    clearRunError: vi.fn(),
    closeNodeLibrary: vi.fn(),
    requestCanvasRefit: vi.fn(),
    refreshNodeRegistry: vi.fn(),
  };
  const updateDocumentName = vi.fn((name: string) => {
    document.name = name;
  });
  return {
    options: {
      workspaceId: "workspace-1",
      workspaceSlug: "local",
      initialGraphId,
      registry,
      document,
      nodes: [],
      isExecutionRunning: () => false,
      uploading: false,
      replaceDocument: callbacks.replaceDocument,
      updateDocumentName,
      attachNodeCallbacks: (data) => data,
      refreshNodeSecretStatuses: callbacks.refreshNodeSecretStatuses,
      clearGraphSecretStatuses: callbacks.clearGraphSecretStatuses,
      clearPendingConnectionRoute: callbacks.clearPendingConnectionRoute,
      clearRunError: callbacks.clearRunError,
      closeNodeLibrary: callbacks.closeNodeLibrary,
      requestCanvasRefit: callbacks.requestCanvasRefit,
      refreshNodeRegistry: callbacks.refreshNodeRegistry,
    },
    callbacks,
  };
}

async function flushAsyncWork(): Promise<void> {
  await React.act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function waitFor(assertion: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (assertion()) return;
    await flushAsyncWork();
  }
  throw new Error("Hook did not reach the expected state");
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  api.getGraphMaterializations.mockImplementation(
    (_workspaceId, graphId, revision) =>
      Promise.resolve({
        graph_id: graphId,
        graph_revision: revision,
        node_runs: [],
      }),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useSavedGraphLifecycle document ownership", () => {
  it("submits the canonical final drag position", async () => {
    const finalPosition = { x: 240, y: 180 };
    const node = {
      id: "source",
      operator_id: "test.source",
      operator_version: 1,
      config: { label: "source" },
      input_plugs: [],
      artifact_type_bindings: [],
      position: finalPosition,
      layout: null,
    };
    const graph: SavedGraph = {
      ...savedGraph(GRAPH_A_ID, "Dragged graph", 3),
      nodes: [node],
    };
    const document = {
      name: "Dragged graph",
      nodes: graph.nodes ?? [],
      edges: [],
    };
    api.createSavedGraph.mockResolvedValue(graph);
    api.getSavedGraph.mockResolvedValue(graph);
    const { options, callbacks } = lifecycleOptions(null, undefined, document);
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await React.act(async () => {
      await hook.result.current.saveCurrentGraph();
    });

    expect(api.createSavedGraph).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        nodes: [expect.objectContaining({ position: finalPosition })],
      }),
    );

    expect(callbacks.replaceDocument).not.toHaveBeenCalled();
  });

  it("restores the canonical final drag position when opening", async () => {
    const finalPosition = { x: 240, y: 180 };
    const graph: SavedGraph = {
      ...savedGraph(GRAPH_A_ID, "Dragged graph", 3),
      nodes: [{
        id: "source",
        operator_id: "test.source",
        operator_version: 1,
        config: { label: "source" },
        input_plugs: [],
        artifact_type_bindings: [],
        position: finalPosition,
        layout: null,
      }],
    };
    api.getSavedGraph.mockResolvedValue(graph);
    const { options, callbacks } = lifecycleOptions(GRAPH_A_ID);
    await renderHook(useSavedGraphLifecycle, options);

    await waitFor(() => callbacks.replaceDocument.mock.calls.length === 1);

    const openedDocument = callbacks.replaceDocument.mock.calls[0]?.[0];
    expect(openedDocument.nodes[0]?.position).toEqual(finalPosition);
  });

  it("opens a graph containing an unavailable operator as a preserved placeholder", async () => {
    const graph: SavedGraph = {
      ...savedGraph(GRAPH_A_ID, "Legacy graph"),
      nodes: [
        {
          id: "legacy-node",
          operator_id: "legacy.operator",
          operator_version: 7,
          config: { preserved: true },
          position: { x: 20, y: 40 },
          input_plugs: [],
          artifact_type_bindings: [],
        },
      ],
    };
    api.getSavedGraph.mockResolvedValue(graph);
    const { options, callbacks } = lifecycleOptions(GRAPH_A_ID);
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_A_ID);

    expect(hook.result.current.persistenceError).toBeNull();
    expect(callbacks.replaceDocument).toHaveBeenCalledOnce();
    const openedDocument = callbacks.replaceDocument.mock.calls[0]?.[0];
    expect(openedDocument.nodes).toHaveLength(1);
    expect(openedDocument.nodes[0]).toMatchObject({
      operator_id: "legacy.operator",
      operator_version: 7,
    });
  });

  it("does not apply trailing UI mutations from an open superseded during secret refresh", async () => {
    const graphA = savedGraph(GRAPH_A_ID, "Graph A");
    const graphB = savedGraph(GRAPH_B_ID, "Graph B");
    const graphASecrets = deferred<boolean>();
    api.getSavedGraph.mockImplementation((_workspaceId, graphId) =>
      Promise.resolve(graphId === GRAPH_A_ID ? graphA : graphB),
    );
    const refreshSecrets = vi.fn((graph: { id: string }) =>
      graph.id === GRAPH_A_ID
        ? graphASecrets.promise
        : Promise.resolve(true),
    );
    const { options, callbacks } = lifecycleOptions(
      GRAPH_A_ID,
      refreshSecrets,
    );
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_A_ID);
    expect(refreshSecrets).toHaveBeenCalledTimes(1);

    await hook.rerender({ ...options, initialGraphId: GRAPH_B_ID });
    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_B_ID);
    await waitFor(() => hook.result.current.openingGraphId === null);

    await React.act(async () => {
      graphASecrets.resolve(true);
      await graphASecrets.promise;
    });

    expect(hook.result.current.graphName).toBe("Graph B");
    expect(callbacks.clearPendingConnectionRoute).toHaveBeenCalledTimes(1);
    expect(callbacks.clearRunError).toHaveBeenCalledTimes(1);
    expect(callbacks.closeNodeLibrary).toHaveBeenCalledTimes(1);
    expect(callbacks.requestCanvasRefit).toHaveBeenCalledTimes(1);
  });

  it("does not let a late create response replace a graph opened by navigation", async () => {
    const graphB = savedGraph(GRAPH_B_ID, "Graph B");
    const createResponse = deferred<SavedGraph>();
    api.createSavedGraph.mockReturnValue(createResponse.promise);
    api.getSavedGraph.mockResolvedValue(graphB);
    const { options } = lifecycleOptions(null);
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await React.act(async () => {
      hook.result.current.setGraphName("Local draft");
    });
    let savePromise!: Promise<void>;
    await React.act(async () => {
      savePromise = hook.result.current.saveCurrentGraph();
      await Promise.resolve();
    });
    await hook.rerender({ ...options, initialGraphId: GRAPH_B_ID });
    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_B_ID);

    await React.act(async () => {
      createResponse.resolve(savedGraph(GRAPH_A_ID, "Local draft"));
      await savePromise;
    });

    expect(hook.result.current.activeGraph?.id).toBe(GRAPH_B_ID);
    expect(hook.result.current.graphName).toBe("Graph B");
    expect(router.replace).not.toHaveBeenCalled();
    expect(savedGraphQuery.mutate).toHaveBeenCalledOnce();
    expect(options.refreshNodeRegistry).toHaveBeenCalledOnce();
  });

  it("does not let a late active-delete response clear a graph opened by navigation", async () => {
    const graphA = savedGraph(GRAPH_A_ID, "Graph A");
    const graphB = savedGraph(GRAPH_B_ID, "Graph B");
    const deleteResponse = deferred<void>();
    api.getSavedGraph.mockImplementation((_workspaceId, graphId) =>
      Promise.resolve(graphId === GRAPH_A_ID ? graphA : graphB),
    );
    api.deleteSavedGraph.mockReturnValue(deleteResponse.promise);
    const { options, callbacks } = lifecycleOptions(GRAPH_A_ID);
    const hook = await renderHook(useSavedGraphLifecycle, options);
    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_A_ID);

    let deletePromise!: Promise<void>;
    await React.act(async () => {
      deletePromise = hook.result.current.removeSavedGraph(
        savedGraphSummary(graphA),
      );
      await Promise.resolve();
    });
    await hook.rerender({ ...options, initialGraphId: GRAPH_B_ID });
    await waitFor(() => hook.result.current.activeGraph?.id === GRAPH_B_ID);
    const canvasReplacementCount = callbacks.replaceDocument.mock.calls.length;

    await React.act(async () => {
      deleteResponse.resolve();
      await deletePromise;
    });

    expect(hook.result.current.activeGraph?.id).toBe(GRAPH_B_ID);
    expect(hook.result.current.graphName).toBe("Graph B");
    expect(router.replace).not.toHaveBeenCalled();
    expect(callbacks.clearGraphSecretStatuses).not.toHaveBeenCalled();
    expect(callbacks.replaceDocument).toHaveBeenCalledTimes(
      canvasReplacementCount,
    );
    expect(savedGraphQuery.mutate).toHaveBeenCalledOnce();
    expect(callbacks.refreshNodeRegistry).toHaveBeenCalledOnce();
  });

  it("aborts an in-flight open and skips its trailing mutations on unmount", async () => {
    const graphA = savedGraph(GRAPH_A_ID, "Graph A");
    const secretRefresh = deferred<boolean>();
    let requestSignal: AbortSignal | undefined;
    api.getSavedGraph.mockImplementation((_workspaceId, _graphId, signal) => {
      requestSignal = signal;
      return Promise.resolve(graphA);
    });
    const refreshSecrets = vi.fn(() => secretRefresh.promise);
    const { options, callbacks } = lifecycleOptions(
      GRAPH_A_ID,
      refreshSecrets,
    );
    const hook = await renderHook(useSavedGraphLifecycle, options);
    await waitFor(() => refreshSecrets.mock.calls.length === 1);

    await hook.unmount();
    expect(requestSignal?.aborted).toBe(true);

    secretRefresh.resolve(true);
    await secretRefresh.promise;
    await flushAsyncWork();

    expect(callbacks.clearPendingConnectionRoute).not.toHaveBeenCalled();
    expect(callbacks.clearRunError).not.toHaveBeenCalled();
    expect(callbacks.closeNodeLibrary).not.toHaveBeenCalled();
    expect(callbacks.requestCanvasRefit).not.toHaveBeenCalled();
  });

  it("accepts a same-document save revision while keeping newer edits dirty", async () => {
    const createResponse = deferred<SavedGraph>();
    api.createSavedGraph.mockReturnValue(createResponse.promise);
    const { options } = lifecycleOptions(null);
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await React.act(async () => {
      hook.result.current.setGraphName("Submitted name");
    });
    let savePromise!: Promise<void>;
    await React.act(async () => {
      savePromise = hook.result.current.saveCurrentGraph();
      await Promise.resolve();
    });
    await React.act(async () => {
      hook.result.current.setGraphName("Newer local name");
    });

    await React.act(async () => {
      createResponse.resolve(savedGraph(GRAPH_A_ID, "Submitted name", 4));
      await savePromise;
    });

    expect(hook.result.current.activeGraph).toMatchObject({
      id: GRAPH_A_ID,
      revision: 4,
    });
    expect(hook.result.current.graphName).toBe("Newer local name");
    expect(hook.result.current.isDirty).toBe(true);
    expect(router.replace).toHaveBeenCalledWith(
      `/workspaces/local/graphs/${GRAPH_A_ID}`,
      { scroll: false },
    );
  });

  it("routes a newly created graph before secret refresh finishes", async () => {
    const secretRefresh = deferred<boolean>();
    api.createSavedGraph.mockResolvedValue(
      savedGraph(GRAPH_A_ID, "Created graph"),
    );
    const refreshSecrets = vi.fn(() => secretRefresh.promise);
    const { options, callbacks } = lifecycleOptions(null, refreshSecrets);
    const hook = await renderHook(useSavedGraphLifecycle, options);

    await React.act(async () => {
      hook.result.current.setGraphName("Created graph");
    });
    let savePromise!: Promise<void>;
    await React.act(async () => {
      savePromise = hook.result.current.saveCurrentGraph();
      await Promise.resolve();
    });
    await waitFor(() => refreshSecrets.mock.calls.length === 1);
    await flushAsyncWork();

    expect(hook.result.current.activeGraph?.id).toBe(GRAPH_A_ID);
    expect(router.replace).toHaveBeenCalledWith(
      `/workspaces/local/graphs/${GRAPH_A_ID}`,
      { scroll: false },
    );
    expect(callbacks.replaceDocument).not.toHaveBeenCalled();
    const routeOrder = router.replace.mock.invocationCallOrder.at(0);
    const secretRefreshOrder = refreshSecrets.mock.invocationCallOrder.at(0);
    if (routeOrder === undefined || secretRefreshOrder === undefined) {
      throw new Error("Expected routing and secret refresh to both run");
    }
    expect(routeOrder).toBeLessThan(secretRefreshOrder);

    await React.act(async () => {
      secretRefresh.resolve(true);
      await savePromise;
    });
  });
});

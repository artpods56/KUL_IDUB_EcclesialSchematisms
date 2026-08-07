// @vitest-environment jsdom

import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  NodeSpec,
  RunExecution,
  RunExecutionEventHandlers,
  RunExecutionEventSubscription,
  RunExecutionNodeProgressEvent,
  RunExecutionStatusEvent,
  RunRequest,
} from "@/lib/api";
import {
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
} from "../canvas/types";
import type { WorkflowNode } from "../model/execution-plan";
import { deferred } from "./test/deferred";
import { renderHook } from "./test/renderHook";
import { useRunExecution } from "./useRunExecution";

const apiMocks = vi.hoisted(() => ({
  cancelRunExecution: vi.fn<(
    workspaceId: string,
    executionId: string,
  ) => Promise<RunExecution>>(),
  getGraphMaterializations: vi.fn(),
  getRunExecution: vi.fn<(
    workspaceId: string,
    executionId: string,
  ) => Promise<RunExecution>>(),
  startRunExecution: vi.fn<(
    workspaceId: string,
    request: RunRequest,
  ) => Promise<RunExecution>>(),
  subscribeRunExecutionEvents: vi.fn<(
    workspaceId: string,
    executionId: string,
    handlers: RunExecutionEventHandlers,
  ) => RunExecutionEventSubscription>(),
}));

vi.mock("@/lib/api", () => apiMocks);

const executionId = "00000000-0000-4000-8000-000000000001";
const liveSubscriptions: Array<{
  executionId: string;
  handlers: RunExecutionEventHandlers;
  subscription: RunExecutionEventSubscription & { close: ReturnType<typeof vi.fn> };
}> = [];

function nodeSpec(): NodeSpec {
  return {
    operator_id: "test.operator",
    operator_version: 1,
    plugin_slug: "test",
    title: "Test operator",
    description: "Test operator",
    catalog_visible: true,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function workflowNode(id = "node-1"): WorkflowNode {
  return {
    id,
    type: WORKFLOW_NODE_TYPE,
    position: { x: 0, y: 0 },
    data: createWorkflowNodeData(nodeSpec()),
  };
}

function execution(
  status: RunExecution["status"],
  overrides: Partial<RunExecution> = {},
): RunExecution {
  return {
    execution_id: executionId,
    status,
    active_node_id: status === "running" || status === "cancelling"
      ? "node-1"
      : null,
    result: null,
    error: null,
    ...overrides,
  };
}

function succeededExecution(nodeId = "node-1"): RunExecution {
  return execution("succeeded", {
    result: {
      status: "succeeded",
      node_runs: [{
        node_id: nodeId,
        status: "succeeded",
        outputs: [],
        error: null,
      }],
    },
  });
}

type HookOptions = Parameters<typeof useRunExecution>[0];

function hookHarness(
  options: Partial<HookOptions> = {},
) {
  let nodes = options.nodes ? [...options.nodes] : [workflowNode()];
  let runError: string | null = null;
  let setNodesCallCount = 0;
  const setNodes: HookOptions["setNodes"] = (update) => {
    setNodesCallCount += 1;
    nodes = typeof update === "function" ? update(nodes) : update;
  };
  const setRunError = vi.fn((message: string | null) => {
    runError = message;
  });
  const hookOptions: HookOptions = {
    workspaceId: "workspace-1",
    registryAvailable: true,
    nodes,
    edges: [],
    activeGraph: null,
    currentFingerprint: "fingerprint-1",
    isDirty: true,
    nodeSecretStatuses: {},
    setNodes,
    setRunError,
    isGraphSnapshotCurrent: () => true,
    onMaterializationsLoaded: vi.fn(),
    ...options,
  };

  return {
    hookOptions,
    nodes: () => nodes,
    runError: () => runError,
    setNodesCallCount: () => setNodesCallCount,
  };
}

async function launchRun(
  result: Awaited<ReturnType<typeof renderHook<HookOptions, ReturnType<typeof useRunExecution>>>>["result"],
) {
  let runPromise!: Promise<void>;
  await React.act(async () => {
    runPromise = result.current.runWorkflow("all");
    await Promise.resolve();
  });
  return { runPromise };
}

function latestLiveSubscription() {
  const subscription = liveSubscriptions.at(-1);
  if (!subscription) throw new Error("No live execution subscription opened");
  return subscription;
}

function nodeProgressEvent(
  sequence: number,
  overrides: Partial<RunExecutionNodeProgressEvent> = {},
): RunExecutionNodeProgressEvent {
  return {
    kind: "node.progress",
    sequence,
    execution_id: executionId,
    occurred_at: "2026-07-19T12:00:00Z",
    node_path: ["node-1"],
    node_id: "node-1",
    node_run_id: null,
    invocation_index: null,
    invocation_path: [],
    message: `Update ${sequence}`,
    current: null,
    total: null,
    ...overrides,
  };
}

function executionStatusEvent(
  sequence: number,
  status: RunExecution["status"],
  activeNodeId: string | null = status === "running" ? "node-1" : null,
): RunExecutionStatusEvent {
  return {
    kind: "execution.status",
    sequence,
    execution_id: executionId,
    occurred_at: "2026-07-19T12:00:00Z",
    status,
    active_node_id: activeNodeId,
  };
}

describe("useRunExecution", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    apiMocks.subscribeRunExecutionEvents.mockImplementation(
      (_workspaceId, subscribedExecutionId, handlers) => {
        const subscription = { close: vi.fn() };
        liveSubscriptions.push({
          executionId: subscribedExecutionId,
          handlers,
          subscription,
        });
        return subscription;
      },
    );
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    liveSubscriptions.length = 0;
  });

  it("applies queued, running, and successful states from the execution API", async () => {
    const started = deferred<RunExecution>();
    apiMocks.startRunExecution.mockReturnValue(started.promise);
    apiMocks.getRunExecution
      .mockResolvedValueOnce(execution("running"))
      .mockResolvedValueOnce(succeededExecution());
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    const { runPromise } = await launchRun(hook.result);

    expect(apiMocks.startRunExecution).toHaveBeenCalledOnce();
    expect(hook.result.current.visibleExecution?.status).toBe("preparing");
    expect(harness.nodes()[0].data.execution.status).toBe("queued");

    await React.act(async () => {
      started.resolve(execution("queued"));
      await Promise.resolve();
    });
    expect(hook.result.current.visibleExecution?.status).toBe("queued");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(hook.result.current.visibleExecution?.status).toBe("running");
    expect(harness.nodes()[0].data.execution.status).toBe("running");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
      await runPromise;
    });
    expect(hook.result.current.running).toBe(false);
    expect(harness.nodes()[0].data.execution.status).toBe("succeeded");
    expect(harness.nodes()[0].data.run?.node_id).toBe("node-1");
    expect(hook.result.current.announcement).toBe(
      "Execution completed successfully.",
    );
  });

  it("does not regress a terminal SSE status when an older poll resolves", async () => {
    const stalePoll = deferred<RunExecution>();
    const terminalPoll = deferred<RunExecution>();
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution
      .mockReturnValueOnce(stalePoll.promise)
      .mockReturnValueOnce(terminalPoll.promise);
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(apiMocks.getRunExecution).toHaveBeenCalledTimes(1);

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(1, "succeeded"));
    });
    expect(hook.result.current.visibleExecution?.status).toBe("succeeded");

    await React.act(async () => {
      stalePoll.resolve(execution("running", { active_node_id: null }));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(apiMocks.getRunExecution).toHaveBeenCalledTimes(2);
    expect(hook.result.current.visibleExecution?.status).toBe("succeeded");
    expect(harness.nodes()[0]?.data.execution.status).toBe("running");

    await React.act(async () => {
      terminalPoll.resolve(succeededExecution());
      await runPromise;
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");
  });

  it("routes replayed module progress while guarding identity, sequence, and graph snapshot", async () => {
    let snapshotCurrent = true;
    apiMocks.startRunExecution.mockResolvedValue(execution("running", {
      active_node_id: "module-1",
    }));
    apiMocks.getRunExecution.mockResolvedValue(succeededExecution("module-1"));
    const harness = hookHarness({
      nodes: [workflowNode("module-1")],
      isGraphSnapshotCurrent: () => snapshotCurrent,
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      live.handlers.onEvent(nodeProgressEvent(10, {
        execution_id: "another-execution",
        node_path: ["module-1", "inner-1"],
        node_id: "inner-1",
      }));
      live.handlers.onEvent(nodeProgressEvent(2, {
        node_path: ["module-1", "nested-a", "inner-1"],
        node_id: "inner-1",
        invocation_index: 3,
        invocation_path: [2, 1],
        message: "Preparing the payload",
        current: 2,
        total: 5,
      }));
      live.handlers.onEvent(nodeProgressEvent(2, {
        node_path: ["module-1", "nested-a", "inner-1"],
        node_id: "inner-1",
        message: "Duplicate",
      }));
      live.handlers.onEvent(nodeProgressEvent(1, {
        node_path: ["module-1"],
        message: "Out of order",
      }));
      await vi.advanceTimersByTimeAsync(20);
    });

    expect(live.executionId).toBe(executionId);
    expect(harness.nodes()[0]?.data.progress).toEqual({
      omittedCount: 0,
      entries: [{
        sequence: 2,
        message: "Preparing the payload",
        current: 2,
        total: 5,
        sourceNodePath: ["nested-a", "inner-1"],
        invocationIndex: 3,
        invocationPath: [2, 1],
      }],
    });

    await React.act(async () => {
      live.handlers.onEvent(nodeProgressEvent(3, {
        node_path: ["module-1"],
        message: "Stale graph",
      }));
      snapshotCurrent = false;
      await vi.advanceTimersByTimeAsync(20);
    });
    expect(harness.nodes()[0]?.data.progress).toEqual({
      omittedCount: 0,
      entries: [{
        sequence: 2,
        message: "Preparing the payload",
        current: 2,
        total: 5,
        sourceNodePath: ["nested-a", "inner-1"],
        invocationIndex: 3,
        invocationPath: [2, 1],
      }],
    });

    snapshotCurrent = true;
    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(4, "succeeded"));
      await runPromise;
    });
    expect(apiMocks.getRunExecution).toHaveBeenCalledWith("workspace-1", executionId);
    expect(live.subscription.close).toHaveBeenCalled();
    expect(harness.nodes()[0]?.data.progress?.entries).toHaveLength(1);
  });

  it("batches progress into one node update per animation frame", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution.mockResolvedValue(succeededExecution());
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();
    const callsBeforeProgress = harness.setNodesCallCount();

    await React.act(async () => {
      live.handlers.onEvent(nodeProgressEvent(1));
      live.handlers.onEvent(nodeProgressEvent(2));
      live.handlers.onEvent(nodeProgressEvent(3));
    });

    expect(harness.nodes()[0]?.data.progress).toBeNull();
    expect(harness.setNodesCallCount()).toBe(callsBeforeProgress);

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(20);
    });

    expect(harness.setNodesCallCount()).toBe(callsBeforeProgress + 1);
    expect(
      harness.nodes()[0]?.data.progress?.entries.map((entry) => entry.sequence),
    ).toEqual([1, 2, 3]);

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(4, "succeeded"));
      await runPromise;
    });
  });

  it("caps pending progress per outer node before an animation frame", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution.mockResolvedValue(execution("cancelled"));
    const harness = hookHarness({
      nodes: [workflowNode("node-1"), workflowNode("node-2")],
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();
    const callsBeforeProgress = harness.setNodesCallCount();

    await React.act(async () => {
      for (let sequence = 1; sequence <= 100; sequence += 1) {
        live.handlers.onEvent(nodeProgressEvent(sequence));
      }
      for (let sequence = 101; sequence <= 200; sequence += 1) {
        live.handlers.onEvent(nodeProgressEvent(sequence, {
          node_path: ["node-2"],
          node_id: "node-2",
        }));
      }
    });
    expect(harness.nodes()[0]?.data.progress).toBeNull();
    expect(harness.nodes()[1]?.data.progress).toBeNull();
    expect(harness.setNodesCallCount()).toBe(callsBeforeProgress);

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(20);
    });

    expect(harness.setNodesCallCount()).toBe(callsBeforeProgress + 1);
    expect(harness.nodes()[0]?.data.progress?.omittedCount).toBe(60);
    expect(harness.nodes()[0]?.data.progress?.entries).toHaveLength(40);
    expect(harness.nodes()[0]?.data.progress?.entries[0]?.sequence).toBe(61);
    expect(harness.nodes()[0]?.data.progress?.entries.at(-1)?.sequence).toBe(
      100,
    );
    expect(harness.nodes()[1]?.data.progress?.omittedCount).toBe(60);
    expect(harness.nodes()[1]?.data.progress?.entries).toHaveLength(40);
    expect(harness.nodes()[1]?.data.progress?.entries[0]?.sequence).toBe(161);
    expect(harness.nodes()[1]?.data.progress?.entries.at(-1)?.sequence).toBe(
      200,
    );

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(201, "cancelled"));
      await runPromise;
    });
  });

  it("keeps nested node statuses detail-only until the outer module reports", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running", {
      active_node_id: "module-1",
    }));
    apiMocks.getRunExecution.mockResolvedValue(succeededExecution("module-1"));
    const harness = hookHarness({ nodes: [workflowNode("module-1")] });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      live.handlers.onEvent({
        kind: "node.status",
        sequence: 1,
        execution_id: executionId,
        occurred_at: "2026-07-19T12:00:00Z",
        node_path: ["module-1", "inner-1"],
        node_id: "inner-1",
        node_run_id: "inner-run",
        invocation_index: null,
        invocation_path: [],
        status: "succeeded",
      });
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("running");

    await React.act(async () => {
      live.handlers.onEvent({
        kind: "node.status",
        sequence: 2,
        execution_id: executionId,
        occurred_at: "2026-07-19T12:00:01Z",
        node_path: ["module-1"],
        node_id: "module-1",
        node_run_id: "module-run",
        invocation_index: null,
        invocation_path: [],
        status: "succeeded",
      });
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(3, "succeeded"));
      await runPromise;
    });
  });

  it("does not regress terminal nodes while later nodes run or cancellation completes", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution
      .mockResolvedValueOnce(execution("running", {
        active_node_id: "node-2",
      }))
      .mockResolvedValueOnce(execution("cancelled"));
    const harness = hookHarness({
      nodes: [workflowNode("node-1"), workflowNode("node-2")],
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      live.handlers.onEvent({
        kind: "node.status",
        sequence: 1,
        execution_id: executionId,
        occurred_at: "2026-07-19T12:00:00Z",
        node_path: ["node-1"],
        node_id: "node-1",
        node_run_id: "node-1-run",
        invocation_index: null,
        invocation_path: [],
        status: "succeeded",
      });
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");

    await React.act(async () => {
      live.handlers.onEvent({
        kind: "node.status",
        sequence: 2,
        execution_id: executionId,
        occurred_at: "2026-07-19T12:00:01Z",
        node_path: ["node-1"],
        node_id: "node-1",
        node_run_id: "node-1-run",
        invocation_index: null,
        invocation_path: [],
        status: "running",
      });
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");
    expect(harness.nodes()[1]?.data.execution.status).toBe("running");

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(3, "running", "node-2"));
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");
    expect(harness.nodes()[1]?.data.execution.status).toBe("running");

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(4, "cancelled"));
      await runPromise;
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");
    expect(harness.nodes()[1]?.data.execution.status).toBe("cancelled");
  });

  it("bounds progress per node, reports omissions, and clears it for the next run", async () => {
    apiMocks.startRunExecution
      .mockResolvedValueOnce(execution("running"))
      .mockResolvedValueOnce(succeededExecution());
    apiMocks.getRunExecution.mockResolvedValue(succeededExecution());
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      for (let sequence = 1; sequence <= 35; sequence += 1) {
        live.handlers.onEvent(nodeProgressEvent(sequence));
      }
      await vi.advanceTimersByTimeAsync(20);
    });
    expect(harness.nodes()[0]?.data.progress?.entries).toHaveLength(35);

    await React.act(async () => {
      for (let sequence = 36; sequence <= 45; sequence += 1) {
        live.handlers.onEvent(nodeProgressEvent(sequence));
      }
      live.handlers.onEvent(executionStatusEvent(46, "succeeded"));
      await runPromise;
    });

    expect(harness.nodes()[0]?.data.progress?.omittedCount).toBe(5);
    expect(harness.nodes()[0]?.data.progress?.entries).toHaveLength(40);
    expect(harness.nodes()[0]?.data.progress?.entries[0]?.message).toBe(
      "Update 6",
    );
    expect(harness.nodes()[0]?.data.progress?.entries.at(-1)?.message).toBe(
      "Update 45",
    );

    await React.act(async () => {
      await hook.result.current.runWorkflow("all");
    });
    expect(harness.nodes()[0]?.data.progress).toBeNull();
  });

  it("clears stale progress across the canvas without resetting unselected results", async () => {
    apiMocks.startRunExecution.mockResolvedValue(succeededExecution());
    const selected = { ...workflowNode(), selected: true };
    const unselected = workflowNode("node-2");
    unselected.data.run = {
      node_id: "node-2",
      status: "succeeded",
      outputs: [],
      error: null,
    };
    unselected.data.execution = { status: "succeeded" };
    unselected.data.progress = {
      omittedCount: 0,
      entries: [{
        sequence: 7,
        message: "Previous execution",
        current: null,
        total: null,
        sourceNodePath: [],
        invocationIndex: null,
        invocationPath: [],
      }],
    };
    const previousRun = unselected.data.run;
    const harness = hookHarness({ nodes: [selected, unselected] });
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    await React.act(async () => {
      await hook.result.current.runWorkflow("selected");
    });

    expect(harness.nodes()[1]?.data.progress).toBeNull();
    expect(harness.nodes()[1]?.data.run).toBe(previousRun);
    expect(harness.nodes()[1]?.data.execution).toEqual({
      status: "succeeded",
    });
  });

  it("falls back to polling when live progress disconnects without cancelling", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution.mockResolvedValue(succeededExecution());
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);

    await React.act(async () => {
      latestLiveSubscription().handlers.onError(new Event("error"));
    });
    expect(hook.result.current.visibleExecution?.statusError).toBe(
      "Live progress disconnected. Status polling continues.",
    );
    expect(apiMocks.cancelRunExecution).not.toHaveBeenCalled();

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
      await runPromise;
    });
    expect(harness.nodes()[0]?.data.execution.status).toBe("succeeded");
    expect(apiMocks.cancelRunExecution).not.toHaveBeenCalled();
  });

  it("closes live progress on unmount without cancelling the execution", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    await React.act(async () => {
      live.handlers.onEvent(nodeProgressEvent(1));
    });
    expect(harness.nodes()[0]?.data.progress).toBeNull();

    await hook.unmount();
    await runPromise;
    await vi.advanceTimersByTimeAsync(20);

    expect(live.subscription.close).toHaveBeenCalled();
    expect(apiMocks.cancelRunExecution).not.toHaveBeenCalled();
    expect(harness.nodes()[0]?.data.progress).toBeNull();
  });

  it("reserves a run before React can publish the running state", async () => {
    const started = deferred<RunExecution>();
    apiMocks.startRunExecution.mockReturnValue(started.promise);
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    let firstRun!: Promise<void>;
    let duplicateRun!: Promise<void>;

    await React.act(async () => {
      firstRun = hook.result.current.runWorkflow("all");
      duplicateRun = hook.result.current.runWorkflow("all");
      await Promise.resolve();
    });

    expect(apiMocks.startRunExecution).toHaveBeenCalledOnce();
    await React.act(async () => {
      started.resolve(succeededExecution());
      await Promise.all([firstRun, duplicateRun]);
    });
    expect(harness.nodes()[0].data.execution.status).toBe("succeeded");
  });

  it("records the user scope on the submitted execution", async () => {
    apiMocks.startRunExecution.mockResolvedValue(succeededExecution());
    const selectedNode = { ...workflowNode(), selected: true };
    const harness = hookHarness({ nodes: [selectedNode] });
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    await React.act(async () => {
      await hook.result.current.runWorkflow("selected-with-dependencies");
    });

    expect(apiMocks.startRunExecution).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        scope: "selected-with-dependencies",
      }),
    );
  });

  it("rejects an unsupported selected node before loading materializations", async () => {
    const unsupported = { ...workflowNode(), selected: true };
    unsupported.data.compatibility = {
      status: "unsupported",
      issues: ["Operator test.operator@1 is unavailable."],
      inputs: [],
      outputs: [],
      persistedNode: {
        id: "node-1",
        operator_id: "test.operator",
        operator_version: 1,
        config: {},
        position: { x: 0, y: 0 },
        input_plugs: [],
        artifact_type_bindings: [],
      },
    };
    const harness = hookHarness({
      nodes: [unsupported],
      activeGraph: {
        id: "graph-1",
        revision: 7,
        nodes: [],
      },
      isDirty: false,
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    await React.act(async () => {
      await hook.result.current.runWorkflow("selected");
    });

    expect(apiMocks.getGraphMaterializations).not.toHaveBeenCalled();
    expect(apiMocks.startRunExecution).not.toHaveBeenCalled();
    expect(harness.nodes()[0]?.data.execution).toEqual({
      status: "failed",
      error:
        "Cannot run Test operator: Operator test.operator@1 is unavailable.",
    });
  });

  it("records clean saved executions against the saved graph revision", async () => {
    apiMocks.startRunExecution.mockResolvedValue(succeededExecution());
    const harness = hookHarness({
      activeGraph: {
        id: "graph-1",
        revision: 7,
        nodes: [],
      },
      isDirty: false,
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    await React.act(async () => {
      await hook.result.current.runWorkflow("all");
    });

    expect(apiMocks.startRunExecution).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        graph_id: "graph-1",
        graph_revision: 7,
      }),
    );
  });

  it("keeps dirty saved executions temporary by omitting graph provenance", async () => {
    apiMocks.startRunExecution.mockResolvedValue(succeededExecution());
    const harness = hookHarness({
      activeGraph: {
        id: "graph-1",
        revision: 7,
        nodes: [],
      },
      isDirty: true,
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);

    await React.act(async () => {
      await hook.result.current.runWorkflow("all");
    });

    const request = apiMocks.startRunExecution.mock.calls[0]?.[1];
    expect(request).not.toHaveProperty("graph_id");
    expect(request).not.toHaveProperty("graph_revision");
  });

  it("suppresses event, poll, and terminal updates after the graph snapshot changes", async () => {
    let snapshotState: "current" | "flip" | "stale" = "current";
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution
      .mockResolvedValueOnce(execution("running"))
      .mockResolvedValueOnce(succeededExecution());
    const harness = hookHarness({
      isGraphSnapshotCurrent: () => {
        if (snapshotState === "flip") {
          snapshotState = "stale";
          return true;
        }
        return snapshotState === "current";
      },
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);
    const live = latestLiveSubscription();

    harness.hookOptions.setNodes((current) => current.map((node) => ({
      ...node,
      data: {
        ...node.data,
        run: null,
        execution: { status: "idle" },
      },
    })));
    snapshotState = "flip";

    await React.act(async () => {
      live.handlers.onEvent(executionStatusEvent(1, "running"));
    });
    expect(harness.nodes()[0].data.execution.status).toBe("idle");
    expect(snapshotState).toBe("stale");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(harness.nodes()[0].data.execution.status).toBe("idle");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
      await runPromise;
    });

    expect(harness.nodes()[0].data.execution.status).toBe("idle");
    expect(harness.nodes()[0].data.run).toBeNull();
    expect(harness.runError()).toContain(
      "The graph changed while it was running",
    );
    expect(hook.result.current.announcement).toContain(
      "graph changes prevented its results",
    );
  });

  it("restores cancellation failures and permits retries after mismatched responses", async () => {
    apiMocks.startRunExecution.mockResolvedValue(execution("running"));
    apiMocks.getRunExecution.mockResolvedValue(execution("cancelled"));
    apiMocks.cancelRunExecution
      .mockRejectedValueOnce(new Error("Cancellation service unavailable."))
      .mockResolvedValueOnce(execution("cancelling", {
        execution_id: "00000000-0000-4000-8000-000000000002",
      }))
      .mockResolvedValueOnce(execution("cancelling"));
    const harness = hookHarness();
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);

    expect(hook.result.current.visibleExecution?.status).toBe("running");
    expect(harness.nodes()[0].data.execution.status).toBe("running");

    await React.act(async () => {
      await hook.result.current.cancelCurrentExecution();
    });
    expect(hook.result.current.visibleExecution).toMatchObject({
      status: "running",
      statusError: "Cancellation service unavailable. You can try again.",
    });
    expect(harness.nodes()[0].data.execution.status).toBe("running");

    await React.act(async () => {
      await hook.result.current.cancelCurrentExecution();
    });
    expect(hook.result.current.visibleExecution).toMatchObject({
      status: "running",
      statusError:
        "Received cancellation status for another execution. You can try again.",
    });
    expect(harness.nodes()[0].data.execution.status).toBe("running");

    await React.act(async () => {
      await hook.result.current.cancelCurrentExecution();
    });
    expect(apiMocks.cancelRunExecution).toHaveBeenCalledTimes(3);
    expect(hook.result.current.visibleExecution?.status).toBe("cancelling");
    expect(harness.nodes()[0].data.execution.status).toBe("cancelling");

    await React.act(async () => {
      await vi.advanceTimersByTimeAsync(500);
      await runPromise;
    });
    expect(harness.nodes()[0].data.execution.status).toBe("cancelled");
    expect(hook.result.current.announcement).toBe("Execution cancelled.");
  });
});

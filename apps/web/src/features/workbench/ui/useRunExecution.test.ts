// @vitest-environment jsdom

import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { NodeSpec, RunExecution, RunRequest } from "@/lib/api";
import {
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
} from "../canvas/types";
import type { WorkflowNode } from "../model/execution-plan";
import { deferred } from "./test/deferred";
import { renderHook } from "./test/renderHook";
import { useRunExecution } from "./useRunExecution";

const apiMocks = vi.hoisted(() => ({
  cancelRunExecution: vi.fn<(executionId: string) => Promise<RunExecution>>(),
  getGraphMaterializations: vi.fn(),
  getRunExecution: vi.fn<(executionId: string) => Promise<RunExecution>>(),
  startRunExecution: vi.fn<(request: RunRequest) => Promise<RunExecution>>(),
}));

vi.mock("@/lib/api", () => apiMocks);

const executionId = "00000000-0000-4000-8000-000000000001";

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

function workflowNode(): WorkflowNode {
  return {
    id: "node-1",
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

function succeededExecution(): RunExecution {
  return execution("succeeded", {
    result: {
      status: "succeeded",
      node_runs: [{
        node_id: "node-1",
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
  let nodes = [workflowNode()];
  let runError: string | null = null;
  const setNodes: HookOptions["setNodes"] = (update) => {
    nodes = typeof update === "function" ? update(nodes) : update;
  };
  const setRunError = vi.fn((message: string | null) => {
    runError = message;
  });
  const hookOptions: HookOptions = {
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

describe("useRunExecution", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
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
      expect.objectContaining({
        scope: "selected-with-dependencies",
      }),
    );
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

    const request = apiMocks.startRunExecution.mock.calls[0]?.[0];
    expect(request).not.toHaveProperty("graph_id");
    expect(request).not.toHaveProperty("graph_revision");
  });

  it("suppresses terminal results when the graph snapshot changed", async () => {
    const started = deferred<RunExecution>();
    let snapshotCurrent = true;
    apiMocks.startRunExecution.mockReturnValue(started.promise);
    const harness = hookHarness({
      isGraphSnapshotCurrent: () => snapshotCurrent,
    });
    const hook = await renderHook(useRunExecution, harness.hookOptions);
    const { runPromise } = await launchRun(hook.result);

    snapshotCurrent = false;
    await React.act(async () => {
      started.resolve(succeededExecution());
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

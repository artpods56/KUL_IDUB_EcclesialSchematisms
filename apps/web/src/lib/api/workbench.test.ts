import { afterEach, describe, expect, it, vi } from "vitest";

import { getGraphExecution, listGraphExecutions } from "./workbench";

afterEach(() => vi.unstubAllGlobals());

describe("execution history API", () => {
  it("serializes list filters and opaque cursors", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ items: [], next_cursor: null }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await listGraphExecutions("graph/1", {
      limit: 50,
      cursor: "timestamp+execution/id",
      graphRevision: 7,
      status: "failed",
      nodeId: "extract/1",
    }, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/graphs/graph%2F1/executions?limit=50&cursor=timestamp%2Bexecution%2Fid&graph_revision=7&status=failed&node_id=extract%2F1",
      expect.objectContaining({ method: "GET", signal: controller.signal }),
    );
  });

  it("addresses one graph execution without conflating it with live polling", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ execution_id: "execution/1", node_results: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getGraphExecution("graph/1", "execution/1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/v1/graphs/graph%2F1/executions/execution%2F1",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

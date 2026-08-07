import { afterEach, describe, expect, it, vi } from "vitest";

import {
  getArtifactGeoRender,
  getArtifactTableCell,
  getArtifactTablePage,
  getGraphExecution,
  listGraphExecutions,
  artifactContentUrl,
  uploadFile,
} from "./workbench";

afterEach(() => vi.unstubAllGlobals());

const WORKSPACE_ID = "workspace/1";

describe("execution history API", () => {
  it("serializes list filters and opaque cursors", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ items: [], next_cursor: null }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await listGraphExecutions(WORKSPACE_ID, "graph/1", {
      limit: 50,
      cursor: "timestamp+execution/id",
      graphRevision: 7,
      status: "failed",
      nodeId: "extract/1",
    }, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/workspaces/workspace%2F1/graphs/graph%2F1/executions?limit=50&cursor=timestamp%2Bexecution%2Fid&graph_revision=7&status=failed&node_id=extract%2F1",
      expect.objectContaining({ method: "GET", signal: controller.signal }),
    );
  });

  it("addresses one graph execution without conflating it with live polling", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ execution_id: "execution/1", node_results: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getGraphExecution(WORKSPACE_ID, "graph/1", "execution/1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/workspaces/workspace%2F1/graphs/graph%2F1/executions/execution%2F1",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

describe("table artifact API", () => {
  it("serializes page bounds and preview limits", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({
        columns: [],
        rows: [],
        offset: 50,
        limit: 25,
        total_rows: 0,
        column_offset: 10,
        column_limit: 20,
        total_columns: 0,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getArtifactTablePage(
      WORKSPACE_ID,
      "artifact/1",
      50,
      25,
      ["geometry/wkt", "source name"],
      256,
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/workspaces/workspace%2F1/artifacts/artifact%2F1/table/page?offset=50&limit=25&max_cell_characters=256&column_ids=geometry%2Fwkt&column_ids=source+name",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("keeps arbitrary column ids in the cell query", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({
        row_index: 3,
        column_id: "geometry/wkt",
        value: "full",
        encoding: "native",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await getArtifactTableCell(WORKSPACE_ID, "artifact/1", 3, "geometry/wkt");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/workspaces/workspace%2F1/artifacts/artifact%2F1/table/cell?row_index=3&column_id=geometry%2Fwkt",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

describe("GIS artifact API", () => {
  it("addresses the immutable render descriptor without paging", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({
        artifact_id: "artifact/1",
        kind: "map_document",
        basemap: "openstreetmap",
        initial_bounds: null,
        layers: [],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await getArtifactGeoRender(WORKSPACE_ID, "artifact/1", controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/workspaces/workspace%2F1/artifacts/artifact%2F1/geo/render",
      expect.objectContaining({ method: "GET", signal: controller.signal }),
    );
  });
});

describe("file upload API", () => {
  it("streams the selected file as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ uploads: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File([new Uint8Array([1, 2, 3])], "scan.tif", {
      type: "image/tiff",
    });

    await uploadFile(WORKSPACE_ID, file);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/workspaces/workspace%2F1/uploads");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ Accept: "application/json" });
    expect(init.body).toBeInstanceOf(FormData);
    const uploaded = (init.body as FormData).get("file") as File;
    expect(uploaded.name).toBe("scan.tif");
    expect(uploaded.type).toBe("image/tiff");
    expect(uploaded.size).toBe(3);
  });
});

describe("artifact content URLs", () => {
  it("resolves relative and API-owned paths under the workspace-scoped API base", () => {
    expect(artifactContentUrl(WORKSPACE_ID, "./artifacts/artifact-1/content"))
      .toBe("/api/v1/workspaces/workspace%2F1/artifacts/artifact-1/content");
    expect(artifactContentUrl(WORKSPACE_ID, "/v1/workspaces/workspace%2F1/artifacts/artifact-1/content"))
      .toBe("/api/v1/workspaces/workspace%2F1/artifacts/artifact-1/content");
  });

  it("preserves absolute HTTP and custom-scheme URLs", () => {
    expect(artifactContentUrl(WORKSPACE_ID, "https://private.example/artifact"))
      .toBe("https://private.example/artifact");
    expect(artifactContentUrl(WORKSPACE_ID, "pmtiles://private.example/archive.pmtiles"))
      .toBe("pmtiles://private.example/archive.pmtiles");
  });
});

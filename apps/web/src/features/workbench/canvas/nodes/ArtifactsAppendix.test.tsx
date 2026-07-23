// @vitest-environment jsdom

import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { SWRConfig } from "swr";
import { afterEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

import type {
  ArtifactSummary,
  ArtifactTypeSpec,
  RunPortOutput,
} from "@/lib/api";
import { ArtifactPortPreview } from "./ArtifactsAppendix";

function outputFor(
  artifacts: readonly ArtifactSummary[],
  kind: "single" | "sequence" = "single",
): RunPortOutput {
  const refs = artifacts.map((artifact) => ({
    artifact_id: artifact.artifact_id,
    artifact_type: artifact.artifact_type,
    schema_version: artifact.schema_version,
  }));
  return {
    port: "result",
    kind,
    value:
      kind === "single"
        ? refs[0]
        : {
            artifact_type: artifacts[0].artifact_type,
            schema_version: artifacts[0].schema_version,
            index_key: "order_index",
            ordered: true,
            item_refs: refs,
          },
    artifacts,
  };
}

async function renderPreview(
  output: RunPortOutput,
  artifactTypes: readonly ArtifactTypeSpec[] = [],
): Promise<{ container: HTMLDivElement; root: Root }> {
  const container = document.createElement("div");
  const root = createRoot(container);
  await act(async () => {
    root.render(
      createElement(
        SWRConfig,
        { value: { provider: () => new Map(), shouldRetryOnError: false } },
        createElement(ArtifactPortPreview, {
          output,
          artifactTypes,
          previewHeight: 320,
        }),
      ),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  return { container, root };
}

afterEach(() => vi.unstubAllGlobals());

describe("artifact payload loading policy", () => {
  it("loads only the bounded page endpoint for table previews", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          columns: [],
          rows: [],
          offset: 0,
          limit: 50,
          total_rows: 0,
          column_offset: 0,
          column_limit: 25,
          total_columns: 0,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const artifact: ArtifactSummary = {
      artifact_id: "table-artifact",
      artifact_type: "table.data",
      schema_version: 1,
      content_type: "application/json",
      byte_size: 50_000_000,
      content_url: "./artifacts/table-artifact/content",
    };

    const { root } = await renderPreview(outputFor([artifact]));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "/v1/artifacts/table-artifact/table/page?",
    );
    expect(String(fetchMock.mock.calls[0][0])).not.toContain("/content");
    await act(async () => root.unmount());
  });

  it.each([
    "geo.feature_collection",
    "geo.raster_scan",
    "geo.map_layer",
    "geo.map_document",
  ])(
    "never loads generic content for %s previews",
    async (artifactType) => {
      const fetchMock = vi.fn().mockResolvedValue(new Response(
        JSON.stringify({ detail: "Render descriptor unavailable in this policy test" }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      ));
      vi.stubGlobal("fetch", fetchMock);
      const artifact: ArtifactSummary = {
        artifact_id: `map-${artifactType}`,
        artifact_type: artifactType,
        schema_version: 1,
        content_type:
          artifactType === "geo.map_document"
            ? "application/json"
            : "application/geo+json",
        byte_size: 120,
        content_url: `./artifacts/map-${artifactType}/content`,
      };

      const { container, root } = await renderPreview(outputFor([artifact]));

      expect(fetchMock).not.toHaveBeenCalled();
      expect(container.textContent).toContain("Load interactive map");
      const loadButton = [...container.querySelectorAll("button")].find(
        (button) => button.textContent === "Load interactive map",
      );
      await act(async () => {
        loadButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(String(fetchMock.mock.calls[0][0])).toContain("/geo/render");
      expect(String(fetchMock.mock.calls[0][0])).not.toContain("/geo/page");
      expect(String(fetchMock.mock.calls[0][0])).not.toContain("/content");
      await act(async () => root.unmount());
    },
  );

  it.each([
    ["known-large", 2_000_000],
    ["unknown-size", null],
  ])("defers %s JSON artifacts until explicit loading", async (id, byteSize) => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const artifact: ArtifactSummary = {
      artifact_id: id,
      artifact_type: "json.object",
      schema_version: 1,
      content_type: "application/json",
      byte_size: byteSize,
      content_url: `./artifacts/${id}/content`,
    };

    const { container, root } = await renderPreview(outputFor([artifact]));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Load complete JSON");
    await act(async () => root.unmount());
  });

  it("does not let field projection bypass the aggregate byte budget", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ statement_index: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const artifacts: ArtifactSummary[] = ["first", "second"].map((id) => ({
      artifact_id: id,
      artifact_type: "sql.result",
      schema_version: 1,
      content_type: "application/json",
      byte_size: 300_000,
      content_url: `./artifacts/${id}/content`,
    }));
    const artifactTypes: ArtifactTypeSpec[] = [
      {
        key: { id: "sql.result", schema_version: 1 },
        title: "SQL result",
        payload_schema: {
          type: "object",
          properties: { statement_index: { type: "integer" } },
        },
        field_projections: [],
      },
    ];

    const { container, root } = await renderPreview(
      outputFor(artifacts, "sequence"),
      artifactTypes,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/artifacts/first/content");
    expect(container.textContent).toContain(
      "Field projection is disabled because this sequence is too large",
    );
    await act(async () => root.unmount());
  });
});

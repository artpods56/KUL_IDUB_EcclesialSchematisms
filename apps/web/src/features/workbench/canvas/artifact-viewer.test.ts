import { describe, expect, it } from "vitest";

import {
  APPENDIX_HEIGHT_MIN,
  LAYOUT_DIMENSION_MAX,
  NODE_WIDTH_MIN,
} from "./node-layout";
import {
  ARTIFACT_VIEWER_EDGE_TYPE,
  ARTIFACT_VIEWER_INPUT_HANDLE,
  ARTIFACT_VIEWER_NODE_TYPE,
  artifactViewerStorageKey,
  hydrateArtifactViewerDocument,
  serializeArtifactViewerDocument,
  type ArtifactViewerEdge,
  type ArtifactViewerNode,
} from "./artifact-viewer";
import type { ArtifactViewerBinding } from "./artifact-interactions";

describe("artifact viewer storage", () => {
  it("scopes documents by user, stable workspace UUID, and graph id", () => {
    const first = artifactViewerStorageKey("user-1", "workspace-1", "graph-1");

    expect(first).toBe(
      "ns-workbench-presentation:v2:user-1:workspace-1:graph-1",
    );
    expect(artifactViewerStorageKey("user-2", "workspace-1", "graph-1")).not.toBe(first);
    expect(artifactViewerStorageKey("user-1", "workspace-2", "graph-1")).not.toBe(first);
    expect(artifactViewerStorageKey("user-1", "workspace-1", "graph-2")).not.toBe(first);
    expect(artifactViewerStorageKey("user-1", "workspace-1", "team/west")).not.toBe(first);
  });

  it("round-trips only presentation geometry and semantic source identity", () => {
    const nodes: ArtifactViewerNode[] = [
      {
        id: "artifact-viewer-1",
        type: ARTIFACT_VIEWER_NODE_TYPE,
        position: { x: 412, y: -88 },
        data: {
          layout: {
            width: 560,
            bodyHeight: 240,
            appendixHeight: 420,
          },
          mode: "raw",
          run: "do-not-persist-run",
          artifact: "do-not-persist-artifact",
          payload: "do-not-persist-payload",
          content_url: "https://private.example/artifact",
          onRemoveNode: () => undefined,
        },
      },
    ];
    const edges: ArtifactViewerEdge[] = [
      {
        id: "artifact-viewer-edge-1",
        type: ARTIFACT_VIEWER_EDGE_TYPE,
        source: "source-node",
        sourceHandle: "output-handle-with-runtime-contract",
        target: "artifact-viewer-1",
        targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
        data: {
          sourcePortName: "features",
          run: "do-not-persist-edge-run",
          artifacts: "do-not-persist-edge-artifacts",
          payload: "do-not-persist-edge-payload",
          content_url: "https://private.example/edge-artifact",
        },
      },
    ];
    const bindings: ArtifactViewerBinding[] = [
      {
        id: "artifact-viewer-binding-1",
        sourceViewerId: "artifact-viewer-1",
        targetViewerId: "artifact-viewer-2",
        mappings: [
          { sourceField: "historical_name", targetField: "transliteration" },
          { sourceField: "district", targetField: "district" },
        ],
        effects: ["highlight", "focus"],
        emptySelection: "show_all",
      },
    ];
    nodes.push({
      id: "artifact-viewer-2",
      type: ARTIFACT_VIEWER_NODE_TYPE,
      position: { x: 940, y: -88 },
      data: { layout: null, mode: "map" },
    });

    const serialized = serializeArtifactViewerDocument(
      nodes,
      edges,
      bindings,
    );

    expect(JSON.parse(serialized)).toEqual({
      schemaVersion: 2,
      viewers: [
        {
          id: "artifact-viewer-1",
          position: { x: 412, y: -88 },
          layout: {
            width: 560,
            bodyHeight: 240,
            appendixHeight: 420,
          },
          mode: "raw",
        },
        {
          id: "artifact-viewer-2",
          position: { x: 940, y: -88 },
          layout: null,
          mode: "map",
        },
      ],
      links: [
        {
          id: "artifact-viewer-edge-1",
          sourceNodeId: "source-node",
          sourcePortName: "features",
          targetViewerId: "artifact-viewer-1",
        },
      ],
      bindings,
    });
    expect(serialized).not.toContain("do-not-persist");
    expect(serialized).not.toContain("private.example");

    expect(
      hydrateArtifactViewerDocument(serialized, "graph-1"),
    ).toEqual({
      graphId: "graph-1",
      nodes: [
        {
          id: "artifact-viewer-1",
          type: ARTIFACT_VIEWER_NODE_TYPE,
          position: { x: 412, y: -88 },
          data: {
            layout: {
              width: 560,
              bodyHeight: 240,
              appendixHeight: 420,
            },
            mode: "raw",
          },
        },
        {
          id: "artifact-viewer-2",
          type: ARTIFACT_VIEWER_NODE_TYPE,
          position: { x: 940, y: -88 },
          data: { layout: null, mode: "map" },
        },
      ],
      edges: [
        {
          id: "artifact-viewer-edge-1",
          type: ARTIFACT_VIEWER_EDGE_TYPE,
          source: "source-node",
          target: "artifact-viewer-1",
          targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
          data: { sourcePortName: "features" },
        },
      ],
      bindings,
    });
  });
});

describe("artifact viewer document hydration", () => {
  it.each([
    ["invalid JSON", "{"],
    ["wrong schema version", JSON.stringify({
      schemaVersion: 3,
      viewers: [],
      links: [],
    })],
    ["v2 without bindings", JSON.stringify({
      schemaVersion: 2,
      viewers: [],
      links: [],
    })],
    ["missing viewers", JSON.stringify({ schemaVersion: 1, links: [] })],
    ["non-array links", JSON.stringify({
      schemaVersion: 1,
      viewers: [],
      links: {},
    })],
  ])("rejects %s", (_case, serialized) => {
    expect(hydrateArtifactViewerDocument(serialized, "graph-1")).toBeNull();
  });

  it("validates records, clamps layout, de-duplicates viewers, and keeps one link per viewer", () => {
    const serialized = JSON.stringify({
      schemaVersion: 1,
      viewers: [
        {
          id: "artifact-viewer-1",
          position: { x: 10, y: 20 },
          layout: {
            width: 10,
            bodyHeight: LAYOUT_DIMENSION_MAX + 1,
            appendixHeight: 20,
          },
          mode: "map",
        },
        {
          id: "artifact-viewer-1",
          position: { x: 999, y: 999 },
          layout: { width: 999 },
          mode: "duplicate-must-not-win",
        },
        {
          id: "artifact-viewer-2",
          position: { x: -30, y: 44 },
          layout: {
            width: "not-a-number",
            appendixHeight: 300,
          },
          mode: null,
        },
        {
          id: "workflow-node",
          position: { x: 0, y: 0 },
          layout: null,
          mode: null,
        },
        {
          id: "artifact-viewer-invalid-position",
          position: { x: "left", y: 0 },
          layout: null,
          mode: null,
        },
        {
          id: "artifact-viewer-invalid-mode",
          position: { x: 0, y: 0 },
          layout: null,
          mode: 42,
        },
      ],
      links: [
        {
          id: "artifact-viewer-edge-1",
          sourceNodeId: "source-a",
          sourcePortName: "image",
          targetViewerId: "artifact-viewer-1",
        },
        {
          id: "artifact-viewer-edge-replacement",
          sourceNodeId: "source-b",
          sourcePortName: "features",
          targetViewerId: "artifact-viewer-1",
        },
        {
          id: "artifact-viewer-edge-1",
          sourceNodeId: "source-c",
          sourcePortName: "duplicate-edge-id",
          targetViewerId: "artifact-viewer-2",
        },
        {
          id: "artifact-viewer-edge-2",
          sourceNodeId: "source-d",
          sourcePortName: "document",
          targetViewerId: "artifact-viewer-2",
        },
        {
          id: "artifact-viewer-edge-unknown-target",
          sourceNodeId: "source-e",
          sourcePortName: "table",
          targetViewerId: "artifact-viewer-missing",
        },
        {
          id: "artifact-viewer-edge-empty-port",
          sourceNodeId: "source-f",
          sourcePortName: "",
          targetViewerId: "artifact-viewer-2",
        },
      ],
    });

    const hydrated = hydrateArtifactViewerDocument(serialized, "graph-1");

    expect(hydrated?.nodes).toEqual([
      {
        id: "artifact-viewer-1",
        type: ARTIFACT_VIEWER_NODE_TYPE,
        position: { x: 10, y: 20 },
        data: {
          layout: {
            width: NODE_WIDTH_MIN,
            bodyHeight: LAYOUT_DIMENSION_MAX,
            appendixHeight: APPENDIX_HEIGHT_MIN,
          },
          mode: "map",
        },
      },
      {
        id: "artifact-viewer-2",
        type: ARTIFACT_VIEWER_NODE_TYPE,
        position: { x: -30, y: 44 },
        data: {
          layout: { appendixHeight: 300 },
          mode: null,
        },
      },
    ]);
    expect(hydrated?.edges).toEqual([
      {
        id: "artifact-viewer-edge-1",
        type: ARTIFACT_VIEWER_EDGE_TYPE,
        source: "source-a",
        target: "artifact-viewer-1",
        targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
        data: { sourcePortName: "image" },
      },
      {
        id: "artifact-viewer-edge-2",
        type: ARTIFACT_VIEWER_EDGE_TYPE,
        source: "source-d",
        target: "artifact-viewer-2",
        targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
        data: { sourcePortName: "document" },
      },
    ]);
    expect(hydrated?.bindings).toEqual([]);
  });
});

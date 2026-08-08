// @vitest-environment jsdom

import * as React from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const flowMocks = vi.hoisted(() => ({
  edges: [] as Record<string, unknown>[],
  nodes: new Map<string, unknown>(),
  updateNodeInternals: vi.fn(),
}));

const previewMocks = vi.hoisted(() => ({
  render: vi.fn(),
}));

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

vi.mock("@xyflow/react", () => ({
  Handle: ({
    id,
    isConnectable,
    ...props
  }: {
    id: string;
    isConnectable?: boolean;
    "aria-label"?: string;
  }) => (
    <span
      data-testid="artifact-viewer-handle"
      data-handle-id={id}
      data-connectable={String(isConnectable)}
      aria-label={props["aria-label"]}
    />
  ),
  Position: { Left: "left" },
  useEdges: () => flowMocks.edges,
  useNodesData: (nodeId: string) => flowMocks.nodes.get(nodeId) ?? null,
  useUpdateNodeInternals: () => flowMocks.updateNodeInternals,
}));

vi.mock("@/hooks/use-api", () => ({
  useNodeRegistry: () => ({ data: { artifact_types: [] } }),
}));

vi.mock("@/features/workspaces/WorkspaceLayout", () => ({
  useWorkspaceContext: () => ({
    workspace: {
      id: "workspace-1",
      slug: "local",
      name: "Local",
      kind: "personal",
      role: "owner",
      capabilities: [],
    },
    workspaces: [],
    refreshWorkspaces: async () => undefined,
  }),
}));

vi.mock("./ArtifactsAppendix", () => ({
  ArtifactPortPreview: (props: {
    output: { port: string; kind: "single" | "sequence" };
    modeChoice: string | null;
    onModeChoiceChange: (mode: string) => void;
  }) => {
    previewMocks.render(props);
    return (
      <button
        type="button"
        data-testid="artifact-port-preview"
        data-port={props.output.port}
        data-kind={props.output.kind}
        onClick={() => props.onModeChoiceChange("raw")}
      >
        Preview {props.output.port}
      </button>
    );
  },
}));

vi.mock("./LayoutResizeHandle", () => ({
  LayoutResizeHandle: ({ ariaLabel }: { ariaLabel: string }) => (
    <button type="button" data-testid="corner-resize" aria-label={ariaLabel} />
  ),
}));

import type {
  ArtifactSummary,
  NodeSpec,
  RunNodeResult,
  RunPortOutput,
} from "@/lib/api";
import {
  ARTIFACT_VIEWER_EDGE_TYPE,
  ARTIFACT_VIEWER_INPUT_HANDLE,
  ARTIFACT_VIEWER_NODE_TYPE,
  type ArtifactViewerEdge,
  type ArtifactViewerNode,
  type ArtifactViewerNodeData,
  type CanvasWorkflowNode,
} from "../artifact-viewer";
import {
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
} from "../types";
import ArtifactViewerNodeCard from "./ArtifactViewerNode";

const roots: Root[] = [];

function sourceSpec(shape: "one" | "many"): NodeSpec {
  return {
    operator_id: "test.render-source",
    operator_version: 1,
    plugin_slug: "test",
    title: "Prepare map",
    description: "Produces multiple outputs for viewer tests.",
    catalog_visible: true,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [
      {
        name: "metadata",
        title: "Metadata",
        description: null,
        direction: "output",
        artifact_type: { id: "json.object", schema_version: 1 },
        artifact_type_variable: null,
        shape: "one",
        accepted_shapes: ["one"],
        instance_plugs: false,
        variadic: false,
        required: true,
      },
      {
        name: "preview",
        title: "Raster preview",
        description: null,
        direction: "output",
        artifact_type: { id: "image.raster", schema_version: 1 },
        artifact_type_variable: null,
        shape,
        accepted_shapes: [shape],
        instance_plugs: false,
        variadic: false,
        required: true,
      },
    ],
  };
}

function artifact(id: string, artifactType: string): ArtifactSummary {
  return {
    artifact_id: id,
    artifact_type: artifactType,
    schema_version: 1,
    content_type: "application/json",
    content_url: `./artifacts/${id}/content`,
    byte_size: 42,
  };
}

function runOutput(
  port: string,
  kind: "single" | "sequence",
  item: ArtifactSummary,
): RunPortOutput {
  const ref = {
    artifact_id: item.artifact_id,
    artifact_type: item.artifact_type,
    schema_version: item.schema_version,
    content_hash: null,
  };
  return {
    port,
    kind,
    value: kind === "single"
      ? ref
      : {
          artifact_type: item.artifact_type,
          schema_version: item.schema_version,
          index_key: "order_index",
          ordered: true,
          item_refs: [ref],
        },
    artifacts: [item],
  };
}

function sourceNode(
  shape: "one" | "many",
  run: RunNodeResult | null,
): CanvasWorkflowNode {
  const data = createWorkflowNodeData(sourceSpec(shape));
  data.run = run;
  data.execution = { status: run?.status ?? "idle" };
  return {
    id: "source-node",
    type: WORKFLOW_NODE_TYPE,
    position: { x: 0, y: 0 },
    data,
  };
}

function viewerEdge(): ArtifactViewerEdge {
  return {
    id: "artifact-viewer-edge-1",
    type: ARTIFACT_VIEWER_EDGE_TYPE,
    source: "source-node",
    target: "artifact-viewer-1",
    targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
    data: { sourcePortName: "preview" },
  };
}

function renderViewer(
  overrides: Partial<ArtifactViewerNodeData> = {},
): HTMLDivElement {
  const container = document.createElement("div");
  const root = createRoot(container);
  roots.push(root);
  const data: ArtifactViewerNodeData = {
    layout: null,
    mode: null,
    ...overrides,
  };
  const node: ArtifactViewerNode = {
    id: "artifact-viewer-1",
    type: ARTIFACT_VIEWER_NODE_TYPE,
    position: { x: 0, y: 0 },
    data,
  };

  React.act(() => {
    root.render(
      <ArtifactViewerNodeCard
        {...({
          id: node.id,
          data: node.data,
          selected: false,
          isConnectable: true,
        } as React.ComponentProps<typeof ArtifactViewerNodeCard>)}
      />,
    );
  });
  return container;
}

beforeEach(() => {
  flowMocks.edges = [];
  flowMocks.nodes.clear();
  flowMocks.updateNodeInternals.mockClear();
  previewMocks.render.mockClear();
});

afterEach(() => {
  React.act(() => {
    for (const root of roots.splice(0)) root.unmount();
  });
});

describe("ArtifactViewerNode", () => {
  it("invites a generic artifact connection while disconnected", () => {
    const container = renderViewer();

    expect(container.textContent).toContain("Any artifact");
    expect(container.textContent).toContain("No output connected");
    expect(container.textContent).toContain("Waiting");
    expect(container.textContent).toContain("Connect an output");
    expect(
      container.querySelector('[aria-label="Input port Artifact, accepts Any artifact"]'),
    ).not.toBeNull();
    expect(previewMocks.render).not.toHaveBeenCalled();
  });

  it("shows provenance but no preview until the named output succeeds", () => {
    const preview = runOutput(
      "preview",
      "single",
      artifact("failed-preview", "image.raster"),
    );
    flowMocks.edges = [viewerEdge()];
    flowMocks.nodes.set(
      "source-node",
      sourceNode("one", {
        node_id: "source-node",
        status: "failed",
        outputs: [preview],
        error: "Rendering failed",
      }),
    );

    const container = renderViewer();

    expect(container.textContent).toContain("Prepare map → Raster preview");
    expect(container.textContent).toContain("image.raster@1 · single");
    expect(container.textContent).toContain("No artifact");
    expect(container.textContent).toContain("No materialization yet");
    expect(previewMocks.render).not.toHaveBeenCalled();
  });

  it.each([
    ["one", "single"],
    ["many", "sequence"],
  ] as const)(
    "renders only the connected %s output as a %s contract and forwards mode changes",
    (shape, kind) => {
      const metadata = runOutput(
        "metadata",
        "single",
        artifact("metadata-artifact", "json.object"),
      );
      const preview = runOutput(
        "preview",
        kind,
        artifact("preview-artifact", "image.raster"),
      );
      flowMocks.edges = [viewerEdge()];
      flowMocks.nodes.set(
        "source-node",
        sourceNode(shape, {
          node_id: "source-node",
          status: "succeeded",
          outputs: [metadata, preview],
          error: null,
        }),
      );
      const onModeChange = vi.fn();

      const container = renderViewer({
        mode: "map",
        onModeChange,
      });

      expect(container.textContent).toContain("Prepare map → Raster preview");
      expect(container.textContent).toContain(`image.raster@1 · ${kind}`);
      expect(container.textContent).toContain("Materialized");
      expect(
        container.querySelector('[data-testid="artifact-port-preview"]'),
      ).not.toBeNull();
      expect(
        container.querySelector('[data-testid="artifact-port-preview"]')
          ?.getAttribute("data-port"),
      ).toBe("preview");
      const previewProps = previewMocks.render.mock.calls.at(-1)?.[0];
      expect(previewProps?.output).toBe(preview);
      expect(previewProps?.output).not.toBe(metadata);
      expect(previewProps?.modeChoice).toBe("map");

      React.act(() => {
        container
          .querySelector<HTMLButtonElement>(
            '[data-testid="artifact-port-preview"]',
          )
          ?.click();
      });
      expect(onModeChange).toHaveBeenCalledWith("artifact-viewer-1", "raw");
    },
  );

  it("keeps a corner resize handle for the viewer viewport", () => {
    const container = renderViewer();
    expect(
      container.querySelector('[data-testid="corner-resize"]'),
    ).not.toBeNull();
  });
});

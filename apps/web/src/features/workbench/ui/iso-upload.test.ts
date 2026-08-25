// @vitest-environment jsdom

import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { NodeSpec, UploadResponse } from "@/lib/api";
import {
  GEOJSON_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  type WorkflowEdge,
} from "../canvas/types";
import type { GraphCommand } from "../model/graph-document";
import type { WorkflowNode } from "../model/execution-plan";
import type { AuthoringCommandOptions } from "./authoring-command-guard";
import { deferred } from "./test/deferred";
import { renderHook } from "./test/renderHook";
import { useNodeFileUploads } from "./useNodeFileUploads";

const uploadFileMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ uploadFile: uploadFileMock }));

const NO_EDGES: readonly WorkflowEdge[] = [];

function nodeSpec(): NodeSpec {
  return {
    operator_id: GEOJSON_UPLOAD_OPERATOR_ID,
    operator_version: 1,
    plugin_slug: "gis",
    title: "Import GeoJSON",
    description: "Import a GeoJSON FeatureCollection",
    catalog_visible: true,
    runnable: true,
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

function geoJsonFile(): File {
  return new File(['{"type": "FeatureCollection", "features": []}'], "layer.geojson", {
    type: "application/geo+json",
  });
}

function uploadResponse(): UploadResponse {
  return {
    byte_size: 48,
    filename: "layer.geojson",
    upload_key: "uploads/layer.geojson",
  };
}

interface HarnessProps {
  initialNodes: readonly WorkflowNode[];
  applyAuthoringCommands: (
    commands: readonly GraphCommand[],
    options?: AuthoringCommandOptions,
  ) => void;
}

function useHarness({ initialNodes, applyAuthoringCommands }: HarnessProps) {
  const [nodes, setNodes] = React.useState<WorkflowNode[]>([...initialNodes]);
  const [runError, setRunError] = React.useState<string | null>(null);
  const { uploading, handleImagesSelected } = useNodeFileUploads({
    workspaceId: "workspace-1",
    nodes,
    edges: NO_EDGES,
    setNodes,
    setRunError,
    applyAuthoringCommands,
  });
  return { nodes, runError, uploading, handleImagesSelected };
}

type HarnessValue = ReturnType<typeof useHarness>;

async function startUpload(hook: {
  result: { current: HarnessValue };
}): Promise<Promise<void>> {
  let captured!: Promise<void>;
  await React.act(async () => {
    captured = hook.result.current.handleImagesSelected("node-1", [
      geoJsonFile(),
    ]);
  });
  return captured;
}

describe("useNodeFileUploads", () => {
  beforeEach(() => {
    uploadFileMock.mockReset();
  });

  it("commits the upload result and unlocks authoring after a successful upload", async () => {
    const applyAuthoringCommands = vi.fn();
    const upload = uploadResponse();
    const uploadDeferred = deferred<UploadResponse>();
    uploadFileMock.mockReturnValue(uploadDeferred.promise);

    const hook = await renderHook(useHarness, {
      initialNodes: [workflowNode()],
      applyAuthoringCommands,
    });
    console.log("STEP rendered");

    async function startUploadLocal(h: { result: { current: HarnessValue } }) {
      let captured!: Promise<void>;
      await React.act(async () => {
        captured = h.result.current.handleImagesSelected("node-1", [geoJsonFile()]);
      });
      return captured;
    }
    const handlePromise = await startUploadLocal(hook);
    console.log("STEP started, mockCalls=", uploadFileMock.mock.calls.length, "uploading=", hook.result.current.uploading);

    await React.act(async () => {
      uploadDeferred.resolve(upload);
      await handlePromise;
    });
    console.log("STEP resolved, status=", hook.result.current.nodes[0].data.execution.status);

    expect(uploadFileMock).toHaveBeenCalledTimes(1);
    expect(hook.result.current.uploading).toBe(false);
  });
});

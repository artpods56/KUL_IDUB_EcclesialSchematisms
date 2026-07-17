import { describe, expect, it } from "vitest";

import type { NodeSpec } from "@/lib/api";
import { decodeHandleId, encodeHandleId } from "./handles";
import {
  bindArtifactTypeVariable,
  createWorkflowNodeData,
  invalidateWorkflowNodeRuns,
  portMetaForPort,
  resetArtifactTypeBinding,
  serializeRunNode,
} from "./types";

const genericNodeSpec: NodeSpec = {
  operator_id: "sequence.collect",
  operator_version: 1,
  plugin_slug: "test",
  title: "Collect",
  description: "Collect ordered artifacts.",
  catalog_visible: true,
  config_schema: {},
  input_schema: {},
  output_schema: {},
  inputs: [
    {
      name: "items",
      title: "Items",
      description: null,
      direction: "input",
      artifact_type: null,
      artifact_type_variable: "T",
      shape: "one",
      accepted_shapes: ["one", "many"],
      instance_plugs: true,
      variadic: false,
      required: true,
    },
  ],
  outputs: [
    {
      name: "items",
      title: "Items",
      description: null,
      direction: "output",
      artifact_type: null,
      artifact_type_variable: "T",
      shape: "many",
      accepted_shapes: ["many"],
      instance_plugs: false,
      variadic: false,
      required: true,
    },
  ],
};

describe("generic artifact type reset", () => {
  it("keeps an incident binding and clears a disconnected binding", () => {
    const data = createWorkflowNodeData(genericNodeSpec);
    data.artifactTypeBindings = {
      T: { id: "scalar.text", schema_version: 1 },
    };
    data.execution = { status: "failed", error: "stale" };

    expect(resetArtifactTypeBinding(data, "T", true)).toBe(data);

    const reset = resetArtifactTypeBinding(data, "T", false);
    expect(reset).not.toBe(data);
    expect(reset.artifactTypeBindings).toEqual({});
    expect(reset.execution).toEqual({ status: "idle" });
  });

  it("resolves every port sharing T after the first binding", () => {
    const data = bindArtifactTypeVariable(
      createWorkflowNodeData(genericNodeSpec),
      "T",
      { id: "scalar.text", schema_version: 1 },
    );

    const handles = [...data.spec.inputs, ...data.spec.outputs].map((port) =>
      decodeHandleId(
        encodeHandleId(
          portMetaForPort(
            port,
            port.shape,
            undefined,
            data.artifactTypeBindings,
          ),
        ),
      ),
    );

    expect(handles).toHaveLength(2);
    expect(handles).toEqual([
      {
        portName: "items",
        artifactTypeId: "scalar.text",
        schemaVersion: 1,
        shape: "one",
        direction: "input",
      },
      {
        portName: "items",
        artifactTypeId: "scalar.text",
        schemaVersion: 1,
        shape: "many",
        direction: "output",
      },
    ]);
  });

  it("preserves instance-plug identity when T binds to a raster image", () => {
    const input = genericNodeSpec.inputs[0]!;
    const plugId = "image-input-1";
    const genericHandle = decodeHandleId(
      encodeHandleId(portMetaForPort(input, input.shape, plugId)),
    );
    const data = bindArtifactTypeVariable(
      createWorkflowNodeData(genericNodeSpec),
      "T",
      { id: "image.raster", schema_version: 1 },
    );
    const concreteHandle = decodeHandleId(
      encodeHandleId(
        portMetaForPort(
          input,
          input.shape,
          plugId,
          data.artifactTypeBindings,
        ),
      ),
    );

    expect(genericHandle).toEqual({
      portName: "items",
      artifactTypeVariable: "T",
      shape: "one",
      direction: "input",
      plugId,
    });
    expect(concreteHandle).toEqual({
      portName: "items",
      artifactTypeId: "image.raster",
      schemaVersion: 1,
      shape: "one",
      direction: "input",
      plugId,
    });
  });
});

describe("run node serialization", () => {
  it("sends ordinary config while omitting all write-only secret UI state", () => {
    const data = createWorkflowNodeData(genericNodeSpec);
    data.config = {
      base_url: "https://api.openai.com/v1",
      model: "gpt-5-mini",
    };
    data.secretStatuses = { api_key: { state: "configured" } };
    data.secretInputReadiness = { api_key: true };
    data.secretInputScope = "graph-1:2";
    data.onApplyNodeSecret = async () => true;

    const request = serializeRunNode("llm-node", data);

    expect(request.config).toEqual(data.config);
    expect(request).not.toHaveProperty("secretStatuses");
    expect(request).not.toHaveProperty("secretInputReadiness");
    expect(request).not.toHaveProperty("secretInputScope");
    expect(request).not.toHaveProperty("onApplyNodeSecret");
    expect(JSON.stringify(request)).not.toContain("api_key");
  });
});

function nodeWithRun(id: string) {
  const data = createWorkflowNodeData(genericNodeSpec);
  data.run = {
    node_id: id,
    status: "succeeded",
    outputs: [],
    error: null,
  };
  data.execution = { status: "succeeded" };
  return { id, data };
}

describe("workflow result invalidation", () => {
  it("preserves upstream and unrelated runs while clearing the target branch", () => {
    const source = nodeWithRun("source");
    const target = nodeWithRun("target");
    const descendant = nodeWithRun("descendant");
    const unrelated = nodeWithRun("unrelated");

    const next = invalidateWorkflowNodeRuns(
      [source, target, descendant, unrelated],
      [
        { source: "source", target: "target" },
        { source: "target", target: "descendant" },
      ],
      ["target"],
    );

    expect(next[0]).toBe(source);
    expect(next[0]?.data.run).toBe(source.data.run);
    expect(next[3]).toBe(unrelated);
    expect(next[3]?.data.run).toBe(unrelated.data.run);
    expect(next[1]?.data.run).toBeNull();
    expect(next[1]?.data.execution).toEqual({ status: "idle" });
    expect(next[2]?.data.run).toBeNull();
    expect(next[2]?.data.execution).toEqual({ status: "idle" });
  });
});

import { describe, expect, it } from "vitest";

import type { NodeSpec } from "@/lib/api";
import { decodeHandleId, encodeHandleId } from "./handles";
import {
  bindArtifactTypeVariable,
  createWorkflowNodeData,
  invalidateWorkflowNodeRuns,
  portMetaForPort,
  resetArtifactTypeBinding,
} from "./types";

const genericNodeSpec: NodeSpec = {
  operator_id: "sequence.collect",
  operator_version: 1,
  plugin_slug: "test",
  title: "Collect",
  description: "Collect ordered artifacts.",
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

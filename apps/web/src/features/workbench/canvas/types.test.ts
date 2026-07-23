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

const artifactQueryNodeSpec: NodeSpec = {
  operator_id: "sql.artifacts.query",
  operator_version: 1,
  plugin_slug: "external.sql",
  title: "Query artifact tables",
  description: "Runs read-only queries over table artifacts.",
  catalog_visible: true,
  config_schema: {
    type: "object",
    properties: { relations: { type: "array" } },
    required: ["relations"],
  },
  input_schema: {},
  output_schema: {},
  inputs: [
    {
      name: "statements",
      title: "Statements",
      description: null,
      direction: "input",
      artifact_type: { id: "sql.statement", schema_version: 1 },
      artifact_type_variable: null,
      shape: "one",
      accepted_shapes: ["one"],
      instance_plugs: true,
      variadic: true,
      required: true,
    },
    {
      name: "relations",
      title: "Relations",
      description: null,
      direction: "input",
      artifact_type: { id: "table.data", schema_version: 1 },
      artifact_type_variable: null,
      shape: "one",
      accepted_shapes: ["one"],
      instance_plugs: true,
      variadic: true,
      required: true,
    },
  ],
  outputs: [
    {
      name: "tables",
      title: "Tables",
      description: null,
      direction: "output",
      artifact_type: { id: "table.data", schema_version: 1 },
      artifact_type_variable: null,
      shape: "many",
      accepted_shapes: ["many"],
      instance_plugs: false,
      variadic: false,
      required: true,
    },
  ],
};

describe("artifact query initialization", () => {
  it("seeds one statement plug and one named relation with shared identity", () => {
    const data = createWorkflowNodeData(artifactQueryNodeSpec);
    const relationPlug = data.inputPlugs.find(
      (plug) => plug.portName === "relations",
    );

    expect(data.inputPlugs.map((plug) => plug.portName)).toEqual([
      "statements",
      "relations",
    ]);
    expect(data.config.relations).toEqual([
      { id: relationPlug?.id, alias: "relation_1" },
    ]);
  });
});

describe("generic artifact type reset", () => {
  it("keeps an incident binding and clears a disconnected binding", () => {
    const data = createWorkflowNodeData(genericNodeSpec);
    data.artifactTypeBindings = {
      T: { id: "scalar.text", schema_version: 1 },
    };
    data.execution = { status: "failed", error: "stale" };
    data.progress = {
      omittedCount: 0,
      entries: [{
        sequence: 1,
        message: "stale",
        current: null,
        total: null,
        sourceNodePath: [],
        invocationIndex: null,
        invocationPath: [],
      }],
    };

    expect(resetArtifactTypeBinding(data, "T", true)).toBe(data);

    const reset = resetArtifactTypeBinding(data, "T", false);
    expect(reset).not.toBe(data);
    expect(reset.artifactTypeBindings).toEqual({});
    expect(reset.execution).toEqual({ status: "idle" });
    expect(reset.progress).toBeNull();
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
      bounds: [19.75, 49.97, 19.82, 50.03],
    };
    data.secretStatuses = { api_key: { state: "configured" } };
    data.secretInputReadiness = { api_key: true };
    data.secretInputScope = "graph-1:2";
    data.onApplyNodeSecret = async () => true;
    data.progress = {
      omittedCount: 0,
      entries: [{
        sequence: 1,
        message: "api_key=must-not-be-serialized",
        current: null,
        total: null,
        sourceNodePath: [],
        invocationIndex: null,
        invocationPath: [],
      }],
    };

    const request = serializeRunNode("llm-node", data);

    expect(request.config).toEqual(data.config);
    expect(request).not.toHaveProperty("secretStatuses");
    expect(request).not.toHaveProperty("secretInputReadiness");
    expect(request).not.toHaveProperty("secretInputScope");
    expect(request).not.toHaveProperty("onApplyNodeSecret");
    expect(request).not.toHaveProperty("progress");
    expect(JSON.stringify(request)).not.toContain("api_key");
  });

  it("omits authoring input plugs that have no active execution edge", () => {
    const data = createWorkflowNodeData(genericNodeSpec);
    data.inputPlugs = [
      { id: "active", portName: "items" },
      { id: "disabled", portName: "items" },
    ];

    const request = serializeRunNode(
      "collect-node",
      data,
      new Set(["active"]),
    );

    expect(request.input_plugs).toEqual([
      { id: "active", port: "items" },
    ]);
    expect(data.inputPlugs).toHaveLength(2);
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
  data.progress = {
    omittedCount: 0,
    entries: [{
      sequence: 1,
      message: `Progress for ${id}`,
      current: 1,
      total: 1,
      sourceNodePath: [],
      invocationIndex: null,
      invocationPath: [],
    }],
  };
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
    expect(next[0]?.data.progress).toBe(source.data.progress);
    expect(next[3]).toBe(unrelated);
    expect(next[3]?.data.run).toBe(unrelated.data.run);
    expect(next[1]?.data.run).toBeNull();
    expect(next[1]?.data.execution).toEqual({ status: "idle" });
    expect(next[1]?.data.progress).toBeNull();
    expect(next[2]?.data.run).toBeNull();
    expect(next[2]?.data.execution).toEqual({ status: "idle" });
    expect(next[2]?.data.progress).toBeNull();
  });

  it("clears only descendants reached through enabled edges", () => {
    const target = nodeWithRun("target");
    const activeDescendant = nodeWithRun("active-descendant");
    const disabledDescendant = nodeWithRun("disabled-descendant");
    const disabledGrandchild = nodeWithRun("disabled-grandchild");

    const next = invalidateWorkflowNodeRuns(
      [target, activeDescendant, disabledDescendant, disabledGrandchild],
      [
        {
          source: "target",
          target: "active-descendant",
          data: { enabled: true },
        },
        {
          source: "target",
          target: "disabled-descendant",
          data: { enabled: false },
        },
        {
          source: "disabled-descendant",
          target: "disabled-grandchild",
          data: { enabled: true },
        },
      ],
      ["target"],
    );

    expect(next[0]?.data.run).toBeNull();
    expect(next[1]?.data.run).toBeNull();
    expect(next[2]).toBe(disabledDescendant);
    expect(next[2]?.data.run).toBe(disabledDescendant.data.run);
    expect(next[3]).toBe(disabledGrandchild);
    expect(next[3]?.data.run).toBe(disabledGrandchild.data.run);
  });
});

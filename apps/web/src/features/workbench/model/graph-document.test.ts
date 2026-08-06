import { describe, expect, it } from "vitest";

import type { CreateSavedGraphRequest } from "@/lib/api";
import {
  applyGraphCommand,
  authoredGraphDocument,
  createSavedGraphRequest,
  executionInvalidatedNodeIds,
  type AuthoredGraphDocument,
} from "./graph-document";

const node = (id: string) => ({
  id,
  operator_id: `test.${id}`,
  operator_version: 1,
  config: { label: id },
  input_plugs: [],
  artifact_type_bindings: [],
  position: { x: 0, y: 0 },
  layout: null,
});

const edge = {
  id: "edge-1",
  from_node: "source",
  from_port: "output",
  to_node: "target",
  to_port: "input",
  to_plug: null,
  enabled: true,
  collection_mode: "direct" as const,
  projection: null,
  conversion_path: [],
  route_offset: null,
};

function document(): AuthoredGraphDocument {
  return authoredGraphDocument({
    name: "Draft",
    nodes: [node("source"), node("target")],
    edges: [edge],
  });
}

describe("authored graph document", () => {
  it("round-trips only saved graph fields", () => {
    const value = document();
    const request = createSavedGraphRequest(value);

    expect(request).toEqual(value);
    expect(JSON.stringify(value)).not.toContain("callback");
    expect(JSON.stringify(value)).not.toContain("selection");
    expect(JSON.stringify(value)).not.toContain("viewport");
    expect(JSON.stringify(value)).not.toContain("privateFieldDraft");
    expect(JSON.stringify(value)).not.toContain("presence");
    expect(JSON.stringify(value)).not.toContain("execution");
  });

  it("projects adversarial React Flow runtime fields at both durable boundaries", () => {
    const runtimeNode = {
      ...node("source"),
      config: { nested: { result: "legitimate config content" } },
      selected: true,
      width: 480,
      height: 220,
      internals: { handleBounds: { source: [] } },
      onConfigChange: () => undefined,
      execution: { status: "failed" },
      progress: { entries: [] },
      run: { status: "succeeded" },
      result: { payload: "runtime result" },
      presence: { userId: "other-user" },
      privateFieldDraft: "draft-only value",
    };
    const runtimeEdge = {
      ...edge,
      selected: true,
      sourceX: 480,
      sourceY: 220,
      internals: { z: 1 },
      onUpdate: () => undefined,
      execution: { status: "failed" },
      progress: { entries: [] },
      run: { status: "succeeded" },
      result: { payload: "runtime result" },
      presence: { userId: "other-user" },
      privateFieldDraft: "draft-only value",
    };
    const input: CreateSavedGraphRequest = {
      name: "Adversarial graph",
      nodes: [runtimeNode as unknown as NonNullable<CreateSavedGraphRequest["nodes"]>[number]],
      edges: [runtimeEdge as unknown as NonNullable<CreateSavedGraphRequest["edges"]>[number]],
    };

    const canonical = authoredGraphDocument(input);
    const request = createSavedGraphRequest(canonical);
    const serialized = JSON.stringify({ canonical, request });

    expect(canonical.nodes[0]?.config).toEqual({
      nested: { result: "legitimate config content" },
    });
    expect(canonical.nodes[0]).not.toHaveProperty("selected");
    expect(canonical.nodes[0]).not.toHaveProperty("internals");
    expect(canonical.nodes[0]).not.toHaveProperty("onConfigChange");
    expect(canonical.nodes[0]).not.toHaveProperty("execution");
    expect(canonical.edges[0]).not.toHaveProperty("sourceX");
    expect(canonical.edges[0]).not.toHaveProperty("onUpdate");
    expect(canonical.edges[0]).not.toHaveProperty("presence");
    expect(serialized).not.toContain("runtime result");
    expect(serialized).not.toContain("draft-only value");
    expect(serialized).not.toContain("other-user");
  });

  it("applies a compound node removal and its incident edges atomically", () => {
    const result = applyGraphCommand(document(), {
      kind: "remove_nodes",
      node_ids: ["source"],
    });

    expect(result.nodes.map((candidate) => candidate.id)).toEqual(["target"]);
    expect(result.edges).toEqual([]);
  });

  it("updates configuration and layout without mutating the input document", () => {
    const original = document();
    const updated = applyGraphCommand(original, {
      kind: "update_node_configuration",
      node_id: "source",
      field: "label",
      value: "new label",
    });
    const laidOut = applyGraphCommand(updated, {
      kind: "update_node_layout",
      node_id: "source",
      layout: { width: 420, body_height: null, appendix_height: null },
    });

    expect(original.nodes[0]?.config).toEqual({ label: "source" });
    expect(laidOut.nodes[0]?.config).toEqual({ label: "new label" });
    expect(laidOut.nodes[0]?.layout).toEqual({
      width: 420,
      body_height: null,
      appendix_height: null,
    });
  });

  it("changes edge transport through a typed semantic command", () => {
    const result = applyGraphCommand(document(), {
      kind: "update_edge",
      edge_id: "edge-1",
      update: { enabled: false, collection_mode: "map" },
    });

    expect(result.edges[0]).toMatchObject({
      id: "edge-1",
      enabled: false,
      collection_mode: "map",
    });
  });

  it("rejects an edge update for a missing edge", () => {
    expect(() => applyGraphCommand(document(), {
      kind: "update_edge",
      edge_id: "missing-edge",
      update: { enabled: false },
    })).toThrow("missing edge missing-edge");
  });

  it("does not invalidate execution for a move, but scopes configuration invalidation", () => {
    const graph = document();
    const move = executionInvalidatedNodeIds(graph, {
      kind: "move_nodes",
      positions: [{ node_id: "source", x: 20, y: 30 }],
    });
    const targetEdit = executionInvalidatedNodeIds(graph, {
      kind: "update_node_configuration",
      node_id: "target",
      field: "label",
      value: "changed",
    });
    const bind = executionInvalidatedNodeIds(graph, {
      kind: "bind_artifact_type",
      node_id: "source",
      variable: "T",
      artifact_type: { id: "artifact.scalar", schema_version: 1 },
    });
    const reset = executionInvalidatedNodeIds(graph, {
      kind: "reset_artifact_type_binding",
      node_id: "source",
      variable: "T",
    });

    expect(move).toEqual(new Set());
    expect(targetEdit).toEqual(new Set(["target"]));
    expect(bind).toEqual(new Set(["source", "target"]));
    expect(reset).toEqual(new Set(["source", "target"]));
  });

  it("preserves edits when stale dispatchers apply sequentially", () => {
    let latest = document();
    const dispatchMove = () => {
      latest = applyGraphCommand(latest, {
        kind: "move_nodes",
        positions: [{ node_id: "source", x: 40, y: 50 }],
      });
    };
    const dispatchConfig = () => {
      latest = applyGraphCommand(latest, {
        kind: "update_node_configuration",
        node_id: "target",
        field: "label",
        value: "edited after move",
      });
    };

    dispatchMove();
    dispatchConfig();

    expect(latest.nodes).toEqual([
      { ...node("source"), position: { x: 40, y: 50 } },
      { ...node("target"), config: { label: "edited after move" } },
    ]);
  });
});

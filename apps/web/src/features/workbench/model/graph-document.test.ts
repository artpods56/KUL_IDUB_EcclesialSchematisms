import { describe, expect, it } from "vitest";

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

    expect(move).toEqual(new Set());
    expect(targetEdit).toEqual(new Set(["target"]));
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

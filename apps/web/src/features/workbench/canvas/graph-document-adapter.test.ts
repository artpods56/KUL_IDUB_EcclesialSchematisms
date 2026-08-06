import { describe, expect, it } from "vitest";

import {
  graphCommandsFromNodeChanges,
  reduceWorkbenchAuthoringState,
  type WorkbenchAuthoringState,
} from "./graph-document-adapter";
import { authoredGraphDocument } from "../model/graph-document";

const source = {
  id: "source",
  operator_id: "test.source",
  operator_version: 1,
  config: { label: "source" },
  input_plugs: [],
  artifact_type_bindings: [],
  position: { x: 0, y: 0 },
  layout: null,
};

const target = {
  id: "target",
  operator_id: "test.target",
  operator_version: 1,
  config: { label: "target" },
  input_plugs: [],
  artifact_type_bindings: [],
  position: { x: 300, y: 0 },
  layout: null,
};

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

function state(): WorkbenchAuthoringState {
  return {
    document: authoredGraphDocument({
      name: "Draft",
      nodes: [source, target],
      edges: [edge],
    }),
    nodeOverlays: {
      source: {
        run: null,
        execution: { status: "succeeded" },
        progress: null,
      },
      target: {
        run: null,
        execution: { status: "succeeded" },
        progress: null,
      },
    },
    error: null,
  };
}

describe("Workbench authored document adapter", () => {
  it("does not author a move while dragging", () => {
    expect(graphCommandsFromNodeChanges([{
      id: "source",
      type: "position",
      position: { x: 40, y: 50 },
      dragging: true,
    }])).toEqual([]);
  });

  it("authors the final position when dragging stops", () => {
    expect(graphCommandsFromNodeChanges([{
      id: "source",
      type: "position",
      position: { x: 40, y: 50 },
      dragging: false,
    }])).toEqual([{
      kind: "move_nodes",
      positions: [{ node_id: "source", x: 40, y: 50 }],
    }]);
  });

  it("keeps move overlays and scopes config invalidation", () => {
    const moved = reduceWorkbenchAuthoringState(state(), {
      kind: "apply_commands",
      commands: [{
        kind: "move_nodes",
        positions: [{ node_id: "source", x: 40, y: 50 }],
      }],
    });
    const edited = reduceWorkbenchAuthoringState(moved, {
      kind: "apply_commands",
      commands: [{
        kind: "update_node_configuration",
        node_id: "target",
        field: "label",
        value: "edited",
      }],
    });

    expect(moved.nodeOverlays.source?.execution.status).toBe("succeeded");
    expect(edited.nodeOverlays.source?.execution.status).toBe("succeeded");
    expect(edited.nodeOverlays.target?.execution.status).toBe("idle");
    expect(edited.document.nodes[0]?.position).toEqual({ x: 40, y: 50 });
  });

  it("applies sequential stale dispatchers to the latest document", () => {
    let current = state();
    const dispatchMove = () => {
      current = reduceWorkbenchAuthoringState(current, {
        kind: "apply_commands",
        commands: [{
          kind: "move_nodes",
          positions: [{ node_id: "source", x: 80, y: 90 }],
        }],
      });
    };
    const dispatchConfig = () => {
      current = reduceWorkbenchAuthoringState(current, {
        kind: "apply_commands",
        commands: [{
          kind: "update_node_configuration",
          node_id: "target",
          field: "label",
          value: "edited after move",
        }],
      });
    };

    dispatchMove();
    dispatchConfig();

    expect(current.document.nodes[0]?.position).toEqual({ x: 80, y: 90 });
    expect(current.document.nodes[1]?.config).toEqual({
      label: "edited after move",
    });
  });

  it("turns stale commands into a bounded error state", () => {
    const initial = state();
    const result = reduceWorkbenchAuthoringState(initial, {
      kind: "apply_commands",
      commands: [{
        kind: "update_node_configuration",
        node_id: "missing",
        field: "label",
        value: "ignored",
      }],
    });

    expect(result.document).toEqual(initial.document);
    expect(result.nodeOverlays).toEqual(initial.nodeOverlays);
    expect(result.error).toContain("missing node missing");
  });

  it("bounds an update for a missing edge as an adapter error", () => {
    const result = reduceWorkbenchAuthoringState(state(), {
      kind: "apply_commands",
      commands: [{
        kind: "update_edge",
        edge_id: "missing-edge",
        update: { enabled: false },
      }],
    });

    expect(result.document).toEqual(state().document);
    expect(result.error).toContain("missing edge missing-edge");
  });

  it("clears an authoring error when the user dismisses it", () => {
    const errored = reduceWorkbenchAuthoringState(state(), {
      kind: "apply_commands",
      commands: [{
        kind: "update_node_configuration",
        node_id: "missing",
        field: "label",
        value: "ignored",
      }],
    });
    const cleared = reduceWorkbenchAuthoringState(errored, {
      kind: "clear_error",
    });

    expect(cleared.error).toBeNull();
  });
});

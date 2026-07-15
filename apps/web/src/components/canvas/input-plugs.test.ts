import { describe, expect, it } from "vitest";

import type { RunNodeResult } from "@/lib/api";
import {
  collectContributionLabel,
  reorderInputPlug,
  type WorkflowInputPlug,
} from "./input-plugs";

describe("ordered input plugs", () => {
  it("reorders one port by stable id without disturbing another port", () => {
    const plugs: WorkflowInputPlug[] = [
      { id: "a", portName: "items" },
      { id: "b", portName: "items" },
      { id: "note", portName: "note" },
      { id: "c", portName: "items" },
    ];

    const moved = reorderInputPlug(plugs, "items", "c", 0);
    expect(moved).toEqual([
      { id: "c", portName: "items" },
      { id: "a", portName: "items" },
      { id: "note", portName: "note" },
      { id: "b", portName: "items" },
    ]);
    expect(reorderInputPlug(moved, "items", "c", 1)).toEqual([
      { id: "a", portName: "items" },
      { id: "c", portName: "items" },
      { id: "note", portName: "note" },
      { id: "b", portName: "items" },
    ]);
    expect(plugs.map((plug) => plug.id)).toEqual(["a", "b", "note", "c"]);
  });
});

describe("Collect contribution ranges", () => {
  it("formats one-based flattened output ranges from sequence metadata", () => {
    const run: RunNodeResult = {
      node_id: "collect",
      status: "succeeded",
      error: null,
      outputs: [
        {
          port: "items",
          kind: "sequence",
          artifacts: [],
          value: {
            artifact_type: "text",
            schema_version: 1,
            item_refs: [],
            ordered: true,
            index_key: "order_index",
            metadata: {
              collect_segments: [
                {
                  input_index: 0,
                  start_index: 0,
                  item_count: 1,
                  source_kind: "single",
                },
                {
                  input_index: 1,
                  start_index: 1,
                  item_count: 3,
                  source_kind: "sequence",
                },
                {
                  input_index: 2,
                  start_index: 4,
                  item_count: 0,
                  source_kind: "sequence",
                },
              ],
            },
          },
        },
      ],
    };

    expect(collectContributionLabel(run, 0)).toBe("output 1");
    expect(collectContributionLabel(run, 1)).toBe("output 2–4");
    expect(collectContributionLabel(run, 2)).toBe("output empty");
    expect(collectContributionLabel(run, 3)).toBeUndefined();
  });
});

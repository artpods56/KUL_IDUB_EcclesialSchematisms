import { describe, expect, it } from "vitest";

import type { SavedGraphSummary } from "@/lib/api";
import {
  filterGraphsByQuery,
  graphAgeLabel,
  sortGraphsByRecency,
} from "./WorkspaceGraphPanel";

const graph = (
  name: string,
  updatedAt: string,
  overrides: Partial<SavedGraphSummary> = {},
): SavedGraphSummary => ({
  id: `graph-${name}`,
  name,
  node_count: 2,
  edge_count: 1,
  revision: 1,
  updated_at: updatedAt,
  ...overrides,
});

describe("workspace graph panel listing", () => {
  it("orders graphs by most recently updated", () => {
    const ordered = sortGraphsByRecency([
      graph("Older", "2026-08-01T10:00:00Z"),
      graph("Newest", "2026-08-09T10:00:00Z"),
      graph("Middle", "2026-08-05T10:00:00Z"),
    ]);

    expect(ordered.map((entry) => entry.name)).toEqual([
      "Newest",
      "Middle",
      "Older",
    ]);
  });

  it("matches graph names case-insensitively and keeps every graph for a blank query", () => {
    const graphs = [
      graph("Invoice intake", "2026-08-01T10:00:00Z"),
      graph("Payroll", "2026-08-02T10:00:00Z"),
    ];

    expect(filterGraphsByQuery(graphs, "invoice").map((e) => e.name)).toEqual([
      "Invoice intake",
    ]);
    expect(filterGraphsByQuery(graphs, "   ")).toHaveLength(2);
    expect(filterGraphsByQuery(graphs, "nothing")).toHaveLength(0);
  });

  it("summarises graph age in coarse buckets", () => {
    const now = Date.parse("2026-08-10T12:00:00Z");
    expect(graphAgeLabel("2026-08-10T11:59:30Z", now)).toBe("just now");
    expect(graphAgeLabel("2026-08-10T11:15:00Z", now)).toBe("45m ago");
    expect(graphAgeLabel("2026-08-10T09:00:00Z", now)).toBe("3h ago");
    expect(graphAgeLabel("2026-08-06T12:00:00Z", now)).toBe("4d ago");
  });
});

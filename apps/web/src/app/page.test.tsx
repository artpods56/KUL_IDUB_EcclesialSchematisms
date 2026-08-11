import { describe, expect, it, vi } from "vitest";

vi.mock("@/features/graphs/GraphBrowser", () => {
  const GraphBrowser = () => null;
  return { GraphBrowser, default: GraphBrowser };
});

import { GraphBrowser } from "@/features/graphs/GraphBrowser";
import HomePage from "./page";

describe("post-login routing", () => {
  it("uses the graph browser as the authenticated root experience", () => {
    expect(HomePage).toBe(GraphBrowser);
  });

  it("uses the same canonical browser at /graphs", async () => {
    const graphsRoute = await import("./graphs/page");
    expect(graphsRoute.default).toBe(GraphBrowser);
  });
});

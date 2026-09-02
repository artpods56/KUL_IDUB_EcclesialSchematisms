import { describe, expect, it, vi } from "vitest";

vi.mock("@/features/graphs/WorkspaceGraphRedirect", () => {
  const WorkspaceGraphRedirect = () => null;
  return { WorkspaceGraphRedirect, default: WorkspaceGraphRedirect };
});

import { WorkspaceGraphRedirect } from "@/features/graphs/WorkspaceGraphRedirect";
import HomePage from "./page";

describe("post-login routing", () => {
  it("resolves the authenticated root to a workspace graph route", () => {
    expect(HomePage).toBe(WorkspaceGraphRedirect);
  });

  it("keeps /graphs as a compatible redirect", async () => {
    const graphsRoute = await import("./graphs/page");
    expect(graphsRoute.default).toBe(WorkspaceGraphRedirect);
  });
});

import { describe, expect, it } from "vitest";

import type { Workspace } from "@/lib/api";
import {
  workspaceCanManageMembers,
  workspaceRouteAccessState,
  workspaceRouteShowsRail,
} from "./WorkspaceLayout";

const workspace = (capabilities: readonly Workspace["capabilities"][number][]): Workspace => ({
  id: "workspace-1",
  name: "Operations",
  slug: "operations",
  kind: "shared",
  role: "owner",
  capabilities,
});

describe("workspace route and capability state", () => {
  it("distinguishes direct missing routes from revoked access", () => {
    expect(workspaceRouteAccessState("unknown", undefined, undefined)).toBe("missing");
    expect(workspaceRouteAccessState("operations", undefined, { slug: "operations", id: "workspace-1" })).toBe("revoked");
    expect(workspaceRouteAccessState("operations", workspace([]), { slug: "operations", id: "workspace-1" })).toBe("available");
  });

  it("only exposes member management while the server capability is present", () => {
    expect(workspaceCanManageMembers(workspace(["manage_members"]))).toBe(true);
    expect(workspaceCanManageMembers(workspace([]))).toBe(false);
  });

  it("hides the workspace rail on graph workbench routes", () => {
    expect(workspaceRouteShowsRail("/workspaces")).toBe(true);
    expect(workspaceRouteShowsRail("/workspaces/local")).toBe(true);
    expect(workspaceRouteShowsRail("/workspaces/local/graphs/new")).toBe(false);
    expect(
      workspaceRouteShowsRail(
        "/workspaces/local/graphs/00000000-0000-0000-0000-000000000001",
      ),
    ).toBe(false);
  });
});

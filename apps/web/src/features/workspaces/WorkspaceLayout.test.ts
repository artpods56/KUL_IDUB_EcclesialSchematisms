import { describe, expect, it } from "vitest";

import type { Workspace } from "@/lib/api";
import {
  resolveSelectedWorkspace,
  sessionDisplayName,
  sessionInitials,
  workspaceCanManageMembers,
  workspaceDisplayName,
  workspaceRouteAccessState,
  workspaceRouteGraphId,
  workspaceSelectorLabel,
} from "./WorkspaceLayout";

const workspace = (
  capabilities: readonly Workspace["capabilities"][number][],
  overrides: Partial<Workspace> = {},
): Workspace => ({
  id: "workspace-1",
  name: "Operations",
  slug: "operations",
  kind: "shared",
  role: "owner",
  capabilities,
  ...overrides,
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

  it("resolves the open graph so the rail can highlight it", () => {
    expect(workspaceRouteGraphId("/workspaces/local")).toBeNull();
    expect(workspaceRouteGraphId("/workspaces/local/graphs/new")).toBeNull();
    expect(
      workspaceRouteGraphId(
        "/workspaces/local/graphs/00000000-0000-0000-0000-000000000001",
      ),
    ).toBe("00000000-0000-0000-0000-000000000001");
    expect(
      workspaceRouteGraphId(
        "/workspaces/local/graphs/00000000-0000-0000-0000-000000000001/runs",
      ),
    ).toBe("00000000-0000-0000-0000-000000000001");
  });

  it("labels a personal workspace as My graphs in the UI", () => {
    expect(
      workspaceDisplayName(
        workspace([], { kind: "personal", name: "Personal workspace" }),
      ),
    ).toBe("My graphs");
    expect(workspaceDisplayName(workspace([], { name: "Operations" }))).toBe(
      "Operations",
    );
  });

  it("labels the workspace switcher as Personal by default", () => {
    expect(workspaceSelectorLabel(undefined)).toBe("Personal");
    expect(
      workspaceSelectorLabel(
        workspace([], { kind: "personal", name: "Personal workspace" }),
      ),
    ).toBe("Personal");
    expect(
      workspaceSelectorLabel(workspace([], { name: "Operations" })),
    ).toBe("Operations");
  });

  it("resolves the selected workspace to the active slug, else personal", () => {
    const personal = workspace([], {
      kind: "personal",
      id: "workspace-personal",
      slug: "personal",
    });
    const team = workspace([], {
      id: "workspace-team",
      slug: "operations",
      name: "Operations",
    });
    expect(
      resolveSelectedWorkspace([personal, team], "operations"),
    ).toBe(team);
    expect(resolveSelectedWorkspace([personal, team], "missing")).toBe(
      personal,
    );
    expect(resolveSelectedWorkspace([personal, team], undefined)).toBe(
      personal,
    );
    expect(resolveSelectedWorkspace(undefined, undefined)).toBeUndefined();
  });

  it("derives account label and initials from the session profile", () => {
    expect(
      sessionDisplayName({
        display_name: "Ada Lovelace",
        email: "ada@example.test",
        user_id: "user-1",
      }),
    ).toBe("Ada Lovelace");
    expect(
      sessionInitials({
        display_name: "Ada Lovelace",
        email: "ada@example.test",
        user_id: "user-1",
      }),
    ).toBe("AL");
    expect(
      sessionInitials({
        display_name: null,
        email: "ada@example.test",
        user_id: "user-1",
      }),
    ).toBe("AD");
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  addWorkspaceMember,
  changeWorkspaceMemberRole,
  createWorkspace,
  listWorkspaceMembers,
  listWorkspaces,
  removeWorkspaceMember,
} from "./workspaces";

afterEach(() => vi.unstubAllGlobals());

describe("workspace API client", () => {
  it("uses UUID workspace paths and generated request bodies", async () => {
    const fetchMock = vi.fn().mockImplementation((_: string, init: RequestInit) => {
      const status = init.method === "DELETE" ? 204 : 200;
      return Promise.resolve(new Response(status === 204 ? null : "{}", { status }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const workspaceId = "workspace-uuid";
    const userId = "user-uuid";

    await listWorkspaces();
    await createWorkspace({ name: "Shared", slug: "shared" });
    await listWorkspaceMembers(workspaceId);
    await addWorkspaceMember(workspaceId, { user_id: userId, role: "viewer" });
    await changeWorkspaceMemberRole(workspaceId, userId, { role: "editor" });
    await removeWorkspaceMember(workspaceId, userId);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/workspaces",
      "/api/v1/workspaces",
      "/api/v1/workspaces/workspace-uuid/members",
      "/api/v1/workspaces/workspace-uuid/members",
      "/api/v1/workspaces/workspace-uuid/members/user-uuid",
      "/api/v1/workspaces/workspace-uuid/members/user-uuid",
    ]);
    expect(JSON.parse(fetchMock.mock.calls[1]?.[1].body as string)).toEqual({ name: "Shared", slug: "shared" });
    expect(JSON.parse(fetchMock.mock.calls[3]?.[1].body as string)).toEqual({ user_id: userId, role: "viewer" });
    expect(JSON.parse(fetchMock.mock.calls[4]?.[1].body as string)).toEqual({ role: "editor" });
  });
});

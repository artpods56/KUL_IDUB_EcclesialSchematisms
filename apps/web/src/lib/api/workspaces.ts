import { request } from "./client";
import type {
  Workspace,
  WorkspaceCreateRequest,
  WorkspaceMember,
  WorkspaceMemberRequest,
  WorkspaceMemberRoleRequest,
} from "./contract";

export function listWorkspaces(signal?: AbortSignal) {
  return request<readonly Workspace[]>("GET", "/v1/workspaces", { signal });
}

export function createWorkspace(body: WorkspaceCreateRequest) {
  return request<Workspace>("POST", "/v1/workspaces", { body });
}

export function listWorkspaceMembers(workspaceId: string, signal?: AbortSignal) {
  return request<readonly WorkspaceMember[]>(
    "GET",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
    { signal },
  );
}

export function addWorkspaceMember(
  workspaceId: string,
  body: WorkspaceMemberRequest,
) {
  return request<WorkspaceMember>(
    "POST",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
    { body },
  );
}

export function changeWorkspaceMemberRole(
  workspaceId: string,
  userId: string,
  body: WorkspaceMemberRoleRequest,
) {
  return request<WorkspaceMember>(
    "PATCH",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
    { body },
  );
}

export function removeWorkspaceMember(workspaceId: string, userId: string) {
  return request<undefined>(
    "DELETE",
    `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
  );
}

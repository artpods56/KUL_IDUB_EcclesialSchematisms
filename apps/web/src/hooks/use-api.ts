"use client";

import useSWR from "swr";
import {
  listWorkspaceMembers,
  listWorkspaces,
  type NodeRegistry,
  type SavedGraphList,
  type Workspace,
  type WorkspaceMember,
} from "@/lib/api";

/** Keyed SWR hooks over the Notarius API (global fetcher is `apiFetcher`). */

export function useNodeRegistry(workspaceId?: string) {
  return useSWR<NodeRegistry>(
    workspaceId
      ? `/v1/workspaces/${encodeURIComponent(workspaceId)}/nodes`
      : null,
  );
}

export function useSavedGraphs(workspaceId?: string) {
  return useSWR<SavedGraphList>(
    workspaceId
      ? `/v1/workspaces/${encodeURIComponent(workspaceId)}/graphs`
      : null,
  );
}

export function useWorkspaces(userId: string | undefined) {
  return useSWR<readonly Workspace[]>(
    userId ? ["workspaces", userId] : null,
    () => listWorkspaces(),
  );
}

export function useWorkspaceMembers(
  userId: string | undefined,
  workspaceId: string | undefined,
) {
  return useSWR<readonly WorkspaceMember[]>(
    userId && workspaceId
      ? ["workspace-members", userId, workspaceId]
      : null,
    () => listWorkspaceMembers(workspaceId!),
  );
}

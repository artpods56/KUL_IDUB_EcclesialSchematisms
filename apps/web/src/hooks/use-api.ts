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

export function useNodeRegistry() {
  return useSWR<NodeRegistry>("/v1/nodes");
}

export function useSavedGraphs() {
  return useSWR<SavedGraphList>("/v1/graphs");
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

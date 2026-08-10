"use client";

import * as React from "react";
import useSWR from "swr";
import {
  listWorkspaceMembers,
  listWorkspaces,
  type NodeRegistry,
  type SavedGraphList,
  type SavedGraphSummary,
  type Workspace,
  type WorkspaceMember,
} from "@/lib/api";

/** Keyed SWR hooks over the Grafy API (global fetcher is `apiFetcher`). */

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

export function useAllWorkspacesGraphs(
  workspaces: readonly { id: string; slug: string }[] | undefined,
) {
  const keys = (workspaces ?? []).map((w) =>
    `/v1/workspaces/${encodeURIComponent(w.id)}/graphs`,
  );
  const graphs = useSWR<SavedGraphList[]>(keys);
  return React.useMemo(() => {
    if (!graphs.data || !workspaces) return null;
    const all: (SavedGraphSummary & {
      _workspace: { id: string; slug: string };
    })[] = [];
    for (let i = 0; i < workspaces.length; i++) {
      for (const g of graphs.data[i]?.graphs ?? []) {
        all.push({ ...g, _workspace: workspaces[i]! });
      }
    }
    return all.sort((a, b) =>
      Date.parse(b.updated_at) - Date.parse(a.updated_at),
    );
  }, [graphs.data, workspaces]);
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

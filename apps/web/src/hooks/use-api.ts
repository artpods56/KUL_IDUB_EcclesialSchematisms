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
import { request } from "@/lib/api/client";

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

export type GraphLocation = Pick<
  Workspace,
  "id" | "slug" | "name" | "kind"
>;

export interface LocatedGraph extends SavedGraphSummary {
  location: GraphLocation;
}

export interface GraphLocationFailure {
  location: GraphLocation;
  error: Error;
}

export interface AllWorkspacesGraphsResult {
  graphs: readonly LocatedGraph[] | null;
  failures: readonly GraphLocationFailure[];
  isLoading: boolean;
  retry: () => Promise<void>;
}

interface WorkspaceGraphLoad {
  workspaceId: string;
  data: SavedGraphList | null;
  error: Error | null;
}

export function useAllWorkspacesGraphs(
  workspaces: readonly Workspace[] | undefined,
): AllWorkspacesGraphsResult {
  const workspaceIds = workspaces?.map((workspace) => workspace.id);
  const loads = useSWR<readonly WorkspaceGraphLoad[]>(
    workspaceIds && workspaceIds.length > 0
      ? ["all-workspaces-graphs", ...workspaceIds]
      : null,
    async ([, ...ids]: readonly string[]) =>
      Promise.all(
        ids.map(async (workspaceId): Promise<WorkspaceGraphLoad> => {
          try {
            const data = await request<SavedGraphList>(
              "GET",
              `/v1/workspaces/${encodeURIComponent(workspaceId)}/graphs`,
            );
            return { workspaceId, data, error: null };
          } catch (caught) {
            const error =
              caught instanceof Error
                ? caught
                : new Error("The graph list request failed.");
            return { workspaceId, data: null, error };
          }
        }),
      ),
    { shouldRetryOnError: false },
  );

  const state = React.useMemo(() => {
    if (!workspaces) {
      return { graphs: null, failures: [] };
    }
    if (workspaces.length === 0) {
      return { graphs: [], failures: [] };
    }
    if (!loads.data) {
      return { graphs: null, failures: [] };
    }

    const locations = new Map(
      workspaces.map((workspace) => [workspace.id, workspace] as const),
    );
    const graphs: LocatedGraph[] = [];
    const failures: GraphLocationFailure[] = [];

    for (const load of loads.data) {
      const workspace = locations.get(load.workspaceId);
      if (!workspace) continue;
      const location: GraphLocation = {
        id: workspace.id,
        slug: workspace.slug,
        name: workspace.name,
        kind: workspace.kind,
      };
      if (load.error) {
        failures.push({ location, error: load.error });
        continue;
      }
      for (const graph of load.data?.graphs ?? []) {
        graphs.push({ ...graph, location });
      }
    }

    graphs.sort(
      (left, right) =>
        Date.parse(right.updated_at) - Date.parse(left.updated_at),
    );
    return { graphs, failures };
  }, [loads.data, workspaces]);

  const retry = React.useCallback(async () => {
    if (!workspaceIds?.length) return;
    await loads.mutate();
  }, [loads, workspaceIds]);

  return {
    ...state,
    isLoading: Boolean(workspaces?.length) && loads.isLoading,
    retry,
  };
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

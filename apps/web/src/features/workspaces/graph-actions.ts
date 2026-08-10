import {
  deleteSavedGraph,
  getSavedGraph,
  updateSavedGraph,
  type SavedGraphSummary,
} from "@/lib/api";

/** Persist a rename for a graph that is not open in the workbench. */
export async function renameSavedGraphRemote(
  workspaceId: string,
  graph: SavedGraphSummary,
  name: string,
): Promise<void> {
  const saved = await getSavedGraph(workspaceId, graph.id);
  await updateSavedGraph(workspaceId, graph.id, {
    name,
    nodes: saved.nodes ?? [],
    edges: saved.edges ?? [],
    presentation: saved.presentation ?? {
      viewers: [],
      links: [],
      bindings: [],
      annotations: [],
    },
    expected_revision: saved.revision,
  });
}

/** Delete a graph when the workbench lifecycle is not mounted. */
export async function deleteSavedGraphRemote(
  workspaceId: string,
  graph: SavedGraphSummary,
): Promise<boolean> {
  if (!window.confirm(`Delete “${graph.name}”? This cannot be undone.`)) {
    return false;
  }
  await deleteSavedGraph(workspaceId, graph.id, graph.revision);
  return true;
}

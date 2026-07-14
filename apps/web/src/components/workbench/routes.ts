export const LOCAL_WORKSPACE_SLUG = "local";
export const NEW_GRAPH_ROUTE_ID = "new";

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function isSupportedWorkbenchGraphRoute(
  workspaceSlug: string,
  graphId: string,
): boolean {
  return (
    workspaceSlug === LOCAL_WORKSPACE_SLUG &&
    (graphId === NEW_GRAPH_ROUTE_ID || UUID_PATTERN.test(graphId))
  );
}

export function workbenchGraphPath(
  workspaceSlug: string,
  graphId: string,
): string {
  return `/workspaces/${encodeURIComponent(workspaceSlug)}/graphs/${encodeURIComponent(graphId)}`;
}

export function saveAsTemplatePath(
  sourceWorkspaceId: string,
  sourceGraphId: string,
  sourceRevision: number,
): string {
  const params = new URLSearchParams({
    sourceWorkspaceId,
    sourceGraphId,
    sourceRevision: String(sourceRevision),
  });
  return `/templates/new?${params.toString()}`;
}

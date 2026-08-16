import {
  SaveAsTemplate,
  type SaveAsTemplateSource,
} from "@/features/templates/SaveAsTemplate";

interface SaveAsTemplatePageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function SaveAsTemplatePage({
  searchParams,
}: SaveAsTemplatePageProps) {
  const values = await searchParams;
  const workspaceId = first(values.sourceWorkspaceId);
  const graphId = first(values.sourceGraphId);
  const rawRevision = first(values.sourceRevision);
  const revision = rawRevision ? Number(rawRevision) : Number.NaN;
  const source: SaveAsTemplateSource | null =
    workspaceId && graphId && Number.isSafeInteger(revision) && revision > 0
      ? { workspaceId, graphId, revision }
      : null;
  return <SaveAsTemplate source={source} />;
}

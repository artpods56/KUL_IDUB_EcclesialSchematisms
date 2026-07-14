import { notFound } from "next/navigation";

import { isSupportedWorkbenchGraphRoute } from "@/components/workbench/routes";

interface GraphPageProps {
  params: Promise<{
    workspaceSlug: string;
    graphId: string;
  }>;
}

export default async function GraphPage({ params }: GraphPageProps) {
  const { workspaceSlug, graphId } = await params;
  if (!isSupportedWorkbenchGraphRoute(workspaceSlug, graphId)) {
    notFound();
  }

  return null;
}

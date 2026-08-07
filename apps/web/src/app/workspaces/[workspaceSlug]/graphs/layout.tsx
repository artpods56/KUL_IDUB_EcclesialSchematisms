"use client";

import type { ReactNode } from "react";
import { useParams } from "next/navigation";

import {
  Workbench,
} from "@/features/workbench";
import {
  NEW_GRAPH_ROUTE_ID,
  isSupportedWorkbenchGraphRoute,
} from "@/features/workbench/routes";
import { useWorkspaceContext } from "@/features/workspaces/WorkspaceLayout";

interface GraphsLayoutProps {
  children: ReactNode;
}

export default function GraphsLayout({ children }: GraphsLayoutProps) {
  const { workspaceSlug, graphId } = useParams<{
    workspaceSlug: string;
    graphId: string;
  }>();
  const { workspace } = useWorkspaceContext();

  if (
    workspace.slug !== "local" ||
    !isSupportedWorkbenchGraphRoute(workspaceSlug, graphId)
  ) {
    return children;
  }

  return (
    <>
      <Workbench
        workspaceSlug={workspaceSlug}
        initialGraphId={graphId === NEW_GRAPH_ROUTE_ID ? null : graphId}
      />
      {children}
    </>
  );
}

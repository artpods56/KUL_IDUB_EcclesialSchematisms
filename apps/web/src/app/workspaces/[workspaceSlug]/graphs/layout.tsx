"use client";

import type { ReactNode } from "react";
import { useParams } from "next/navigation";

import { Workbench } from "@/components/workbench/Workbench";
import {
  NEW_GRAPH_ROUTE_ID,
  isSupportedWorkbenchGraphRoute,
} from "@/components/workbench/routes";

interface GraphsLayoutProps {
  children: ReactNode;
}

export default function GraphsLayout({ children }: GraphsLayoutProps) {
  const { workspaceSlug, graphId } = useParams<{
    workspaceSlug: string;
    graphId: string;
  }>();

  if (!isSupportedWorkbenchGraphRoute(workspaceSlug, graphId)) {
    return children;
  }

  return (
    <>
      <Workbench
        workspaceSlug={workspaceSlug}
        initialGraphId={graphId === NEW_GRAPH_ROUTE_ID ? null : graphId}
        seedExample={false}
      />
      {children}
    </>
  );
}

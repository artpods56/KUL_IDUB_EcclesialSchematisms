"use client";

import * as React from "react";
import { Copy, LogOut, Users, Workflow } from "lucide-react";
import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";

import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { useWorkspaces } from "@/hooks/use-api";
import type { Workspace } from "@/lib/api";

interface WorkspaceContextValue {
  workspace: Workspace;
  workspaces: readonly Workspace[];
  refreshWorkspaces: () => Promise<readonly Workspace[] | undefined>;
}

export type WorkspaceRouteAccessState = "available" | "missing" | "revoked";

export function workspaceCanManageMembers(workspace: Workspace): boolean {
  return workspace.capabilities.includes("manage_members");
}

/** Graph workbench is full-bleed; keep the rail on directory/overview only. */
export function workspaceRouteShowsRail(pathname: string): boolean {
  return !/\/graphs(?:\/|$)/.test(pathname);
}

export function workspaceRouteAccessState(
  workspaceSlug: string,
  workspace: Workspace | undefined,
  previouslyResolvedWorkspace: Pick<Workspace, "slug" | "id"> | undefined,
): WorkspaceRouteAccessState {
  if (workspace) return "available";
  return previouslyResolvedWorkspace?.slug === workspaceSlug ? "revoked" : "missing";
}

const WorkspaceContext = React.createContext<WorkspaceContextValue | null>(null);

export function useWorkspaceContext(): WorkspaceContextValue {
  const context = React.useContext(WorkspaceContext);
  if (!context) throw new Error("useWorkspaceContext must be used inside a workspace route");
  return context;
}

export function WorkspaceRail({
  workspaces,
  activeSlug,
  userId,
  onLogout,
}: {
  workspaces: readonly Workspace[];
  activeSlug?: string;
  userId: string;
  onLogout: () => Promise<void>;
}) {
  const router = useRouter();
  const [copied, setCopied] = React.useState(false);

  const copyIdentity = async () => {
    try {
      await navigator.clipboard.writeText(userId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <aside className="ns-workspace-rail" aria-label="Workspace navigation">
      <button
        type="button"
        className="ns-workspace-rail__brand"
        aria-label="All workspaces"
        onClick={() => router.push("/workspaces")}
      >
        N
      </button>
      <div className="ns-workspace-rail__items">
        {workspaces.map((workspace) => (
          <button
            key={workspace.id}
            type="button"
            className={`ns-workspace-rail__item${activeSlug === workspace.slug ? " is-active" : ""}`}
            title={`${workspace.name} · ${workspace.role}`}
            aria-label={workspace.name}
            onClick={() => router.push(`/workspaces/${encodeURIComponent(workspace.slug)}`)}
          >
            {workspace.kind === "personal" ? <Workflow size={15} /> : <Users size={15} />}
            <span>{workspace.name}</span>
          </button>
        ))}
      </div>
      <div className="ns-workspace-rail__footer">
        <button
          type="button"
          className="ns-workspace-rail__identity"
          onClick={() => void copyIdentity()}
          title={userId}
        >
          <span className="ns-workspace-rail__identity-label">{copied ? "Copied" : "User ID"}</span>
          <span className="ns-workspace-rail__identity-value">{userId.slice(0, 8)}…</span>
          <Copy size={12} />
        </button>
        <button
          type="button"
          className="ns-workspace-rail__logout"
          onClick={() => void onLogout()}
          title="Sign out"
          aria-label="Sign out"
        >
          <LogOut size={15} />
        </button>
      </div>
    </aside>
  );
}

function WorkspaceRouteStatus({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="ns-workspace-route-status">
      <p className="ns-workspace-route-status__eyebrow">WORKSPACE</p>
      <h1>{title}</h1>
      <p>{detail}</p>
      <Link href="/workspaces">Return to workspaces</Link>
    </main>
  );
}

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  const { workspaceSlug } = useParams<{ workspaceSlug: string }>();
  const pathname = usePathname() ?? "";
  const { session, logout } = useAuthSession();
  const { data, error, mutate } = useWorkspaces(session.user_id);
  const [previouslyResolvedWorkspace, setPreviouslyResolvedWorkspace] =
    React.useState<Pick<Workspace, "slug" | "id"> | undefined>(undefined);
  const showRail = workspaceRouteShowsRail(pathname);

  const workspace = data?.find((candidate) => candidate.slug === workspaceSlug);
  if (
    workspace &&
    (previouslyResolvedWorkspace?.slug !== workspace.slug ||
      previouslyResolvedWorkspace.id !== workspace.id)
  ) {
    setPreviouslyResolvedWorkspace({ slug: workspace.slug, id: workspace.id });
  }
  const routeAccessState = workspaceRouteAccessState(
    workspaceSlug,
    workspace,
    previouslyResolvedWorkspace,
  );

  if (error) {
    return <WorkspaceRouteStatus title="Workspaces unavailable" detail="Notarius could not confirm access to this workspace." />;
  }
  if (!data) {
    return <WorkspaceRouteStatus title="Loading workspace" detail="Checking your current workspace access…" />;
  }

  if (!workspace) {
    return routeAccessState === "revoked"
      ? <WorkspaceRouteStatus title="Workspace access revoked" detail="Your access to this workspace is no longer available." />
      : <WorkspaceRouteStatus title="Workspace not found" detail="No workspace with this slug is available to this session." />;
  }

  return (
    <WorkspaceContext.Provider
      value={{
        workspace,
        workspaces: data,
        refreshWorkspaces: () => mutate(),
      }}
    >
      <div className="ns-workspace-frame">
        {showRail ? (
          <WorkspaceRail
            workspaces={data}
            activeSlug={workspace.slug}
            userId={session.user_id}
            onLogout={logout}
          />
        ) : null}
        <main className="ns-workspace-frame__main">{children}</main>
      </div>
    </WorkspaceContext.Provider>
  );
}

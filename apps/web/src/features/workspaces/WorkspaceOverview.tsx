"use client";

import { ArrowUpRight, ShieldCheck, Users, Workflow } from "lucide-react";
import Link from "next/link";

import { useWorkspaceContext } from "./WorkspaceLayout";
import { WorkspaceMembersDialog } from "./WorkspaceMembersDialog";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";

export function WorkspaceOverview() {
  const { workspace } = useWorkspaceContext();
  const { session } = useAuthSession();
  const canManageMembers = workspace.capabilities.includes("manage_members");
  const isLocal = workspace.slug === "local";

  return (
    <div className="ns-workspace-overview">
      <header className="ns-workspace-overview__header">
        <div>
          <p className="ns-workspace-overview__eyebrow">WORKSPACE / {workspace.kind}</p>
          <h1>{workspace.name}</h1>
          <p className="ns-workspace-overview__slug">/{workspace.slug} · {workspace.role}</p>
        </div>
        <div className="ns-workspace-overview__actions">
          {canManageMembers ? <WorkspaceMembersDialog /> : null}
          {isLocal ? (
            <Link className="ns-workspace-button ns-workspace-button--primary" href="/workspaces/local/graphs/new">
              Open workbench <ArrowUpRight size={14} />
            </Link>
          ) : null}
        </div>
      </header>

      <section className="ns-workspace-overview__section" aria-labelledby="workspace-access-heading">
        <div className="ns-workspace-overview__section-heading">
          <div>
            <p className="ns-workspace-overview__eyebrow">ACCESS</p>
            <h2 id="workspace-access-heading">Server-authorized capabilities</h2>
          </div>
          <ShieldCheck size={18} aria-hidden="true" />
        </div>
        <div className="ns-workspace-capabilities">
          {workspace.capabilities.map((capability) => <span key={capability}>{capability.replaceAll("_", " ")}</span>)}
        </div>
      </section>

      <section className="ns-workspace-overview__section" aria-labelledby="workspace-surface-heading">
        <div className="ns-workspace-overview__section-heading">
          <div>
            <p className="ns-workspace-overview__eyebrow">SURFACE</p>
            <h2 id="workspace-surface-heading">{isLocal ? "Graph workbench" : "Workspace operations"}</h2>
          </div>
          {isLocal ? <Workflow size={18} aria-hidden="true" /> : <Users size={18} aria-hidden="true" />}
        </div>
        <p className="ns-workspace-overview__copy">
          {isLocal
            ? "The current node canvas remains available for the local workspace while tenant graph routes are prepared."
            : "This workspace is ready for overview and member operations. Tenant graph resources are not enabled here yet."}
        </p>
      </section>

      <p className="ns-workspace-overview__identity">Signed in as user <button type="button" onClick={() => void navigator.clipboard.writeText(session.user_id)}>{session.user_id}</button></p>
    </div>
  );
}

"use client";

import { ArrowUpRight, ShieldCheck, Users, Workflow } from "lucide-react";
import Link from "next/link";

import {
  useWorkspaceContext,
  workspaceCanManageMembers,
} from "./WorkspaceLayout";
import { WorkspaceMembersDialog } from "./WorkspaceMembersDialog";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";

export function WorkspaceOverview() {
  const { workspace } = useWorkspaceContext();
  const { session } = useAuthSession();
  const canManageMembers = workspaceCanManageMembers(workspace);
  const workbenchHref = `/workspaces/${encodeURIComponent(workspace.slug)}/graphs/new`;

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
          <Link className="ns-workspace-button ns-workspace-button--primary" href={workbenchHref}>
            Open workbench <ArrowUpRight size={14} />
          </Link>
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
          {workspace.capabilities.map((capability) => (
            <span key={capability}>{capability.replaceAll("_", " ")}</span>
          ))}
        </div>
      </section>

      <section className="ns-workspace-overview__section" aria-labelledby="workspace-surface-heading">
        <div className="ns-workspace-overview__section-heading">
          <div>
            <p className="ns-workspace-overview__eyebrow">SURFACE</p>
            <h2 id="workspace-surface-heading">Graph workbench</h2>
          </div>
          {workspace.kind === "personal" ? (
            <Workflow size={18} aria-hidden="true" />
          ) : (
            <Users size={18} aria-hidden="true" />
          )}
        </div>
        <p className="ns-workspace-overview__copy">
          Open the workbench to author graphs in this workspace. Create additional
          shared workspaces from the{" "}
          <Link href="/workspaces">workspaces directory</Link>.
        </p>
      </section>

      <p className="ns-workspace-overview__identity">
        Signed in as user{" "}
        <button
          type="button"
          onClick={() => void navigator.clipboard.writeText(session.user_id)}
        >
          {session.user_id}
        </button>
      </p>
    </div>
  );
}

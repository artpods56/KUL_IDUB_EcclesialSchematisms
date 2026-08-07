"use client";

import * as React from "react";
import { ArrowUpRight, Plus, Users, Workflow } from "lucide-react";
import Link from "next/link";

import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { WorkspaceRail } from "@/features/workspaces/WorkspaceLayout";
import { createWorkspace } from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { useWorkspaces } from "@/hooks/use-api";

function slugFromName(name: string): string {
  const slug = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug || "shared-workspace";
}

export default function WorkspacesPage() {
  const { session, logout } = useAuthSession();
  const { data: workspaces, error, mutate } = useWorkspaces(session.user_id);
  const [name, setName] = React.useState("");
  const [createOpen, setCreateOpen] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState<string | null>(null);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedName = name.trim();
    if (!normalizedName) return;
    setBusy(true);
    setMessage(null);
    try {
      const workspace = await createWorkspace({ name: normalizedName, slug: slugFromName(normalizedName) });
      await mutate((current) => current ? [...current, workspace] : [workspace], { revalidate: false });
      setName("");
      setCreateOpen(false);
    } catch (caught) {
      setMessage(caught instanceof ApiError && caught.status === 409 ? "That workspace slug is already in use." : "The shared workspace could not be created.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ns-workspace-directory">
      {workspaces ? <WorkspaceRail workspaces={workspaces} userId={session.user_id} onLogout={logout} /> : null}
      <main className="ns-workspace-directory__main">
        <header className="ns-workspace-directory__header">
          <div>
            <p className="ns-workspace-overview__eyebrow">NOTARIUS / WORKSPACES</p>
            <h1>Your workspaces</h1>
            <p>Choose an operational surface. Access is returned by the server for this session.</p>
          </div>
          <button type="button" className="ns-workspace-button ns-workspace-button--primary" onClick={() => setCreateOpen((current) => !current)}>
            <Plus size={14} /> New shared workspace
          </button>
        </header>

        {createOpen ? (
          <form className="ns-workspace-create" onSubmit={(event) => void submit(event)}>
            <label>Workspace name<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Planning room" /></label>
            <button type="submit" className="ns-workspace-button ns-workspace-button--primary" disabled={busy}>{busy ? "Creating…" : "Create workspace"}</button>
            {message ? <span className="ns-member-message" role="status">{message}</span> : null}
          </form>
        ) : null}
        {error ? <p className="ns-workspace-route-status__inline">Workspaces could not be loaded.</p> : null}
        {!workspaces ? <p className="ns-workspace-directory__loading">Loading workspaces…</p> : workspaces.length === 0 ? <p className="ns-workspace-directory__loading">No workspaces are available to this session.</p> : (
          <div className="ns-workspace-list">
            {workspaces.map((workspace) => (
              <Link className="ns-workspace-list__row" key={workspace.id} href={`/workspaces/${encodeURIComponent(workspace.slug)}`}>
                <span className="ns-workspace-list__icon">{workspace.kind === "personal" ? <Workflow size={16} /> : <Users size={16} />}</span>
                <span className="ns-workspace-list__copy"><strong>{workspace.name}</strong><small>/{workspace.slug} · {workspace.role}</small></span>
                <span className="ns-workspace-list__caps">{workspace.capabilities.length} capabilities</span>
                <ArrowUpRight size={15} aria-hidden="true" />
              </Link>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

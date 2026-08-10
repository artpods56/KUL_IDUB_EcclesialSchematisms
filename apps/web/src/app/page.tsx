"use client";

import * as React from "react";
import { ArrowUpRight, Plus, Workflow } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { BrandLoader } from "@/components/brand";
import {
  WorkspaceRail,
  isPersonalWorkspace,
  isSharedWithMe,
  sessionDisplayName,
  workspaceDisplayName,
} from "@/features/workspaces/WorkspaceLayout";
import { graphAgeLabel } from "@/features/workspaces/WorkspaceGraphPanel";
import { createWorkspace } from "@/lib/api";
import type { Workspace } from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { useAllWorkspacesGraphs, useWorkspaces } from "@/hooks/use-api";
import { workbenchGraphPath } from "@/features/workbench/routes";

function slugFromName(name: string): string {
  return (
    name
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "shared-workspace"
  );
}

const RECENT_GRAPH_LIMIT = 6;

export default function HomePage() {
  const router = useRouter();
  const { session, logout } = useAuthSession();
  const { data: workspaces } = useWorkspaces(session.user_id);
  const recentGraphs = useAllWorkspacesGraphs(workspaces)?.slice(0, RECENT_GRAPH_LIMIT) ?? null;

  const displayName = sessionDisplayName(session);

  const personal = React.useMemo(
    () => (workspaces ?? []).filter(isPersonalWorkspace),
    [workspaces],
  );
  const shared = React.useMemo(
    () => (workspaces ?? []).filter(isSharedWithMe),
    [workspaces],
  );
  const ownedShared = React.useMemo(
    () =>
      (workspaces ?? []).filter(
        (w) => w.kind === "shared" && w.role === "owner",
      ),
    [workspaces],
  );

  const [createOpen, setCreateOpen] = React.useState(false);
  const [name, setName] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState<string | null>(null);

  const handleCreate = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    setMessage(null);
    try {
      const created = await createWorkspace({
        name: trimmed,
        slug: slugFromName(trimmed),
      });
      setName("");
      setCreateOpen(false);
      router.push(`/workspaces/${encodeURIComponent(created.slug)}`);
    } catch (err) {
      setMessage(
        err instanceof ApiError && err.status === 409
          ? "That workspace slug is already in use."
          : "The workspace could not be created.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ns-home">
      {workspaces ? (
        <WorkspaceRail
          workspaces={workspaces}
          session={session}
          onLogout={logout}
          onBrandClick={() => window.scrollTo({ top: 0 })}
        />
      ) : null}
      <main className="ns-home__main">
        <header className="ns-home__header">
          <div>
            <h1>
              Hello, {displayName}
            </h1>
            <p className="ns-home__sub">
              Pick up where you left off, or start something new.
            </p>
          </div>
          <button
            type="button"
            className="ns-workspace-button"
            onClick={() => setCreateOpen((v) => !v)}
          >
            <Plus size={14} />
            {createOpen ? "Cancel" : "Create workspace"}
          </button>
        </header>

        {createOpen ? (
          <form className="ns-workspace-create" onSubmit={handleCreate}>
            <label>
              Workspace name
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Planning room"
                autoFocus
              />
              <span className="ns-workspace-create__hint">
                Slug: <code>/{slugFromName(name || "shared-workspace")}</code>
              </span>
            </label>
            <button
              type="submit"
              className="ns-workspace-button ns-workspace-button--primary"
              disabled={busy || !name.trim()}
            >
              {busy ? "Creating…" : "Create"}
            </button>
            {message ? (
              <span className="ns-member-message" role="status">
                {message}
              </span>
            ) : null}
          </form>
        ) : null}

        {!workspaces ? (
          <div className="ns-home__loading">
            <BrandLoader size={32} label="Loading" />
          </div>
        ) : workspaces.length === 0 ? (
          <div className="ns-home__empty">
            <p className="ns-home__empty-text">
              You don&apos;t have any workspaces yet. Create one to start building
              graphs.
            </p>
          </div>
        ) : (
          <>
            {/* Recent graphs */}
            {(recentGraphs && recentGraphs.length > 0) ||
            !workspaces ? null : (
              <section className="ns-home__recent">
                <h2>Recent graphs</h2>
                <Link
                  href="/workspaces"
                  className="ns-home__link"
                >
                  View all
                  <ArrowUpRight size={13} />
                </Link>
              </section>
            )}
            {recentGraphs && recentGraphs.length > 0 ? (
              <section className="ns-home__section" aria-label="Recent graphs">
                <div className="ns-home__section-header">
                  <h2>Recent graphs</h2>
                  <Link
                    href="/workspaces"
                    className="ns-home__link"
                  >
                    View all
                    <ArrowUpRight size={13} />
                  </Link>
                </div>
                <ul className="ns-home__graph-list">
                  {recentGraphs.map((graph) => (
                    <li key={`${graph._workspace.id}/${graph.id}`}>
                      <Link
                        href={workbenchGraphPath(
                          graph._workspace.slug,
                          graph.id,
                        )}
                        className="ns-home__graph-item"
                      >
                        <span className="ns-home__graph-icon">
                          <Workflow size={15} />
                        </span>
                        <span className="ns-home__graph-copy">
                          <span className="ns-home__graph-name">
                            {graph.name}
                          </span>
                          <span className="ns-home__graph-meta">
                            {graphAgeLabel(graph.updated_at)} ·{" "}
                            {workspaceDisplayName(
                              graph._workspace as Workspace,
                            )}
                          </span>
                        </span>
                        <span className="ns-home__graph-arrow">
                          <ArrowUpRight size={13} />
                        </span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}

            {/* Open a workspace */}
            <section className="ns-home__section" aria-label="Your workspaces">
              <div className="ns-home__section-header">
                <h2>Workspaces</h2>
                <Link href="/workspaces" className="ns-home__link">
                  All workspaces
                  <ArrowUpRight size={13} />
                </Link>
              </div>
              <ul className="ns-home__workspace-list">
                {personal.map((w) => (
                  <li key={w.id}>
                    <Link
                      href={`/workspaces/${encodeURIComponent(w.slug)}`}
                      className="ns-home__workspace-item"
                    >
                      <span className="ns-home__workspace-name">
                        {workspaceDisplayName(w)}
                      </span>
                      <span className="ns-home__workspace-meta">
                        /{w.slug}
                      </span>
                      <span className="ns-home__workspace-arrow">
                        <ArrowUpRight size={13} />
                      </span>
                    </Link>
                  </li>
                ))}
                {shared.length > 0 || ownedShared.length > 0 ? (
                  <li
                    className={
                      personal.length > 0 ? "ns-home__workspace-divider" : ""
                    }
                  >
                    {shared.length > 0 || ownedShared.length > 0 ? (
                      <>
                        <span className="ns-home__workspace-group-label">
                          Shared
                        </span>
                        {ownedShared.concat(shared).map((w) => (
                          <Link
                            key={w.id}
                            href={`/workspaces/${encodeURIComponent(w.slug)}`}
                            className="ns-home__workspace-item"
                          >
                            <span className="ns-home__workspace-name">
                              {workspaceDisplayName(w)}
                            </span>
                            <span className="ns-home__workspace-meta">
                              /{w.slug} · {w.role}
                            </span>
                            <span className="ns-home__workspace-arrow">
                              <ArrowUpRight size={13} />
                            </span>
                          </Link>
                        ))}
                      </>
                    ) : null}
                  </li>
                ) : null}
              </ul>
            </section>
          </>
        )}

        <footer className="ns-home__identity">
          <p>
            User{" "}
            <button
              type="button"
              onClick={() =>
                void navigator.clipboard.writeText(session.user_id)
              }
              className="ns-home__identity-btn"
            >
              {session.user_id}
            </button>
          </p>
        </footer>
      </main>
    </div>
  );
}

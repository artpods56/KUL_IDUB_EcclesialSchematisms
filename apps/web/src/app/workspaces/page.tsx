"use client";

import * as React from "react";
import {
  ArrowUpRight,
  Plus,
  Search,
  Share2,
  Users,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { BrandLoader } from "@/components/brand";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import {
  WorkspaceRail,
  workspaceDisplayName,
} from "@/features/workspaces/WorkspaceLayout";
import { createWorkspace } from "@/lib/api";
import type { Workspace } from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { useWorkspaces } from "@/hooks/use-api";

function slugFromName(name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return slug || "shared-workspace";
}

function matchesQuery(workspace: Workspace, query: string): boolean {
  if (!query) return true;
  const haystack =
    `${workspaceDisplayName(workspace)} ${workspace.name} ${workspace.slug} ${workspace.role}`.toLowerCase();
  return haystack.includes(query);
}

type SortMode = "name" | "kind";

function sortWorkspaces(workspaces: readonly Workspace[], mode: SortMode): Workspace[] {
  const copy = [...workspaces];
  if (mode === "kind") {
    copy.sort((a, b) => {
      if (a.kind !== b.kind) return a.kind.localeCompare(b.kind);
      return workspaceDisplayName(a).localeCompare(workspaceDisplayName(b));
    });
    return copy;
  }
  copy.sort((a, b) =>
    workspaceDisplayName(a).localeCompare(workspaceDisplayName(b)),
  );
  return copy;
}

function WorkspaceRow({ workspace }: { workspace: Workspace }) {
  return (
    <Link
      className="ns-workspace-list__row"
      href={`/workspaces/${encodeURIComponent(workspace.slug)}`}
    >
      <span className="ns-workspace-list__icon">
        {workspace.kind === "personal" ? <Workflow size={16} /> : <Users size={16} />}
      </span>
      <span className="ns-workspace-list__copy">
        <strong>{workspaceDisplayName(workspace)}</strong>
        <small>
          /{workspace.slug} · {workspace.role}
        </small>
      </span>
      <span className="ns-workspace-list__caps">
        {workspace.capabilities.length} capabilities
      </span>
      <span className="ns-workspace-list__open">Open</span>
      <ArrowUpRight size={15} aria-hidden="true" />
    </Link>
  );
}

function WorkspaceSection({
  title,
  workspaces,
  empty,
}: {
  title: string;
  workspaces: readonly Workspace[];
  empty: React.ReactNode;
}) {
  return (
    <section className="ns-workspace-section" aria-label={title}>
      <div className="ns-workspace-section__heading">
        <h2>{title}</h2>
        {workspaces.length > 0 ? (
          <span className="ns-workspace-section__meta">
            {workspaces.length} {workspaces.length === 1 ? "workspace" : "workspaces"}
          </span>
        ) : null}
      </div>
      {workspaces.length === 0 ? (
        <div className="ns-workspace-empty">{empty}</div>
      ) : (
        <div className="ns-workspace-list">
          {workspaces.map((workspace) => (
            <WorkspaceRow key={workspace.id} workspace={workspace} />
          ))}
        </div>
      )}
    </section>
  );
}

export default function WorkspacesPage() {
  const router = useRouter();
  const { session, logout } = useAuthSession();
  const { data: workspaces, error, mutate } = useWorkspaces(session.user_id);
  
  const [query, setQuery] = React.useState("");
  const [sortMode, setSortMode] = React.useState<SortMode>("name");
  const [name, setName] = React.useState("");
  const [createOpen, setCreateOpen] = React.useState(false);
  const [joinOpen, setJoinOpen] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState<string | null>(null);
  const [joinCopied, setJoinCopied] = React.useState(false);

  const normalizedQuery = query.trim().toLowerCase();

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedName = name.trim();
    if (!normalizedName) return;
    setBusy(true);
    setMessage(null);
    try {
      const workspace = await createWorkspace({
        name: normalizedName,
        slug: slugFromName(normalizedName),
      });
      await mutate(
        (current) => (current ? [...current, workspace] : [workspace]),
        { revalidate: false },
      );
      setName("");
      setCreateOpen(false);
      router.push(`/workspaces/${encodeURIComponent(workspace.slug)}`);
    } catch (caught) {
      setMessage(
        caught instanceof ApiError && caught.status === 409
          ? "That workspace slug is already in use."
          : "The shared workspace could not be created.",
      );
    } finally {
      setBusy(false);
    }
  };

  const copyUserId = async () => {
    try {
      await navigator.clipboard.writeText(session.user_id);
      setJoinCopied(true);
      window.setTimeout(() => setJoinCopied(false), 1_500);
    } catch {
      setJoinCopied(false);
    }
  };

  const filtered = React.useMemo(() => {
    if (!workspaces) return [];
    return sortWorkspaces(
      workspaces.filter((workspace) => matchesQuery(workspace, normalizedQuery)),
      sortMode,
    );
  }, [workspaces, normalizedQuery, sortMode]);

  const personal = filtered.filter((w) => w.kind === "personal");
  const ownedShared = filtered.filter(
    (workspace) => workspace.kind === "shared" && workspace.role === "owner",
  );

  const description =
    "Workspaces organize your projects, data, and collaborators. Choose a workspace to continue or create a new one.";

  return (
    <div className="ns-workspace-directory">
      {workspaces ? (
        <WorkspaceRail
          workspaces={workspaces}
          session={session}
          onLogout={logout}
        />
      ) : null}
      <main className="ns-workspace-directory__main">
        <header className="ns-workspace-directory__header">
          <div>
            <p className="ns-workspace-overview__eyebrow">Grafy / Workspaces</p>
            <h1>Workspaces</h1>
            <p>{description}</p>
          </div>
          <div className="ns-workspace-directory__actions">
            <button
              type="button"
              className="ns-workspace-button ns-workspace-button--primary"
              onClick={() => {
                setCreateOpen((current) => !current);
                setJoinOpen(false);
              }}
            >
              <Plus size={14} /> Create workspace
            </button>
            <button
              type="button"
              className="ns-workspace-button"
              onClick={() => {
                setJoinOpen((current) => !current);
                setCreateOpen(false);
              }}
            >
              <Share2 size={14} /> Join workspace
            </button>
          </div>
        </header>

        {createOpen ? (
          <form
            className="ns-workspace-create"
            onSubmit={(event) => void submit(event)}
          >
            <label>
              Workspace name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
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
              disabled={busy || name.trim() === ""}
            >
              {busy ? "Creating…" : "Create workspace"}
            </button>
            {message ? (
              <span className="ns-member-message" role="status">
                {message}
              </span>
            ) : null}
          </form>
        ) : null}

        {joinOpen ? (
          <div className="ns-workspace-join" role="region" aria-label="Join a workspace">
            <p>
              Shared workspaces are joined when an owner adds your user ID as a
              member. Copy your ID and send it to them.
            </p>
            <button
              type="button"
              className="ns-workspace-button"
              onClick={() => void copyUserId()}
            >
              {joinCopied ? "Copied user ID" : "Copy user ID"}
            </button>
          </div>
        ) : null}

        <div className="ns-workspace-toolbar">
          <label className="ns-workspace-search">
            <Search size={15} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search workspaces…"
              aria-label="Search workspaces"
            />
          </label>
          <label className="ns-workspace-sort">
            <span>Sort</span>
            <select
              value={sortMode}
              onChange={(event) => setSortMode(event.target.value as SortMode)}
            >
              <option value="name">Name</option>
              <option value="kind">Kind</option>
            </select>
          </label>
        </div>

        {error ? (
          <p className="ns-workspace-route-status__inline">
            Workspaces could not be loaded.
          </p>
        ) : null}
        {!workspaces ? (
          <div className="ns-workspace-directory__loading">
            <BrandLoader size={40} label="Loading workspaces" />
            <span>Loading workspaces…</span>
          </div>
        ) : (
          <>
            <WorkspaceSection
              title="Recent"
              workspaces={filtered.slice(0, 6)}
              empty={<p>No workspaces are available to this session.</p>}
            />
            <WorkspaceSection
              title="Personal"
              workspaces={personal}
              empty={<p>Your personal workspace will appear here after first login.</p>}
            />
            {ownedShared.length > 0 ? (
              <WorkspaceSection
                title="Owned shared"
                workspaces={ownedShared}
                empty={null}
              />
            ) : null}
          </>
        )}

        <aside className="ns-workspace-edu" aria-label="What is a workspace?">
          <div>
            <h2>What is a workspace?</h2>
            <p>
              A workspace is the tenancy boundary for graphs, collaborators, and
              module libraries. Personal workspaces stay private; shared
              workspaces let a team author together.
            </p>
          </div>
          <span className="ns-workspace-edu__mark" aria-hidden="true" />
        </aside>
      </main>
    </div>
  );
}

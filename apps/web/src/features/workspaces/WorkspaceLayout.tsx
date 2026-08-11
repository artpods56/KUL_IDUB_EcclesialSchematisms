"use client";

import * as React from "react";
import { Popover } from "@base-ui/react/popover";
import {
  LayoutGrid,
  LoaderCircle,
  LogOut,
  Plus,
  Save,
  Settings,
  Users,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";

import { BrandIcon, BrandWordmark } from "@/components/brand";
import { useTheme } from "@/components/theme";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { useWorkbenchChrome } from "@/features/workbench/ui/WorkbenchChromeContext";
import {
  NEW_GRAPH_ROUTE_ID,
  workbenchGraphPath,
} from "@/features/workbench/routes";
import { useSavedGraphs, useWorkspaces } from "@/hooks/use-api";
import type { SavedGraphSummary, Session, Workspace } from "@/lib/api";
import {
  deleteSavedGraphRemote,
  renameSavedGraphRemote,
} from "./graph-actions";
import { GraphRowMenu, promptGraphRename } from "./GraphRowMenu";
import {
  WorkspaceGraphPanel,
  sortGraphsByRecency,
} from "./WorkspaceGraphPanel";

interface WorkspaceContextValue {
  workspace: Workspace;
  workspaces: readonly Workspace[];
  refreshWorkspaces: () => Promise<readonly Workspace[] | undefined>;
}

export type WorkspaceRouteAccessState = "available" | "missing" | "revoked";



const RAIL_COLLAPSED_KEY = "ns-workspace-rail-collapsed";
const RAIL_EXPANDED_WIDTH = 200;
const RAIL_COLLAPSED_WIDTH = 64;
const RAIL_COLLAPSE_THRESHOLD = 132;
const RAIL_DESKTOP_QUERY = "(min-width: 861px)";

const railCollapsedListeners = new Set<() => void>();

function subscribeRailCollapsed(listener: () => void): () => void {
  railCollapsedListeners.add(listener);
  return () => railCollapsedListeners.delete(listener);
}

function readRailCollapsed(): boolean {
  try {
    return window.localStorage.getItem(RAIL_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function writeRailCollapsed(next: boolean): void {
  try {
    window.localStorage.setItem(RAIL_COLLAPSED_KEY, next ? "1" : "0");
  } catch {
    // Ignore storage errors.
  }
  for (const listener of railCollapsedListeners) listener();
}

function clearRailWidthOverride(): void {
  const root = document.documentElement;
  root.style.removeProperty("--ns-rail-width");
  root.classList.remove("ns-rail-resizing");
  document.body.classList.remove("ns-rail-resizing");
}

function setRailWidthOverride(width: number): void {
  const root = document.documentElement;
  root.style.setProperty("--ns-rail-width", `${Math.round(width)}px`);
  root.classList.add("ns-rail-resizing");
  document.body.classList.add("ns-rail-resizing");
}

function useRailCollapsed(): [boolean, (next: boolean) => void] {
  const collapsed = React.useSyncExternalStore(
    subscribeRailCollapsed,
    readRailCollapsed,
    () => false,
  );

  React.useEffect(() => {
    document.documentElement.dataset.railCollapsed = collapsed ? "true" : "false";
  }, [collapsed]);

  const setCollapsed = React.useCallback((next: boolean) => {
    writeRailCollapsed(next);
  }, []);

  return [collapsed, setCollapsed];
}

export function workspaceCanManageMembers(workspace: Workspace): boolean {
  return workspace.capabilities.includes("manage_members");
}

/** Graph currently open in the workbench, or null on non-graph routes. */
export function workspaceRouteGraphId(pathname: string): string | null {
  const match = /\/graphs\/([^/?#]+)/.exec(pathname);
  const graphId = match?.[1];
  if (!graphId || graphId === NEW_GRAPH_ROUTE_ID) return null;
  return decodeURIComponent(graphId);
}

const RAIL_RECENT_GRAPH_LIMIT = 6;

export function workspaceRouteAccessState(
  workspaceSlug: string,
  workspace: Workspace | undefined,
  previouslyResolvedWorkspace: Pick<Workspace, "slug" | "id"> | undefined,
): WorkspaceRouteAccessState {
  if (workspace) return "available";
  return previouslyResolvedWorkspace?.slug === workspaceSlug ? "revoked" : "missing";
}

/** User-facing save/share location label. */
export function workspaceDisplayName(
  workspace: Pick<Workspace, "name" | "kind">,
): string {
  if (workspace.kind === "personal") return "My graphs";
  return workspace.name;
}

export function sessionDisplayName(session: Pick<Session, "display_name" | "email" | "user_id">): string {
  const named = session.display_name?.trim();
  if (named) return named;
  const email = session.email?.trim();
  if (email) return email.split("@")[0] || email;
  return "User";
}

export function sessionInitials(session: Pick<Session, "display_name" | "email" | "user_id">): string {
  const named = session.display_name?.trim();
  if (named) {
    const parts = named.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0]![0] ?? ""}${parts[1]![0] ?? ""}`.toUpperCase();
    }
    return named.slice(0, 2).toUpperCase();
  }
  const email = session.email?.trim();
  if (email) return email.slice(0, 2).toUpperCase();
  return session.user_id.slice(0, 2).toUpperCase();
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
  session,
  onLogout,
  onBrandClick,
  onNewGraph,
}: {
  workspaces: readonly Workspace[];
  activeSlug?: string;
  session: Session;
  onLogout: () => Promise<void>;
  /** Optional graph-browser behavior when creation may require a location choice. */
  onNewGraph?: () => void;
  /** Optional same-page behavior for the brand button. */
  onBrandClick?: () => void;
}) {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const { cycleTheme, preference } = useTheme();
  const [collapsed, setCollapsed] = useRailCollapsed();
  const [accountMenuOpen, setAccountMenuOpen] = React.useState(false);
  const [graphPanelOpen, setGraphPanelOpen] = React.useState(false);
  const [graphActionBusyId, setGraphActionBusyId] = React.useState<string | null>(
    null,
  );
  const [previewCollapsed, setPreviewCollapsed] = React.useState<boolean | null>(null);
  const chrome = useWorkbenchChrome();
  const dragRef = React.useRef<{
    pointerId: number;
    startX: number;
    startWidth: number;
    moved: boolean;
  } | null>(null);

  const goGraphs = () => {
    router.push("/graphs");
  };

  const finishResize = React.useCallback(
    (clientX: number) => {
      const drag = dragRef.current;
      if (!drag) return;
      const nextWidth = Math.min(
        RAIL_EXPANDED_WIDTH,
        Math.max(
          RAIL_COLLAPSED_WIDTH,
          drag.startWidth + (clientX - drag.startX),
        ),
      );
      const wasClick = !drag.moved;
      const nextCollapsed = wasClick ? !collapsed : nextWidth < RAIL_COLLAPSE_THRESHOLD;
      dragRef.current = null;
      setPreviewCollapsed(null);
      setCollapsed(nextCollapsed);
      clearRailWidthOverride();
    },
    [collapsed, setCollapsed],
  );

  const onResizePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    if (!window.matchMedia(RAIL_DESKTOP_QUERY).matches) return;
    event.preventDefault();
    const startWidth = collapsed ? RAIL_COLLAPSED_WIDTH : RAIL_EXPANDED_WIDTH;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setPreviewCollapsed(collapsed);
    setRailWidthOverride(startWidth);
  };

  const onResizePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.startX;
    if (!drag.moved && Math.abs(deltaX) >= 4) {
      drag.moved = true;
    }
    if (!drag.moved) return;
    const nextWidth = Math.min(
      RAIL_EXPANDED_WIDTH,
      Math.max(RAIL_COLLAPSED_WIDTH, drag.startWidth + deltaX),
    );
    setRailWidthOverride(nextWidth);
    setPreviewCollapsed(nextWidth < RAIL_COLLAPSE_THRESHOLD);
  };

  const onResizePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    finishResize(event.clientX);
  };

  React.useEffect(() => () => clearRailWidthOverride(), []);

  const themeLabel =
    preference === "light" ? "Light theme" : preference === "dark" ? "Dark theme" : "System theme";
  const displayName = sessionDisplayName(session);
  const initials = sessionInitials(session);
  const email = session.email?.trim() || null;
  const visuallyCollapsed = previewCollapsed ?? collapsed;
  const graphBrowserActive = pathname === "/" || pathname === "/graphs";
  const teamSettingsActive =
    pathname === "/workspaces" ||
    Boolean(
      activeSlug &&
        pathname === `/workspaces/${encodeURIComponent(activeSlug)}`,
    );
  const activeWorkspace = activeSlug
    ? workspaces.find((candidate) => candidate.slug === activeSlug)
    : undefined;
  const activeGraphId = workspaceRouteGraphId(pathname);
  const { data: savedGraphs, mutate: mutateGraphs } = useSavedGraphs(
    activeWorkspace?.id,
  );
  const recentGraphs = React.useMemo(
    () =>
      sortGraphsByRecency(savedGraphs?.graphs ?? []).slice(
        0,
        RAIL_RECENT_GRAPH_LIMIT,
      ),
    [savedGraphs],
  );

  const renameGraph = React.useCallback(
    async (graph: SavedGraphSummary) => {
      if (!activeWorkspace) return;
      const next = promptGraphRename(graph.name);
      if (!next) return;
      setGraphActionBusyId(graph.id);
      try {
        if (chrome) {
          await chrome.renameGraph(graph, next);
        } else {
          await renameSavedGraphRemote(activeWorkspace.id, graph, next);
        }
        void mutateGraphs();
      } catch (error) {
        window.alert(
          error instanceof Error
            ? error.message
            : "The graph could not be renamed.",
        );
      } finally {
        setGraphActionBusyId(null);
      }
    },
    [activeWorkspace, chrome, mutateGraphs],
  );

  const deleteGraph = React.useCallback(
    async (graph: SavedGraphSummary) => {
      if (!activeWorkspace) return;
      setGraphActionBusyId(graph.id);
      try {
        if (chrome) {
          await chrome.deleteGraph(graph);
        } else {
          const deleted = await deleteSavedGraphRemote(
            activeWorkspace.id,
            graph,
          );
          if (!deleted) return;
        }
        void mutateGraphs();
      } catch (error) {
        window.alert(
          error instanceof Error
            ? error.message
            : "The graph could not be deleted.",
        );
      } finally {
        setGraphActionBusyId(null);
      }
    },
    [activeWorkspace, chrome, mutateGraphs],
  );

  return (
    <aside
      className={`ns-workspace-rail${visuallyCollapsed ? " is-collapsed" : ""}`}
      aria-label="Primary navigation"
    >
      <button
        type="button"
        className="ns-workspace-rail__brand"
        aria-label="Graphs"
        onClick={() => {
          if (onBrandClick) {
            onBrandClick();
          } else {
            goGraphs();
          }
        }}
      >
        <BrandWordmark className="ns-workspace-rail__brand-wordmark" height={24} />
        <BrandIcon className="ns-workspace-rail__brand-icon" size={28} alt="" />
      </button>

      <nav className="ns-workspace-rail__nav" aria-label="Graphs">
        <p className="ns-workspace-rail__section-label">Graphs</p>
        <button
          type="button"
          className={`ns-workspace-rail__item${graphBrowserActive ? " is-active" : ""}`}
          title="Browse all graphs"
          onClick={goGraphs}
        >
          <Workflow size={15} aria-hidden="true" />
          <span>All graphs</span>
        </button>
        {activeWorkspace || onNewGraph ? (
          <button
            type="button"
            className="ns-workspace-rail__item"
            title="New graph"
            onClick={() => {
              if (onNewGraph) {
                onNewGraph();
                return;
              }
              if (activeWorkspace) {
                router.push(
                  workbenchGraphPath(activeWorkspace.slug, NEW_GRAPH_ROUTE_ID),
                );
              }
            }}
          >
            <Plus size={15} aria-hidden="true" />
            <span>New graph</span>
          </button>
        ) : null}
        {activeWorkspace ? (
          <>
            <button
              type="button"
              data-graph-panel-trigger=""
              aria-expanded={graphPanelOpen}
              className={`ns-workspace-rail__item${graphPanelOpen ? " is-active" : ""}`}
              title={`Quickly switch graphs in ${workspaceDisplayName(activeWorkspace)}`}
              onClick={() => setGraphPanelOpen((open) => !open)}
            >
              <LayoutGrid size={15} aria-hidden="true" />
              <span>Quick switch</span>
            </button>
            {chrome ? (
              <button
                type="button"
                className={`ns-workspace-rail__item${chrome.isDirty ? " is-active" : ""}`}
                title={
                  chrome.saving
                    ? "Saving graph…"
                    : chrome.isDirty || !chrome.activeGraphId
                      ? "Save graph"
                      : "All changes are saved"
                }
                disabled={!chrome.canSave}
                onClick={() => void chrome.save()}
              >
                {chrome.saving ? (
                  <LoaderCircle
                    size={15}
                    aria-hidden="true"
                    className="ns-workspace-rail__spin"
                  />
                ) : (
                  <Save size={15} aria-hidden="true" />
                )}
                <span>
                  {chrome.saving
                    ? "Saving…"
                    : chrome.isDirty || !chrome.activeGraphId
                      ? "Save"
                      : "Saved"}
                </span>
              </button>
            ) : null}
          </>
        ) : null}
      </nav>

      {activeWorkspace ? (
        <nav className="ns-workspace-rail__nav" aria-label="Graph location">
          <p className="ns-workspace-rail__section-label">Location</p>
          <div
            className="ns-workspace-rail__item ns-workspace-rail__location"
            title={`Current location · ${workspaceDisplayName(activeWorkspace)}`}
            aria-label={`Current graph location ${workspaceDisplayName(activeWorkspace)}`}
          >
            {activeWorkspace.kind === "personal" ? (
              <Workflow size={15} aria-hidden="true" />
            ) : (
              <Users size={15} aria-hidden="true" />
            )}
            <span>{workspaceDisplayName(activeWorkspace)}</span>
          </div>
        </nav>
      ) : null}

      {activeWorkspace && !visuallyCollapsed && recentGraphs.length ? (
        <nav
          className="ns-workspace-rail__nav ns-workspace-rail__nav--open"
          aria-label="Recent graphs"
        >
          <p className="ns-workspace-rail__section-label">Recent</p>
          <div className="ns-workspace-rail__items">
            {recentGraphs.map((graph) => (
              <div
                key={graph.id}
                className={`ns-graph-row${activeGraphId === graph.id ? " is-active" : ""}`}
              >
                <button
                  type="button"
                  className="ns-workspace-rail__item ns-graph-row__open"
                  title={graph.name}
                  aria-label={graph.name}
                  onClick={() =>
                    router.push(
                      workbenchGraphPath(activeWorkspace.slug, graph.id),
                    )
                  }
                >
                  <Workflow size={15} aria-hidden="true" />
                  <span>{graph.name}</span>
                </button>
                <GraphRowMenu
                  graph={graph}
                  busy={graphActionBusyId === graph.id}
                  onRename={(entry) => void renameGraph(entry)}
                  onDelete={(entry) => void deleteGraph(entry)}
                />
              </div>
            ))}
          </div>
        </nav>
      ) : null}

      <button
        type="button"
        className={`ns-workspace-rail__settings${teamSettingsActive ? " is-active" : ""}`}
        onClick={() => router.push("/workspaces")}
        title="Teams & access"
      >
        <Settings size={15} aria-hidden="true" />
        <span>Teams &amp; access</span>
      </button>

      <div className="ns-workspace-rail__footer">
        <Popover.Root open={accountMenuOpen} onOpenChange={setAccountMenuOpen}>
          <Popover.Trigger
            className="ns-workspace-rail__account"
            title={displayName}
            aria-label="Account menu"
          >
            <span className="ns-workspace-rail__avatar" aria-hidden="true">
              {initials}
            </span>
            <span className="ns-workspace-rail__account-copy">
              <span className="ns-workspace-rail__account-name">{displayName}</span>
              {email ? (
                <span className="ns-workspace-rail__account-email">{email}</span>
              ) : null}
            </span>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Positioner
              className="ns-workspace-rail__account-positioner"
              side="top"
              align="start"
              sideOffset={8}
            >
              <Popover.Popup className="ns-workspace-rail__account-menu">
                <button
                  type="button"
                  className="ns-workspace-rail__account-menu-item"
                  onClick={() => {
                    cycleTheme();
                    setAccountMenuOpen(false);
                  }}
                >
                  <Settings size={14} aria-hidden="true" />
                  {themeLabel}
                </button>
                <button
                  type="button"
                  className="ns-workspace-rail__account-menu-item"
                  onClick={() => {
                    setAccountMenuOpen(false);
                    void onLogout();
                  }}
                >
                  <LogOut size={14} aria-hidden="true" />
                  Log out
                </button>
              </Popover.Popup>
            </Popover.Positioner>
          </Popover.Portal>
        </Popover.Root>
      </div>

      <div
        className="ns-workspace-rail__resize"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        aria-valuemin={RAIL_COLLAPSED_WIDTH}
        aria-valuemax={RAIL_EXPANDED_WIDTH}
        aria-valuenow={collapsed ? RAIL_COLLAPSED_WIDTH : RAIL_EXPANDED_WIDTH}
        title="Click to collapse or expand · drag to resize"
        onPointerDown={onResizePointerDown}
        onPointerMove={onResizePointerMove}
        onPointerUp={onResizePointerUp}
        onPointerCancel={onResizePointerUp}
      />

      {graphPanelOpen && activeWorkspace ? (
        <WorkspaceGraphPanel
          workspaceId={activeWorkspace.id}
          workspaceSlug={activeWorkspace.slug}
          activeGraphId={activeGraphId}
          busyGraphId={graphActionBusyId}
          onRename={(graph) => void renameGraph(graph)}
          onDelete={(graph) => void deleteGraph(graph)}
          onClose={() => setGraphPanelOpen(false)}
        />
      ) : null}
    </aside>
  );
}

function WorkspaceRouteStatus({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="ns-workspace-route-status">
      <p className="ns-workspace-route-status__eyebrow">Graph location</p>
      <h1>{title}</h1>
      <p>{detail}</p>
      <Link href="/graphs">Return to graphs</Link>
    </main>
  );
}

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  const { workspaceSlug } = useParams<{ workspaceSlug: string }>();
  const { session, logout } = useAuthSession();
  const { data, error, mutate } = useWorkspaces(session.user_id);
  const [previouslyResolvedWorkspace, setPreviouslyResolvedWorkspace] =
    React.useState<Pick<Workspace, "slug" | "id"> | undefined>(undefined);

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
    return <WorkspaceRouteStatus title="Graph location unavailable" detail="Grafy could not confirm access to this graph location." />;
  }
  if (!data) {
    return <WorkspaceRouteStatus title="Loading graph location" detail="Checking your current access…" />;
  }

  if (!workspace) {
    return routeAccessState === "revoked"
      ? <WorkspaceRouteStatus title="Graph location access removed" detail="Your access to this location is no longer available." />
      : <WorkspaceRouteStatus title="Graph location not found" detail="This graph location is not available to your account." />;
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
        <WorkspaceRail
          workspaces={data}
          activeSlug={workspace.slug}
          session={session}
          onLogout={logout}
        />
        <div className="ns-workspace-frame__main">{children}</div>
      </div>
    </WorkspaceContext.Provider>
  );
}

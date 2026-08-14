"use client";

import * as React from "react";
import { Search, X } from "lucide-react";
import { useRouter } from "next/navigation";

import { useSavedGraphs } from "@/hooks/use-api";
import type { SavedGraphSummary } from "@/lib/api";
import { workbenchGraphPath } from "@/features/workbench/routes";
import { GraphRowMenu } from "./GraphRowMenu";

export function sortGraphsByRecency(
  graphs: readonly SavedGraphSummary[],
): readonly SavedGraphSummary[] {
  return [...graphs].sort(
    (left, right) =>
      Date.parse(right.updated_at) - Date.parse(left.updated_at),
  );
}

export function filterGraphsByQuery(
  graphs: readonly SavedGraphSummary[],
  query: string,
): readonly SavedGraphSummary[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return graphs;
  return graphs.filter((graph) => graph.name.toLowerCase().includes(needle));
}

/** Relative age, coarse enough to stay stable without a re-render timer. */
export function graphAgeLabel(updatedAt: string, now = Date.now()): string {
  const elapsed = now - Date.parse(updatedAt);
  if (!Number.isFinite(elapsed)) return "";
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(updatedAt).toLocaleDateString();
}

export function WorkspaceGraphPanel({
  workspaceId,
  workspaceSlug,
  activeGraphId,
  busyGraphId = null,
  onRename,
  onDelete,
  onClose,
}: {
  workspaceId: string;
  workspaceSlug: string;
  activeGraphId: string | null;
  busyGraphId?: string | null;
  onRename: (graph: SavedGraphSummary) => void;
  onDelete: (graph: SavedGraphSummary) => void;
  onClose: (restoreFocus?: boolean) => void;
}) {
  const router = useRouter();
  const { data, isLoading } = useSavedGraphs(workspaceId);
  const [query, setQuery] = React.useState("");
  const panelRef = React.useRef<HTMLDivElement | null>(null);
  const searchRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    searchRef.current?.focus();
  }, []);

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose(true);
    };
    const onPointerDown = (event: PointerEvent) => {
      const panel = panelRef.current;
      if (!panel) return;
      const target = event.target;
      if (!(target instanceof Node) || panel.contains(target)) return;
      // The rail trigger toggles itself; closing here would re-open it.
      if (
        target instanceof Element &&
        (target.closest("[data-graph-panel-trigger]") ||
          target.closest(".ns-workspace-rail__account-menu"))
      ) {
        return;
      }
      onClose(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [onClose]);

  const graphs = React.useMemo(
    () => filterGraphsByQuery(sortGraphsByRecency(data?.graphs ?? []), query),
    [data, query],
  );
  const total = data?.graphs.length ?? 0;

  const openGraph = (graphId: string) => {
    router.push(workbenchGraphPath(workspaceSlug, graphId));
    onClose(false);
  };

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label="Quick graph switcher"
      className="ns-graph-panel"
    >
      <div className="ns-graph-panel__header">
        <p className="ns-graph-panel__title">
          Quick switch
          {total > 0 ? (
            <span className="ns-graph-panel__count">{total}</span>
          ) : null}
        </p>
        <button
          type="button"
          className="ns-graph-panel__icon-button"
          aria-label="Close quick graph switcher"
          onClick={() => onClose(true)}
        >
          <X size={14} aria-hidden="true" />
        </button>
      </div>

      <div className="ns-graph-panel__search">
        <Search size={14} aria-hidden="true" />
        <input
          ref={searchRef}
          type="search"
          value={query}
          placeholder="Search graphs"
          aria-label="Search graphs"
          onChange={(event) => setQuery(event.currentTarget.value)}
        />
      </div>

      <div className="ns-graph-panel__list">
        {isLoading ? (
          <p className="ns-graph-panel__empty">Loading graphs…</p>
        ) : graphs.length === 0 ? (
          <p className="ns-graph-panel__empty">
            {query.trim()
              ? "No graphs match that search."
              : "No graphs in this location yet. Use New graph in the sidebar to start one."}
          </p>
        ) : (
          graphs.map((graph) => (
            <div
              key={graph.id}
              className={`ns-graph-panel__row${activeGraphId === graph.id ? " is-active" : ""}`}
            >
              <button
                type="button"
                className="ns-graph-panel__row-open"
                onClick={() => openGraph(graph.id)}
              >
                <span className="ns-graph-panel__row-name">{graph.name}</span>
                <span className="ns-graph-panel__row-meta">
                  {`${graphAgeLabel(graph.updated_at)} · ${graph.node_count} ${
                    graph.node_count === 1 ? "node" : "nodes"
                  }`}
                </span>
              </button>
              <GraphRowMenu
                graph={graph}
                busy={busyGraphId === graph.id}
                onRename={onRename}
                onDelete={onDelete}
              />
            </div>
          ))
        )}
      </div>
    </div>
  );
}

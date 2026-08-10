"use client";

import * as React from "react";
import { Popover } from "@base-ui/react/popover";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import type { SavedGraphSummary } from "@/lib/api";

export function GraphRowMenu({
  graph,
  busy = false,
  onRename,
  onDelete,
}: {
  graph: SavedGraphSummary;
  busy?: boolean;
  onRename: (graph: SavedGraphSummary) => void;
  onDelete: (graph: SavedGraphSummary) => void;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger
        className="ns-graph-row__menu-trigger"
        disabled={busy}
        aria-label={`Actions for ${graph.name}`}
        title="Graph actions"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
        }}
      >
        <MoreHorizontal size={14} aria-hidden="true" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner
          className="ns-workspace-rail__account-positioner"
          side="right"
          align="start"
          sideOffset={4}
        >
          <Popover.Popup className="ns-workspace-rail__account-menu">
            <button
              type="button"
              className="ns-workspace-rail__account-menu-item"
              onClick={() => {
                setOpen(false);
                onRename(graph);
              }}
            >
              <Pencil size={14} aria-hidden="true" />
              Rename
            </button>
            <button
              type="button"
              className="ns-workspace-rail__account-menu-item ns-workspace-rail__account-menu-item--danger"
              onClick={() => {
                setOpen(false);
                onDelete(graph);
              }}
            >
              <Trash2 size={14} aria-hidden="true" />
              Delete
            </button>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}

export function promptGraphRename(currentName: string): string | null {
  const next = window.prompt("Rename graph", currentName);
  if (next === null) return null;
  const trimmed = next.trim();
  if (!trimmed || trimmed === currentName) return null;
  return trimmed.slice(0, 160);
}

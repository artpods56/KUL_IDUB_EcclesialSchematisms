"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { ViewportPortal } from "@xyflow/react";

import type { PresenceParticipant } from "./protocol";

const ACTOR_COLOR: Record<string, string> = {
  indigo: "#4f46e5",
  emerald: "#059669",
  amber: "#d97706",
  rose: "#e11d48",
  sky: "#0284c7",
  violet: "#7c3aed",
  teal: "#0d9488",
  orange: "#ea580c",
};

const s = stylex.create({
  cursor: {
    position: "absolute",
    pointerEvents: "none",
    zIndex: 5,
    display: "grid",
    justifyItems: "start",
    gap: "2px",
    transform: "translate(-2px, -2px)",
  },
  pointer: {
    width: "10px",
    height: "10px",
    borderRadius: "2px 10px 10px 10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "light-dark(#ffffff, #111214)",
    boxShadow: "0 1px 2px rgba(0, 0, 0, 0.25)",
  },
  label: {
    maxWidth: "120px",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    padding: "1px 5px",
    borderRadius: "4px",
    color: "#fff",
    fontSize: "10px",
    fontWeight: 650,
    lineHeight: 1.3,
  },
  strip: {
    position: "absolute",
    zIndex: 25,
    top: "62px",
    right: "13px",
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "flex-end",
    gap: "6px",
    maxWidth: "min(280px, 40vw)",
    pointerEvents: "none",
  },
  chip: {
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    minHeight: "24px",
    padding: "2px 8px 2px 6px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "light-dark(rgba(15, 23, 42, 0.12), rgba(255, 255, 255, 0.14))",
    borderRadius: "999px",
    backgroundColor: "light-dark(rgba(255, 255, 255, 0.88), rgba(24, 26, 30, 0.88))",
    color: "light-dark(#1f2937, #e5e7eb)",
    fontSize: "11px",
    fontWeight: 600,
    lineHeight: 1,
    boxShadow: "0 1px 2px rgba(0, 0, 0, 0.08)",
  },
  dot: {
    width: "8px",
    height: "8px",
    borderRadius: "999px",
    flex: "0 0 auto",
  },
  activity: {
    color: "light-dark(#6b7280, #9ca3af)",
    fontWeight: 500,
  },
});

function actorColor(color: string): string {
  return ACTOR_COLOR[color] ?? ACTOR_COLOR.indigo ?? "#4f46e5";
}

function activityLabel(activity: PresenceParticipant["activity"]): string | null {
  if (activity === "moving_nodes") return "moving";
  if (activity === "editing_node") return "editing";
  if (activity === "connecting") return "connecting";
  return null;
}

export interface PresenceOverlayProps {
  participants: readonly PresenceParticipant[];
  localSessionId: string | null;
}

export function PresenceOverlay({
  participants,
  localSessionId,
}: PresenceOverlayProps) {
  const remote = React.useMemo(
    () =>
      participants.filter(
        (participant) => participant.graph_room_session_id !== localSessionId,
      ),
    [localSessionId, participants],
  );

  if (remote.length === 0) return null;

  return (
    <>
      <div {...stylex.props(s.strip)} aria-label="Collaborators">
        {remote.map((participant) => {
          const color = actorColor(participant.actor.color);
          const activity = activityLabel(participant.activity);
          return (
            <span
              key={participant.graph_room_session_id}
              {...stylex.props(s.chip)}
              title={participant.actor.display_name}
            >
              <span {...stylex.props(s.dot)} style={{ backgroundColor: color }} />
              {participant.actor.display_name}
              {activity ? (
                <span {...stylex.props(s.activity)}>{activity}</span>
              ) : null}
            </span>
          );
        })}
      </div>
      <ViewportPortal>
        {remote.map((participant) => {
          if (!participant.cursor) return null;
          const color = actorColor(participant.actor.color);
          return (
            <div
              key={`cursor-${participant.graph_room_session_id}`}
              {...stylex.props(s.cursor)}
              style={{
                left: participant.cursor.x,
                top: participant.cursor.y,
              }}
            >
              <span
                {...stylex.props(s.pointer)}
                style={{ backgroundColor: color }}
              />
              <span
                {...stylex.props(s.label)}
                style={{ backgroundColor: color }}
              >
                {participant.actor.display_name}
              </span>
            </div>
          );
        })}
      </ViewportPortal>
    </>
  );
}

export function remoteSelectedNodeIds(
  participants: readonly PresenceParticipant[],
  localSessionId: string | null,
): ReadonlySet<string> {
  const ids = new Set<string>();
  for (const participant of participants) {
    if (participant.graph_room_session_id === localSessionId) continue;
    for (const nodeId of participant.selected_node_ids) {
      ids.add(nodeId);
    }
  }
  return ids;
}

export function remoteSelectionColor(
  participants: readonly PresenceParticipant[],
  localSessionId: string | null,
  nodeId: string,
): string | null {
  for (const participant of participants) {
    if (participant.graph_room_session_id === localSessionId) continue;
    if (participant.selected_node_ids.includes(nodeId)) {
      return actorColor(participant.actor.color);
    }
  }
  return null;
}

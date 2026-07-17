"use client";

import * as stylex from "@stylexjs/stylex";
import type { Connection } from "@xyflow/react";

import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { RunEdgeCollectionMode } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import type { ConnectionRoute } from "../canvas/handles";
import {
  connectionRouteDescription,
  connectionRouteTitle,
} from "../model/graph-authoring";

interface ConnectionEndpoint {
  nodeTitle: string;
  portName: string;
  artifactType: string;
}

export interface PendingConnectionRoute {
  connection: Connection;
  collectionMode: RunEdgeCollectionMode;
  candidates: ConnectionRoute[];
  source: ConnectionEndpoint;
  target: ConnectionEndpoint;
}

interface ConnectionRouteDialogProps {
  pendingRoute: PendingConnectionRoute | null;
  onSelect: (route: ConnectionRoute) => void;
  onClose: () => void;
}

const s = stylex.create({
  projectionFlow: {
    display: "grid",
    gridTemplateColumns: "minmax(0,1fr) 24px minmax(0,1fr)",
    alignItems: "center",
    gap: "7px",
    marginBottom: "14px",
  },
  projectionEndpoint: {
    minWidth: 0,
    display: "grid",
    gap: "3px",
    padding: "9px 10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "6px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  projectionDirection: {
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
  },
  projectionEndpointName: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 700,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  projectionEndpointType: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  projectionArrow: {
    color: tokens.colorSubtle,
    fontSize: "15px",
    textAlign: "center",
  },
  projectionPrompt: {
    marginBottom: "7px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeSm,
  },
  projectionChoices: { display: "grid", gap: "6px" },
  projectionChoice: {
    width: "100%",
    minHeight: "44px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    padding: "8px 10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorder,
      ":hover": tokens.colorAccentBorder,
      ":focus-visible": tokens.colorAccent,
    },
    borderRadius: "6px",
    outline: "none",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorAccentSoft,
    },
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  projectionChoiceTitle: { fontSize: tokens.fontSizeSm, fontWeight: 720 },
  projectionChoicePath: {
    color: tokens.colorProjectionPath,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  projectionActions: {
    display: "flex",
    justifyContent: "flex-end",
    marginTop: "12px",
  },
  projectionCancel: {
    minHeight: "29px",
    paddingInline: "10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorMuted,
    cursor: "pointer",
    fontSize: tokens.fontSizeSm,
  },
});

export function ConnectionRouteDialog({
  pendingRoute,
  onSelect,
  onClose,
}: ConnectionRouteDialogProps) {
  return (
    <Dialog
      open={pendingRoute !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent style={{ width: "430px" }}>
        <DialogHeader>
          <DialogTitle>Choose a connection route</DialogTitle>
          <DialogDescription>
            More than one declared route can satisfy this input.
          </DialogDescription>
        </DialogHeader>
        <DialogBody>
          {pendingRoute ? (
            <>
              <div {...stylex.props(s.projectionFlow)}>
                <div {...stylex.props(s.projectionEndpoint)}>
                  <span {...stylex.props(s.projectionDirection)}>Source</span>
                  <span {...stylex.props(s.projectionEndpointName)}>
                    {pendingRoute.source.nodeTitle} · {pendingRoute.source.portName}
                  </span>
                  <span {...stylex.props(s.projectionEndpointType)}>
                    {pendingRoute.source.artifactType}
                  </span>
                </div>
                <span
                  aria-hidden="true"
                  {...stylex.props(s.projectionArrow)}
                >
                  →
                </span>
                <div {...stylex.props(s.projectionEndpoint)}>
                  <span {...stylex.props(s.projectionDirection)}>Target</span>
                  <span {...stylex.props(s.projectionEndpointName)}>
                    {pendingRoute.target.nodeTitle} · {pendingRoute.target.portName}
                  </span>
                  <span {...stylex.props(s.projectionEndpointType)}>
                    {pendingRoute.target.artifactType}
                  </span>
                </div>
              </div>
              <p {...stylex.props(s.projectionPrompt)}>
                Choose how this edge carries the value:
              </p>
              <div {...stylex.props(s.projectionChoices)}>
                {pendingRoute.candidates.map((candidate, index) => (
                  <button
                    key={`${candidate.kind}-${connectionRouteDescription(pendingRoute.source.portName, candidate)}`}
                    type="button"
                    autoFocus={index === 0}
                    aria-label={`Use ${connectionRouteTitle(candidate)} from ${pendingRoute.source.nodeTitle}`}
                    {...stylex.props(s.projectionChoice)}
                    onClick={() => {
                      onSelect(candidate);
                      onClose();
                    }}
                  >
                    <span {...stylex.props(s.projectionChoiceTitle)}>
                      {connectionRouteTitle(candidate)}
                    </span>
                    <span {...stylex.props(s.projectionChoicePath)}>
                      {connectionRouteDescription(
                        pendingRoute.source.portName,
                        candidate,
                      )}
                    </span>
                  </button>
                ))}
              </div>
              <div {...stylex.props(s.projectionActions)}>
                <button
                  type="button"
                  {...stylex.props(s.projectionCancel)}
                  onClick={onClose}
                >
                  Cancel
                </button>
              </div>
            </>
          ) : null}
        </DialogBody>
      </DialogContent>
    </Dialog>
  );
}

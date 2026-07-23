"use client";

import * as stylex from "@stylexjs/stylex";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
  type EdgeProps,
} from "@xyflow/react";
import { Eye, X } from "lucide-react";

import { tokens } from "@/lib/stylex/tokens.stylex";
import type {
  ArtifactViewerEdge,
  CanvasEdge,
  CanvasNode,
} from "../artifact-viewer";

const s = stylex.create({
  positioner: {
    position: "absolute",
    zIndex: 10,
    pointerEvents: "all",
  },
  label: {
    minHeight: "23px",
    display: "inline-flex",
    alignItems: "center",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "9999px",
    backgroundColor: tokens.colorSurfaceRaised,
    boxShadow: tokens.shadowNode,
    color: tokens.colorMuted,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "9px",
    fontWeight: 700,
    pointerEvents: "all",
  },
  labelSelected: {
    borderColor: tokens.colorAccentBorder,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorTextEmphasis,
  },
  copy: {
    minWidth: 0,
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    maxWidth: "170px",
    paddingInline: "8px",
  },
  text: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  removeButton: {
    width: "23px",
    height: "23px",
    display: "grid",
    placeItems: "center",
    flexShrink: 0,
    padding: 0,
    borderWidth: 0,
    borderLeftWidth: 1,
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorBorder,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
});

export default function ArtifactViewerEdgeControl({
  id,
  data,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  selected,
}: EdgeProps<ArtifactViewerEdge>) {
  const { deleteElements } = useReactFlow<CanvasNode, CanvasEdge>();
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const sourcePortName = data?.sourcePortName ?? "output";

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        interactionWidth={24}
        style={{
          ...style,
          opacity: selected ? 0.9 : 0.62,
          strokeDasharray: "6 5",
          strokeWidth: selected ? 2.5 : (style?.strokeWidth ?? 2),
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan nowheel"
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
          {...stylex.props(s.positioner)}
        >
          <div
            aria-label={`Artifact viewer link from ${sourcePortName}`}
            {...stylex.props(
              s.label,
              selected ? s.labelSelected : null,
            )}
          >
            <span {...stylex.props(s.copy)}>
              <Eye size={10} aria-hidden="true" />
              <span {...stylex.props(s.text)}>
                preview · {sourcePortName}
              </span>
            </span>
            {selected ? (
              <button
                type="button"
                aria-label={`Remove artifact viewer connection from ${sourcePortName}`}
                title="Remove viewer connection"
                {...stylex.props(s.removeButton)}
                onClick={(event) => {
                  event.stopPropagation();
                  void deleteElements({ edges: [{ id }] });
                }}
              >
                <X size={11} aria-hidden="true" />
              </button>
            ) : null}
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

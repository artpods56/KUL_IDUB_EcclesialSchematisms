"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";
import {
  BaseEdge,
  EdgeLabelRenderer,
  Position,
  type EdgeProps,
  useReactFlow,
  useViewport,
} from "@xyflow/react";
import { Check, ChevronDown, GripVertical } from "lucide-react";

import { tokens } from "@/lib/stylex/tokens.stylex";
import type {
  WorkflowEdge,
  WorkflowEdgeData,
  WorkflowEdgeRoute,
  WorkflowEdgeRouteOption,
  WorkflowEdgeRouteOffset,
} from "../types";

const s = stylex.create({
  positioner: {
    position: "absolute",
    pointerEvents: "all",
    zIndex: 10,
  },
  controls: {
    display: "inline-flex",
    alignItems: "stretch",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "5px",
    backgroundColor: tokens.colorSurfaceRaised,
    boxShadow: tokens.shadowNode,
    pointerEvents: "all",
  },
  controlsSelected: {
    borderColor: tokens.colorAccentBorder,
    boxShadow: tokens.shadowNodeSelected,
  },
  controlsDisabled: {
    opacity: 0.78,
  },
  routeButton: {
    width: "23px",
    display: "grid",
    placeItems: "center",
    borderWidth: 0,
    borderRightWidth: 1,
    borderRightStyle: "solid",
    borderRightColor: tokens.colorBorder,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorAccentSoft,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorAccent },
    cursor: "grab",
    touchAction: "none",
  },
  routeButtonDragging: {
    backgroundColor: tokens.colorAccentSoft,
    color: tokens.colorAccent,
    cursor: "grabbing",
  },
  editButton: {
    minHeight: "25px",
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    maxWidth: "190px",
    paddingInline: "7px",
    borderWidth: 0,
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorTextEmphasis,
    cursor: "pointer",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
  },
  editLabel: {
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  removeButton: {
    order: 2,
    width: "25px",
    display: "grid",
    placeItems: "center",
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
  popup: {
    width: "300px",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "7px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
  header: {
    display: "grid",
    gap: "3px",
    padding: "10px 11px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  title: { fontSize: tokens.fontSizeSm, fontWeight: 750 },
  summary: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
  },
  section: {
    display: "grid",
    gap: "5px",
    padding: "9px 11px 11px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  sectionLast: { borderBottomWidth: 0 },
  sectionTitle: {
    color: tokens.colorMuted,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  option: {
    width: "100%",
    minHeight: "35px",
    display: "grid",
    gridTemplateColumns: "minmax(0,1fr) 16px",
    alignItems: "center",
    gap: "7px",
    padding: "6px 7px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorder,
      ":hover": tokens.colorAccentBorder,
      ":disabled": tokens.colorBorder,
    },
    borderRadius: "4px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorAccentSoft,
      ":disabled": tokens.colorSurfaceSunken,
    },
    color: { default: tokens.colorText, ":disabled": tokens.colorTextDisabled },
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    textAlign: "left",
  },
  optionActive: {
    borderColor: tokens.colorAccentBorder,
    backgroundColor: tokens.colorAccentSoft,
  },
  optionCopy: { minWidth: 0, display: "grid", gap: "2px" },
  optionTitle: {
    overflow: "hidden",
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  optionDescription: {
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1.35,
  },
  check: { color: tokens.colorAccent },
});

function projectionsEqual(
  left: WorkflowEdgeRoute["projection"],
  right: WorkflowEdgeRoute["projection"],
): boolean {
  if (!left || !right) return left === right;
  return (
    left.path.length === right.path.length &&
    left.path.every((segment, index) => segment === right.path[index])
  );
}

function conversionPathsEqual(
  left: WorkflowEdgeRoute["conversionPath"],
  right: WorkflowEdgeRoute["conversionPath"],
): boolean {
  return (
    left.length === right.length &&
    left.every(
      (conversion, index) =>
        conversion.id === right[index]?.id &&
        conversion.version === right[index]?.version,
    )
  );
}

function routeSelection(option: WorkflowEdgeRouteOption): WorkflowEdgeRoute {
  return {
    projection: option.projection
      ? { path: [...option.projection.path] }
      : undefined,
    conversionPath: option.conversionPath.map((conversion) => ({
      id: conversion.id,
      version: conversion.version,
    })),
  };
}

function projectionLabel(
  sourcePortName: string,
  projection: WorkflowEdgeData["projection"],
): string {
  if (!projection?.path.length) return sourcePortName;
  return `${sourcePortName}.${projection.path.join(".")}`;
}

interface Point {
  x: number;
  y: number;
}

function midpoint(left: Point, right: Point): Point {
  return {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
  };
}

function controlOffset(distance: number): number {
  return distance >= 0 ? distance / 2 : 6.25 * Math.sqrt(-distance);
}

function bezierControlPoint(
  position: Position,
  start: Point,
  end: Point,
): Point {
  switch (position) {
    case Position.Left:
      return {
        x: start.x - controlOffset(start.x - end.x),
        y: start.y,
      };
    case Position.Right:
      return {
        x: start.x + controlOffset(end.x - start.x),
        y: start.y,
      };
    case Position.Top:
      return {
        x: start.x,
        y: start.y - controlOffset(start.y - end.y),
      };
    case Position.Bottom:
      return {
        x: start.x,
        y: start.y + controlOffset(end.y - start.y),
      };
  }
}

function routedBezierPath({
  source,
  target,
  sourcePosition,
  targetPosition,
  routeOffset,
}: {
  source: Point;
  target: Point;
  sourcePosition: Position;
  targetPosition: Position;
  routeOffset: WorkflowEdgeRouteOffset;
}): { anchor: Point; path: string } {
  const sourceControl = bezierControlPoint(
    sourcePosition,
    source,
    target,
  );
  const targetControl = bezierControlPoint(
    targetPosition,
    target,
    source,
  );

  const sourceHalfControl = midpoint(source, sourceControl);
  const controlMidpoint = midpoint(sourceControl, targetControl);
  const targetHalfControl = midpoint(targetControl, target);
  const sourceAnchorControl = midpoint(sourceHalfControl, controlMidpoint);
  const targetAnchorControl = midpoint(controlMidpoint, targetHalfControl);
  const naturalAnchor = midpoint(sourceAnchorControl, targetAnchorControl);
  const anchor = {
    x: naturalAnchor.x + routeOffset.x,
    y: naturalAnchor.y + routeOffset.y,
  };
  const routedSourceAnchorControl = {
    x: sourceAnchorControl.x + routeOffset.x,
    y: sourceAnchorControl.y + routeOffset.y,
  };
  const routedTargetAnchorControl = {
    x: targetAnchorControl.x + routeOffset.x,
    y: targetAnchorControl.y + routeOffset.y,
  };

  return {
    anchor,
    path: [
      `M${source.x},${source.y}`,
      `C${sourceHalfControl.x},${sourceHalfControl.y}`,
      `${routedSourceAnchorControl.x},${routedSourceAnchorControl.y}`,
      `${anchor.x},${anchor.y}`,
      `C${routedTargetAnchorControl.x},${routedTargetAnchorControl.y}`,
      `${targetHalfControl.x},${targetHalfControl.y}`,
      `${target.x},${target.y}`,
    ].join(" "),
  };
}

function EdgeOption({
  title,
  description,
  active,
  onSelect,
}: {
  title: string;
  description: string;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      {...stylex.props(s.option, active ? s.optionActive : null)}
      onClick={onSelect}
    >
      <span {...stylex.props(s.optionCopy)}>
        <span {...stylex.props(s.optionTitle)}>{title}</span>
        <span {...stylex.props(s.optionDescription)}>
          {description}
        </span>
      </span>
      {active ? <Check size={12} {...stylex.props(s.check)} /> : null}
    </button>
  );
}

export default function WorkflowEdgeControl({
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
}: EdgeProps<WorkflowEdge>) {
  const { deleteElements, screenToFlowPosition } = useReactFlow();
  const { zoom } = useViewport();
  const edgeData: WorkflowEdgeData = data ?? {
    enabled: true,
    collectionMode: "direct",
  };
  const onUpdate = edgeData.onUpdate;
  const enabled = edgeData.enabled !== false;
  const canToggleEnabled = Boolean(onUpdate && (!enabled || edgeData.canDisable));
  const savedRouteOffset = edgeData.routeOffset ?? { x: 0, y: 0 };
  const [draftRouteOffset, setDraftRouteOffset] =
    React.useState<WorkflowEdgeRouteOffset | null>(null);
  const dragRef = React.useRef<{
    grabOffset: Point;
    latestOffset: WorkflowEdgeRouteOffset;
    pointerId: number;
  } | null>(null);
  const routeOffset = draftRouteOffset ?? savedRouteOffset;
  const { anchor, path: edgePath } = routedBezierPath({
    source: { x: sourceX, y: sourceY },
    target: { x: targetX, y: targetY },
    sourcePosition,
    targetPosition,
    routeOffset,
  });
  const sourcePortName = edgeData.sourcePortName ?? "output";
  const activeRoute: WorkflowEdgeRoute = {
    projection: edgeData.projection,
    conversionPath: edgeData.conversionPath ?? [],
  };
  const routeOptions = edgeData.routeOptions?.length
    ? edgeData.routeOptions
    : [
        {
          ...activeRoute,
          conversionTitles: edgeData.conversionTitles ?? [],
        },
      ];
  const valueOptions: Array<{
    title: string;
    description: string;
    routes: WorkflowEdgeRouteOption[];
  }> = [];
  for (const route of routeOptions) {
    const existing = valueOptions.find((option) =>
      projectionsEqual(option.routes[0]?.projection, route.projection),
    );
    if (existing) {
      existing.routes.push(route);
      continue;
    }
    valueOptions.push({
      title: route.projection
        ? (route.projectionTitle ?? route.projection.path.join("."))
        : "Whole output",
      description: route.projection
        ? route.projection.path.join(".")
        : "Pass the complete source artifact.",
      routes: [route],
    });
  }
  const conversionOptions: Array<{
    title: string;
    description: string;
    route: WorkflowEdgeRouteOption;
  }> = [];
  for (const route of routeOptions) {
    if (!projectionsEqual(route.projection, activeRoute.projection)) continue;
    if (
      conversionOptions.some((option) =>
        conversionPathsEqual(
          option.route.conversionPath,
          route.conversionPath,
        ),
      )
    ) {
      continue;
    }
    conversionOptions.push({
      title: route.conversionPath.length
        ? route.conversionTitles.join(" → ")
        : "No conversion",
      description: route.conversionPath.length
        ? route.conversionPath
            .map((conversion) => `${conversion.id}@${conversion.version}`)
            .join(" → ")
        : "Keep the selected artifact contract.",
      route,
    });
  }
  const allowedModes = edgeData.allowedCollectionModes ?? [edgeData.collectionMode];
  const conversionLabel = activeRoute.conversionPath.length
    ? ` → ${activeRoute.conversionPath
        .map(
          (conversion, index) =>
            edgeData.conversionTitles?.[index] ?? conversion.id,
        )
        .join(" → ")}`
    : "";
  const routeLabel = `${projectionLabel(sourcePortName, edgeData.projection)}${conversionLabel}${
    edgeData.collectionMode === "map" ? " · each" : ""
  }`;
  const label = enabled ? routeLabel : `${routeLabel} · disabled`;
  const statusDescription = enabled
    ? edgeData.canDisable
      ? "Included in dependency planning and workflow runs."
      : "Only connections to optional inputs can be disabled."
    : edgeData.canDisable
      ? "Saved on the canvas but omitted from dependency planning and runs."
      : "This required input is disconnected until the connection is enabled."

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          opacity: enabled ? style?.opacity : selected ? 0.58 : 0.42,
          strokeDasharray: enabled ? style?.strokeDasharray : "7 5",
          strokeWidth: selected ? 2.7 : (style?.strokeWidth ?? 2),
        }}
        interactionWidth={24}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan nowheel"
          style={{
            transform: `translate(-50%, -50%) translate(${anchor.x}px, ${anchor.y}px)`,
          }}
          {...stylex.props(s.positioner)}
        >
          <div
            style={{
              transform: `scale(${Math.min(1 / Math.max(zoom, 0.01), 1.75)})`,
              transformOrigin: "center",
            }}
            {...stylex.props(
              s.controls,
              selected ? s.controlsSelected : null,
              enabled ? null : s.controlsDisabled,
            )}
          >
            <button
              type="button"
              aria-label={`Bend connection ${label}`}
              aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home"
              title="Drag or use arrow keys to bend · double-click or press Home to reset"
              {...stylex.props(
                s.routeButton,
                draftRouteOffset ? s.routeButtonDragging : null,
              )}
              onClick={(event) => event.stopPropagation()}
              onDoubleClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                edgeData.onRouteOffsetChange?.(id, { x: 0, y: 0 });
              }}
              onKeyDown={(event) => {
                const step = (event.shiftKey ? 24 : 8) / Math.max(zoom, 0.01);
                let nextOffset: WorkflowEdgeRouteOffset | null = null;
                if (event.key === "Home") {
                  nextOffset = { x: 0, y: 0 };
                } else if (event.key === "ArrowLeft") {
                  nextOffset = {
                    x: savedRouteOffset.x - step,
                    y: savedRouteOffset.y,
                  };
                } else if (event.key === "ArrowRight") {
                  nextOffset = {
                    x: savedRouteOffset.x + step,
                    y: savedRouteOffset.y,
                  };
                } else if (event.key === "ArrowUp") {
                  nextOffset = {
                    x: savedRouteOffset.x,
                    y: savedRouteOffset.y - step,
                  };
                } else if (event.key === "ArrowDown") {
                  nextOffset = {
                    x: savedRouteOffset.x,
                    y: savedRouteOffset.y + step,
                  };
                }
                if (!nextOffset) return;
                event.preventDefault();
                event.stopPropagation();
                edgeData.onRouteOffsetChange?.(id, nextOffset);
              }}
              onPointerDown={(event) => {
                if (event.button !== 0 || !edgeData.onRouteOffsetChange) return;
                event.preventDefault();
                event.stopPropagation();
                const pointer = screenToFlowPosition({
                  x: event.clientX,
                  y: event.clientY,
                });
                event.currentTarget.setPointerCapture(event.pointerId);
                dragRef.current = {
                  pointerId: event.pointerId,
                  grabOffset: {
                    x: pointer.x - anchor.x,
                    y: pointer.y - anchor.y,
                  },
                  latestOffset: savedRouteOffset,
                };
                setDraftRouteOffset(savedRouteOffset);
              }}
              onPointerMove={(event) => {
                const drag = dragRef.current;
                if (!drag || drag.pointerId !== event.pointerId) return;
                event.preventDefault();
                event.stopPropagation();
                const pointer = screenToFlowPosition({
                  x: event.clientX,
                  y: event.clientY,
                });
                const nextOffset = {
                  x: pointer.x - drag.grabOffset.x - (anchor.x - routeOffset.x),
                  y: pointer.y - drag.grabOffset.y - (anchor.y - routeOffset.y),
                };
                drag.latestOffset = nextOffset;
                setDraftRouteOffset(nextOffset);
              }}
              onPointerUp={(event) => {
                const drag = dragRef.current;
                if (!drag || drag.pointerId !== event.pointerId) return;
                event.preventDefault();
                event.stopPropagation();
                dragRef.current = null;
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }
                edgeData.onRouteOffsetChange?.(id, drag.latestOffset);
                setDraftRouteOffset(null);
              }}
              onPointerCancel={(event) => {
                const drag = dragRef.current;
                if (!drag || drag.pointerId !== event.pointerId) return;
                event.stopPropagation();
                dragRef.current = null;
                setDraftRouteOffset(null);
              }}
              onLostPointerCapture={(event) => {
                const drag = dragRef.current;
                if (!drag || drag.pointerId !== event.pointerId) return;
                dragRef.current = null;
                setDraftRouteOffset(null);
              }}
            >
              <GripVertical size={12} aria-hidden="true" />
            </button>
            <button
              type="button"
              aria-label={`Remove connection ${label}`}
              title="Remove connection"
              {...stylex.props(s.removeButton)}
              onClick={(event) => {
                event.stopPropagation();
                void deleteElements({ edges: [{ id }] });
              }}
            >
              <span aria-hidden="true">×</span>
            </button>
            <Popover.Root>
              <Popover.Trigger
                type="button"
                aria-label={`Edit connection ${label}`}
                title="Edit projection, conversion, and collection handling"
                {...stylex.props(s.editButton)}
              >
                <span {...stylex.props(s.editLabel)}>{label}</span>
                <ChevronDown size={11} />
              </Popover.Trigger>
              <Popover.Portal>
                <Popover.Positioner
                  side="bottom"
                  align="center"
                  sideOffset={7}
                >
                  <Popover.Popup
                    className="nodrag nopan nowheel"
                    {...stylex.props(s.popup)}
                  >
                    <header {...stylex.props(s.header)}>
                      <span {...stylex.props(s.title)}>Connection</span>
                      <span {...stylex.props(s.summary)}>
                        This edge owns the value passed between the two ports.
                      </span>
                    </header>

                    <section {...stylex.props(s.section)}>
                      <span {...stylex.props(s.sectionTitle)}>Status</span>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={enabled}
                        aria-label={`Connection ${routeLabel} enabled`}
                        disabled={!canToggleEnabled}
                        title={
                          canToggleEnabled
                            ? enabled
                              ? "Disable connection"
                              : "Enable connection"
                            : "Required input connections must remain enabled"
                        }
                        {...stylex.props(
                          s.option,
                          enabled ? s.optionActive : null,
                        )}
                        onClick={() => onUpdate?.(id, { enabled: !enabled })}
                      >
                        <span {...stylex.props(s.optionCopy)}>
                          <span {...stylex.props(s.optionTitle)}>
                            {enabled ? "Enabled" : "Disabled"}
                          </span>
                          <span {...stylex.props(s.optionDescription)}>
                            {statusDescription}
                          </span>
                        </span>
                        {enabled ? (
                          <Check size={12} {...stylex.props(s.check)} />
                        ) : null}
                      </button>
                    </section>

                    <section {...stylex.props(s.section)}>
                      <span {...stylex.props(s.sectionTitle)}>Value</span>
                      {onUpdate
                        ? valueOptions.map((option) => {
                            const route =
                              option.routes.find((candidate) =>
                                conversionPathsEqual(
                                  candidate.conversionPath,
                                  activeRoute.conversionPath,
                                ),
                              ) ??
                              option.routes.find(
                                (candidate) =>
                                  candidate.conversionPath.length === 0,
                              ) ??
                              [...option.routes].sort(
                                (left, right) =>
                                  left.conversionPath.length -
                                  right.conversionPath.length,
                              )[0];
                            if (!route) return null;
                            return (
                              <EdgeOption
                                key={option.routes[0]?.projection?.path.join(".") ?? "whole"}
                                title={option.title}
                                description={option.description}
                                active={projectionsEqual(
                                  activeRoute.projection,
                                  route.projection,
                                )}
                                onSelect={() =>
                                  onUpdate(id, { route: routeSelection(route) })
                                }
                              />
                            );
                          })
                        : null}
                    </section>

                    <section {...stylex.props(s.section)}>
                      <span {...stylex.props(s.sectionTitle)}>Conversion path</span>
                      {onUpdate
                        ? conversionOptions.map((option) => (
                            <EdgeOption
                              key={
                                option.route.conversionPath.length
                                  ? option.route.conversionPath
                                      .map(
                                        (conversion) =>
                                          `${conversion.id}@${conversion.version}`,
                                      )
                                      .join("|")
                                  : "none"
                              }
                              title={option.title}
                              description={option.description}
                              active={conversionPathsEqual(
                                activeRoute.conversionPath,
                                option.route.conversionPath,
                              )}
                              onSelect={() =>
                                onUpdate(id, {
                                  route: routeSelection(option.route),
                                })
                              }
                            />
                          ))
                        : null}
                    </section>

                    <section {...stylex.props(s.section, s.sectionLast)}>
                      <span {...stylex.props(s.sectionTitle)}>Collection</span>
                      <button
                        type="button"
                        disabled={!allowedModes.includes("direct")}
                        {...stylex.props(
                          s.option,
                          edgeData.collectionMode === "direct"
                            ? s.optionActive
                            : null,
                        )}
                        onClick={() =>
                          edgeData.onUpdate?.(id, { collectionMode: "direct" })
                        }
                      >
                        <span {...stylex.props(s.optionCopy)}>
                          <span {...stylex.props(s.optionTitle)}>
                            Pass whole value
                          </span>
                          <span {...stylex.props(s.optionDescription)}>
                            Invoke the target once with this edge value.
                          </span>
                        </span>
                        {edgeData.collectionMode === "direct" ? (
                          <Check size={12} {...stylex.props(s.check)} />
                        ) : null}
                      </button>
                      <button
                        type="button"
                        disabled={!allowedModes.includes("map")}
                        {...stylex.props(
                          s.option,
                          edgeData.collectionMode === "map"
                            ? s.optionActive
                            : null,
                        )}
                        onClick={() =>
                          edgeData.onUpdate?.(id, { collectionMode: "map" })
                        }
                      >
                        <span {...stylex.props(s.optionCopy)}>
                          <span {...stylex.props(s.optionTitle)}>
                            Map each item
                          </span>
                          <span {...stylex.props(s.optionDescription)}>
                            Invoke the target once for every item in the list.
                          </span>
                        </span>
                        {edgeData.collectionMode === "map" ? (
                          <Check size={12} {...stylex.props(s.check)} />
                        ) : null}
                      </button>
                    </section>
                  </Popover.Popup>
                </Popover.Positioner>
              </Popover.Portal>
            </Popover.Root>
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

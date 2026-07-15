"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";
import {
  Handle,
  Position,
  useNodeConnections,
  useUpdateNodeInternals,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import {
  ArrowDown,
  ArrowUp,
  CircleHelp,
  GripVertical,
  LoaderCircle,
  Plus,
  RotateCcw,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import type { Port } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { schemaFields, type SchemaField } from "../config-schema";
import { encodeHandleId, handleStyle } from "../handles";
import { inputPlugsForPort } from "../input-plugs";
import { ARTIFACT_TYPE_COLOR } from "../nodes.css";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  acceptedPortShapes,
  declaredArtifactTypeVariables,
  effectivePortShape,
  portHasInstancePlugs,
  portMetaForPort,
  resolvedPortArtifactType,
  selectionSizeLabel,
  selectedSourceItems,
  type WorkflowNodeData,
} from "../types";
import { ArtifactsAppendix } from "./ArtifactsAppendix";
import { PortTypePopover } from "./type-inspector";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

const ACCEPTED_IMAGE_TYPES =
  ".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,image/png,image/jpeg,image/webp,image/tiff,image/bmp";

const s = stylex.create({
  shell: {
    position: "relative",
    width: "300px",
    overflow: "visible",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    boxShadow:
      "0 1px 2px light-dark(rgba(20, 24, 32, 0.1), rgba(0, 0, 0, 0.38)), 0 8px 22px light-dark(rgba(20, 24, 32, 0.1), rgba(0, 0, 0, 0.32))",
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    boxSizing: "border-box",
  },
  selected: {
    boxShadow: `0 2px 5px light-dark(rgba(107, 82, 212, 0.16), rgba(128, 103, 232, 0.22)), 0 12px 30px light-dark(rgba(20, 24, 32, 0.14), rgba(0, 0, 0, 0.46)), 0 0 0 2px ${tokens.colorAccentBorder}`,
  },
  header: {
    display: "grid",
    gap: "2px",
    padding: "12px 16px 12px 12px",
  },
  titleRow: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "4px",
  },
  headerButton: {
    width: "22px",
    height: "22px",
    display: "grid",
    placeItems: "center",
    flexShrink: 0,
    borderWidth: 0,
    borderRadius: "9999px",
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorText },
    cursor: "pointer",
  },
  removeButton: {
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
  },
  title: {
    minWidth: 0,
    overflow: "hidden",
    marginLeft: "4px",
    color: tokens.colorText,
    fontSize: tokens.fontSizeMd,
    fontWeight: 650,
    letterSpacing: "-0.01em",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  operator: {
    overflow: "hidden",
    marginLeft: "56px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  helpPopup: {
    width: "280px",
    display: "grid",
    gap: "6px",
    padding: "11px 13px",
    borderRadius: "12px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
  helpTitle: { fontSize: tokens.fontSizeSm, fontWeight: 750 },
  helpDescription: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.5,
  },
  tabs: {
    display: "grid",
    gap: "5px",
    paddingBlock: "2px",
  },
  tabsOutput: {
    display: "grid",
    gap: "5px",
    paddingTop: "2px",
    paddingBottom: "14px",
  },
  genericTypes: {
    display: "grid",
    gap: "5px",
    padding: "0 10px 8px",
  },
  genericTypeRow: {
    minHeight: "30px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    padding: "5px 7px",
    borderRadius: "7px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  genericTypeDot: {
    width: "6px",
    height: "6px",
    flexShrink: 0,
    borderRadius: "9999px",
    backgroundColor: tokens.colorAccent,
  },
  genericTypeCopy: {
    minWidth: 0,
    flex: 1,
    overflow: "hidden",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  genericTypeBound: {
    color: tokens.colorTextEmphasis,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontWeight: 650,
  },
  resetType: {
    minHeight: "22px",
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    paddingInline: "6px",
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: { default: tokens.colorMuted, ":hover": tokens.colorText },
    cursor: "pointer",
    fontSize: "10px",
    fontWeight: 650,
  },
  resetTypeDisabled: {
    color: tokens.colorSubtle,
    cursor: "not-allowed",
    opacity: 0.55,
  },
  plugGroup: {
    display: "grid",
    gap: "5px",
    paddingBottom: "4px",
  },
  plugPortHeader: {
    minHeight: "24px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
    paddingInline: "12px 10px",
  },
  plugPortTitle: {
    display: "flex",
    minWidth: 0,
    alignItems: "center",
    gap: "7px",
    padding: 0,
    overflow: "hidden",
    borderWidth: 0,
    backgroundColor: "transparent",
    color: tokens.colorTextEmphasis,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  plugPortRule: {
    flexShrink: 0,
    color: tokens.colorSubtle,
    fontSize: "10px",
  },
  plugList: {
    display: "grid",
    gap: "4px",
    paddingInline: "8px",
  },
  plugRow: {
    position: "relative",
    minWidth: 0,
    minHeight: "38px",
    display: "grid",
    gridTemplateColumns: "20px 20px minmax(0, 1fr) auto",
    alignItems: "center",
    gap: "4px",
    padding: "3px 4px 3px 28px",
    borderRadius: "9px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  plugRowDragging: {
    backgroundColor: tokens.colorAccentSoft,
    boxShadow: `inset 0 0 0 1px ${tokens.colorAccentBorder}`,
  },
  plugGrip: {
    width: "20px",
    height: "26px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorSubtle,
    cursor: "grab",
    touchAction: "none",
  },
  plugIndex: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
    textAlign: "center",
  },
  plugCopy: {
    minWidth: 0,
    display: "grid",
    gap: "1px",
  },
  plugSource: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  plugSourceEmpty: { color: tokens.colorMuted, fontWeight: 550 },
  plugMeta: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontSize: "10px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  plugActions: { display: "flex", alignItems: "center", gap: "1px" },
  plugAction: {
    width: "18px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorText },
    cursor: "pointer",
  },
  plugActionDisabled: {
    color: tokens.colorTextDisabled,
    cursor: "default",
    opacity: 0.45,
  },
  plugRemove: {
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
  },
  addPlug: {
    minHeight: "26px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "5px",
    marginInline: "8px",
    paddingInline: "8px",
    borderWidth: 0,
    borderRadius: "8px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHoverStrong,
    },
    color: tokens.colorMuted,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
    fontWeight: 650,
  },
  tabRow: {
    position: "relative",
    display: "flex",
    minHeight: "28px",
    alignItems: "center",
  },
  tabRowOut: { justifyContent: "flex-end" },
  tab: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
    maxWidth: "calc(100% - 12px)",
    height: "24px",
    paddingInline: "14px 12px",
    borderWidth: 0,
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHoverStrong,
    },
    color: tokens.colorTextEmphasis,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
  },
  tabLabel: {
    minWidth: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  tabShape: { flexShrink: 0 },
  tabIn: { borderRadius: "0 9999px 9999px 0" },
  tabOut: {
    flexDirection: "row-reverse",
    paddingInline: "12px 14px",
    borderRadius: "9999px 0 0 9999px",
  },
  dot: {
    width: "6px",
    height: "6px",
    flexShrink: 0,
    borderRadius: "9999px",
  },
  body: {
    display: "grid",
    gap: "9px",
    padding: "0 16px 14px",
  },
  upload: {
    width: "100%",
    minHeight: "34px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    borderWidth: 0,
    borderRadius: "10px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHover,
    },
    color: tokens.colorTextEmphasis,
    cursor: "pointer",
    fontSize: tokens.fontSizeSm,
    fontWeight: 600,
  },
  hiddenInput: {
    position: "absolute",
    width: "1px",
    height: "1px",
    overflow: "hidden",
    clip: "rect(0 0 0 0)",
    whiteSpace: "nowrap",
  },
  fileList: {
    maxHeight: "132px",
    display: "grid",
    gap: "5px",
    overflowY: "auto",
  },
  fileRow: {
    minWidth: 0,
    display: "grid",
    gridTemplateColumns: "18px minmax(0,1fr) auto 22px",
    alignItems: "center",
    gap: "6px",
    minHeight: "28px",
    paddingInline: "10px 4px",
    borderRadius: "8px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  fileIndex: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  fileName: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 550,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  fileSize: { color: tokens.colorSubtle, fontSize: tokens.fontSizeXs },
  fileRemove: {
    width: "22px",
    height: "22px",
    display: "grid",
    placeItems: "center",
    borderWidth: 0,
    borderRadius: "6px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  moreFiles: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  configList: { display: "grid", gap: "9px" },
  field: { display: "grid", gap: "4px" },
  fieldLabel: {
    display: "flex",
    alignItems: "center",
    gap: "3px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
    textTransform: "capitalize",
  },
  fieldDescription: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
  },
  input: {
    width: "100%",
    height: "31px",
    paddingInline: "10px",
    borderWidth: 0,
    borderRadius: "8px",
    outline: {
      default: "none",
      ":focus": `2px solid ${tokens.colorAccentBorder}`,
    },
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  textarea: {
    height: "96px",
    paddingBlock: "8px",
    lineHeight: 1.45,
    resize: "none",
  },
  checkRow: {
    minHeight: "30px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
  },
  check: { accentColor: tokens.colorAccent },
  required: { color: tokens.colorWarning, fontSize: tokens.fontSizeSm },
  error: {
    overflow: "hidden",
    color: tokens.colorDanger,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  emptyBody: {
    padding: "0 16px 14px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  spacer: { minHeight: "4px" },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
});

function nodeInteractionProps(props: ReturnType<typeof stylex.props>) {
  return {
    ...props,
    className: `nodrag nowheel${props.className ? ` ${props.className}` : ""}`,
  };
}

function PortTab({
  data,
  port,
  shape,
}: {
  data: WorkflowNodeData;
  port: Port;
  shape: Port["shape"];
}) {
  const input = port.direction === "input";
  const visibleName = port.title ?? port.name;
  const artifactType = resolvedPortArtifactType(
    port,
    data.artifactTypeBindings,
  );
  const color = artifactType
    ? ARTIFACT_TYPE_COLOR[artifactType.id] ?? tokens.colorAccent
    : tokens.colorAccent;
  const artifactContract = artifactType
    ? `${artifactType.id}@${artifactType.schema_version}`
    : "Any artifact";
  const effectiveContract =
    shape === "many" ? `list[${artifactContract}]` : artifactContract;
  const accessibleLabel = input
    ? `Input port ${visibleName}, accepts ${effectiveContract}${port.required ? ", required" : ""}`
    : `Output port ${visibleName}, provides ${effectiveContract}`;

  return (
    <div {...stylex.props(s.tabRow, input ? null : s.tabRowOut)}>
      <PortTypePopover
        port={port}
        shape={shape}
        artifactTypeBindings={data.artifactTypeBindings}
      >
        <Popover.Trigger
          type="button"
          aria-label={`Inspect ${visibleName} type`}
          title={port.description ?? `Inspect ${visibleName} type`}
          {...nodeInteractionProps(
            stylex.props(s.tab, input ? s.tabIn : s.tabOut),
          )}
        >
          <span {...stylex.props(s.tabLabel)}>{visibleName}</span>
          {input && port.required ? (
            <span {...stylex.props(s.required, s.tabShape)}>*</span>
          ) : null}
          {shape === "many" ? (
            <span {...stylex.props(s.tabShape)}>· many</span>
          ) : null}
        </Popover.Trigger>
      </PortTypePopover>
      <Handle
        type={input ? "target" : "source"}
        position={input ? Position.Left : Position.Right}
        id={encodeHandleId(
          portMetaForPort(
            port,
            input ? port.shape : shape,
            undefined,
            data.artifactTypeBindings,
          ),
        )}
        aria-label={accessibleLabel}
        title={
          input
            ? `${accessibleLabel}. Connect a compatible output here.${port.description ? ` ${port.description}` : ""}`
            : `${accessibleLabel}. Drag to a compatible input.${port.description ? ` ${port.description}` : ""}`
        }
        style={handleStyle("50%", color, port.variadic)}
      />
    </div>
  );
}

function InstancePlugPort({
  id,
  data,
  port,
}: {
  id: string;
  data: WorkflowNodeData;
  port: Port;
}) {
  const plugs = inputPlugsForPort(data.inputPlugs, port.name);
  const [draggedPlugId, setDraggedPlugId] = React.useState<string | null>(null);
  const draggedPlugIdRef = React.useRef<string | null>(null);
  const lastPointerTargetRef = React.useRef<string | null>(null);
  const visibleName = port.title ?? port.name;
  const artifactType = resolvedPortArtifactType(
    port,
    data.artifactTypeBindings,
  );
  const color = artifactType
    ? ARTIFACT_TYPE_COLOR[artifactType.id] ?? tokens.colorAccent
    : tokens.colorAccent;
  const acceptedShapeLabel = acceptedPortShapes(port)
    .map((shape) => (shape === "many" ? "sequence" : "single"))
    .join(" or ");

  const finishPointerDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    draggedPlugIdRef.current = null;
    lastPointerTargetRef.current = null;
    setDraggedPlugId(null);
  };

  return (
    <section {...stylex.props(s.plugGroup)} aria-label={`${visibleName} inputs`}>
      <div {...stylex.props(s.plugPortHeader)}>
        <PortTypePopover
          port={port}
          shape={port.shape}
          artifactTypeBindings={data.artifactTypeBindings}
        >
          <button
            type="button"
            aria-label={`Inspect ${visibleName} type`}
            title={port.description ?? `Inspect ${visibleName} type`}
            {...nodeInteractionProps(stylex.props(s.plugPortTitle))}
          >
            <span {...stylex.props(s.dot)} style={{ backgroundColor: color }} />
            <span {...stylex.props(s.tabLabel)}>{visibleName}</span>
            {port.required ? <span {...stylex.props(s.required)}>*</span> : null}
          </button>
        </PortTypePopover>
        <span {...stylex.props(s.plugPortRule)}>
          {acceptedShapeLabel} · plug order
        </span>
      </div>

      <div {...stylex.props(s.plugList)}>
        {plugs.map((plug, index) => {
          const binding = data.inputPlugBindings[plug.id];
          const connectionMeta = binding
            ? [
                binding.sourceShape === "many" ? "sequence" : "single",
                binding.conversionLabel
                  ? `via ${binding.conversionLabel}`
                  : null,
                binding.contributionLabel,
              ]
                .filter((label): label is string => Boolean(label))
                .join(" · ")
            : `Accepts ${acceptedShapeLabel}`;
          const accessibleLabel = `${visibleName} input ${index + 1}, accepts ${acceptedShapeLabel}`;
          return (
            <div
              key={plug.id}
              data-input-plug-id={plug.id}
              data-input-plug-port={port.name}
              {...stylex.props(
                s.plugRow,
                draggedPlugId === plug.id ? s.plugRowDragging : null,
              )}
            >
              <Handle
                className="nodrag nowheel"
                type="target"
                position={Position.Left}
                id={encodeHandleId(
                  portMetaForPort(
                    port,
                    port.shape,
                    plug.id,
                    data.artifactTypeBindings,
                  ),
                )}
                aria-label={accessibleLabel}
                title={`${accessibleLabel}. Connect one compatible output here.`}
                style={handleStyle("50%", color, true)}
              />
              <button
                type="button"
                aria-label={`Drag to reorder ${visibleName} input ${index + 1}`}
                title="Drag to reorder; arrow buttons also move this input"
                {...nodeInteractionProps(stylex.props(s.plugGrip))}
                onPointerDown={(event) => {
                  if (event.button !== 0) return;
                  event.stopPropagation();
                  event.currentTarget.setPointerCapture(event.pointerId);
                  draggedPlugIdRef.current = plug.id;
                  lastPointerTargetRef.current = plug.id;
                  setDraggedPlugId(plug.id);
                }}
                onPointerMove={(event) => {
                  const activePlugId = draggedPlugIdRef.current;
                  if (!activePlugId) return;
                  event.preventDefault();
                  event.stopPropagation();
                  const target = document
                    .elementFromPoint(event.clientX, event.clientY)
                    ?.closest<HTMLElement>("[data-input-plug-id]");
                  const targetPlugId = target?.dataset.inputPlugId;
                  if (targetPlugId === activePlugId) {
                    lastPointerTargetRef.current = null;
                    return;
                  }
                  if (
                    !targetPlugId ||
                    target?.dataset.inputPlugPort !== port.name ||
                    targetPlugId === lastPointerTargetRef.current
                  ) {
                    return;
                  }
                  const targetIndex = plugs.findIndex(
                    (candidate) => candidate.id === targetPlugId,
                  );
                  if (targetIndex === -1) return;
                  lastPointerTargetRef.current = targetPlugId;
                  data.onReorderInputPlug?.(
                    id,
                    port.name,
                    activePlugId,
                    targetIndex,
                  );
                }}
                onPointerUp={finishPointerDrag}
                onPointerCancel={finishPointerDrag}
              >
                <GripVertical size={12} />
              </button>
              <span {...stylex.props(s.plugIndex)}>{index + 1}</span>
              <span {...stylex.props(s.plugCopy)}>
                <span
                  {...stylex.props(
                    s.plugSource,
                    binding ? null : s.plugSourceEmpty,
                  )}
                  title={binding?.sourceLabel}
                >
                  {binding?.sourceLabel ?? "Connect input"}
                </span>
                <span {...stylex.props(s.plugMeta)} title={connectionMeta}>
                  {connectionMeta}
                </span>
              </span>
              <span {...stylex.props(s.plugActions)}>
                <button
                  type="button"
                  disabled={index === 0}
                  aria-label={`Move ${visibleName} input ${index + 1} up`}
                  title="Move input up"
                  {...nodeInteractionProps(
                    stylex.props(
                      s.plugAction,
                      index === 0 ? s.plugActionDisabled : null,
                    ),
                  )}
                  onClick={() =>
                    data.onReorderInputPlug?.(
                      id,
                      port.name,
                      plug.id,
                      index - 1,
                    )
                  }
                >
                  <ArrowUp size={10} />
                </button>
                <button
                  type="button"
                  disabled={index === plugs.length - 1}
                  aria-label={`Move ${visibleName} input ${index + 1} down`}
                  title="Move input down"
                  {...nodeInteractionProps(
                    stylex.props(
                      s.plugAction,
                      index === plugs.length - 1
                        ? s.plugActionDisabled
                        : null,
                    ),
                  )}
                  onClick={() =>
                    data.onReorderInputPlug?.(
                      id,
                      port.name,
                      plug.id,
                      index + 1,
                    )
                  }
                >
                  <ArrowDown size={10} />
                </button>
                <button
                  type="button"
                  aria-label={`Remove ${visibleName} input ${index + 1}`}
                  title="Remove input and its connection"
                  {...nodeInteractionProps(
                    stylex.props(s.plugAction, s.plugRemove),
                  )}
                  onClick={() => data.onRemoveInputPlug?.(id, plug.id)}
                >
                  <Trash2 size={10} />
                </button>
              </span>
            </div>
          );
        })}
      </div>
      <button
        type="button"
        {...nodeInteractionProps(stylex.props(s.addPlug))}
        onClick={() => data.onAddInputPlug?.(id, port.name)}
      >
        <Plus size={11} />
        Add input
      </button>
    </section>
  );
}

function GenericArtifactTypeState({
  id,
  data,
  resettable,
}: {
  id: string;
  data: WorkflowNodeData;
  resettable: boolean;
}) {
  const variables = declaredArtifactTypeVariables(data.spec);
  if (!variables.length) return null;

  return (
    <div {...stylex.props(s.genericTypes)} aria-label="Generic artifact types">
      {variables.map((variable) => {
        const artifactType = data.artifactTypeBindings[variable];
        const label = artifactType
          ? `${artifactType.id}@${artifactType.schema_version}`
          : "Any artifact · binds on connect";
        return (
          <div key={variable} {...stylex.props(s.genericTypeRow)}>
            <span
              aria-hidden="true"
              {...stylex.props(s.genericTypeDot)}
              style={
                artifactType
                  ? {
                      backgroundColor:
                        ARTIFACT_TYPE_COLOR[artifactType.id] ??
                        tokens.colorAccent,
                    }
                  : undefined
              }
            />
            <span
              title={`${variable}: ${label}`}
              {...stylex.props(
                s.genericTypeCopy,
                artifactType ? s.genericTypeBound : null,
              )}
            >
              {label}
            </span>
            {artifactType ? (
              <button
                type="button"
                disabled={!resettable || !data.onResetArtifactTypeBinding}
                aria-label={`Reset artifact type ${variable}`}
                title={
                  resettable
                    ? "Reset type"
                    : "Disconnect this node before resetting its type"
                }
                {...nodeInteractionProps(
                  stylex.props(
                    s.resetType,
                    resettable ? null : s.resetTypeDisabled,
                  ),
                )}
                onClick={() => data.onResetArtifactTypeBinding?.(id, variable)}
              >
                <RotateCcw size={10} />
                Reset type
              </button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function ConfigField({
  field,
  value,
  onChange,
}: {
  field: SchemaField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (field.type === "boolean") {
    return (
      <label {...stylex.props(s.field)}>
        <span {...stylex.props(s.checkRow)}>
          <input
            type="checkbox"
            checked={value === true}
            {...nodeInteractionProps(stylex.props(s.check))}
            onChange={(event) => onChange(event.currentTarget.checked)}
          />
          {field.title}
          {field.required ? <span {...stylex.props(s.required)}>*</span> : null}
        </span>
        {field.description ? (
          <span {...stylex.props(s.fieldDescription)}>
            {field.description}
          </span>
        ) : null}
      </label>
    );
  }

  return (
    <label {...stylex.props(s.field)}>
      <span {...stylex.props(s.fieldLabel)}>
        {field.title}
        {field.required ? <span {...stylex.props(s.required)}>*</span> : null}
      </span>
      {field.description ? (
        <span {...stylex.props(s.fieldDescription)}>{field.description}</span>
      ) : null}
      {field.enumValues ? (
        <select
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          {...nodeInteractionProps(stylex.props(s.input))}
          onChange={(event) => {
            const selected = event.currentTarget.value;
            onChange(
              field.type === "number" || field.type === "integer"
                ? Number(selected)
                : selected,
            );
          }}
        >
          {typeof value !== "string" && typeof value !== "number" ? (
            <option value="" disabled>
              Choose an option
            </option>
          ) : null}
          {field.enumValues.map((option) => (
            <option key={String(option)} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : field.type === "string" && field.format === "textarea" ? (
        <textarea
          value={typeof value === "string" ? value : ""}
          minLength={field.minLength}
          maxLength={field.maxLength}
          {...nodeInteractionProps(stylex.props(s.input, s.textarea))}
          onChange={(event) => onChange(event.currentTarget.value)}
        />
      ) : (
        <input
          type={
            field.type === "number" || field.type === "integer"
              ? "number"
              : "text"
          }
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          min={field.minimum}
          max={field.maximum}
          minLength={field.minLength}
          maxLength={field.maxLength}
          pattern={field.pattern}
          step={field.type === "integer" ? 1 : undefined}
          {...nodeInteractionProps(stylex.props(s.input))}
          onChange={(event) => {
            const raw = event.currentTarget.value;
            onChange(
              field.type === "number" || field.type === "integer"
                ? raw === ""
                  ? undefined
                  : Number(raw)
                : raw,
            );
          }}
        />
      )}
    </label>
  );
}

function SourceBody({ id, data }: { id: string; data: WorkflowNodeData }) {
  const items = selectedSourceItems(data);
  const inputRef = React.useRef<HTMLInputElement>(null);

  return (
    <div {...stylex.props(s.body)}>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_IMAGE_TYPES}
        {...nodeInteractionProps(stylex.props(s.hiddenInput))}
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          if (files.length) data.onFilesSelected?.(id, files);
        }}
      />
      <button
        type="button"
        {...nodeInteractionProps(stylex.props(s.upload))}
        onClick={() => inputRef.current?.click()}
      >
        {data.execution.status === "uploading" ? (
          <LoaderCircle size={12} {...stylex.props(s.spinner)} />
        ) : (
          <Upload size={12} />
        )}
        {data.execution.status === "uploading"
          ? "Uploading…"
          : items.length
            ? "Replace images"
            : "Choose images"}
      </button>
      {items.length ? (
        <div {...nodeInteractionProps(stylex.props(s.fileList))}>
          {items.map((item, index) => (
            <div
              key={`${item.external_uri}-${index}`}
              {...stylex.props(s.fileRow)}
            >
              <span {...stylex.props(s.fileIndex)}>
                {String(index + 1).padStart(2, "0")}
              </span>
              <span {...stylex.props(s.fileName)}>{item.display_name}</span>
              <span {...stylex.props(s.fileSize)}>
                {selectionSizeLabel(item.size_bytes)}
              </span>
              <button
                type="button"
                aria-label={`Remove ${item.display_name}`}
                title={`Remove ${item.display_name}`}
                {...nodeInteractionProps(stylex.props(s.fileRemove))}
                onClick={() => data.onRemoveSelection?.(id, index)}
              >
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p {...stylex.props(s.moreFiles)}>
          PNG, JPEG, WebP, TIFF or BMP · ordered as selected
        </p>
      )}
      {data.execution.error ? (
        <div {...stylex.props(s.error)} title={data.execution.error}>
          {data.execution.error}
        </div>
      ) : null}
    </div>
  );
}

function GenericBody({ id, data }: { id: string; data: WorkflowNodeData }) {
  const fields = schemaFields(data.spec.config_schema);
  if (!fields.length && !data.execution.error) return null;

  return (
    <div {...stylex.props(s.body)}>
      {fields.length ? (
        <div {...stylex.props(s.configList)}>
          {fields.map((field) => (
            <ConfigField
              key={field.name}
              field={field}
              value={data.config[field.name]}
              onChange={(value) =>
                data.onConfigChange?.(id, field.name, value)
              }
            />
          ))}
        </div>
      ) : null}
      {data.execution.error ? (
        <div {...stylex.props(s.error)} title={data.execution.error}>
          {data.execution.error}
        </div>
      ) : null}
    </div>
  );
}

function NodeHeader({ id, data }: { id: string; data: WorkflowNodeData }) {
  return (
    <header {...stylex.props(s.header)}>
      <span {...stylex.props(s.titleRow)}>
        <Popover.Root>
          <Popover.Trigger
            type="button"
            aria-label={`About ${data.spec.title}`}
            title={`About ${data.spec.title}`}
            {...nodeInteractionProps(stylex.props(s.headerButton))}
          >
            <CircleHelp size={13} />
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Positioner side="top" align="start" sideOffset={7}>
              <Popover.Popup
                {...nodeInteractionProps(stylex.props(s.helpPopup))}
              >
                <span {...stylex.props(s.helpTitle)}>{data.spec.title}</span>
                <span {...stylex.props(s.helpDescription)}>
                  {data.spec.description ||
                    "No description is available for this node."}
                </span>
              </Popover.Popup>
            </Popover.Positioner>
          </Popover.Portal>
        </Popover.Root>
        <button
          type="button"
          aria-label={`Remove ${data.spec.title}`}
          title={`Remove ${data.spec.title}`}
          {...nodeInteractionProps(
            stylex.props(s.headerButton, s.removeButton),
          )}
          onClick={() => data.onRemoveNode?.(id)}
        >
          <X size={13} />
        </button>
        <span {...stylex.props(s.title)} title={data.spec.title}>
          {data.spec.title}
        </span>
      </span>
      <span {...stylex.props(s.operator)} title={data.spec.description}>
        {data.spec.operator_id}@{data.spec.operator_version}
      </span>
    </header>
  );
}

function WorkflowNodeCard({ id, data, selected }: NodeProps<WorkflowNode>) {
  const fields = schemaFields(data.spec.config_schema);
  const isUpload = data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID;
  const hasConfig = fields.length > 0;
  const producedArtifactCount = (data.run?.outputs ?? []).reduce(
    (count, output) => count + output.artifacts.length,
    0,
  );
  const inputPlugRevision = data.inputPlugs
    .map((plug) => `${plug.portName}:${plug.id}`)
    .join("|");
  const artifactTypeBindingRevision = Object.entries(data.artifactTypeBindings)
    .map(
      ([variable, artifactType]) =>
        `${variable}:${artifactType.id}@${artifactType.schema_version}`,
    )
    .sort()
    .join("|");
  const incidentConnections = useNodeConnections({ id });
  const updateNodeInternals = useUpdateNodeInternals();

  React.useEffect(() => {
    const frame = window.requestAnimationFrame(() => updateNodeInternals(id));
    return () => window.cancelAnimationFrame(frame);
  }, [
    data.mappedInputPort,
    artifactTypeBindingRevision,
    inputPlugRevision,
    data.spec.inputs.length,
    data.spec.outputs.length,
    fields.length,
    producedArtifactCount,
    id,
    updateNodeInternals,
  ]);

  return (
    <>
      <article {...stylex.props(s.shell, selected ? s.selected : null)}>
        <NodeHeader id={id} data={data} />
        <GenericArtifactTypeState
          id={id}
          data={data}
          resettable={incidentConnections.length === 0}
        />
        {data.spec.inputs.length ? (
          <div {...stylex.props(s.tabs)}>
            {data.spec.inputs.map((port) => (
              portHasInstancePlugs(port) ? (
                <InstancePlugPort
                  key={`in-${port.name}`}
                  id={id}
                  data={data}
                  port={port}
                />
              ) : (
                <PortTab
                  key={`in-${port.name}`}
                  data={data}
                  port={port}
                  shape={effectivePortShape(data, port)}
                />
              )
            ))}
          </div>
        ) : null}
        {isUpload ? (
          <SourceBody id={id} data={data} />
        ) : hasConfig || data.execution.error ? (
          <GenericBody id={id} data={data} />
        ) : (
          <div {...stylex.props(s.spacer)} aria-hidden />
        )}
        {data.spec.outputs.length ? (
          <div {...stylex.props(s.tabsOutput)}>
            {data.spec.outputs.map((port) => (
              <PortTab
                key={`out-${port.name}`}
                data={data}
                port={port}
                shape={effectivePortShape(data, port)}
              />
            ))}
          </div>
        ) : null}
        {!isUpload &&
        !hasConfig &&
        !data.execution.error &&
        !data.spec.inputs.length &&
        !data.spec.outputs.length ? (
          <p {...stylex.props(s.emptyBody)}>
            {data.spec.description || "No configuration for this operator."}
          </p>
        ) : null}
      </article>
      <ArtifactsAppendix data={data} />
    </>
  );
}

export default WorkflowNodeCard;

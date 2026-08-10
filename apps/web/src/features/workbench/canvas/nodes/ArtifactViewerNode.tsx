"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  Handle,
  Position,
  useEdges,
  useNodesData,
  useUpdateNodeInternals,
  type NodeProps,
} from "@xyflow/react";
import {
  CheckCircle2,
  Link2,
  LoaderCircle,
  TriangleAlert,
  X,
} from "lucide-react";

import { useNodeRegistry } from "@/hooks/use-api";
import { useWorkspaceContext } from "@/features/workspaces/WorkspaceLayout";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  ARTIFACT_VIEWER_EDGE_TYPE,
  ARTIFACT_VIEWER_INPUT_HANDLE,
  ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE,
  ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE,
  type ArtifactViewerEdge,
  type ArtifactViewerNode,
  type CanvasEdge,
  type CanvasNode,
} from "../artifact-viewer";
import {
  EMPTY_ARTIFACT_KEY_SELECTION,
  type ArtifactViewerInteractionContext,
} from "../artifact-interactions";
import { RemoteSelectionRing } from "../../room/RemoteSelectionRing";
import { handleStyle } from "../handle-style";
import {
  resolvedAppendixHeight,
  resolvedNodeWidth,
  type WorkflowNodeLayout,
} from "../node-layout";
import {
  WORKFLOW_NODE_TYPE,
  effectivePortShape,
  resolvedPortArtifactType,
} from "../types";
import { ArtifactPortPreview } from "./ArtifactsAppendix";
import { LayoutResizeHandle } from "./LayoutResizeHandle";
import { usePickupLift } from "./usePickupLift";
import { useShellGridFill } from "./useShellGridFill";

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const s = stylex.create({
  shellFrame: {
    position: "relative",
    boxSizing: "border-box",
    // Option C pickup: release settles quicker than the spring lift.
    transitionProperty: {
      default: "transform",
      "@media (prefers-reduced-motion: reduce)": "none",
    },
    transitionDuration: "120ms",
    transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
  },
  shellFrameActive: {
    // Slight active-tier lift; the spring overshoot reads as the node waking
    // up under the pointer.
    transform: "translate3d(0, -2px, 0)",
    transitionProperty: {
      default: "transform",
      "@media (prefers-reduced-motion: reduce)": "none",
    },
    transitionDuration: "200ms",
    transitionTimingFunction: "cubic-bezier(0.34, 1.56, 0.64, 1)",
  },
  shellFrameDragged: {
    // Full pickup: the node is carried above the canvas.
    transform: "translate3d(0, -8px, 0)",
    transitionProperty: {
      default: "transform",
      "@media (prefers-reduced-motion: reduce)": "none",
    },
    transitionDuration: "200ms",
    transitionTimingFunction: "cubic-bezier(0.34, 1.56, 0.64, 1)",
  },
  shell: {
    position: "relative",
    width: "520px",
    overflow: "visible",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorChrome,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    boxSizing: "border-box",
    cursor: "grab",
    transitionProperty: {
      default: "box-shadow",
      "@media (prefers-reduced-motion: reduce)": "none",
    },
    // Hide (release) is quicker than pickup so the card never appears to snap
    // away from a lingering ground plate.
    transitionDuration: "90ms",
    transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
  },
  shellContent: {
    display: "grid",
    gap: "10px",
    padding: "10px 12px 12px",
    boxSizing: "border-box",
    flexShrink: 0,
    width: "100%",
  },
  pickedUp: {
    // Pair the lifted frame with a near-card shadow and a lower ground shadow.
    boxShadow: tokens.shadowNodeRaised,
    transitionDuration: "120ms",
  },
  dragging: { cursor: "grabbing" },
  pickupShadow: {
    // Geometry-neutral ground plate. Sits BEFORE the card in DOM order so the
    // opaque shell paints over the overlap; the inline gutter-aware inset keeps
    // the plate box tucked behind the lifted shell, so only the offset shadow
    // reads as ground.
    position: "absolute",
    display: "block",
    borderRadius: tokens.radiusLg,
    boxShadow: tokens.shadowNodeActive,
    opacity: 0,
    pointerEvents: "none",
    transform: "translate3d(0, 2px, 0) scale(0.97)",
    transformOrigin: "50% 45%",
    transitionProperty: {
      default: "opacity, transform, box-shadow",
      "@media (prefers-reduced-motion: reduce)": "none",
    },
    // Release settles quicker than pickup (opacity, transform, box-shadow).
    transitionDuration: "70ms, 120ms, 120ms",
    transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
  },
  pickupShadowActive: {
    opacity: 0.5,
    transform: "translate3d(0, 3px, 0)",
    // Opacity and box-shadow ease while the transform rides the lift spring.
    transitionDuration: "120ms, 200ms, 200ms",
    transitionTimingFunction:
      "cubic-bezier(0.22, 1, 0.36, 1), cubic-bezier(0.34, 1.56, 0.64, 1), cubic-bezier(0.22, 1, 0.36, 1)",
  },
  pickupShadowDragged: {
    opacity: 0.9,
    transform: "translate3d(0, 9px, 0) scale(1.02)",
    boxShadow: tokens.shadowNodeDragged,
    transitionDuration: "120ms, 200ms, 200ms",
    transitionTimingFunction:
      "cubic-bezier(0.22, 1, 0.36, 1), cubic-bezier(0.34, 1.56, 0.64, 1), cubic-bezier(0.22, 1, 0.36, 1)",
  },
  header: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  removeButton: {
    width: "22px",
    height: "22px",
    display: "grid",
    placeItems: "center",
    flexShrink: 0,
    padding: 0,
    borderWidth: 0,
    borderRadius: "9999px",
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  heading: {
    minWidth: 0,
    display: "grid",
    flex: 1,
    gap: "2px",
  },
  titleRow: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "7px",
  },
  title: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeMd,
    fontWeight: 500,
    letterSpacing: "-0.01em",
  },
  contract: {
    minWidth: 0,
    maxWidth: "220px",
    overflow: "hidden",
    padding: "2px 7px",
    borderRadius: "9999px",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "9px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  source: {
    maxWidth: "360px",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "9px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  status: {
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    flexShrink: 0,
    color: tokens.colorSuccess,
    fontSize: "9px",
    fontWeight: 600,
    letterSpacing: "0.02em",
    textTransform: "uppercase",
  },
  statusBusy: { color: tokens.colorInfo },
  statusIdle: { color: tokens.colorSubtle },
  statusUnavailable: { color: tokens.colorWarning },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
  inputRow: {
    position: "relative",
    display: "flex",
    alignItems: "center",
    minHeight: "28px",
    marginLeft: "-12px",
  },
  inputTab: {
    height: "24px",
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    paddingInline: "14px 11px",
    borderRadius: "0 9999px 9999px 0",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 500,
  },
  inputKind: {
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "9px",
  },
  interactionRow: {
    position: "relative",
    minHeight: "28px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginInline: "-12px",
  },
  interactionTab: {
    minHeight: "24px",
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    paddingInline: "14px 10px",
    borderRadius: "0 9999px 9999px 0",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorMuted,
    fontSize: "9px",
    fontWeight: 500,
  },
  interactionTabOutput: {
    paddingInline: "10px 14px",
    borderRadius: "9999px 0 0 9999px",
  },
  viewport: {
    minHeight: 0,
    overflow: "hidden",
  },
  empty: {
    height: "100%",
    display: "grid",
    placeItems: "center",
    alignContent: "center",
    gap: "8px",
    padding: "24px",
    borderRadius: tokens.radiusMd,
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorSubtle,
    textAlign: "center",
    boxSizing: "border-box",
  },
  emptyTitle: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 600,
  },
  emptyCopy: {
    maxWidth: "290px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.5,
  },
});

function interactionProps(props: ReturnType<typeof stylex.props>) {
  return {
    ...props,
    className: `nodrag nopan nowheel${props.className ? ` ${props.className}` : ""}`,
  };
}

export default function ArtifactViewerNodeCard({
  id,
  data,
  isConnectable,
  selected,
  dragging,
}: NodeProps<ArtifactViewerNode>) {
  const edges = useEdges<CanvasEdge>();
  const incomingEdge = edges.find(
    (edge): edge is ArtifactViewerEdge =>
      edge.type === ARTIFACT_VIEWER_EDGE_TYPE &&
      edge.target === id &&
      edge.targetHandle === ARTIFACT_VIEWER_INPUT_HANDLE,
  );
  const sourceNodeCandidate = useNodesData<CanvasNode>(
    incomingEdge?.source ?? "",
  );
  const sourceNode = sourceNodeCandidate?.type === WORKFLOW_NODE_TYPE
    ? sourceNodeCandidate
    : null;
  const sourcePortName = incomingEdge?.data?.sourcePortName ?? null;
  const sourcePort = sourceNode && sourcePortName
    ? sourceNode.data.spec.outputs.find(
        (candidate) => candidate.name === sourcePortName,
      )
    : undefined;
  const succeededRun = sourceNode?.data.run?.status === "succeeded"
    ? sourceNode.data.run
    : null;
  const output = succeededRun && sourcePortName
    ? succeededRun.outputs.find(
        (candidate) => candidate.port === sourcePortName,
      )
    : undefined;
  const renderableOutput = output?.artifacts.length ? output : null;
  const firstArtifact = renderableOutput?.artifacts[0];
  const declaredArtifactType = sourcePort && sourceNode
    ? resolvedPortArtifactType(
        sourcePort,
        sourceNode.data.artifactTypeBindings,
      )
    : null;
  const artifactTypeLabel = firstArtifact
    ? `${firstArtifact.artifact_type}@${firstArtifact.schema_version}`
    : declaredArtifactType
      ? `${declaredArtifactType.id}@${declaredArtifactType.schema_version}`
      : "Any artifact";
  const artifactShapeLabel = output
    ? output.kind
    : sourcePort && sourceNode
      ? effectivePortShape(sourceNode.data, sourcePort) === "many"
        ? "sequence"
        : "single"
      : null;
  const artifactContract = artifactShapeLabel
    ? `${artifactTypeLabel} · ${artifactShapeLabel}`
    : artifactTypeLabel;
  const feedLabel = incomingEdge?.data?.projection?.path.length
    ? `${sourcePort?.title ?? sourcePortName ?? "output"}.${incomingEdge.data.projection.path.join(".")}`
    : (sourcePort?.title ?? sourcePortName ?? "output");
  const sourceLabel = !incomingEdge
    ? "No output connected"
    : sourceNode
      ? `${sourceNode.data.spec.title} → ${feedLabel}`
      : `${incomingEdge.source} → ${feedLabel}`;
  const sourceIsBusy =
    sourceNode?.data.execution.status === "uploading" ||
    sourceNode?.data.execution.status === "queued" ||
    sourceNode?.data.execution.status === "running" ||
    sourceNode?.data.execution.status === "cancelling";
  const stateLabel = !incomingEdge
    ? "Waiting"
    : !sourceNode
      ? "Unavailable"
      : sourceIsBusy
        ? "Updating"
        : renderableOutput
          ? "Materialized"
          : "No artifact";
  const updateNodeInternals = useUpdateNodeInternals();
  const { tier, pickedUp, draggedTier, liftRef, holdHandlers } = usePickupLift({
    id,
    selected,
    dragging,
    updateNodeInternals,
  });
  const [draftLayout, setDraftLayout] = React.useState<WorkflowNodeLayout | null>(
    null,
  );
  const layout = draftLayout ?? data.layout;
  const width = resolvedNodeWidth(layout);
  const previewHeight = resolvedAppendixHeight(layout);
  const {
    contentRef,
    frameStyle,
    shellStyle,
    gridWidth,
    gutter,
    fillMinHeight,
  } = useShellGridFill(width);
  const outputRevision = renderableOutput?.artifacts
    .map((artifact) => artifact.artifact_id)
    .join(":") ?? "";
  const { workspace } = useWorkspaceContext();
  const { data: registry } = useNodeRegistry(workspace.id);
  const interaction = React.useMemo<ArtifactViewerInteractionContext>(
    () => ({
      outgoingFields: data.outgoingFields ?? [],
      selection: data.selection ?? EMPTY_ARTIFACT_KEY_SELECTION,
      incoming: data.incomingBindings ?? [],
      onFieldsChange: (fields) =>
        data.onFieldsChange?.(id, fields),
      onSelectionChange: (selection) =>
        data.onSelectionChange?.(id, selection),
      onActivityChange: (activity) =>
        data.onActivityChange?.(id, activity),
    }),
    [data, id],
  );

  React.useLayoutEffect(() => {
    updateNodeInternals(id);
  }, [
    fillMinHeight,
    gridWidth,
    id,
    incomingEdge?.id,
    outputRevision,
    previewHeight,
    updateNodeInternals,
  ]);

  const commitLayout = (next: WorkflowNodeLayout | null) => {
    setDraftLayout(null);
    data.onLayoutChange?.(id, next);
    window.requestAnimationFrame(() => updateNodeInternals(id));
  };

  return (
    <div
      ref={liftRef}
      {...holdHandlers}
      {...stylex.props(
        s.shellFrame,
        tier === "active" ? s.shellFrameActive : null,
        tier === "dragged" ? s.shellFrameDragged : null,
      )}
      style={frameStyle}
    >
      <span
        aria-hidden="true"
        data-node-pickup-shadow="true"
        data-picked-up={pickedUp}
        data-dragging={draggedTier}
        {...stylex.props(
          s.pickupShadow,
          tier === "active" ? s.pickupShadowActive : null,
          tier === "dragged" ? s.pickupShadowDragged : null,
        )}
        style={{
          inset: `${gutter}px ${gutter + 10}px ${gutter + 12}px ${gutter + 10}px`,
        }}
      />
      <article
        aria-label="Artifact viewer"
        data-testid="artifact-viewer-node"
        {...stylex.props(
          s.shell,
          pickedUp ? s.pickedUp : null,
          draggedTier ? s.dragging : null,
        )}
        style={shellStyle}
      >
      <div ref={contentRef} {...stylex.props(s.shellContent)}>
        {!selected && data.remoteSelectionColor ? (
          <RemoteSelectionRing color={data.remoteSelectionColor} />
        ) : null}
        <header {...stylex.props(s.header)}>
          <button
            type="button"
            aria-label="Remove Artifact viewer"
            title="Remove Artifact viewer"
            {...interactionProps(stylex.props(s.removeButton))}
            onClick={() => data.onRemoveNode?.(id)}
          >
            <X size={13} aria-hidden="true" />
          </button>
          <span {...stylex.props(s.heading)}>
            <span {...stylex.props(s.titleRow)}>
              <span {...stylex.props(s.title)}>Artifact Viewer</span>
              <span title={artifactContract} {...stylex.props(s.contract)}>
                {artifactContract}
              </span>
            </span>
            <span title={sourceLabel} {...stylex.props(s.source)}>
              {sourceLabel}
            </span>
          </span>
          <span
            aria-live="polite"
            {...stylex.props(
              s.status,
              sourceIsBusy ? s.statusBusy : null,
              !sourceIsBusy &&
                  (!incomingEdge ||
                    (incomingEdge && !renderableOutput && sourceNode))
                ? s.statusIdle
                : null,
              incomingEdge && !sourceNode ? s.statusUnavailable : null,
            )}
          >
            {sourceIsBusy ? (
              <LoaderCircle
                size={10}
                aria-hidden="true"
                {...stylex.props(s.spinner)}
              />
            ) : !incomingEdge ? (
              <Link2 size={10} aria-hidden="true" />
            ) : !sourceNode ? (
              <TriangleAlert size={10} aria-hidden="true" />
            ) : renderableOutput ? (
              <CheckCircle2 size={10} aria-hidden="true" />
            ) : (
              <Link2 size={10} aria-hidden="true" />
            )}
            {stateLabel}
          </span>
        </header>

        <div {...stylex.props(s.inputRow)}>
          <span {...stylex.props(s.inputTab)}>
            Artifact <span {...stylex.props(s.inputKind)}>any</span>
          </span>
          <Handle
            id={ARTIFACT_VIEWER_INPUT_HANDLE}
            type="target"
            position={Position.Left}
            isConnectable={isConnectable}
            aria-label="Input port Artifact, accepts Any artifact"
            title="Accepts any artifact or artifact sequence. Connect an output here."
            style={handleStyle("50%", tokens.colorAccent)}
          />
        </div>

        <div {...stylex.props(s.interactionRow)}>
          <span {...stylex.props(s.interactionTab)}>
            <Link2 size={10} aria-hidden="true" />
            linked input
          </span>
          <Handle
            id={ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE}
            type="target"
            position={Position.Left}
            isConnectable={isConnectable}
            aria-label="Viewer interaction input"
            title="Accept a key selection from another Artifact Viewer."
            style={handleStyle("50%", tokens.colorInfo)}
          />
          <span {...stylex.props(s.interactionTab, s.interactionTabOutput)}>
            selection
            <Link2 size={10} aria-hidden="true" />
          </span>
          <Handle
            id={ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE}
            type="source"
            position={Position.Right}
            isConnectable={isConnectable}
            aria-label="Viewer selection output"
            title="Send this viewer's key selection to another Artifact Viewer."
            style={handleStyle("50%", tokens.colorInfo)}
          />
        </div>

        <div
          {...interactionProps(stylex.props(s.viewport))}
          style={{ height: previewHeight }}
        >
          {renderableOutput ? (
            <ArtifactPortPreview
              key={`${incomingEdge?.source}:${renderableOutput.port}:${outputRevision}:${incomingEdge?.data?.projection?.path.join(".") ?? "whole"}`}
              output={renderableOutput}
              artifactTypes={registry?.artifact_types ?? []}
              previewHeight={Math.max(120, previewHeight - 44)}
              modeChoice={data.mode}
              onModeChoiceChange={(mode) => data.onModeChange?.(id, mode)}
              feedProjection={incomingEdge?.data?.projection ?? null}
              interaction={interaction}
            />
          ) : (
            <div {...stylex.props(s.empty)}>
              <Link2 size={19} aria-hidden="true" />
              <span {...stylex.props(s.emptyTitle)}>
                {incomingEdge ? "No materialization yet" : "Connect an output"}
              </span>
              <span {...stylex.props(s.emptyCopy)}>
                {incomingEdge
                  ? "The connected output has not produced an artifact for the current run."
                  : "The viewer chooses its renderer from the artifact connected to this generic input."}
              </span>
            </div>
          )}
        </div>
      </div>

      <LayoutResizeHandle
        layout={layout}
        axes={["width", "appendixHeight"]}
        ariaLabel="Resize artifact viewer"
        onDraft={setDraftLayout}
        onCommit={commitLayout}
      />
      </article>
    </div>
  );
}

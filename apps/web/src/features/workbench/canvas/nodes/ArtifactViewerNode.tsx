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

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const s = stylex.create({
  shell: {
    position: "relative",
    display: "grid",
    gap: "10px",
    width: "520px",
    padding: "10px 12px 12px",
    overflow: "visible",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    boxSizing: "border-box",
  },
  selected: {
    boxShadow: tokens.shadowNodeSelected,
    outlineWidth: "2px",
    outlineStyle: "solid",
    outlineColor: tokens.colorAccentBorder,
    outlineOffset: "1px",
  },
  header: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "8px",
    cursor: "grab",
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
    fontWeight: 700,
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
    fontWeight: 750,
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
    fontWeight: 650,
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
    fontWeight: 700,
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
    borderRadius: "10px",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorSubtle,
    textAlign: "center",
    boxSizing: "border-box",
  },
  emptyTitle: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 750,
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
  const sourceLabel = !incomingEdge
    ? "No output connected"
    : sourceNode
      ? `${sourceNode.data.spec.title} → ${sourcePort?.title ?? sourcePortName ?? "output"}`
      : `${incomingEdge.source} → ${sourcePortName ?? "output"}`;
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
  const [draftLayout, setDraftLayout] = React.useState<WorkflowNodeLayout | null>(
    null,
  );
  const layout = draftLayout ?? data.layout;
  const width = resolvedNodeWidth(layout);
  const previewHeight = resolvedAppendixHeight(layout);
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

  React.useEffect(() => {
    updateNodeInternals(id);
  }, [id, incomingEdge?.id, outputRevision, previewHeight, updateNodeInternals, width]);

  const commitLayout = (next: WorkflowNodeLayout | null) => {
    setDraftLayout(null);
    data.onLayoutChange?.(id, next);
    window.requestAnimationFrame(() => updateNodeInternals(id));
  };

  return (
    <article
      aria-label="Artifact viewer"
      data-testid="artifact-viewer-node"
      {...stylex.props(s.shell, selected ? s.selected : null)}
      style={{ width }}
    >
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
            key={`${incomingEdge?.source}:${renderableOutput.port}:${outputRevision}`}
            output={renderableOutput}
            artifactTypes={registry?.artifact_types ?? []}
            previewHeight={Math.max(120, previewHeight - 44)}
            modeChoice={data.mode}
            onModeChoiceChange={(mode) => data.onModeChange?.(id, mode)}
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

      <LayoutResizeHandle
        layout={layout}
        axes={["width", "appendixHeight"]}
        ariaLabel="Resize artifact viewer"
        onDraft={setDraftLayout}
        onCommit={commitLayout}
      />
    </article>
  );
}

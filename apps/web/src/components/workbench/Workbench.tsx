"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Toast } from "@base-ui/react/toast";
import { useRouter } from "next/navigation";
import {
  NodeToolbar,
  Position,
  type Connection,
  type IsValidConnection,
  type Node,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  ChevronDown,
  CircleAlert,
  Copy,
  LoaderCircle,
  Maximize2,
  Monitor,
  Moon,
  Play,
  Plus,
  Save,
  Sun,
  Trash2,
  Workflow,
  X,
} from "lucide-react";

import { NodeSelector } from "@/components/workbench/NodeSelector";
import { SavedGraphBrowser } from "@/components/workbench/SavedGraphBrowser";
import {
  NEW_GRAPH_ROUTE_ID,
  workbenchGraphPath,
} from "@/components/workbench/routes";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  WorkflowCanvas,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
} from "@/components/canvas/WorkflowCanvas";
import {
  connectionRouteForSelection,
  connectionRouteMatchesSelection,
  connectionRouteSelection,
  connectionRoutesFor,
  decodedHandleArtifactType,
  decodeHandleId,
  encodeHandleId,
  type ConnectionRoute,
} from "@/components/canvas/handles";
import {
  appendInputPlug,
  collectContributionLabel,
  inputPlugsForPort,
  removeInputPlug as withoutInputPlug,
  reorderInputPlug as withReorderedInputPlug,
} from "@/components/canvas/input-plugs";
import {
  hydrateSavedGraph,
  savedGraphDraft,
  savedGraphFingerprint,
  withMaterializedNodeRuns,
} from "@/components/canvas/saved-graph";
import {
  nodeSecretBindingReady,
  nodeSecretInputs,
  reconciledNodeSecretStatuses,
  type WorkflowNodeSecretInput,
  type WorkflowNodeSecretStatus,
  type WorkflowNodeSecretStatuses,
} from "@/components/canvas/node-secrets";
import {
  ARTIFACT_TYPE_COLOR,
} from "@/components/canvas/nodes.css";
import type { SchemaBuilderField } from "@/components/canvas/schema-builder";
import { useTheme } from "@/components/theme";
import {
  IMAGE_UPLOAD_OPERATOR_ID,
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  acceptedPortShapes,
  bindArtifactTypeVariable,
  createWorkflowNodeData,
  effectivePortShape,
  imageUploads,
  invalidateWorkflowNodeRuns,
  portHasInstancePlugs,
  removeImageUpload,
  replaceImageUploads,
  resetArtifactTypeBinding,
  serializeRunNode,
  serializeWorkflowEdgeTransport,
  type WorkflowEdge,
  type WorkflowEdgeRouteOption,
  type WorkflowEdgeRouteOffset,
  type WorkflowEdgeUpdate,
  type WorkflowArtifactTypeBindings,
  type WorkflowNodeData,
  type WorkflowInputPlugBinding,
  type WorkflowInputPlug,
} from "@/components/canvas/types";
import { useNodeRegistry, useSavedGraphs } from "@/hooks/use-api";
import {
  applyNodeSecret,
  createSavedGraph,
  deleteSavedGraph,
  fileToBase64,
  getGraphMaterializations,
  getGraphNodeSecrets,
  getSavedGraph,
  removeNodeSecret,
  runGraph,
  updateSavedGraph,
  uploadImage,
  type ArtifactTypeKey,
  type NodeRegistry,
  type NodeSecretStatus,
  type NodeSpec,
  type PinnedOutputInput,
  type Port,
  type RunEdgeCollectionMode,
  type RunEdgeInput,
  type RunNodeResult,
  type SavedGraphNode,
  type SavedGraphSummary,
} from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { tokens } from "@/lib/stylex/tokens.stylex";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;
type RunScope = "all" | "selected" | "selected-with-dependencies";

interface ActiveSavedGraph {
  id: string;
  revision: number;
  nodes: readonly SavedGraphNode[];
}

type NodeSecretStatusesByNode = Readonly<
  Record<string, WorkflowNodeSecretStatuses>
>;

interface WorkbenchProps {
  workspaceSlug: string;
  initialGraphId: string | null;
}

type GlobalIssueId = "registry" | "graph" | "run";

interface GlobalIssue {
  id: GlobalIssueId;
  title: string;
  message: string;
}

const NEW_GRAPH_NAME = "Untitled workflow";
const WORKBENCH_FIT_VIEW_OPTIONS = {
  padding: {
    top: "90px",
    right: "48px",
    bottom: "64px",
    left: "165px",
  },
  maxZoom: 0.88,
} as const;

function graphNodeSecretStatuses(
  nodes: readonly WorkflowNode[],
  remote: readonly NodeSecretStatus[],
): NodeSecretStatusesByNode {
  return Object.fromEntries(
    nodes
      .filter((node) => nodeSecretInputs(node.data.spec).length > 0)
      .map((node) => [
        node.id,
        reconciledNodeSecretStatuses(node.data.spec, node.id, remote),
      ]),
  );
}

function nodeSecretStatusesWithState(
  nodes: readonly WorkflowNode[],
  state: WorkflowNodeSecretStatus["state"],
  message?: string,
): NodeSecretStatusesByNode {
  return Object.fromEntries(
    nodes
      .filter((node) => nodeSecretInputs(node.data.spec).length > 0)
      .map((node) => [
        node.id,
        Object.fromEntries(
          nodeSecretInputs(node.data.spec).map((input) => [
            input.name,
            { state, message } satisfies WorkflowNodeSecretStatus,
          ]),
        ),
      ]),
  );
}

interface ConnectionEndpoint {
  nodeTitle: string;
  portName: string;
  artifactType: string;
}

interface PendingConnectionRoute {
  connection: Connection;
  collectionMode: RunEdgeCollectionMode;
  candidates: ConnectionRoute[];
  source: ConnectionEndpoint;
  target: ConnectionEndpoint;
}

interface PendingBoundEdge {
  nodeId: string;
  variable: string;
  artifactType: ArtifactTypeKey;
  edge: WorkflowEdge;
}

function connectionRouteTitle(route: ConnectionRoute): string {
  const conversionTitle = route.conversionPath
    .map((conversion) => conversion.title)
    .join(" → ");
  let title = "Whole output";
  if (route.kind === "projection") title = route.projection.title;
  if (route.kind === "conversion") title = conversionTitle;
  if (route.kind === "projection-conversion") {
    title = `${route.projection.title} → ${conversionTitle}`;
  }
  const binding = route.artifactTypeBinding;
  return binding
    ? `${title} · ${binding.artifactType.id}@${binding.artifactType.schema_version}`
    : title;
}

function connectionRouteDescription(
  sourcePortName: string,
  route: ConnectionRoute,
): string {
  const conversionDescription = route.conversionPath
    .map(
      (conversion) =>
        `${conversion.title} · ${conversion.key.id}@${conversion.key.version}`,
    )
    .join(" → ");
  if (route.kind === "projection") {
    return `${sourcePortName}.${route.projection.path.join(".")}`;
  }
  if (route.kind === "conversion") {
    return `${sourcePortName} → ${conversionDescription}`;
  }
  if (route.kind === "projection-conversion") {
    return `${sourcePortName}.${route.projection.path.join(".")} → ${conversionDescription}`;
  }
  return sourcePortName;
}

function workflowEdgeRouteOption(
  route: ConnectionRoute,
): WorkflowEdgeRouteOption {
  const selection = connectionRouteSelection(route);
  return {
    ...selection,
    projectionTitle:
      route.kind === "projection" || route.kind === "projection-conversion"
        ? route.projection.title
        : undefined,
    conversionTitles: route.conversionPath.map(
      (conversion) => conversion.title,
    ),
  };
}

function mappedInputPortForNode(
  nodeId: string,
  edges: readonly WorkflowEdge[],
): string | null {
  const edge = edges.find(
    (candidate) =>
      candidate.target === nodeId && candidate.data?.collectionMode === "map",
  );
  return decodeHandleId(edge?.targetHandle)?.portName ?? null;
}

function effectiveShapeForPort(
  node: WorkflowNode,
  port: Port,
  edges: readonly WorkflowEdge[],
): Port["shape"] {
  return effectivePortShape(
    {
      ...node.data,
      mappedInputPort: mappedInputPortForNode(node.id, edges),
    },
    port,
  );
}

function collectionModeForConnection(
  connection: Connection,
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): RunEdgeCollectionMode | null {
  const sourceHandle = decodeHandleId(connection.sourceHandle);
  const targetHandle = decodeHandleId(connection.targetHandle);
  const sourceNode = nodes.find((node) => node.id === connection.source);
  const targetNode = nodes.find((node) => node.id === connection.target);
  const sourcePort = sourceNode?.data.spec.outputs.find(
    (port) => port.name === sourceHandle?.portName,
  );
  const targetPort = targetNode?.data.spec.inputs.find(
    (port) => port.name === targetHandle?.portName,
  );
  if (!sourceNode || !targetNode || !sourcePort || !targetPort) return null;

  const sourceShape = effectiveShapeForPort(sourceNode, sourcePort, edges);
  if (acceptedPortShapes(targetPort).includes(sourceShape)) return "direct";
  if (portHasInstancePlugs(targetPort)) return null;

  const targetShape = effectiveShapeForPort(targetNode, targetPort, edges);
  if (sourceShape === targetShape) return "direct";
  if (sourceShape === "many" && targetShape === "one") return "map";
  return null;
}

function inputPlugBindingsForNode(
  node: WorkflowNode,
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
  registry: NodeRegistry | undefined,
): Readonly<Record<string, WorkflowInputPlugBinding>> {
  const bindings: Record<string, WorkflowInputPlugBinding> = {};
  for (const port of node.data.spec.inputs.filter(portHasInstancePlugs)) {
    const portPlugs = inputPlugsForPort(node.data.inputPlugs, port.name);
    portPlugs.forEach((plug, inputIndex) => {
      const edge = edges.find(
        (candidate) =>
          candidate.target === node.id &&
          decodeHandleId(candidate.targetHandle)?.plugId === plug.id,
      );
      if (!edge) return;

      const sourceHandle = decodeHandleId(edge.sourceHandle);
      const sourceNode = nodes.find((candidate) => candidate.id === edge.source);
      const sourcePort = sourceNode?.data.spec.outputs.find(
        (candidate) => candidate.name === sourceHandle?.portName,
      );
      if (!sourceHandle || !sourceNode || !sourcePort) return;

      const projectionLabel = edge.data?.projection?.path.join(".");
      const conversionLabels = (edge.data?.conversionPath ?? []).map(
        (requestedConversion) =>
          registry?.artifact_conversions.find(
            (conversion) =>
              conversion.key.id === requestedConversion.id &&
              conversion.key.version === requestedConversion.version,
          )?.title ??
          `${requestedConversion.id}@${requestedConversion.version}`,
      );
      const conversionLabel = [projectionLabel, ...conversionLabels]
        .filter((label): label is string => Boolean(label))
        .join(" → ");
      const contributionLabel = collectContributionLabel(
        node.data.run,
        inputIndex,
      );
      bindings[plug.id] = {
        sourceLabel: `${sourceNode.data.spec.title} · ${sourcePort.title ?? sourcePort.name}`,
        sourceShape: effectiveShapeForPort(sourceNode, sourcePort, edges),
        ...(conversionLabel ? { conversionLabel } : {}),
        ...(contributionLabel ? { contributionLabel } : {}),
      };
    });
  }
  return bindings;
}

function nodeAndDescendantIds(
  nodeId: string,
  edges: readonly WorkflowEdge[],
): Set<string> {
  const invalidatedNodeIds = new Set([nodeId]);
  const pendingNodeIds = [nodeId];

  while (pendingNodeIds.length) {
    const sourceNodeId = pendingNodeIds.shift();
    if (sourceNodeId === undefined) continue;

    for (const edge of edges) {
      if (
        edge.source !== sourceNodeId ||
        invalidatedNodeIds.has(edge.target)
      ) {
        continue;
      }
      invalidatedNodeIds.add(edge.target);
      pendingNodeIds.push(edge.target);
    }
  }

  return invalidatedNodeIds;
}

function selectedNodeAndAncestorIds(
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): Set<string> {
  const knownNodeIds = new Set(nodes.map((node) => node.id));
  const executionNodeIds = new Set(
    nodes.filter((node) => node.selected).map((node) => node.id),
  );
  const pendingNodeIds = [...executionNodeIds];

  while (pendingNodeIds.length) {
    const targetNodeId = pendingNodeIds.shift();
    if (targetNodeId === undefined) continue;

    for (const edge of edges) {
      if (
        edge.target !== targetNodeId ||
        !knownNodeIds.has(edge.source) ||
        executionNodeIds.has(edge.source)
      ) {
        continue;
      }
      executionNodeIds.add(edge.source);
      pendingNodeIds.push(edge.source);
    }
  }

  return executionNodeIds;
}

interface MissingRequiredInput {
  nodeId: string;
  nodeTitle: string;
  portName: string;
}

function missingRequiredInputsFor(
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): MissingRequiredInput[] {
  return nodes.flatMap((node) =>
    node.data.spec.inputs.flatMap((port) => {
      if (portHasInstancePlugs(port)) {
        const plugs = inputPlugsForPort(node.data.inputPlugs, port.name);
        if (!plugs.length) {
          return port.required
            ? [{
                nodeId: node.id,
                nodeTitle: node.data.spec.title,
                portName: port.name,
              }]
            : [];
        }
        return plugs.flatMap((plug, index) =>
          edges.some(
            (edge) =>
              edge.target === node.id &&
              decodeHandleId(edge.targetHandle)?.plugId === plug.id,
          )
            ? []
            : [{
                nodeId: node.id,
                nodeTitle: node.data.spec.title,
                portName: `${port.name} input ${index + 1}`,
              }],
        );
      }
      if (!port.required) return [];
      return edges.some(
        (edge) =>
          edge.target === node.id &&
          decodeHandleId(edge.targetHandle)?.portName === port.name,
      )
        ? []
        : [{
            nodeId: node.id,
            nodeTitle: node.data.spec.title,
            portName: port.name,
          }];
    }),
  );
}

interface ExecutionValidationIssue {
  nodeId: string | null;
  message: string;
}

function executionValidationIssue(
  scope: RunScope,
  executionNodes: readonly WorkflowNode[],
  executionEdges: readonly WorkflowEdge[],
): ExecutionValidationIssue | null {
  if (!executionNodes.length) {
    return {
      nodeId: null,
      message: scope !== "all"
        ? "Select at least one node before running a selection."
        : "Add at least one node before running the workflow.",
    };
  }

  const imageUploadWithoutImages = executionNodes.find(
    (node) =>
      node.data.spec.operator_id === IMAGE_UPLOAD_OPERATOR_ID &&
      !imageUploads(node.data).length,
  );
  if (imageUploadWithoutImages) {
    return {
      nodeId: imageUploadWithoutImages.id,
      message: `Choose images for ${imageUploadWithoutImages.data.spec.title} before running.`,
    };
  }

  const missingInputs = missingRequiredInputsFor(
    executionNodes,
    executionEdges,
  );
  if (!missingInputs.length) return null;

  const first = missingInputs[0];
  return {
    nodeId: first.nodeId,
    message: `${first.nodeTitle}.${first.portName} is required but unconnected in this run.`,
  };
}

const s = stylex.create({
  shell: {
    position: "relative",
    width: "100%",
    height: "100svh",
    overflow: "hidden",
    backgroundColor: tokens.colorBg,
    color: tokens.colorText,
  },
  canvas: { position: "absolute", inset: 0 },
  topBar: {
    position: "absolute",
    zIndex: 20,
    top: "13px",
    left: "13px",
    right: "13px",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: "12px",
    pointerEvents: "none",
  },
  chrome: {
    display: "flex",
    alignItems: "center",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "8px",
    backgroundColor: tokens.colorChrome,
    pointerEvents: "auto",
  },
  identity: {
    minHeight: "43px",
    gap: "9px",
    padding: "6px 9px 6px 11px",
    borderRadius: "12px",
    boxShadow: tokens.shadowNode,
  },
  identityCopy: {
    width: "min(230px, 42vw)",
    minWidth: 0,
    display: "grid",
    gap: "1px",
  },
  identityMenu: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "5px",
    padding: 0,
    borderWidth: 0,
    backgroundColor: "transparent",
    color: tokens.colorMuted,
    cursor: "pointer",
    textAlign: "left",
  },
  brand: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    fontWeight: 800,
    letterSpacing: "0.16em",
    lineHeight: 1.1,
  },
  saveState: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1.1,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  saveStateDirty: { color: tokens.colorWarning },
  workflowName: {
    width: "100%",
    minWidth: 0,
    padding: 0,
    overflow: "hidden",
    borderWidth: 0,
    outline: "none",
    backgroundColor: "transparent",
    color: tokens.colorTextEmphasis,
    fontFamily: "inherit",
    fontSize: tokens.fontSizeMd,
    fontWeight: 700,
    lineHeight: 1.2,
    textOverflow: "ellipsis",
  },
  identityDivider: {
    width: "1px",
    height: "28px",
    flexShrink: 0,
    backgroundColor: tokens.colorDivider,
  },
  identityActions: {
    display: "flex",
    alignItems: "center",
    gap: "2px",
  },
  identityAction: {
    width: "29px",
    paddingInline: 0,
  },
  identityActionActive: {
    backgroundColor: {
      default: tokens.colorAccentSoft,
      ":hover": tokens.colorAccentSoft,
    },
    color: tokens.colorAccent,
  },
  identityStats: {
    minWidth: {
      default: "116px",
      "@media (max-width: 520px)": "52px",
    },
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "5px",
    flexShrink: 0,
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontVariantNumeric: "tabular-nums",
    whiteSpace: "nowrap",
  },
  identityStatValue: {
    color: tokens.colorTextEmphasis,
    fontWeight: 750,
  },
  identityStatLabel: {
    display: {
      default: "inline",
      "@media (max-width: 520px)": "none",
    },
  },
  identityStatSeparator: { color: tokens.colorDivider },
  graphStatusDot: {
    width: "5px",
    height: "5px",
    flexShrink: 0,
    borderRadius: "99px",
    backgroundColor: tokens.colorSuccess,
  },
  graphStatusDotIncomplete: { backgroundColor: tokens.colorWarning },
  graphStatusDotError: { backgroundColor: tokens.colorDanger },
  graphStatusDotRunning: { backgroundColor: tokens.colorInfo },
  toolButton: {
    height: "31px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    paddingInline: "9px",
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover, ":disabled": "transparent" },
    color: { default: tokens.colorMuted, ":disabled": tokens.colorTextDisabled },
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    fontSize: tokens.fontSizeSm,
  },
  primaryButton: {
    backgroundColor: {
      default: tokens.colorAccent,
      ":hover": tokens.colorAccentHover,
      ":disabled": tokens.colorAccentDisabled,
    },
    color: { default: tokens.colorOnAccent, ":disabled": tokens.colorTextDisabled },
    fontWeight: 700,
  },
  actionRail: {
    position: "absolute",
    zIndex: 20,
    top: "70px",
    left: "13px",
    width: {
      default: "118px",
      "@media (max-width: 640px)": "44px",
    },
    display: "grid",
    gap: "3px",
    padding: "6px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "14px",
    backgroundColor: tokens.colorChrome,
    boxShadow: tokens.shadowNode,
  },
  toastViewport: {
    position: "fixed",
    zIndex: 80,
    top: "70px",
    right: "13px",
    width: "min(380px, calc(100vw - 26px))",
    maxHeight: "calc(100svh - 84px)",
    display: "flex",
    flexDirection: "column",
    alignItems: "stretch",
    gap: "8px",
    outline: "none",
    pointerEvents: "none",
  },
  toastRoot: {
    width: "100%",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "12px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    pointerEvents: "auto",
  },
  toastContent: {
    display: "grid",
    gridTemplateColumns: "26px minmax(0, 1fr) 28px",
    alignItems: "start",
    gap: "10px",
    padding: "11px",
  },
  toastIcon: {
    width: "26px",
    height: "26px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "8px",
    backgroundColor: tokens.colorDangerHover,
    color: tokens.colorDanger,
  },
  toastCopy: {
    minWidth: 0,
    display: "grid",
    gap: "3px",
    paddingTop: "1px",
  },
  toastTitle: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 750,
    lineHeight: 1.35,
  },
  toastDescription: {
    margin: 0,
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
    overflowWrap: "anywhere",
    userSelect: "text",
    whiteSpace: "pre-wrap",
  },
  toastClose: {
    width: "28px",
    height: "28px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 0,
    borderRadius: "7px",
    outline: "none",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
      ":focus-visible": tokens.colorHoverStrong,
    },
    color: {
      default: tokens.colorSubtle,
      ":hover": tokens.colorText,
      ":focus-visible": tokens.colorText,
    },
    cursor: "pointer",
  },
  railButton: {
    width: "100%",
    height: "34px",
    display: "flex",
    alignItems: "center",
    justifyContent: {
      default: "flex-start",
      "@media (max-width: 640px)": "center",
    },
    gap: "8px",
    paddingInline: {
      default: "9px",
      "@media (max-width: 640px)": 0,
    },
    borderWidth: 0,
    borderRadius: "9px",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
      ":disabled": "transparent",
    },
    color: {
      default: tokens.colorMuted,
      ":hover": tokens.colorTextEmphasis,
      ":disabled": tokens.colorTextDisabled,
    },
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
    textAlign: "left",
    transitionDuration: "120ms",
    transitionProperty: "background-color, color",
  },
  railPrimary: {
    backgroundColor: {
      default: tokens.colorAccentSoft,
      ":hover": tokens.colorAccentSoft,
    },
    color: { default: tokens.colorAccent, ":hover": tokens.colorAccent },
  },
  railDanger: {
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorDangerHover,
      ":disabled": "transparent",
    },
    color: {
      default: tokens.colorMuted,
      ":hover": tokens.colorDanger,
      ":disabled": tokens.colorTextDisabled,
    },
  },
  railLabel: {
    display: {
      default: "inline",
      "@media (max-width: 640px)": "none",
    },
  },
  railDivider: {
    height: "1px",
    marginBlock: "3px",
    backgroundColor: tokens.colorDivider,
  },
  selectionToolbar: {
    zIndex: 25,
    minHeight: "42px",
    display: "flex",
    alignItems: "center",
    gap: "3px",
    padding: "5px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "12px",
    backgroundColor: tokens.colorChrome,
    boxShadow: tokens.shadowNodeSelected,
    pointerEvents: "auto",
  },
  selectionLabel: {
    paddingInline: "7px 9px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
    whiteSpace: "nowrap",
  },
  selectionDivider: {
    width: "1px",
    height: "24px",
    marginInline: "2px",
    backgroundColor: tokens.colorDivider,
  },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
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

function GlobalIssueToastList({
  issues,
  onDismiss,
}: {
  issues: readonly GlobalIssue[];
  onDismiss: (issue: GlobalIssue) => void;
}) {
  const { toasts, add, close } = Toast.useToastManager();
  const activeIssueIds = React.useRef<Set<string>>(new Set());

  React.useEffect(() => {
    const nextIssueIds = new Set<string>();
    for (const issue of issues) {
      const toastId = `workflow-${issue.id}`;
      nextIssueIds.add(toastId);
      add({
        id: toastId,
        title: issue.id === "run" ? "Workflow issue" : `${issue.title} issue`,
        description: issue.message,
        type: "error",
        priority: "high",
        timeout: issue.id === "registry" ? 0 : 8000,
        onClose: () => onDismiss(issue),
      });
    }
    for (const toastId of activeIssueIds.current) {
      if (!nextIssueIds.has(toastId)) close(toastId);
    }
    activeIssueIds.current = nextIssueIds;
  }, [add, close, issues, onDismiss]);

  return (
    <Toast.Portal>
      <Toast.Viewport
        aria-label="Workflow notifications"
        {...stylex.props(s.toastViewport)}
      >
        {toasts.map((toast) => (
          <Toast.Root
            key={toast.id}
            toast={toast}
            swipeDirection="right"
            className={`ns-workbench-toast ${stylex.props(s.toastRoot).className}`}
          >
            <Toast.Content {...stylex.props(s.toastContent)}>
              <span aria-hidden="true" {...stylex.props(s.toastIcon)}>
                <CircleAlert size={15} />
              </span>
              <span {...stylex.props(s.toastCopy)}>
                <Toast.Title {...stylex.props(s.toastTitle)} />
                <Toast.Description {...stylex.props(s.toastDescription)} />
              </span>
              <Toast.Close
                aria-label="Dismiss workflow notification"
                {...stylex.props(s.toastClose)}
              >
                <X size={14} />
              </Toast.Close>
            </Toast.Content>
          </Toast.Root>
        ))}
      </Toast.Viewport>
    </Toast.Portal>
  );
}

export function Workbench({
  workspaceSlug,
  initialGraphId,
}: WorkbenchProps) {
  const router = useRouter();
  const {
    data: registry,
    error: registryError,
    mutate: refreshNodeRegistry,
  } = useNodeRegistry();
  const {
    data: savedGraphList,
    error: savedGraphListError,
    isLoading: savedGraphsLoading,
    isValidating: savedGraphsRefreshing,
    mutate: refreshSavedGraphs,
  } = useSavedGraphs();
  const { preference, cycleTheme } = useTheme();
  const [nodes, setNodes] = React.useState<WorkflowNode[]>([]);
  const [edges, setEdges] = React.useState<WorkflowEdge[]>([]);
  const [nodeSecretStatuses, setNodeSecretStatuses] =
    React.useState<NodeSecretStatusesByNode>({});
  const [graphName, setGraphName] = React.useState(NEW_GRAPH_NAME);
  const [activeGraph, setActiveGraph] =
    React.useState<ActiveSavedGraph | null>(null);
  const [savedFingerprint, setSavedFingerprint] =
    React.useState<string | null>(null);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<WorkflowNode, WorkflowEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [graphBrowserOpen, setGraphBrowserOpen] = React.useState(false);
  const [runningScope, setRunningScope] = React.useState<RunScope | null>(null);
  const running = runningScope !== null;
  const [runError, setRunError] = React.useState<string | null>(null);
  const [pendingConnectionRoute, setPendingConnectionRoute] =
    React.useState<PendingConnectionRoute | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [openingGraphId, setOpeningGraphId] = React.useState<string | null>(null);
  const [deletingGraphId, setDeletingGraphId] = React.useState<string | null>(null);
  const [persistenceError, setPersistenceError] = React.useState<string | null>(null);
  const [fitRevision, setFitRevision] = React.useState(0);
  const initializedRef = React.useRef(false);
  const approvedRouteGraphIdRef = React.useRef<string | null>(null);
  const openRequestRef = React.useRef<AbortController | null>(null);
  const currentFingerprintRef = React.useRef("");
  const activeGraphRef = React.useRef<ActiveSavedGraph | null>(null);
  const pendingBoundEdgesRef = React.useRef<PendingBoundEdge[]>([]);
  const nodesByIdRef = React.useRef<ReadonlyMap<string, WorkflowNode>>(new Map());

  React.useEffect(() => {
    nodesByIdRef.current = new Map(
      nodes.map((node) => [node.id, node]),
    );
  }, [nodes]);

  const handleNodeHandlesMeasured = React.useCallback((
    nodeId: string,
    artifactTypeBindings: WorkflowArtifactTypeBindings,
  ) => {
    const ready: PendingBoundEdge[] = [];
    const waiting: PendingBoundEdge[] = [];
    for (const pending of pendingBoundEdgesRef.current) {
      const measuredBinding = artifactTypeBindings[pending.variable];
      if (
        pending.nodeId === nodeId &&
        measuredBinding?.id === pending.artifactType.id &&
        measuredBinding.schema_version === pending.artifactType.schema_version
      ) {
        ready.push(pending);
      } else {
        waiting.push(pending);
      }
    }
    if (!ready.length) return;

    pendingBoundEdgesRef.current = waiting;
    setEdges((current) =>
      ready.reduce(
        (next, pending) => addEdge(pending.edge, next),
        current,
      ),
    );
  }, []);

  const updateConfig = React.useCallback(
    (nodeId: string, name: string, value: unknown) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              config:
                node.id === nodeId
                  ? { ...node.data.config, [name]: value }
                  : node.data.config,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const refreshNodeSecretStatuses = React.useCallback(async (
    graph: ActiveSavedGraph,
    graphNodes: readonly WorkflowNode[],
    signal?: AbortSignal,
  ): Promise<boolean> => {
    if (!graphNodes.some((node) => nodeSecretInputs(node.data.spec).length > 0)) {
      setNodeSecretStatuses({});
      return true;
    }
    setNodeSecretStatuses(
      nodeSecretStatusesWithState(graphNodes, "loading"),
    );
    try {
      const response = await getGraphNodeSecrets(graph.id, signal);
      if (
        response.graph_id !== graph.id ||
        response.graph_revision !== graph.revision
      ) {
        throw new Error("Node secret status revision mismatch");
      }
      setNodeSecretStatuses(
        graphNodeSecretStatuses(graphNodes, response.secrets),
      );
      return true;
    } catch {
      if (signal?.aborted) return false;
      setNodeSecretStatuses(
        nodeSecretStatusesWithState(
          graphNodes,
          "error",
          "Secret status could not be loaded.",
        ),
      );
      return false;
    }
  }, []);

  const applyConfiguredNodeSecret = React.useCallback(async (
    nodeId: string,
    name: string,
    value: string,
  ): Promise<boolean> => {
    const graph = activeGraphRef.current;
    if (!graph) return false;
    const node = nodesByIdRef.current.get(nodeId);
    const input = node
      ? nodeSecretInputs(node.data.spec).find((candidate) => candidate.name === name)
      : undefined;
    const savedNode = graph.nodes.find((candidate) => candidate.id === nodeId);
    if (
      !node ||
      !input ||
      !nodeSecretBindingReady(input, {
        id: node.id,
        operator_id: node.data.spec.operator_id,
        operator_version: node.data.spec.operator_version,
        config: node.data.config,
      }, savedNode)
    ) {
      return false;
    }

    setNodeSecretStatuses((current) => ({
      ...current,
      [nodeId]: {
        ...(current[nodeId] ?? {}),
        [name]: { state: "applying" },
      },
    }));
    try {
      const response = await applyNodeSecret(graph.id, nodeId, name, {
        value,
        expected_graph_revision: graph.revision,
      });
      if (
        response.node_id !== nodeId ||
        response.name !== name ||
        response.configured !== true
      ) {
        throw new Error("Node secret response mismatch");
      }
      if (
        activeGraphRef.current?.id !== graph.id ||
        activeGraphRef.current.revision !== graph.revision
      ) {
        return true;
      }
      setNodeSecretStatuses((current) => ({
        ...current,
        [nodeId]: {
          ...(current[nodeId] ?? {}),
          [name]: { state: "configured" },
        },
      }));
      return true;
    } catch {
      if (
        activeGraphRef.current?.id === graph.id &&
        activeGraphRef.current.revision === graph.revision
      ) {
        setNodeSecretStatuses((current) => ({
          ...current,
          [nodeId]: {
            ...(current[nodeId] ?? {}),
            [name]: {
              state: "error",
              message: "The secret could not be applied.",
            },
          },
        }));
      }
      return false;
    }
  }, []);

  const removeConfiguredNodeSecret = React.useCallback(async (
    nodeId: string,
    name: string,
  ): Promise<boolean> => {
    const graph = activeGraphRef.current;
    if (!graph) return false;
    const node = nodesByIdRef.current.get(nodeId);
    const input = node
      ? nodeSecretInputs(node.data.spec).find((candidate) => candidate.name === name)
      : undefined;
    const savedNode = graph.nodes.find((candidate) => candidate.id === nodeId);
    if (
      !node ||
      !input ||
      !nodeSecretBindingReady(input, {
        id: node.id,
        operator_id: node.data.spec.operator_id,
        operator_version: node.data.spec.operator_version,
        config: node.data.config,
      }, savedNode)
    ) {
      return false;
    }

    setNodeSecretStatuses((current) => ({
      ...current,
      [nodeId]: {
        ...(current[nodeId] ?? {}),
        [name]: { state: "removing" },
      },
    }));
    try {
      await removeNodeSecret(graph.id, nodeId, name, graph.revision);
      if (
        activeGraphRef.current?.id !== graph.id ||
        activeGraphRef.current.revision !== graph.revision
      ) {
        return true;
      }
      setNodeSecretStatuses((current) => ({
        ...current,
        [nodeId]: {
          ...(current[nodeId] ?? {}),
          [name]: { state: "unconfigured" },
        },
      }));
      return true;
    } catch {
      if (
        activeGraphRef.current?.id === graph.id &&
        activeGraphRef.current.revision === graph.revision
      ) {
        setNodeSecretStatuses((current) => ({
          ...current,
          [nodeId]: {
            ...(current[nodeId] ?? {}),
            [name]: {
              state: "error",
              message: "The stored secret could not be removed.",
            },
          },
        }));
      }
      return false;
    }
  }, []);

  const removeNode = React.useCallback((nodeId: string) => {
    const changedTargetNodeIds = edges
      .filter((edge) => edge.source === nodeId)
      .map((edge) => edge.target);
    setNodes((current) =>
      invalidateWorkflowNodeRuns(
        current.filter((node) => node.id !== nodeId),
        edges,
        changedTargetNodeIds,
      ),
    );
    setEdges((current) =>
      current.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
    );
    setNodeSecretStatuses((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== nodeId),
      ),
    );
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [edges]);

  const handleRemoveImageUpload = React.useCallback(
    (nodeId: string, index: number) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...(node.id === nodeId
                ? removeImageUpload(node.data, index)
                : node.data),
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const addNodeInputPlug = React.useCallback(
    (nodeId: string, portName: string) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? appendInputPlug(node.data.inputPlugs, portName)
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const removeNodeInputPlug = React.useCallback(
    (nodeId: string, plugId: string) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? withoutInputPlug(node.data.inputPlugs, plugId)
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setEdges((current) =>
        current.filter(
          (edge) =>
            edge.target !== nodeId ||
            decodeHandleId(edge.targetHandle)?.plugId !== plugId,
        ),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const reorderNodeInputPlug = React.useCallback(
    (
      nodeId: string,
      portName: string,
      plugId: string,
      toIndex: number,
    ) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              inputPlugs:
                node.id === nodeId
                  ? withReorderedInputPlug(
                      node.data.inputPlugs,
                      portName,
                      plugId,
                      toIndex,
                    )
                  : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setRunError(null);
    },
    [edges],
  );

  const updateSchemaBuilderFields = React.useCallback(
    (
      nodeId: string,
      fields: readonly SchemaBuilderField[],
      inputPlugs: readonly WorkflowInputPlug[],
    ) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      const retainedPlugIds = new Set(inputPlugs.map((plug) => plug.id));
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...node.data,
              config:
                node.id === nodeId
                  ? { ...node.data.config, fields }
                  : node.data.config,
              inputPlugs:
                node.id === nodeId ? inputPlugs : node.data.inputPlugs,
              run: null,
              execution: { status: "idle" },
            },
          };
        }),
      );
      setEdges((current) =>
        current.filter((edge) => {
          if (edge.target !== nodeId) return true;
          const plugId = decodeHandleId(edge.targetHandle)?.plugId;
          return !plugId || retainedPlugIds.has(plugId);
        }),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const handleImagesSelected = React.useCallback(async (nodeId: string, files: File[]) => {
    const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
    setNodes((current) => current.map((node) => {
      if (!invalidatedNodeIds.has(node.id)) return node;
      return {
        ...node,
        data: {
          ...node.data,
          run: null,
          execution: node.id === nodeId
            ? { status: "uploading" }
            : { status: "idle" },
        },
      };
    }));
    setRunError(null);
    try {
      const uploads = await Promise.all(files.map(async (file) =>
        uploadImage(file.name, await fileToBase64(file)),
      ));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: invalidatedNodeIds.has(node.id)
          ? {
              ...(node.id === nodeId
                ? replaceImageUploads(node.data, uploads)
                : node.data),
              execution: { status: "idle" },
              run: null,
            }
          : node.data,
      })));
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "Image upload failed";
      setNodes((current) => current.map((node) => node.id === nodeId ? {
        ...node,
        data: { ...node.data, execution: { status: "failed", error: message } },
      } : node));
    }
  }, [edges]);

  const resetNodeArtifactTypeBinding = React.useCallback(
    (nodeId: string, variable: string) => {
      const hasIncidentEdges = edges.some(
        (edge) => edge.source === nodeId || edge.target === nodeId,
      );
      if (hasIncidentEdges) return;

      setNodes((current) =>
        current.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: resetArtifactTypeBinding(node.data, variable, false),
              }
            : node,
        ),
      );
      setPendingConnectionRoute(null);
      setRunError(null);
    },
    [edges],
  );

  const attachNodeCallbacks = React.useCallback(
    (data: WorkflowNodeData): WorkflowNodeData => ({
      ...data,
      onConfigChange: updateConfig,
      onRemoveNode: removeNode,
      onImagesSelected:
        data.spec.operator_id === IMAGE_UPLOAD_OPERATOR_ID
          ? handleImagesSelected
          : undefined,
      onRemoveImageUpload: handleRemoveImageUpload,
      onAddInputPlug: addNodeInputPlug,
      onRemoveInputPlug: removeNodeInputPlug,
      onReorderInputPlug: reorderNodeInputPlug,
      onSchemaBuilderFieldsChange: updateSchemaBuilderFields,
      onResetArtifactTypeBinding: resetNodeArtifactTypeBinding,
      onHandlesMeasured: handleNodeHandlesMeasured,
    }),
    [
      addNodeInputPlug,
      handleImagesSelected,
      handleNodeHandlesMeasured,
      removeNode,
      removeNodeInputPlug,
      handleRemoveImageUpload,
      reorderNodeInputPlug,
      resetNodeArtifactTypeBinding,
      updateConfig,
      updateSchemaBuilderFields,
    ],
  );

  React.useEffect(() => {
    if (!registry) return;
    if (!initializedRef.current) {
      initializedRef.current = true;
      return;
    }
    const byOperator = new Map(
      registry.nodes.map((spec) => [
        `${spec.operator_id}@${spec.operator_version}`,
        spec,
      ]),
    );
    setNodes((current) => current.map((node) => {
      const spec = byOperator.get(
        `${node.data.spec.operator_id}@${node.data.spec.operator_version}`,
      );
      if (!spec) return { ...node, data: attachNodeCallbacks(node.data) };
      return {
        ...node,
        data: attachNodeCallbacks({ ...node.data, spec }),
      };
    }));
  }, [attachNodeCallbacks, registry]);

  React.useEffect(() => {
    if (!flow || !nodes.length) return;
    const frame = window.requestAnimationFrame(
      () => void flow.fitView(WORKBENCH_FIT_VIEW_OPTIONS),
    );
    return () => window.cancelAnimationFrame(frame);
  }, [fitRevision, flow, nodes.length]);

  const currentDraft = React.useMemo(
    () => savedGraphDraft(graphName, nodes, edges),
    [edges, graphName, nodes],
  );
  const currentFingerprint = React.useMemo(
    () => savedGraphFingerprint(currentDraft),
    [currentDraft],
  );
  React.useEffect(() => {
    currentFingerprintRef.current = currentFingerprint;
  }, [currentFingerprint]);
  React.useEffect(() => {
    activeGraphRef.current = activeGraph;
  }, [activeGraph]);
  const hasUnsavedDraft =
    nodes.length > 0 ||
    edges.length > 0 ||
    graphName.trim() !== NEW_GRAPH_NAME;
  const isDirty = activeGraph
    ? savedFingerprint !== currentFingerprint
    : hasUnsavedDraft;
  const uploading = nodes.some(
    (node) => node.data.execution.status === "uploading",
  );
  const graphOperationBusy = Boolean(
    saving || openingGraphId || deletingGraphId || running || uploading,
  );

  React.useEffect(() => {
    if (!isDirty) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [isDirty]);

  React.useEffect(
    () => () => openRequestRef.current?.abort(),
    [],
  );

  const imageUploadWithoutImages = nodes.some(
    (node) =>
      node.data.spec.operator_id === IMAGE_UPLOAD_OPERATOR_ID &&
      !imageUploads(node.data).length,
  );
  const selectedNodeIds = React.useMemo(
    () => nodes.flatMap((node) => (node.selected ? [node.id] : [])),
    [nodes],
  );
  const selectedNodeCount = selectedNodeIds.length;
  const selectedWithDependenciesCount = selectedNodeAndAncestorIds(
    nodes,
    edges,
  ).size;
  const missingRequiredInputs = missingRequiredInputsFor(nodes, edges);
  const connectionInstruction = missingRequiredInputs.length
    ? `${missingRequiredInputs.length} required input${missingRequiredInputs.length === 1 ? "" : "s"} unconnected · drag between ports to connect them`
    : null;
  const runSelectedDisabled =
    !registry || running || selectedNodeCount === 0;
  const nodeErrorCount = nodes.filter(
    (node) => Boolean(node.data.execution.error),
  ).length;
  const globalIssues = React.useMemo<GlobalIssue[]>(() => {
    const issues: GlobalIssue[] = [];
    if (registryError) {
      issues.push({
        id: "registry",
        title: "Registry",
        message: registryError instanceof Error
          ? registryError.message
          : "The live node registry is unavailable.",
      });
    }
    if (persistenceError) {
      issues.push({
        id: "graph",
        title: "Graph",
        message: persistenceError,
      });
    }
    if (runError) {
      issues.push({
        id: "run",
        title: "Run",
        message: runError,
      });
    }
    return issues;
  }, [persistenceError, registryError, runError]);
  const dismissGlobalIssue = React.useCallback((issue: GlobalIssue) => {
    if (issue.id === "graph") {
      setPersistenceError((current) =>
        current === issue.message ? null : current,
      );
    }
    if (issue.id === "run") {
      setRunError((current) => current === issue.message ? null : current);
    }
  }, []);
  const graphHasErrors = globalIssues.length > 0 || nodeErrorCount > 0;
  const graphNeedsAttention = imageUploadWithoutImages || missingRequiredInputs.length > 0;
  const canvasStatusMessage = runningScope === "selected"
    ? "running selected nodes · latest upstream outputs are pinned"
    : runningScope === "selected-with-dependencies"
      ? "running selected nodes and all upstream dependencies"
      : globalIssues.length
        ? `${globalIssues.length} workflow issue${globalIssues.length === 1 ? "" : "s"}`
        : nodeErrorCount
          ? `${nodeErrorCount} node issue${nodeErrorCount === 1 ? "" : "s"}`
          : !registry
            ? "loading live registry…"
            : imageUploadWithoutImages
              ? "choose images before running"
              : connectionInstruction ?? "all required inputs connected · ready to run";

  const onNodesChange: OnNodesChange<WorkflowNode> = React.useCallback(
    (changes) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );

  const invalidateWorkflowResults = React.useCallback(
    (
      changedTargetNodeIds: readonly string[],
      workflowEdges: readonly WorkflowEdge[],
    ) => {
      if (!changedTargetNodeIds.length) return;
      setNodes((current) =>
        invalidateWorkflowNodeRuns(
          current,
          workflowEdges,
          changedTargetNodeIds,
        ),
      );
      setRunError(null);
    },
    [],
  );

  const onEdgesChange: OnEdgesChange<WorkflowEdge> = React.useCallback(
    (changes) => {
      const changedTargetNodeIds = new Set<string>();
      for (const change of changes) {
        if (change.type === "remove" || change.type === "replace") {
          const previousEdge = edges.find((edge) => edge.id === change.id);
          if (previousEdge) changedTargetNodeIds.add(previousEdge.target);
        }
        if (change.type === "add" || change.type === "replace") {
          changedTargetNodeIds.add(change.item.target);
        }
      }
      setEdges((current) => applyEdgeChanges(changes, current));
      invalidateWorkflowResults([...changedTargetNodeIds], edges);
    },
    [edges, invalidateWorkflowResults],
  );

  const updateEdge = React.useCallback(
    (edgeId: string, update: WorkflowEdgeUpdate) => {
      const changedEdge = edges.find((edge) => edge.id === edgeId);
      if (!changedEdge) return;
      setEdges((current) =>
        current.map((edge) => {
          if (edge.id !== edgeId) return edge;
          const projection = update.route
            ? update.route.projection
            : edge.data?.projection;
          const conversionPath = update.route
            ? update.route.conversionPath.map((conversion) => ({
                id: conversion.id,
                version: conversion.version,
              }))
            : (edge.data?.conversionPath ?? []);
          return {
            ...edge,
            data: {
              ...edge.data,
              collectionMode:
                update.collectionMode ??
                edge.data?.collectionMode ??
                "direct",
              projection,
              conversionPath,
            },
          };
        }),
      );
      invalidateWorkflowResults([changedEdge.target], edges);
    },
    [edges, invalidateWorkflowResults],
  );

  const updateEdgeRoute = React.useCallback(
    (edgeId: string, routeOffset: WorkflowEdgeRouteOffset) => {
      setEdges((current) =>
        current.map((edge) =>
          edge.id === edgeId
            ? {
                ...edge,
                data: {
                  ...edge.data,
                  collectionMode: edge.data?.collectionMode ?? "direct",
                  routeOffset,
                },
              }
            : edge,
        ),
      );
    },
    [],
  );

  const addWorkflowEdge = React.useCallback((
    connection: Connection,
    collectionMode: RunEdgeCollectionMode,
    route: ConnectionRoute,
  ) => {
    let committedConnection = connection;
    let newlyBoundNodeId: string | null = null;
    const binding = route.artifactTypeBinding;
    if (binding) {
      const handleId = binding.endpoint === "source"
        ? connection.sourceHandle
        : connection.targetHandle;
      const handle = decodeHandleId(handleId);
      const nodeId = binding.endpoint === "source"
        ? connection.source
        : connection.target;
      const node = nodes.find((candidate) => candidate.id === nodeId);
      const existingBinding = node?.data.artifactTypeBindings[binding.variable];
      if (
        !handle ||
        handle.artifactTypeVariable !== binding.variable ||
        !node ||
        (existingBinding &&
          (existingBinding.id !== binding.artifactType.id ||
            existingBinding.schema_version !==
              binding.artifactType.schema_version))
      ) {
        return;
      }

      const concreteHandleId = encodeHandleId({
        portName: handle.portName,
        artifactTypeId: binding.artifactType.id,
        schemaVersion: binding.artifactType.schema_version,
        shape: handle.shape,
        direction: handle.direction,
        ...(handle.plugId ? { plugId: handle.plugId } : {}),
      });
      committedConnection = binding.endpoint === "source"
        ? { ...connection, sourceHandle: concreteHandleId }
        : { ...connection, targetHandle: concreteHandleId };
      if (!existingBinding) newlyBoundNodeId = nodeId;
    }

    const source = decodeHandleId(committedConnection.sourceHandle);
    const sourceArtifactType = source
      ? decodedHandleArtifactType(source)
      : null;
    const color = sourceArtifactType
      ? ARTIFACT_TYPE_COLOR[sourceArtifactType.id] ?? tokens.colorAccent
      : tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 2,
    };
    const selection = connectionRouteSelection(route);
    const edge: WorkflowEdge = {
      ...committedConnection,
      id: `edge-${crypto.randomUUID()}`,
      type: WORKFLOW_EDGE_TYPE,
      animated: false,
      data: {
        collectionMode,
        projection: selection.projection
          ? { path: [...selection.projection.path] }
          : undefined,
        conversionPath: selection.conversionPath.map((conversion) => ({
          id: conversion.id,
          version: conversion.version,
        })),
      },
      style: edgeStyle,
    };
    if (binding && newlyBoundNodeId) {
      const bindingNodeId = newlyBoundNodeId;
      // Binding replaces the generic handle ID. Keep the concrete edge pending
      // until WorkflowNode confirms React Flow has measured the replacement.
      pendingBoundEdgesRef.current = [
        ...pendingBoundEdgesRef.current,
        {
          nodeId: bindingNodeId,
          variable: binding.variable,
          artifactType: binding.artifactType,
          edge,
        },
      ];
      setNodes((current) =>
        current.map((candidate) =>
          candidate.id === bindingNodeId
            ? {
                ...candidate,
                data: bindArtifactTypeVariable(
                  candidate.data,
                  binding.variable,
                  binding.artifactType,
                ),
              }
            : candidate,
        ),
      );
    } else {
      setEdges((current) => addEdge(edge, current));
    }
    invalidateWorkflowResults([edge.target], [...edges, edge]);
  }, [edges, invalidateWorkflowResults, nodes]);

  const isValidConnection = React.useCallback<
    IsValidConnection<WorkflowEdge>
  >((connection) => {
    const candidate: Connection = {
      source: connection.source,
      sourceHandle: connection.sourceHandle ?? null,
      target: connection.target,
      targetHandle: connection.targetHandle ?? null,
    };
    const collectionMode = collectionModeForConnection(
      candidate,
      nodes,
      edges,
    );
    if (
      !collectionMode ||
      !connectionRoutesFor(
        candidate,
        registry?.artifact_types ?? [],
        registry?.artifact_conversions ?? [],
      ).length
    ) return false;

    const target = decodeHandleId(candidate.targetHandle);
    const targetNode = nodes.find((node) => node.id === candidate.target);
    const input = targetNode?.data.spec.inputs.find(
      (port) => port.name === target?.portName,
    );
    if (!target || !input) return false;
    if (portHasInstancePlugs(input)) {
      if (
        !target.plugId ||
        !targetNode?.data.inputPlugs.some(
          (plug) =>
            plug.id === target.plugId && plug.portName === input.name,
        )
      ) {
        return false;
      }
    } else if (target.plugId) {
      return false;
    }

    if (
      collectionMode === "map" &&
      edges.some(
        (edge) =>
          edge.target === connection.target &&
          edge.data?.collectionMode === "map",
      )
    ) {
      return false;
    }

    const connectionEdgeId = "id" in connection ? connection.id : null;
    const pendingNodeIds = [candidate.target];
    const visitedNodeIds = new Set<string>();
    while (pendingNodeIds.length) {
      const nodeId = pendingNodeIds.pop();
      if (!nodeId || visitedNodeIds.has(nodeId)) continue;
      if (nodeId === candidate.source) return false;
      visitedNodeIds.add(nodeId);
      for (const edge of edges) {
        if (edge.id !== connectionEdgeId && edge.source === nodeId) {
          pendingNodeIds.push(edge.target);
        }
      }
    }

    if (portHasInstancePlugs(input)) {
      return !edges.some(
        (edge) =>
          edge.id !== connectionEdgeId &&
          edge.target === candidate.target &&
          decodeHandleId(edge.targetHandle)?.plugId === target.plugId,
      );
    }
    if (input.variadic) return true;
    return !edges.some((edge) =>
      edge.id !== connectionEdgeId &&
      edge.target === candidate.target &&
      decodeHandleId(edge.targetHandle)?.portName === target.portName,
    );
  }, [
    edges,
    nodes,
    registry?.artifact_conversions,
    registry?.artifact_types,
  ]);

  const onConnect: OnConnect = React.useCallback((connection) => {
    if (!isValidConnection(connection)) return;
    const collectionMode = collectionModeForConnection(
      connection,
      nodes,
      edges,
    );
    if (!collectionMode) return;

    const candidates = connectionRoutesFor(
      connection,
      registry?.artifact_types ?? [],
      registry?.artifact_conversions ?? [],
    );
    const candidate = candidates[0];
    if (!candidate) return;
    if (candidates.length === 1) {
      addWorkflowEdge(connection, collectionMode, candidate);
      return;
    }

    const source = decodeHandleId(connection.sourceHandle);
    const sourceNode = nodes.find((node) => node.id === connection.source);

    const target = decodeHandleId(connection.targetHandle);
    const targetNode = nodes.find((node) => node.id === connection.target);
    if (!source || !target || !sourceNode || !targetNode) {
      return;
    }
    const sourceArtifactType = decodedHandleArtifactType(source);
    const targetArtifactType = decodedHandleArtifactType(target);

    setPendingConnectionRoute({
      connection,
      collectionMode,
      candidates,
      source: {
        nodeTitle: sourceNode.data.spec.title,
        portName: source.portName,
        artifactType: sourceArtifactType
          ? `${sourceArtifactType.id}@${sourceArtifactType.schema_version}`
          : `Any artifact · ${source.artifactTypeVariable}`,
      },
      target: {
        nodeTitle: targetNode.data.spec.title,
        portName: target.portName,
        artifactType: targetArtifactType
          ? `${targetArtifactType.id}@${targetArtifactType.schema_version}`
          : `Any artifact · ${target.artifactTypeVariable}`,
      },
    });
  }, [
    addWorkflowEdge,
    edges,
    isValidConnection,
    nodes,
    registry?.artifact_conversions,
    registry?.artifact_types,
  ]);

  const addCatalogNode = React.useCallback((spec: NodeSpec) => {
    const id = `node-${crypto.randomUUID()}`;
    const center = flow?.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }) ?? { x: 600, y: 280 };
    const data = attachNodeCallbacks(createWorkflowNodeData(spec));
    setNodes((current) => [
      ...current.map((node) => ({ ...node, selected: false })),
      { id, type: WORKFLOW_NODE_TYPE, position: { x: center.x - 140, y: center.y - 110 }, selected: true, data },
    ]);
    setLibraryOpen(false);
  }, [attachNodeCallbacks, flow]);

  const duplicateSelectedNodes = React.useCallback(() => {
    const selectedNodes = nodes.filter((node) => node.selected);
    if (!selectedNodes.length || running) return;

    const duplicates = selectedNodes.map((node) => ({
      node,
      id: `node-${crypto.randomUUID()}`,
    }));
    const duplicatedNodeIds = new Map(
      duplicates.map(({ node, id }) => [node.id, id]),
    );
    const duplicatedNodes: WorkflowNode[] = duplicates.map(({ node, id }) => ({
      ...node,
      id,
      position: { x: node.position.x + 36, y: node.position.y + 36 },
      selected: true,
      dragging: false,
      data: {
        ...node.data,
        inputPlugs: node.data.inputPlugs.map((plug) => ({ ...plug })),
        inputPlugBindings: {},
        artifactTypeBindings: structuredClone(
          node.data.artifactTypeBindings,
        ),
        mappedInputPort: null,
        config: structuredClone(node.data.config),
        run: null,
        execution: { status: "idle" },
      },
    }));
    const duplicatedEdges: WorkflowEdge[] = edges.flatMap((edge) => {
      const source = duplicatedNodeIds.get(edge.source);
      const target = duplicatedNodeIds.get(edge.target);
      if (!source || !target) return [];
      return [{
        ...edge,
        id: `edge-${crypto.randomUUID()}`,
        source,
        target,
        selected: false,
        data: edge.data ? structuredClone(edge.data) : undefined,
      }];
    });

    setNodes([
      ...nodes.map((node) => ({ ...node, selected: false })),
      ...duplicatedNodes,
    ]);
    setEdges([
      ...edges.map((edge) => ({ ...edge, selected: false })),
      ...duplicatedEdges,
    ]);
    setPendingConnectionRoute(null);
    setRunError(null);
  }, [edges, nodes, running]);

  const deleteSelectedNodes = React.useCallback(() => {
    if (!flow || !selectedNodeIds.length || running) return;
    setPendingConnectionRoute(null);
    setRunError(null);
    void flow.deleteElements({
      nodes: selectedNodeIds.map((id) => ({ id })),
    });
  }, [flow, running, selectedNodeIds]);

  const confirmDiscard = React.useCallback(
    (action: string): boolean =>
      !isDirty ||
      window.confirm(
        `“${graphName.trim() || NEW_GRAPH_NAME}” has unsaved changes. Discard them and ${action}?`,
      ),
    [graphName, isDirty],
  );

  const showBlankGraph = React.useCallback(() => {
    openRequestRef.current?.abort();
    setNodes([]);
    setEdges([]);
    setNodeSecretStatuses({});
    setGraphName(NEW_GRAPH_NAME);
    activeGraphRef.current = null;
    setActiveGraph(null);
    setSavedFingerprint(null);
    setPendingConnectionRoute(null);
    setRunError(null);
    setPersistenceError(null);
    setLibraryOpen(false);
    setFitRevision((current) => current + 1);
  }, []);

  const requestNewGraph = React.useCallback(() => {
    if (!confirmDiscard("start a new graph")) return;
    setGraphBrowserOpen(false);
    const path = workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID);
    if (window.location.pathname === path) {
      showBlankGraph();
      return;
    }
    approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
    router.push(path, { scroll: false });
  }, [confirmDiscard, router, showBlankGraph, workspaceSlug]);

  const saveCurrentGraph = React.useCallback(async () => {
    if (running || saving || openingGraphId || deletingGraphId) return;
    if (!currentDraft.name) {
      setPersistenceError("Enter a graph name before saving.");
      return;
    }

    const submittedDraft = currentDraft;
    setSaving(true);
    setPersistenceError(null);
    try {
      const savedGraph = activeGraph
        ? await updateSavedGraph(activeGraph.id, {
            ...submittedDraft,
            expected_revision: activeGraph.revision,
          })
        : await createSavedGraph(submittedDraft);
      const responseDraft = {
        name: savedGraph.name,
        nodes: savedGraph.nodes ?? [],
        edges: savedGraph.edges ?? [],
      };
      const nextActiveGraph = {
        id: savedGraph.id,
        revision: savedGraph.revision,
        nodes: savedGraph.nodes ?? [],
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      setGraphName((current) =>
        current.trim() === submittedDraft.name ? savedGraph.name : current,
      );
      await refreshNodeSecretStatuses(nextActiveGraph, nodes);
      if (!activeGraph) {
        approvedRouteGraphIdRef.current = savedGraph.id;
        router.replace(
          workbenchGraphPath(workspaceSlug, savedGraph.id),
          { scroll: false },
        );
      }
      void refreshSavedGraphs();
      void refreshNodeRegistry();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setPersistenceError(
          "This graph changed in another session. Your canvas is unchanged; refresh the list before deciding whether to reopen it.",
        );
      } else {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be saved.",
        );
      }
    } finally {
      setSaving(false);
    }
  }, [
    activeGraph,
    currentDraft,
    deletingGraphId,
    openingGraphId,
    refreshSavedGraphs,
    refreshNodeRegistry,
    refreshNodeSecretStatuses,
    router,
    running,
    saving,
    nodes,
    workspaceSlug,
  ]);

  const openSavedGraph = React.useCallback(async (
    graphId: string,
    confirmBeforeOpen = true,
    updateAddress = true,
  ) => {
    if (!registry) {
      setPersistenceError("The live node registry must load before a graph can open.");
      return;
    }
    if (confirmBeforeOpen && !confirmDiscard("open another graph")) return;
    if (updateAddress) {
      setGraphBrowserOpen(false);
      if (activeGraph?.id !== graphId) {
        approvedRouteGraphIdRef.current = graphId;
        router.push(
          workbenchGraphPath(workspaceSlug, graphId),
          { scroll: false },
        );
      }
      return;
    }
    const openingFingerprint = currentFingerprint;

    openRequestRef.current?.abort();
    const controller = new AbortController();
    openRequestRef.current = controller;
    setOpeningGraphId(graphId);
    setPersistenceError(null);
    try {
      const savedGraph = await getSavedGraph(graphId, controller.signal);
      let materializationWarning: string | null = null;
      let materializedNodeRuns: RunNodeResult[] = [];
      try {
        const materializations = await getGraphMaterializations(
          graphId,
          savedGraph.revision,
          controller.signal,
        );
        materializedNodeRuns = [...materializations.node_runs];
      } catch (error) {
        if (controller.signal.aborted) return;
        const message = error instanceof Error
          ? error.message
          : "Latest materialized outputs could not be loaded.";
        materializationWarning = `Graph opened without its latest materialized outputs: ${message}`;
      }
      const hydrated = hydrateSavedGraph(
        savedGraph,
        registry,
        materializedNodeRuns,
      );
      if (controller.signal.aborted) return;
      if (currentFingerprintRef.current !== openingFingerprint) {
        setPersistenceError(
          "The canvas changed while the graph was loading. Your newer edits were kept; open the graph again when you are ready to replace them.",
        );
        return;
      }

      const responseDraft = {
        name: savedGraph.name,
        nodes: savedGraph.nodes ?? [],
        edges: savedGraph.edges ?? [],
      };
      const openedNodes = hydrated.nodes.map((node) => ({
          ...node,
          data: attachNodeCallbacks(node.data),
        }));
      setNodes(openedNodes);
      setEdges(hydrated.edges);
      setGraphName(savedGraph.name);
      const nextActiveGraph = {
        id: savedGraph.id,
        revision: savedGraph.revision,
        nodes: savedGraph.nodes ?? [],
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      await refreshNodeSecretStatuses(
        nextActiveGraph,
        openedNodes,
        controller.signal,
      );
      setPendingConnectionRoute(null);
      setRunError(null);
      setPersistenceError(materializationWarning);
      setLibraryOpen(false);
      setGraphBrowserOpen(false);
      setFitRevision((current) => current + 1);
    } catch (error) {
      if (!controller.signal.aborted) {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be opened.",
        );
      }
    } finally {
      if (openRequestRef.current === controller) {
        openRequestRef.current = null;
        setOpeningGraphId(null);
      }
    }
  }, [
    activeGraph?.id,
    attachNodeCallbacks,
    confirmDiscard,
    currentFingerprint,
    registry,
    refreshNodeSecretStatuses,
    router,
    workspaceSlug,
  ]);

  React.useEffect(() => {
    if (!registry) {
      return;
    }

    const routeGraphId = initialGraphId ?? NEW_GRAPH_ROUTE_ID;
    const displayedGraphId = activeGraph?.id ?? NEW_GRAPH_ROUTE_ID;
    if (routeGraphId === displayedGraphId) {
      if (approvedRouteGraphIdRef.current === routeGraphId) {
        approvedRouteGraphIdRef.current = null;
      }
      return;
    }

    if (
      approvedRouteGraphIdRef.current !== null &&
      approvedRouteGraphIdRef.current !== routeGraphId
    ) {
      return;
    }

    const explicitlyApproved =
      approvedRouteGraphIdRef.current === routeGraphId;
    approvedRouteGraphIdRef.current = null;
    if (!explicitlyApproved && !confirmDiscard("navigate with browser history")) {
      approvedRouteGraphIdRef.current = displayedGraphId;
      router.push(
        workbenchGraphPath(workspaceSlug, displayedGraphId),
        { scroll: false },
      );
      return;
    }

    if (!initialGraphId) {
      // The App Router retains this workbench so history navigation can be confirmed first.
      // Once accepted, the route change is the boundary that replaces the canvas draft.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      showBlankGraph();
      return;
    }

    void openSavedGraph(initialGraphId, false, false);
  }, [
    activeGraph?.id,
    confirmDiscard,
    initialGraphId,
    openSavedGraph,
    registry,
    router,
    showBlankGraph,
    workspaceSlug,
  ]);

  const removeSavedGraph = React.useCallback(async (
    graph: SavedGraphSummary,
  ) => {
    const deletingActiveGraph = activeGraph?.id === graph.id;
    const warning = deletingActiveGraph && isDirty
      ? `Delete “${graph.name}”? Its unsaved canvas changes will also be discarded.`
      : `Delete “${graph.name}”? This cannot be undone.`;
    if (!window.confirm(warning)) return;

    const expectedRevision = deletingActiveGraph
      ? activeGraph.revision
      : graph.revision;
    const deletingFingerprint = currentFingerprint;
    setDeletingGraphId(graph.id);
    setPersistenceError(null);
    try {
      await deleteSavedGraph(graph.id, expectedRevision);
      if (deletingActiveGraph) {
        if (currentFingerprintRef.current === deletingFingerprint) {
          showBlankGraph();
          approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
          router.replace(
            workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID),
            { scroll: false },
          );
        } else {
          activeGraphRef.current = null;
          setActiveGraph(null);
          setSavedFingerprint(null);
          setPersistenceError(
            "The saved graph was deleted. Changes made while deletion was in progress remain as an unsaved draft.",
          );
          approvedRouteGraphIdRef.current = NEW_GRAPH_ROUTE_ID;
          router.replace(
            workbenchGraphPath(workspaceSlug, NEW_GRAPH_ROUTE_ID),
            { scroll: false },
          );
        }
      }
      void refreshSavedGraphs();
      void refreshNodeRegistry();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setPersistenceError(
          "This graph changed before it could be deleted. Refresh the saved graph list and try again.",
        );
      } else {
        setPersistenceError(
          error instanceof Error ? error.message : "The graph could not be deleted.",
        );
      }
    } finally {
      setDeletingGraphId(null);
    }
  }, [
    activeGraph,
    currentFingerprint,
    isDirty,
    refreshSavedGraphs,
    refreshNodeRegistry,
    router,
    showBlankGraph,
    workspaceSlug,
  ]);

  const toggleGraphBrowser = React.useCallback(() => {
    setLibraryOpen(false);
    setGraphBrowserOpen((open) => !open);
  }, []);

  const canvasNodes = React.useMemo(
    () =>
      nodes.map((node) => {
        const savedNode = activeGraph?.nodes.find(
          (candidate) => candidate.id === node.id,
        );
        return {
          ...node,
          data: {
            ...node.data,
            secretStatuses: nodeSecretStatuses[node.id] ?? {},
            secretInputReadiness: Object.fromEntries(
              nodeSecretInputs(node.data.spec).map((input) => [
                input.name,
                nodeSecretBindingReady(input, {
                  id: node.id,
                  operator_id: node.data.spec.operator_id,
                  operator_version: node.data.spec.operator_version,
                  config: node.data.config,
                }, savedNode),
              ]),
            ),
            secretInputScope: `${activeGraph?.id ?? "unsaved"}:${activeGraph?.revision ?? "none"}`,
            onApplyNodeSecret: applyConfiguredNodeSecret,
            onRemoveNodeSecret: removeConfiguredNodeSecret,
            mappedInputPort: mappedInputPortForNode(node.id, edges),
            inputPlugBindings: inputPlugBindingsForNode(
              node,
              nodes,
              edges,
              registry,
            ),
          },
        };
      }),
    [
      activeGraph,
      applyConfiguredNodeSecret,
      edges,
      nodeSecretStatuses,
      nodes,
      registry,
      removeConfiguredNodeSecret,
    ],
  );

  const canvasEdges = React.useMemo(
    () =>
      edges.map((edge) => {
        const connection: Connection = {
          source: edge.source,
          sourceHandle: edge.sourceHandle ?? null,
          target: edge.target,
          targetHandle: edge.targetHandle ?? null,
        };
        const source = decodeHandleId(edge.sourceHandle);
        const activeSelection = {
          projection: edge.data?.projection,
          conversionPath: edge.data?.conversionPath ?? [],
        };
        const routes = connectionRoutesFor(
          connection,
          registry?.artifact_types ?? [],
          registry?.artifact_conversions ?? [],
        );
        const activeRoute = connectionRouteForSelection(
          connection,
          registry?.artifact_types ?? [],
          registry?.artifact_conversions ?? [],
          activeSelection,
        );
        if (
          activeRoute &&
          !routes.some((route) =>
            connectionRouteMatchesSelection(route, activeSelection),
          )
        ) {
          routes.push(activeRoute);
        }
        const routeOptions = routes.map(workflowEdgeRouteOption);
        const conversionTitles = activeSelection.conversionPath.map(
          (requestedConversion) =>
            registry?.artifact_conversions.find(
              (conversion) =>
                conversion.key.id === requestedConversion.id &&
                conversion.key.version === requestedConversion.version,
            )?.title ?? `${requestedConversion.id}@${requestedConversion.version}`,
        );
        const otherEdges = edges.filter((candidate) => candidate.id !== edge.id);
        const validMode = collectionModeForConnection(
          connection,
          nodes,
          otherEdges,
        );
        return {
          ...edge,
          type: WORKFLOW_EDGE_TYPE,
          data: {
            ...edge.data,
            collectionMode: edge.data?.collectionMode ?? "direct",
            sourcePortName: source?.portName,
            conversionTitles,
            routeOptions,
            allowedCollectionModes: validMode ? [validMode] : [],
            onUpdate: updateEdge,
            onRouteOffsetChange: updateEdgeRoute,
          },
        };
      }),
    [
      edges,
      nodes,
      registry?.artifact_conversions,
      registry?.artifact_types,
      updateEdge,
      updateEdgeRoute,
    ],
  );

  const runWorkflow = async (scope: RunScope) => {
    if (!registry || running) return;
    const planningFingerprint = currentFingerprint;
    const planningActiveGraph = activeGraph;
    let planningNodes = nodes;

    if (scope === "selected" && activeGraph && !isDirty) {
      try {
        const materializations = await getGraphMaterializations(
          activeGraph.id,
          activeGraph.revision,
        );
        const currentActiveGraph = activeGraphRef.current;
        if (
          currentFingerprintRef.current !== planningFingerprint ||
          currentActiveGraph?.id !== planningActiveGraph?.id ||
          currentActiveGraph?.revision !== planningActiveGraph?.revision
        ) {
          setRunError(
            "The active graph changed while latest materialized outputs were loading. Run the selection again.",
          );
          return;
        }
        planningNodes = withMaterializedNodeRuns(
          planningNodes,
          materializations.node_runs,
        );
        setNodes((current) =>
          withMaterializedNodeRuns(current, materializations.node_runs),
        );
        setPersistenceError(null);
      } catch (error) {
        const message = error instanceof Error
          ? error.message
          : "Latest materialized outputs could not be loaded.";
        setRunError(
          `Cannot verify the latest upstream outputs for this saved graph: ${message}`,
        );
        return;
      }
    }

    const executionNodeIds = scope === "all"
      ? new Set(planningNodes.map((node) => node.id))
      : scope === "selected-with-dependencies"
        ? selectedNodeAndAncestorIds(planningNodes, edges)
        : new Set(
            planningNodes
              .filter((node) => node.selected)
              .map((node) => node.id),
          );
    const executionNodes = planningNodes.filter((node) =>
      executionNodeIds.has(node.id),
    );
    const executionEdges = scope === "all"
      ? edges
      : scope === "selected-with-dependencies"
        ? edges.filter(
            (edge) =>
              executionNodeIds.has(edge.source) &&
              executionNodeIds.has(edge.target),
          )
        : edges.filter((edge) => executionNodeIds.has(edge.target));
    const secretBackedNodes = executionNodes
      .map((node) => ({ node, inputs: nodeSecretInputs(node.data.spec) }))
      .filter(({ inputs }) => inputs.length > 0);
    if (secretBackedNodes.length && !activeGraph) {
      setRunError(
        "Save the graph before running nodes that use stored secrets.",
      );
      return;
    }
    let changedSecretBinding: {
      node: WorkflowNode;
      input: WorkflowNodeSecretInput;
    } | undefined;
    if (activeGraph) {
      for (const { node, inputs } of secretBackedNodes) {
        const savedNode = activeGraph.nodes.find(
          (candidate) => candidate.id === node.id,
        );
        const changedInput = inputs.find((input) =>
          !nodeSecretBindingReady(input, {
            id: node.id,
            operator_id: node.data.spec.operator_id,
            operator_version: node.data.spec.operator_version,
            config: node.data.config,
          }, savedNode)
        );
        if (changedInput) {
          changedSecretBinding = { node, input: changedInput };
          break;
        }
      }
    }
    if (changedSecretBinding) {
      setRunError(
        `Save the graph before running ${changedSecretBinding.node.data.spec.title}: its ${changedSecretBinding.input.title} binding is new or changed.`,
      );
      return;
    }
    const unavailableSecret = secretBackedNodes.find(({ node, inputs }) =>
      inputs.some(
        (input) =>
          nodeSecretStatuses[node.id]?.[input.name]?.state !== "configured",
      ),
    );
    if (unavailableSecret) {
      setRunError(
        `Configure every required secret for ${unavailableSecret.node.data.spec.title} before running.`,
      );
      return;
    }
    const validationIssue = executionValidationIssue(
      scope,
      executionNodes,
      executionEdges,
    );
    if (validationIssue) {
      if (validationIssue.nodeId) {
        setRunError(null);
        setNodes((current) => current.map((node) =>
          node.id === validationIssue.nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  run: null,
                  execution: {
                    status: "failed",
                    error: validationIssue.message,
                  },
                },
              }
            : node,
        ));
      } else {
        setRunError(validationIssue.message);
      }
      return;
    }

    const pinnedOutputs: PinnedOutputInput[] = [];
    if (scope === "selected") {
      const nodesById = new Map(
        planningNodes.map((node) => [node.id, node]),
      );
      const pinnedSourcePorts = new Map<string, Set<string>>();
      const missingPinnedOutputs: string[] = [];

      for (const edge of executionEdges) {
        if (executionNodeIds.has(edge.source)) continue;

        const source = decodeHandleId(edge.sourceHandle);
        const target = decodeHandleId(edge.targetHandle);
        if (!source || !target) {
          setRunError(
            `Cannot run the selection because edge ${edge.id} does not identify both source and target ports.`,
          );
          return;
        }

        const sourcePorts = pinnedSourcePorts.get(edge.source) ?? new Set<string>();
        if (sourcePorts.has(source.portName)) continue;
        sourcePorts.add(source.portName);
        pinnedSourcePorts.set(edge.source, sourcePorts);

        const sourceNode = nodesById.get(edge.source);
        const output = sourceNode?.data.run?.status === "succeeded"
          ? sourceNode.data.run.outputs.find(
              (candidate) => candidate.port === source.portName,
            )
          : undefined;
        if (!output) {
          const sourceName = sourceNode?.data.spec.title ?? edge.source;
          missingPinnedOutputs.push(`${sourceName}.${source.portName}`);
          continue;
        }

        pinnedOutputs.push({
          from_node: edge.source,
          from_port: source.portName,
          value: output.value,
        });
      }

      if (missingPinnedOutputs.length) {
        const endpoints = missingPinnedOutputs.join(", ");
        setRunError(
          `Cannot run the selection because no accessible materialized output is available for ${endpoints}. Select the missing upstream nodes too, or choose “Run with dependencies”.`,
        );
        return;
      }
    }

    setRunningScope(scope);
    setRunError(null);
    setNodes((current) =>
      current.map((node) =>
        executionNodeIds.has(node.id)
          ? {
              ...node,
              data: {
                ...node.data,
                run: null,
                execution: { status: "running" },
              },
            }
          : node,
      ),
    );
    try {
      const runEdges = executionEdges.flatMap<RunEdgeInput>(
        (edge) => {
          const source = decodeHandleId(edge.sourceHandle);
          const target = decodeHandleId(edge.targetHandle);
          if (!source || !target) return [];
          const runEdge = {
            from_node: edge.source,
            from_port: source.portName,
            to_node: edge.target,
            to_port: target.portName,
            to_plug: target.plugId ?? null,
            ...serializeWorkflowEdgeTransport(edge.data),
          };
          return [runEdge];
        },
      );
      const runNodes = executionNodes.map((node) =>
        serializeRunNode(node.id, node.data),
      );
      const materializesSavedGraph = Boolean(activeGraph && !isDirty);
      const graphContext = materializesSavedGraph && activeGraph
        ? {
            graph_id: activeGraph.id,
            graph_revision: activeGraph.revision,
          }
        : {};
      const secretGraphContext = secretBackedNodes.length && activeGraph
        ? {
            secret_graph_id: activeGraph.id,
            secret_graph_revision: activeGraph.revision,
          }
        : {};
      const response = await runGraph({
        nodes: runNodes,
        edges: runEdges,
        ...(scope === "selected" ? { pinned_outputs: pinnedOutputs } : {}),
        ...graphContext,
        ...secretGraphContext,
      });
      const currentActiveGraph = activeGraphRef.current;
      if (
        currentFingerprintRef.current !== planningFingerprint ||
        currentActiveGraph?.id !== planningActiveGraph?.id ||
        currentActiveGraph?.revision !== planningActiveGraph?.revision
      ) {
        setNodes((current) => current.map((node) =>
          executionNodeIds.has(node.id) &&
          node.data.execution.status === "running"
            ? {
                ...node,
                data: {
                  ...node.data,
                  execution: { status: "idle" },
                },
              }
            : node,
        ));
        setRunError(
          materializesSavedGraph
            ? "The graph changed while it was running. Results were recorded for the original saved revision and were not applied to this canvas."
            : "The graph changed while it was running. The completed run was not applied to this canvas.",
        );
        return;
      }
      const byNode = new Map(response.node_runs.map((run) => [run.node_id, run]));
      setNodes((current) => current.map((node) => {
        if (!executionNodeIds.has(node.id)) return node;
        const run = byNode.get(node.id);
        return {
          ...node,
          data: {
            ...node.data,
            run: run ?? null,
            execution: run
              ? {
                  status: run.status,
                  error: run.error ?? (run.status === "failed"
                    ? "This node failed without error details."
                    : undefined),
                }
              : { status: "skipped", error: "The server did not return a result for this node." },
          },
        };
      }));
    } catch (runFailure) {
      const currentActiveGraph = activeGraphRef.current;
      setNodes((current) => current.map((node) =>
        executionNodeIds.has(node.id) &&
        node.data.execution.status === "running"
          ? {
              ...node,
              data: {
                ...node.data,
                execution: { status: "idle" },
              },
            }
          : node,
      ));
      if (
        currentFingerprintRef.current !== planningFingerprint ||
        currentActiveGraph?.id !== planningActiveGraph?.id ||
        currentActiveGraph?.revision !== planningActiveGraph?.revision
      ) {
        setRunError(
          "The graph changed while it was running. The completed run was not applied to this canvas.",
        );
        return;
      }
      const missingPinnedArtifact =
        scope === "selected" &&
        runFailure instanceof ApiError &&
        runFailure.detail.includes("references missing artifact");
      const message = missingPinnedArtifact
        ? "A previously materialized upstream artifact is no longer accessible. Run the missing upstream nodes too, or choose “Run with dependencies”."
        : runFailure instanceof Error
          ? runFailure.message
          : "Workflow run failed";
      setRunError(message);
    } finally {
      setRunningScope(null);
    }
  };

  // Firefox uses autocomplete to control restored dynamic button state, but
  // React's button typings omit that browser-specific attribute.
  const firefoxDynamicButtonProps: React.ButtonHTMLAttributes<HTMLButtonElement> & {
    autoComplete: "off";
  } = { autoComplete: "off" };

  return (
    <main {...stylex.props(s.shell)}>
      <section {...stylex.props(s.canvas)} aria-label="Workflow canvas">
        <WorkflowCanvas
          fitViewOptions={WORKBENCH_FIT_VIEW_OPTIONS}
          nodes={canvasNodes}
          edges={canvasEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onPaneReady={setFlow}
          onPaneClick={() => {
            setLibraryOpen(false);
            setGraphBrowserOpen(false);
          }}
          animateEdges={running}
        >
          {selectedNodeIds.length ? (
            <NodeToolbar
              nodeId={selectedNodeIds}
              isVisible
              position={Position.Bottom}
              offset={16}
              className={`ns-node-detail ${stylex.props(s.selectionToolbar).className}`}
            >
              <span {...stylex.props(s.selectionLabel)}>
                {selectedNodeCount} selected
              </span>
              <span {...stylex.props(s.selectionDivider)} />
              <button
                type="button"
                disabled={runSelectedDisabled}
                title="Run only the selected nodes; latest accessible upstream outputs are pinned"
                {...stylex.props(s.toolButton, s.primaryButton)}
                onClick={() => void runWorkflow("selected")}
              >
                {runningScope === "selected" ? (
                  <LoaderCircle size={13} {...stylex.props(s.spinner)} />
                ) : (
                  <Play size={13} />
                )}
                {runningScope === "selected" ? "Running…" : "Run"}
              </button>
              <button
                type="button"
                disabled={runSelectedDisabled}
                title={`Run the selection and every upstream dependency (${selectedWithDependenciesCount} total)`}
                {...stylex.props(s.toolButton)}
                onClick={() => void runWorkflow("selected-with-dependencies")}
              >
                {runningScope === "selected-with-dependencies" ? (
                  <LoaderCircle size={13} {...stylex.props(s.spinner)} />
                ) : (
                  <Workflow size={13} />
                )}
                {runningScope === "selected-with-dependencies"
                  ? "Running…"
                  : "With dependencies"}
              </button>
            </NodeToolbar>
          ) : null}
        </WorkflowCanvas>
      </section>

      <div {...stylex.props(s.topBar)}>
        <div {...stylex.props(s.chrome, s.identity)}>
          <span {...stylex.props(s.identityCopy)}>
            <button
              type="button"
              {...stylex.props(s.identityMenu)}
              onClick={toggleGraphBrowser}
            >
              <span {...stylex.props(s.brand)}>NOTARIUS</span>
              <span {...stylex.props(s.saveState, isDirty ? s.saveStateDirty : null)}>
                {saving
                  ? "saving…"
                  : activeGraph
                    ? isDirty ? "unsaved" : `saved · r${activeGraph.revision}`
                    : "not saved"}
              </span>
              <ChevronDown size={11} />
            </button>
            <input
              aria-label="Graph name"
              value={graphName}
              maxLength={160}
              {...stylex.props(s.workflowName)}
              onChange={(event) => {
                setGraphName(event.currentTarget.value);
                setPersistenceError(null);
              }}
            />
          </span>
          <span {...stylex.props(s.identityDivider)} />
          <span
            aria-label={`${nodes.length} node${nodes.length === 1 ? "" : "s"}, ${edges.length} connection${edges.length === 1 ? "" : "s"}. ${canvasStatusMessage}`}
            title={canvasStatusMessage}
            {...stylex.props(s.identityStats)}
          >
            <span
              aria-hidden="true"
              {...stylex.props(
                s.graphStatusDot,
                graphHasErrors ? s.graphStatusDotError : null,
                !graphHasErrors && running ? s.graphStatusDotRunning : null,
                !graphHasErrors && !running && graphNeedsAttention
                  ? s.graphStatusDotIncomplete
                  : null,
              )}
            />
            <span>
              <span {...stylex.props(s.identityStatValue)}>{nodes.length}</span>{" "}
              <span {...stylex.props(s.identityStatLabel)}>
                node{nodes.length === 1 ? "" : "s"}
              </span>
            </span>
            <span aria-hidden="true" {...stylex.props(s.identityStatSeparator)}>·</span>
            <span>
              <span {...stylex.props(s.identityStatValue)}>{edges.length}</span>{" "}
              <span {...stylex.props(s.identityStatLabel)}>
                connection{edges.length === 1 ? "" : "s"}
              </span>
            </span>
          </span>
          <span {...stylex.props(s.identityDivider)} />
          <span {...stylex.props(s.identityActions)}>
            <button
              type="button"
              aria-label={
                saving
                  ? "Saving graph"
                  : activeGraph && !isDirty
                    ? "Graph saved"
                    : "Save graph"
              }
              disabled={
                saving ||
                running ||
                Boolean(openingGraphId) ||
                Boolean(deletingGraphId) ||
                !graphName.trim() ||
                Boolean(activeGraph && !isDirty)
              }
              title={
                activeGraph && !isDirty
                  ? "All changes are saved"
                  : "Save graph"
              }
              {...stylex.props(
                s.toolButton,
                s.identityAction,
                isDirty ? s.identityActionActive : null,
              )}
              onClick={() => void saveCurrentGraph()}
            >
              {saving ? (
                <LoaderCircle size={13} {...stylex.props(s.spinner)} />
              ) : (
                <Save size={13} />
              )}
            </button>
            <button
              type="button"
              aria-label={
                preference === "light"
                  ? "Switch to dark theme"
                  : preference === "dark"
                    ? "Switch to system theme"
                    : "Switch to light theme"
              }
              title={
                preference === "light"
                  ? "Light theme"
                  : preference === "dark"
                    ? "Dark theme"
                    : "System theme"
              }
              {...stylex.props(s.toolButton, s.identityAction)}
              onClick={cycleTheme}
            >
              {preference === "light" ? (
                <Sun size={13} />
              ) : preference === "dark" ? (
                <Moon size={13} />
              ) : (
                <Monitor size={13} />
              )}
            </button>
          </span>
        </div>
      </div>

      <aside aria-label="Canvas actions" {...stylex.props(s.actionRail)}>
        <button
          type="button"
          {...firefoxDynamicButtonProps}
          aria-label="Add node"
          disabled={!registry || running}
          title="Add node"
          {...stylex.props(s.railButton, s.railPrimary)}
          onClick={() => {
            setGraphBrowserOpen(false);
            setLibraryOpen((open) => !open);
          }}
        >
          <Plus size={14} />
          <span {...stylex.props(s.railLabel)}>Node</span>
        </button>
        <button
          type="button"
          {...firefoxDynamicButtonProps}
          disabled={!flow}
          title="Fit workflow"
          {...stylex.props(s.railButton)}
          onClick={() => void flow?.fitView(WORKBENCH_FIT_VIEW_OPTIONS)}
        >
          <Maximize2 size={14} />
          <span {...stylex.props(s.railLabel)}>Fit</span>
        </button>
        <span {...stylex.props(s.railDivider)} />
        <button
          type="button"
          disabled={!selectedNodeCount || running}
          title={
            selectedNodeCount
              ? `Duplicate ${selectedNodeCount} selected node${selectedNodeCount === 1 ? "" : "s"}`
              : "Select one or more nodes to duplicate"
          }
          {...stylex.props(s.railButton)}
          onClick={duplicateSelectedNodes}
        >
          <Copy size={14} />
          <span {...stylex.props(s.railLabel)}>Duplicate</span>
        </button>
        <button
          type="button"
          disabled={!flow || !selectedNodeCount || running}
          title={
            selectedNodeCount
              ? `Delete ${selectedNodeCount} selected node${selectedNodeCount === 1 ? "" : "s"}`
              : "Select one or more nodes to delete"
          }
          {...stylex.props(s.railButton, s.railDanger)}
          onClick={deleteSelectedNodes}
        >
          <Trash2 size={14} />
          <span {...stylex.props(s.railLabel)}>Delete</span>
        </button>
      </aside>

      <Toast.Provider timeout={8000} limit={3}>
        <GlobalIssueToastList
          issues={globalIssues}
          onDismiss={dismissGlobalIssue}
        />
      </Toast.Provider>

      {graphBrowserOpen ? (
        <SavedGraphBrowser
          graphs={savedGraphList?.graphs ?? []}
          activeGraphId={activeGraph?.id ?? null}
          openingGraphId={openingGraphId}
          deletingGraphId={deletingGraphId}
          busy={graphOperationBusy}
          loading={savedGraphsLoading}
          refreshing={savedGraphsRefreshing}
          error={
            savedGraphListError instanceof Error
              ? savedGraphListError.message
              : savedGraphListError
                ? "Saved graphs are unavailable."
                : null
          }
          onClose={() => setGraphBrowserOpen(false)}
          onNew={requestNewGraph}
          onOpen={(graphId) => void openSavedGraph(graphId)}
          onDelete={(graph) => void removeSavedGraph(graph)}
          onRefresh={() => void refreshSavedGraphs()}
        />
      ) : null}

      {registry ? (
        <NodeSelector
          open={libraryOpen}
          registry={registry}
          activeGraphId={activeGraph?.id ?? null}
          onOpenChange={setLibraryOpen}
          onAddNode={addCatalogNode}
        />
      ) : null}

      <Dialog
        open={pendingConnectionRoute !== null}
        onOpenChange={(open) => {
          if (!open) setPendingConnectionRoute(null);
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
            {pendingConnectionRoute ? (
              <>
                <div {...stylex.props(s.projectionFlow)}>
                  <div {...stylex.props(s.projectionEndpoint)}>
                    <span {...stylex.props(s.projectionDirection)}>Source</span>
                    <span {...stylex.props(s.projectionEndpointName)}>
                      {pendingConnectionRoute.source.nodeTitle} · {pendingConnectionRoute.source.portName}
                    </span>
                    <span {...stylex.props(s.projectionEndpointType)}>
                      {pendingConnectionRoute.source.artifactType}
                    </span>
                  </div>
                  <span aria-hidden="true" {...stylex.props(s.projectionArrow)}>→</span>
                  <div {...stylex.props(s.projectionEndpoint)}>
                    <span {...stylex.props(s.projectionDirection)}>Target</span>
                    <span {...stylex.props(s.projectionEndpointName)}>
                      {pendingConnectionRoute.target.nodeTitle} · {pendingConnectionRoute.target.portName}
                    </span>
                    <span {...stylex.props(s.projectionEndpointType)}>
                      {pendingConnectionRoute.target.artifactType}
                    </span>
                  </div>
                </div>
                <p {...stylex.props(s.projectionPrompt)}>
                  Choose how this edge carries the value:
                </p>
                <div {...stylex.props(s.projectionChoices)}>
                  {pendingConnectionRoute.candidates.map((candidate, index) => (
                    <button
                      key={`${candidate.kind}-${connectionRouteDescription(pendingConnectionRoute.source.portName, candidate)}`}
                      type="button"
                      autoFocus={index === 0}
                      aria-label={`Use ${connectionRouteTitle(candidate)} from ${pendingConnectionRoute.source.nodeTitle}`}
                      {...stylex.props(s.projectionChoice)}
                      onClick={() => {
                        addWorkflowEdge(
                          pendingConnectionRoute.connection,
                          pendingConnectionRoute.collectionMode,
                          candidate,
                        );
                        setPendingConnectionRoute(null);
                      }}
                    >
                      <span {...stylex.props(s.projectionChoiceTitle)}>
                        {connectionRouteTitle(candidate)}
                      </span>
                      <span {...stylex.props(s.projectionChoicePath)}>
                        {connectionRouteDescription(
                          pendingConnectionRoute.source.portName,
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
                    onClick={() => setPendingConnectionRoute(null)}
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : null}
          </DialogBody>
        </DialogContent>
      </Dialog>
    </main>
  );
}

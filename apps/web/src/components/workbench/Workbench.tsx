"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { useRouter } from "next/navigation";
import {
  type Connection,
  type IsValidConnection,
  type Node,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Cable,
  ChevronDown,
  LoaderCircle,
  Maximize2,
  Monitor,
  MousePointer2,
  Moon,
  Play,
  Plus,
  Save,
  Search,
  Sun,
  Workflow,
  X,
} from "lucide-react";

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
  connectionArtifactContractIsValid,
  connectionRouteSelection,
  connectionRoutesFor,
  decodeHandleId,
  encodeHandleId,
  projectionCandidatesForConnection,
  type ConnectionRoute,
} from "@/components/canvas/handles";
import {
  hydrateSavedGraph,
  savedGraphDraft,
  savedGraphFingerprint,
  withMaterializedNodeRuns,
} from "@/components/canvas/saved-graph";
import {
  ARTIFACT_TYPE_COLOR,
} from "@/components/canvas/nodes.css";
import { useTheme } from "@/components/theme";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  WORKFLOW_EDGE_TYPE,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  effectivePortShape,
  portMetaForPort,
  removeSelectionItem,
  replaceSelection,
  selectedSourceItems,
  serializeNodeConfig,
  type WorkflowEdge,
  type WorkflowEdgeRouteOption,
  type WorkflowEdgeRouteOffset,
  type WorkflowEdgeUpdate,
  type WorkflowNodeData,
} from "@/components/canvas/types";
import { useNodeRegistry, useSavedGraphs } from "@/hooks/use-api";
import {
  createSavedGraph,
  deleteSavedGraph,
  fileToBase64,
  getGraphMaterializations,
  getSavedGraph,
  runGraph,
  updateSavedGraph,
  uploadFile,
  type ArtifactTypeSpec,
  type NodeSpec,
  type PinnedOutputInput,
  type Port,
  type RunEdgeCollectionMode,
  type RunEdgeInput,
  type RunNodeResult,
  type SavedGraphSummary,
} from "@/lib/api";
import { ApiError } from "@/lib/api/client";
import { tokens } from "@/lib/stylex/tokens.stylex";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;
type RunScope = "all" | "selected" | "selected-with-dependencies";

interface ActiveSavedGraph {
  id: string;
  revision: number;
}

interface WorkbenchProps {
  workspaceSlug: string;
  initialGraphId: string | null;
  seedExample: boolean;
}

const INITIAL_GRAPH_NAME = "Arithmetic field projection";
const NEW_GRAPH_NAME = "Untitled workflow";

const ARITHMETIC_OPERATORS = [
  "arithmetic.number",
  "arithmetic.add_subtract",
  "arithmetic.multiply",
] as const;

const ARITHMETIC_NODE_LAYOUT = [
  {
    id: "node-number-nine",
    operatorId: "arithmetic.number",
    position: { x: 40, y: 150 },
    config: { value: 9 },
  },
  {
    id: "node-number-four",
    operatorId: "arithmetic.number",
    position: { x: 40, y: 470 },
    config: { value: 4 },
  },
  {
    id: "node-add-subtract",
    operatorId: "arithmetic.add_subtract",
    position: { x: 470, y: 310 },
    config: {},
  },
  {
    id: "node-multiply",
    operatorId: "arithmetic.multiply",
    position: { x: 900, y: 310 },
    config: {},
  },
] as const;

const ARITHMETIC_LINKS = [
  {
    sourceId: "node-number-nine",
    sourcePort: "value",
    targetId: "node-add-subtract",
    targetPort: "left",
  },
  {
    sourceId: "node-number-four",
    sourcePort: "value",
    targetId: "node-add-subtract",
    targetPort: "right",
  },
  {
    sourceId: "node-add-subtract",
    sourcePort: "result",
    targetId: "node-multiply",
    targetPort: "left",
    projectionPath: ["addition"],
  },
  {
    sourceId: "node-add-subtract",
    sourcePort: "result",
    targetId: "node-multiply",
    targetPort: "right",
    projectionPath: ["subtraction"],
  },
] as const;

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

function connectionRouteTitle(route: ConnectionRoute): string {
  if (route.kind === "projection") return route.projection.title;
  if (route.kind === "conversion") return route.conversion.title;
  if (route.kind === "projection-conversion") {
    return `${route.projection.title} → ${route.conversion.title}`;
  }
  return "Whole output";
}

function connectionRouteDescription(
  sourcePortName: string,
  route: ConnectionRoute,
): string {
  if (route.kind === "projection") {
    return `${sourcePortName}.${route.projection.path.join(".")}`;
  }
  if (route.kind === "conversion") {
    return `${sourcePortName} → ${route.conversion.title}`;
  }
  if (route.kind === "projection-conversion") {
    return `${sourcePortName}.${route.projection.path.join(".")} → ${route.conversion.title}`;
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
    conversionTitle:
      route.kind === "conversion" || route.kind === "projection-conversion"
        ? route.conversion.title
        : undefined,
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
  const targetShape = effectiveShapeForPort(targetNode, targetPort, edges);
  if (sourceShape === targetShape) return "direct";
  if (sourceShape === "many" && targetShape === "one") return "map";
  return null;
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
  nodeTitle: string;
  portName: string;
}

function missingRequiredInputsFor(
  nodes: readonly WorkflowNode[],
  edges: readonly WorkflowEdge[],
): MissingRequiredInput[] {
  return nodes.flatMap((node) =>
    node.data.spec.inputs
      .filter(
        (port) =>
          port.required &&
          !edges.some(
            (edge) =>
              edge.target === node.id &&
              decodeHandleId(edge.targetHandle)?.portName === port.name,
          ),
      )
      .map((port) => ({
        nodeTitle: node.data.spec.title,
        portName: port.name,
      })),
  );
}

function executionValidationError(
  scope: RunScope,
  executionNodes: readonly WorkflowNode[],
  executionEdges: readonly WorkflowEdge[],
): string | null {
  if (!executionNodes.length) {
    return scope !== "all"
      ? "Select at least one node before running a selection."
      : "Add at least one node before running the workflow.";
  }

  const sourceWithoutFiles = executionNodes.find(
    (node) =>
      node.data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID &&
      !selectedSourceItems(node.data).length,
  );
  if (sourceWithoutFiles) {
    return `Choose source files for ${sourceWithoutFiles.data.spec.title} before running.`;
  }

  const missingInputs = missingRequiredInputsFor(
    executionNodes,
    executionEdges,
  );
  if (!missingInputs.length) return null;

  const first = missingInputs[0];
  return `${first.nodeTitle}.${first.portName} is required but unconnected in this run.`;
}

function exampleWorkflowEdges(
  nodes: readonly WorkflowNode[],
  artifactTypes: readonly ArtifactTypeSpec[],
): WorkflowEdge[] {
  const edges: WorkflowEdge[] = [];

  for (const link of ARITHMETIC_LINKS) {
    const source = nodes.find((node) => node.id === link.sourceId);
    const target = nodes.find((node) => node.id === link.targetId);
    const output = source?.data.spec.outputs.find(
      (port) => port.name === link.sourcePort,
    );
    const input = target?.data.spec.inputs.find(
      (port) => port.name === link.targetPort,
    );
    if (!source || !target || !output || !input) continue;

    const connection: Connection = {
      source: source.id,
      sourceHandle: encodeHandleId(portMetaForPort(output)),
      target: target.id,
      targetHandle: encodeHandleId(portMetaForPort(input)),
    };
    const color =
      ARTIFACT_TYPE_COLOR[output.artifact_type.id] ?? tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 2,
    };

    if (!("projectionPath" in link)) {
      if (!connectionArtifactContractIsValid(connection)) continue;
      edges.push({
        ...connection,
        id: `edge-${source.id}-${output.name}-${target.id}-${input.name}`,
        type: WORKFLOW_EDGE_TYPE,
        data: { collectionMode: "direct" },
        style: edgeStyle,
      });
      continue;
    }

    const projection = projectionCandidatesForConnection(
      connection,
      artifactTypes,
    ).find(
      (candidate) =>
        candidate.path.join(".") === link.projectionPath.join("."),
    );
    if (!projection) continue;
    edges.push({
      ...connection,
      id: `edge-${source.id}-${output.name}-${target.id}-${input.name}`,
      type: WORKFLOW_EDGE_TYPE,
      data: {
        collectionMode: "direct",
        projection: { path: [...projection.path] },
      },
      style: edgeStyle,
    });
  }
  return edges;
}

function workflowNodes(
  specs: readonly NodeSpec[],
): WorkflowNode[] {
  const byOperator = new Map(specs.map((spec) => [spec.operator_id, spec]));
  return ARITHMETIC_NODE_LAYOUT.flatMap((definition, index) => {
    const spec = byOperator.get(definition.operatorId);
    if (!spec) return [];
    const data = createWorkflowNodeData(spec);
    data.config = { ...data.config, ...definition.config };
    return [
      {
        id: definition.id,
        type: WORKFLOW_NODE_TYPE,
        position: definition.position,
        selected: index === 0,
        data,
      },
    ];
  });
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
    padding: "6px 9px 6px 7px",
  },
  brandMark: {
    width: "29px",
    height: "29px",
    display: "grid",
    placeItems: "center",
    borderRadius: "6px",
    borderWidth: 0,
    backgroundColor: tokens.colorAccent,
    color: tokens.colorOnAccent,
    cursor: "pointer",
  },
  identityCopy: {
    width: "230px",
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
  actions: { height: "43px", gap: "4px", padding: "5px" },
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
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
  library: {
    position: "absolute",
    zIndex: 30,
    top: "66px",
    left: "13px",
    width: "310px",
    maxHeight: "min(620px, calc(100vh - 92px))",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "9px",
    backgroundColor: tokens.colorSurface,
  },
  libraryHeader: {
    height: "40px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    paddingInline: "11px 7px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  libraryTitle: { flex: 1, fontSize: tokens.fontSizeSm, fontWeight: 700 },
  closeButton: {
    width: "24px",
    height: "24px",
    display: "grid",
    placeItems: "center",
    borderWidth: 0,
    borderRadius: "4px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorSubtle,
    cursor: "pointer",
  },
  searchWrap: { position: "relative", padding: "9px" },
  searchIcon: { position: "absolute", top: "19px", left: "19px", color: tokens.colorSubtle },
  searchInput: {
    width: "100%",
    height: "32px",
    padding: "0 9px 0 29px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: tokens.colorBorderStrong, ":focus": tokens.colorAccent },
    borderRadius: "5px",
    outline: "none",
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  libraryGroup: { maxHeight: "500px", overflowY: "auto", padding: "0 9px 9px" },
  groupLabel: {
    padding: "5px 2px 6px",
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
  },
  libraryItem: {
    position: "relative",
    width: "100%",
    minHeight: "52px",
    display: "grid",
    gridTemplateColumns: "minmax(0,1fr) auto",
    alignItems: "center",
    gap: "9px",
    marginTop: "4px",
    padding: "7px 8px",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: tokens.colorBorder, ":hover": tokens.colorBorderStrong },
    borderRadius: "5px",
    backgroundColor: { default: tokens.colorSurfaceMuted, ":hover": tokens.colorHover },
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  libraryCopy: { minWidth: 0, display: "grid", gap: "2px" },
  libraryTitleText: { fontSize: tokens.fontSizeSm, fontWeight: 700 },
  libraryMeta: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  libraryPorts: { color: tokens.colorSubtle, fontSize: tokens.fontSizeXs, whiteSpace: "nowrap" },
  empty: { padding: "18px", color: tokens.colorSubtle, fontSize: tokens.fontSizeSm, textAlign: "center" },
  statusBar: {
    position: "absolute",
    zIndex: 15,
    left: "13px",
    bottom: "13px",
    minHeight: "29px",
    gap: "7px",
    padding: "5px 8px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
  },
  statusDot: { width: "5px", height: "5px", borderRadius: "99px", backgroundColor: tokens.colorSuccess },
  statusDotIncomplete: { backgroundColor: tokens.colorWarning },
  statusDotError: { backgroundColor: tokens.colorDanger },
  statusValue: { color: tokens.colorTextEmphasis, fontWeight: 700 },
  statusDivider: { width: "1px", height: "12px", backgroundColor: tokens.colorDivider },
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

export function Workbench({
  workspaceSlug,
  initialGraphId,
  seedExample,
}: WorkbenchProps) {
  const router = useRouter();
  const { data: registry, error: registryError } = useNodeRegistry();
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
  const [graphName, setGraphName] = React.useState(
    seedExample ? INITIAL_GRAPH_NAME : NEW_GRAPH_NAME,
  );
  const [activeGraph, setActiveGraph] =
    React.useState<ActiveSavedGraph | null>(null);
  const [savedFingerprint, setSavedFingerprint] =
    React.useState<string | null>(null);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<WorkflowNode, WorkflowEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [graphBrowserOpen, setGraphBrowserOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
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

  const removeNode = React.useCallback((nodeId: string) => {
    setNodes((current) =>
      current
        .filter((node) => node.id !== nodeId)
        .map((node) => ({
          ...node,
          data: {
            ...node.data,
            run: null,
            execution: { status: "idle" },
          },
        })),
    );
    setEdges((current) =>
      current.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId,
      ),
    );
    setPendingConnectionRoute(null);
    setRunError(null);
  }, []);

  const removeSelection = React.useCallback(
    (nodeId: string, index: number) => {
      const invalidatedNodeIds = nodeAndDescendantIds(nodeId, edges);
      setNodes((current) =>
        current.map((node) => {
          if (!invalidatedNodeIds.has(node.id)) return node;
          return {
            ...node,
            data: {
              ...(node.id === nodeId
                ? removeSelectionItem(node.data, index)
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

  const handleFilesSelected = React.useCallback(async (nodeId: string, files: File[]) => {
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
      const selections = await Promise.all(files.map(async (file) =>
        uploadFile(file.name, await fileToBase64(file)),
      ));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: invalidatedNodeIds.has(node.id)
          ? {
              ...(node.id === nodeId
                ? replaceSelection(node.data, selections)
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

  const attachNodeCallbacks = React.useCallback(
    (data: WorkflowNodeData): WorkflowNodeData => ({
      ...data,
      onConfigChange: updateConfig,
      onRemoveNode: removeNode,
      onFilesSelected:
        data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID
          ? handleFilesSelected
          : undefined,
      onRemoveSelection: removeSelection,
    }),
    [
      handleFilesSelected,
      removeNode,
      removeSelection,
      updateConfig,
    ],
  );

  React.useEffect(() => {
    if (!registry) return;
    if (!initializedRef.current) {
      initializedRef.current = true;
      if (initialGraphId || !seedExample) return;
      const initialNodes = workflowNodes(registry.nodes).map((node) => ({
        ...node,
        data: attachNodeCallbacks(node.data),
      }));
      // Registry arrival is the one-time boundary that creates the mutable canvas draft.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setNodes(initialNodes);
      setEdges([]);
      setFitRevision((current) => current + 1);
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
  }, [attachNodeCallbacks, initialGraphId, registry, seedExample]);

  React.useEffect(() => {
    if (!flow || !nodes.length) return;
    const frame = window.requestAnimationFrame(() => void flow.fitView({ padding: 0.12, maxZoom: 0.88 }));
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

  const catalog = registry?.nodes ?? [];
  const filteredCatalog = catalog.filter((spec) => {
    const query = search.trim().toLowerCase();
    return !query || spec.title.toLowerCase().includes(query) || spec.operator_id.toLowerCase().includes(query);
  });
  const sourceWithoutFiles = nodes.some(
    (node) => node.data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID && !selectedSourceItems(node.data).length,
  );
  const selectedNodeCount = nodes.filter((node) => node.selected).length;
  const selectedWithDependenciesCount = selectedNodeAndAncestorIds(
    nodes,
    edges,
  ).size;
  const missingOperators = registry
    ? ARITHMETIC_OPERATORS.filter((operatorId) => !registry.nodes.some((spec) => spec.operator_id === operatorId))
    : [];
  const missingRequiredInputs = missingRequiredInputsFor(nodes, edges);
  const connectionInstruction = missingRequiredInputs.length
    ? `${missingRequiredInputs.length} required input${missingRequiredInputs.length === 1 ? "" : "s"} unconnected · drag between ports or use Wire example`
    : null;
  const canWireExample = Boolean(
    registry &&
    missingOperators.length === 0 &&
    ARITHMETIC_NODE_LAYOUT.every((definition) =>
      nodes.some((node) => node.id === definition.id),
    ),
  );
  const runAllDisabled =
    !registry ||
    !nodes.length ||
    running ||
    sourceWithoutFiles ||
    missingRequiredInputs.length > 0;
  const runSelectedDisabled =
    !registry || running || selectedNodeCount === 0;
  const statusMessage = runningScope === "selected"
    ? "running selected nodes · latest upstream outputs are pinned"
    : runningScope === "selected-with-dependencies"
      ? "running selected nodes and all upstream dependencies"
      : !registry
        ? registryError
          ? "registry unavailable · run disabled"
          : "loading live registry…"
        : persistenceError
          ? persistenceError
          : runError
            ? runError
            : sourceWithoutFiles
              ? "choose source files before running"
              : connectionInstruction ?? "all required inputs connected · ready to run";

  const onNodesChange: OnNodesChange<WorkflowNode> = React.useCallback(
    (changes) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );

  const clearWorkflowResults = React.useCallback(() => {
    setNodes((current) => current.map((node) => ({
      ...node,
      data: {
        ...node.data,
        run: null,
        execution: { status: "idle" },
      },
    })));
    setRunError(null);
  }, []);

  const onEdgesChange: OnEdgesChange<WorkflowEdge> = React.useCallback(
    (changes) => {
      setEdges((current) => applyEdgeChanges(changes, current));
      if (changes.some((change) => change.type !== "select")) {
        clearWorkflowResults();
      }
    },
    [clearWorkflowResults],
  );

  const updateEdge = React.useCallback(
    (edgeId: string, update: WorkflowEdgeUpdate) => {
      setEdges((current) =>
        current.map((edge) => {
          if (edge.id !== edgeId) return edge;
          const projection = update.route
            ? update.route.projection
            : edge.data?.projection;
          const conversion = update.route
            ? update.route.conversion
            : edge.data?.conversion;
          return {
            ...edge,
            data: {
              ...edge.data,
              collectionMode:
                update.collectionMode ??
                edge.data?.collectionMode ??
                "direct",
              projection,
              conversion,
            },
          };
        }),
      );
      clearWorkflowResults();
    },
    [clearWorkflowResults],
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
    const source = decodeHandleId(connection.sourceHandle);
    const color = source
      ? ARTIFACT_TYPE_COLOR[source.artifactTypeId] ?? tokens.colorAccent
      : tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 2,
    };
    const selection = connectionRouteSelection(route);
    const edge: WorkflowEdge = {
      ...connection,
      id: `edge-${crypto.randomUUID()}`,
      type: WORKFLOW_EDGE_TYPE,
      animated: false,
      data: {
        collectionMode,
        projection: selection.projection
          ? { path: [...selection.projection.path] }
          : undefined,
        conversion: selection.conversion
          ? {
              id: selection.conversion.id,
              version: selection.conversion.version,
            }
          : undefined,
      },
      style: edgeStyle,
    };
    setEdges((current) => addEdge(edge, current));
    clearWorkflowResults();
  }, [clearWorkflowResults]);

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
    const firstCandidate = candidates[0];
    if (!firstCandidate) return;
    if (firstCandidate.kind === "exact" || candidates.length === 1) {
      addWorkflowEdge(connection, collectionMode, firstCandidate);
      return;
    }

    const source = decodeHandleId(connection.sourceHandle);
    const sourceNode = nodes.find((node) => node.id === connection.source);

    const target = decodeHandleId(connection.targetHandle);
    const targetNode = nodes.find((node) => node.id === connection.target);
    if (!source || !target || !sourceNode || !targetNode) {
      return;
    }

    setPendingConnectionRoute({
      connection,
      collectionMode,
      candidates,
      source: {
        nodeTitle: sourceNode.data.spec.title,
        portName: source.portName,
        artifactType: `${source.artifactTypeId}@${source.schemaVersion}`,
      },
      target: {
        nodeTitle: targetNode.data.spec.title,
        portName: target.portName,
        artifactType: `${target.artifactTypeId}@${target.schemaVersion}`,
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
    setSearch("");
  }, [attachNodeCallbacks, flow]);

  const wireExample = React.useCallback(() => {
    if (!registry || !canWireExample || running) return;
    setEdges(exampleWorkflowEdges(nodes, registry.artifact_types));
    setPendingConnectionRoute(null);
    clearWorkflowResults();
  }, [canWireExample, clearWorkflowResults, nodes, registry, running]);

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
    setGraphName(NEW_GRAPH_NAME);
    activeGraphRef.current = null;
    setActiveGraph(null);
    setSavedFingerprint(null);
    setPendingConnectionRoute(null);
    setRunError(null);
    setPersistenceError(null);
    setLibraryOpen(false);
    setSearch("");
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
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      setGraphName((current) =>
        current.trim() === submittedDraft.name ? savedGraph.name : current,
      );
      clearWorkflowResults();
      if (!activeGraph) {
        approvedRouteGraphIdRef.current = savedGraph.id;
        router.replace(
          workbenchGraphPath(workspaceSlug, savedGraph.id),
          { scroll: false },
        );
      }
      void refreshSavedGraphs();
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
    clearWorkflowResults,
    currentDraft,
    deletingGraphId,
    openingGraphId,
    refreshSavedGraphs,
    router,
    running,
    saving,
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
      setNodes(
        hydrated.nodes.map((node) => ({
          ...node,
          data: attachNodeCallbacks(node.data),
        })),
      );
      setEdges(hydrated.edges);
      setGraphName(savedGraph.name);
      const nextActiveGraph = {
        id: savedGraph.id,
        revision: savedGraph.revision,
      };
      activeGraphRef.current = nextActiveGraph;
      setActiveGraph(nextActiveGraph);
      setSavedFingerprint(savedGraphFingerprint(responseDraft));
      setPendingConnectionRoute(null);
      setRunError(null);
      setPersistenceError(materializationWarning);
      setLibraryOpen(false);
      setSearch("");
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
    router,
    workspaceSlug,
  ]);

  React.useEffect(() => {
    if (seedExample || !registry) {
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
    return () => openRequestRef.current?.abort();
  }, [
    activeGraph?.id,
    confirmDiscard,
    initialGraphId,
    openSavedGraph,
    registry,
    router,
    seedExample,
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
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          mappedInputPort: mappedInputPortForNode(node.id, edges),
        },
      })),
    [edges, nodes],
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
        const routeOptions = connectionRoutesFor(
          connection,
          registry?.artifact_types ?? [],
          registry?.artifact_conversions ?? [],
        ).map(workflowEdgeRouteOption);
        const conversionTitle = edge.data?.conversion
          ? registry?.artifact_conversions.find(
              (conversion) =>
                conversion.key.id === edge.data?.conversion?.id &&
                conversion.key.version === edge.data.conversion.version,
            )?.title
          : undefined;
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
            conversionTitle,
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
    const validationError = executionValidationError(
      scope,
      executionNodes,
      executionEdges,
    );
    if (validationError) {
      setRunError(validationError);
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
            collection_mode: edge.data?.collectionMode ?? "direct",
            projection: edge.data?.projection
              ? { path: [...edge.data.projection.path] }
              : null,
            conversion: edge.data?.conversion
              ? {
                  id: edge.data.conversion.id,
                  version: edge.data.conversion.version,
                }
              : null,
          };
          return [runEdge];
        },
      );
      const runNodes = executionNodes.map((node) => ({
        id: node.id,
        operator_id: node.data.spec.operator_id,
        operator_version: node.data.spec.operator_version,
        config: serializeNodeConfig(node.data),
      }));
      const graphContext = activeGraph && !isDirty
        ? {
            graph_id: activeGraph.id,
            graph_revision: activeGraph.revision,
          }
        : {};
      const response = await runGraph({
        nodes: runNodes,
        edges: runEdges,
        ...(scope === "selected" ? { pinned_outputs: pinnedOutputs } : {}),
        ...graphContext,
      });
      const currentActiveGraph = activeGraphRef.current;
      if (
        currentFingerprintRef.current !== planningFingerprint ||
        currentActiveGraph?.id !== planningActiveGraph?.id ||
        currentActiveGraph?.revision !== planningActiveGraph?.revision
      ) {
        setRunError(
          "The graph changed while it was running. Results were recorded for the original saved revision and were not applied to this canvas.",
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
              ? { status: run.status, error: run.error ?? undefined }
              : { status: "skipped", error: "The server did not return a result for this node." },
          },
        };
      }));
      if (response.status === "failed") {
        setRunError("The workflow completed with node errors. Check the failed node for details.");
      }
    } catch (runFailure) {
      const currentActiveGraph = activeGraphRef.current;
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
      setNodes((current) => current.map((node) =>
        executionNodeIds.has(node.id)
          ? {
              ...node,
              data: {
                ...node.data,
                execution: { status: "failed", error: message },
              },
            }
          : node,
      ));
    } finally {
      setRunningScope(null);
    }
  };

  return (
    <main {...stylex.props(s.shell)}>
      <section {...stylex.props(s.canvas)} aria-label="Workflow canvas">
        <WorkflowCanvas
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
        />
      </section>

      <div {...stylex.props(s.topBar)}>
        <div {...stylex.props(s.chrome, s.identity)}>
          <button
            type="button"
            aria-label="Browse saved graphs"
            title="Browse saved graphs"
            {...stylex.props(s.brandMark)}
            onClick={toggleGraphBrowser}
          >
            <Workflow size={16} strokeWidth={2.2} />
          </button>
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
        </div>

        <div {...stylex.props(s.chrome, s.actions)}>
          <button
            type="button"
            disabled={
              saving ||
              running ||
              Boolean(openingGraphId) ||
              Boolean(deletingGraphId) ||
              !graphName.trim() ||
              Boolean(activeGraph && !isDirty)
            }
            title={activeGraph && !isDirty ? "All changes are saved" : "Save graph"}
            {...stylex.props(s.toolButton)}
            onClick={() => void saveCurrentGraph()}
          >
            {saving ? (
              <LoaderCircle size={13} {...stylex.props(s.spinner)} />
            ) : (
              <Save size={13} />
            )}
            {saving ? "Saving…" : activeGraph && !isDirty ? "Saved" : "Save"}
          </button>
          <button
            type="button"
            {...stylex.props(s.toolButton)}
            disabled={!registry}
            onClick={() => {
              setGraphBrowserOpen(false);
              setLibraryOpen((open) => !open);
            }}
          >
            <Plus size={13} /> Add node
          </button>
          <button
            type="button"
            disabled={!canWireExample || running}
            title={canWireExample
              ? "Replace current connections with the canonical four-edge example"
              : "Waiting for all arithmetic example nodes"}
            {...stylex.props(s.toolButton)}
            onClick={wireExample}
          >
            <Cable size={13} /> Wire example
          </button>
          <button type="button" aria-label="Fit workflow" {...stylex.props(s.toolButton)} onClick={() => void flow?.fitView({ padding: 0.12 })}>
            <Maximize2 size={13} />
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
            {...stylex.props(s.toolButton)}
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
          <button
            type="button"
            disabled={runSelectedDisabled}
            title={
              selectedNodeCount
                ? "Run only the selected nodes; latest accessible upstream outputs are pinned"
                : "Drag-select or shift-click at least one node"
            }
            {...stylex.props(s.toolButton)}
            onClick={() => void runWorkflow("selected")}
          >
            {runningScope === "selected" ? (
              <LoaderCircle size={13} {...stylex.props(s.spinner)} />
            ) : (
              <MousePointer2 size={13} />
            )}
            {runningScope === "selected"
              ? "Running…"
              : `Run selected${selectedNodeCount ? ` (${selectedNodeCount})` : ""}`}
          </button>
          <button
            type="button"
            disabled={runSelectedDisabled}
            title={
              selectedNodeCount
                ? `Run the selected nodes and every upstream dependency (${selectedWithDependenciesCount} total)`
                : "Drag-select or shift-click at least one node"
            }
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
              : `Run with dependencies${selectedNodeCount ? ` (${selectedWithDependenciesCount})` : ""}`}
          </button>
          <button
            type="button"
            disabled={runAllDisabled}
            title={!registry
              ? "Waiting for the live node registry"
              : sourceWithoutFiles
                  ? "Choose at least one source image"
                  : connectionInstruction ?? undefined}
            {...stylex.props(s.toolButton, s.primaryButton)}
            onClick={() => void runWorkflow("all")}
          >
            {runningScope === "all" ? (
              <LoaderCircle size={13} {...stylex.props(s.spinner)} />
            ) : (
              <Play size={13} />
            )}
            {runningScope === "all" ? "Running…" : "Run all"}
          </button>
        </div>
      </div>

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

      {libraryOpen ? (
        <aside {...stylex.props(s.library)} aria-label="Node library">
          <header {...stylex.props(s.libraryHeader)}>
            <Plus size={13} />
            <h2 {...stylex.props(s.libraryTitle)}>Add a node</h2>
            <button type="button" aria-label="Close node library" {...stylex.props(s.closeButton)} onClick={() => setLibraryOpen(false)}><X size={13} /></button>
          </header>
          <div {...stylex.props(s.searchWrap)}>
            <Search size={12} {...stylex.props(s.searchIcon)} />
            <input
              autoFocus
              aria-label="Search registered nodes"
              value={search}
              placeholder="Search registered nodes…"
              {...stylex.props(s.searchInput)}
              onChange={(event) => setSearch(event.currentTarget.value)}
            />
          </div>
          <div {...stylex.props(s.libraryGroup)}>
            <div {...stylex.props(s.groupLabel)}>Live operators</div>
            {filteredCatalog.length ? filteredCatalog.map((spec) => (
              <button key={`${spec.operator_id}@${spec.operator_version}`} type="button" {...stylex.props(s.libraryItem)} onClick={() => addCatalogNode(spec)}>
                <span {...stylex.props(s.libraryCopy)}>
                  <span {...stylex.props(s.libraryTitleText)}>{spec.title}</span>
                  <span {...stylex.props(s.libraryMeta)}>{spec.operator_id}@{spec.operator_version}</span>
                </span>
                <span {...stylex.props(s.libraryPorts)}>{spec.inputs.length} in · {spec.outputs.length} out</span>
              </button>
            )) : <div {...stylex.props(s.empty)}>No matching registered nodes.</div>}
          </div>
        </aside>
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

      <div
        role="status"
        aria-live="polite"
        {...stylex.props(s.chrome, s.statusBar)}
        title={statusMessage}
      >
        <span {...stylex.props(
          s.statusDot,
          registryError || runError || persistenceError ? s.statusDotError : null,
          !registryError && !runError && !persistenceError &&
            (sourceWithoutFiles || missingRequiredInputs.length)
            ? s.statusDotIncomplete
            : null,
        )} />
        <span {...stylex.props(s.statusValue)}>{nodes.length}</span> nodes
        <span {...stylex.props(s.statusDivider)} />
        <span {...stylex.props(s.statusValue)}>{edges.length}</span> connections
        <span {...stylex.props(s.statusDivider)} />
        {statusMessage}
      </div>
    </main>
  );
}

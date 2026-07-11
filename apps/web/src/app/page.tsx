"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
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
  LoaderCircle,
  Maximize2,
  Play,
  Plus,
  Search,
  Workflow,
  X,
} from "lucide-react";

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
  connectionIsValid,
  decodeHandleId,
  encodeHandleId,
  projectionAwareConnectionIsValid,
  projectionCandidatesForConnection,
} from "@/components/canvas/handles";
import { ARTIFACT_TYPE_COLOR } from "@/components/canvas/nodes.css";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  PROTOTYPE_NODE_TYPE,
  createPrototypeNodeData,
  portMetaForPrototypePort,
  removePrototypeSelectionItem,
  replacePrototypeSelection,
  selectedPrototypeItems,
  serializePrototypeConfig,
  type PrototypeFlowEdge,
  type PrototypeNodeData,
} from "@/components/canvas/types";
import { usePrototypeRegistry } from "@/hooks/use-api";
import {
  fileToBase64,
  runPrototypeGraph,
  uploadPrototypeFile,
  type PrototypeArtifactTypeSpec,
  type PrototypeFieldProjection,
  type PrototypeNodeSpec,
  type PrototypeRunEdgeInput,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

type PrototypeFlowNode = Node<PrototypeNodeData, typeof PROTOTYPE_NODE_TYPE>;

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

const PROJECTION_EDGE_PRESENTATION: Pick<
  PrototypeFlowEdge,
  | "labelStyle"
  | "labelShowBg"
  | "labelBgStyle"
  | "labelBgPadding"
  | "labelBgBorderRadius"
> = {
  labelStyle: { fill: "#f0ecff", fontSize: 10, fontWeight: 700 },
  labelShowBg: true,
  labelBgStyle: {
    fill: "#292532",
    fillOpacity: 1,
    stroke: "rgba(190,168,255,0.72)",
    strokeWidth: 1,
  },
  labelBgPadding: [5, 3],
  labelBgBorderRadius: 4,
};

interface ProjectionEndpoint {
  nodeTitle: string;
  portName: string;
  artifactType: string;
}

interface PendingProjection {
  connection: Connection;
  candidates: PrototypeFieldProjection[];
  source: ProjectionEndpoint;
  target: ProjectionEndpoint;
}

function exampleWorkflowEdges(
  nodes: readonly PrototypeFlowNode[],
  artifactTypes: readonly PrototypeArtifactTypeSpec[],
): PrototypeFlowEdge[] {
  const edges: PrototypeFlowEdge[] = [];

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
      sourceHandle: encodeHandleId(portMetaForPrototypePort(output)),
      target: target.id,
      targetHandle: encodeHandleId(portMetaForPrototypePort(input)),
    };
    const color =
      ARTIFACT_TYPE_COLOR[output.artifact_type.id] ?? tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 4,
    };

    if (!("projectionPath" in link)) {
      if (!connectionIsValid(connection)) continue;
      edges.push({
        ...connection,
        id: `edge-${source.id}-${output.name}-${target.id}-${input.name}`,
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
      ...PROJECTION_EDGE_PRESENTATION,
      id: `edge-${source.id}-${output.name}-${target.id}-${input.name}`,
      data: { projection: { path: [...projection.path] } },
      label: `${output.name}.${projection.path.join(".")}`,
      style: edgeStyle,
    });
  }
  return edges;
}

function workflowNodes(
  specs: readonly PrototypeNodeSpec[],
): PrototypeFlowNode[] {
  const byOperator = new Map(specs.map((spec) => [spec.operator_id, spec]));
  return ARITHMETIC_NODE_LAYOUT.flatMap((definition, index) => {
    const spec = byOperator.get(definition.operatorId);
    if (!spec) return [];
    const data = createPrototypeNodeData(spec);
    data.config = { ...data.config, ...definition.config };
    return [
      {
        id: definition.id,
        type: PROTOTYPE_NODE_TYPE,
        position: definition.position,
        selected: index === 0,
        data,
      },
    ];
  });
}

function groupColor(operatorId: string): string {
  if (operatorId.startsWith("source.")) return "#43c59e";
  if (operatorId.startsWith("ocr.")) return "#9a7cf2";
  if (operatorId.includes("export")) return "#f0a65a";
  return "#57a5ef";
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
    borderColor: "rgba(255,255,255,0.11)",
    borderRadius: "8px",
    backgroundColor: "rgba(27,29,33,0.9)",
    boxShadow: "0 10px 28px rgba(0,0,0,0.28)",
    backdropFilter: "blur(16px)",
    pointerEvents: "auto",
  },
  identity: { minHeight: "43px", gap: "10px", padding: "6px 9px 6px 7px" },
  brandMark: {
    width: "29px",
    height: "29px",
    display: "grid",
    placeItems: "center",
    borderRadius: "6px",
    backgroundColor: tokens.colorAccent,
    color: "#fff",
  },
  identityCopy: { display: "grid", gap: "1px" },
  brand: {
    color: tokens.colorMuted,
    fontSize: "8px",
    fontWeight: 800,
    letterSpacing: "0.16em",
    lineHeight: 1.1,
  },
  workflowName: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    color: "#f0f1f3",
    fontSize: "11.5px",
    fontWeight: 700,
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
    backgroundColor: { default: "transparent", ":hover": "rgba(255,255,255,0.07)", ":disabled": "transparent" },
    color: { default: tokens.colorMuted, ":disabled": "#5f636a" },
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    fontSize: "10.5px",
  },
  primaryButton: {
    backgroundColor: { default: tokens.colorAccent, ":hover": "#9077f0", ":disabled": "#3b3847" },
    color: { default: "#fff", ":disabled": "#77727f" },
    fontWeight: 700,
    boxShadow: "0 4px 14px rgba(125,93,239,0.28)",
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
    borderColor: "rgba(255,255,255,0.12)",
    borderRadius: "9px",
    backgroundColor: "rgba(25,27,30,0.97)",
    boxShadow: "0 18px 46px rgba(0,0,0,0.42)",
    animationName: "ns-panel-enter",
    animationDuration: "150ms",
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
  libraryTitle: { flex: 1, fontSize: "11px", fontWeight: 700 },
  closeButton: {
    width: "24px",
    height: "24px",
    display: "grid",
    placeItems: "center",
    borderWidth: 0,
    borderRadius: "4px",
    backgroundColor: { default: "transparent", ":hover": "rgba(255,255,255,0.07)" },
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
    backgroundColor: "#1c1e21",
    color: tokens.colorText,
    fontSize: "10px",
  },
  libraryGroup: { maxHeight: "500px", overflowY: "auto", padding: "0 9px 9px" },
  groupLabel: {
    padding: "5px 2px 6px",
    color: tokens.colorSubtle,
    fontSize: "8px",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
  },
  libraryItem: {
    position: "relative",
    width: "100%",
    minHeight: "52px",
    display: "grid",
    gridTemplateColumns: "4px minmax(0,1fr) auto",
    alignItems: "center",
    gap: "9px",
    marginTop: "4px",
    padding: "7px 8px 7px 5px",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: "rgba(255,255,255,0.07)", ":hover": tokens.colorBorderStrong },
    borderRadius: "5px",
    backgroundColor: { default: "rgba(255,255,255,0.025)", ":hover": "rgba(255,255,255,0.05)" },
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  libraryAccent: { width: "3px", height: "29px", borderRadius: "2px" },
  libraryCopy: { minWidth: 0, display: "grid", gap: "2px" },
  libraryTitleText: { fontSize: "10px", fontWeight: 700 },
  libraryMeta: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "7.5px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  libraryPorts: { color: tokens.colorSubtle, fontSize: "8px", whiteSpace: "nowrap" },
  empty: { padding: "18px", color: tokens.colorSubtle, fontSize: "10px", textAlign: "center" },
  statusBar: {
    position: "absolute",
    zIndex: 15,
    left: "13px",
    bottom: "13px",
    minHeight: "29px",
    gap: "7px",
    padding: "5px 8px",
    color: tokens.colorSubtle,
    fontSize: "8.5px",
  },
  statusDot: { width: "5px", height: "5px", borderRadius: "99px", backgroundColor: tokens.colorSuccess },
  statusDotIncomplete: { backgroundColor: tokens.colorWarning },
  statusDotError: { backgroundColor: tokens.colorDanger },
  statusValue: { color: "#d1d4d8", fontWeight: 700 },
  statusDivider: { width: "1px", height: "12px", backgroundColor: "rgba(255,255,255,0.1)" },
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
    borderColor: "rgba(255,255,255,0.08)",
    borderRadius: "6px",
    backgroundColor: "rgba(255,255,255,0.025)",
  },
  projectionDirection: {
    color: tokens.colorSubtle,
    fontSize: "8px",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
  },
  projectionEndpointName: {
    overflow: "hidden",
    color: "#e2e4e7",
    fontSize: "10px",
    fontWeight: 700,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  projectionEndpointType: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "8px",
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
    fontSize: "9px",
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
      default: "rgba(255,255,255,0.1)",
      ":hover": "rgba(167,139,250,0.55)",
      ":focus-visible": tokens.colorAccent,
    },
    borderRadius: "6px",
    outline: "none",
    backgroundColor: {
      default: "rgba(255,255,255,0.025)",
      ":hover": "rgba(128,103,232,0.1)",
    },
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  projectionChoiceTitle: { fontSize: "10.5px", fontWeight: 720 },
  projectionChoicePath: {
    color: "#bdb4e7",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "8.5px",
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
    backgroundColor: { default: "transparent", ":hover": "rgba(255,255,255,0.05)" },
    color: tokens.colorMuted,
    cursor: "pointer",
    fontSize: "9px",
  },
});

export default function Home() {
  const { data: registry, error: registryError } = usePrototypeRegistry();
  const [nodes, setNodes] = React.useState<PrototypeFlowNode[]>([]);
  const [edges, setEdges] = React.useState<PrototypeFlowEdge[]>([]);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<PrototypeFlowNode, PrototypeFlowEdge>
  >();
  const [libraryOpen, setLibraryOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [running, setRunning] = React.useState(false);
  const [runError, setRunError] = React.useState<string | null>(null);
  const [pendingProjection, setPendingProjection] =
    React.useState<PendingProjection | null>(null);
  const initializedRef = React.useRef(false);
  const nodeCounterRef = React.useRef(10);
  const edgeCounterRef = React.useRef(10);

  const updateConfig = React.useCallback(
    (nodeId: string, name: string, value: unknown) => {
      setNodes((current) =>
        current.map((node) => ({
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
        })),
      );
      setRunError(null);
    },
    [],
  );

  const removeSelection = React.useCallback(
    (nodeId: string, index: number) => {
      setNodes((current) =>
        current.map((node) => ({
          ...node,
          data: {
            ...(node.id === nodeId
              ? removePrototypeSelectionItem(node.data, index)
              : node.data),
            run: null,
            execution: { status: "idle" },
          },
        })),
      );
      setRunError(null);
    },
    [],
  );

  const handleFilesSelected = React.useCallback(async (nodeId: string, files: File[]) => {
    setNodes((current) => current.map((node) => ({
      ...node,
      data: {
        ...node.data,
        run: null,
        execution: node.id === nodeId
          ? { status: "uploading" }
          : { status: "idle" },
      },
    })));
    setRunError(null);
    try {
      const selections = await Promise.all(files.map(async (file) =>
        uploadPrototypeFile(file.name, await fileToBase64(file)),
      ));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: {
          ...(node.id === nodeId
            ? replacePrototypeSelection(node.data, selections)
            : node.data),
          execution: { status: "idle" },
          run: null,
        },
      })));
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "Image upload failed";
      setNodes((current) => current.map((node) => node.id === nodeId ? {
        ...node,
        data: { ...node.data, execution: { status: "failed", error: message } },
      } : node));
    }
  }, []);

  React.useEffect(() => {
    if (!registry) return;
    if (!initializedRef.current) {
      const initialNodes = workflowNodes(registry.nodes).map((node) => ({
        ...node,
        data: {
          ...node.data,
          onConfigChange: updateConfig,
          onFilesSelected:
            node.data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID
              ? handleFilesSelected
              : undefined,
          onRemoveSelection: removeSelection,
        },
      }));
      setNodes(initialNodes);
      setEdges([]);
      initializedRef.current = true;
      return;
    }
    const byOperator = new Map(registry.nodes.map((spec) => [spec.operator_id, spec]));
    setNodes((current) => current.map((node) => {
      const spec = byOperator.get(node.data.spec.operator_id);
      if (!spec) return node;
      return {
        ...node,
        data: {
          ...node.data,
          spec,
          onConfigChange: updateConfig,
          onFilesSelected: spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID ? handleFilesSelected : undefined,
          onRemoveSelection: removeSelection,
        },
      };
    }));
  }, [handleFilesSelected, registry, removeSelection, updateConfig]);

  React.useEffect(() => {
    if (!flow || !nodes.length) return;
    const frame = window.requestAnimationFrame(() => void flow.fitView({ padding: 0.12, maxZoom: 0.88 }));
    return () => window.cancelAnimationFrame(frame);
  }, [flow, nodes.length]);

  const catalog = registry?.nodes ?? [];
  const filteredCatalog = catalog.filter((spec) => {
    const query = search.trim().toLowerCase();
    return !query || spec.title.toLowerCase().includes(query) || spec.operator_id.toLowerCase().includes(query);
  });
  const sourceWithoutFiles = nodes.some(
    (node) => node.data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID && !selectedPrototypeItems(node.data).length,
  );
  const missingOperators = registry
    ? ARITHMETIC_OPERATORS.filter((operatorId) => !registry.nodes.some((spec) => spec.operator_id === operatorId))
    : [];
  const missingRequiredInputs = nodes.flatMap((node) =>
    node.data.spec.inputs
      .filter((port) => port.required && !edges.some((edge) => {
        if (edge.target !== node.id) return false;
        return decodeHandleId(edge.targetHandle)?.portName === port.name;
      }))
      .map((port) => ({ nodeTitle: node.data.spec.title, portName: port.name })),
  );
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
  const runDisabled =
    !registry ||
    !nodes.length ||
    running ||
    sourceWithoutFiles ||
    missingRequiredInputs.length > 0 ||
    missingOperators.length > 0;
  const statusMessage = !registry
    ? registryError
      ? "registry unavailable · run disabled"
      : "loading live registry…"
    : missingOperators.length
      ? `missing ${missingOperators.length} arithmetic operator${missingOperators.length === 1 ? "" : "s"}`
      : runError
        ? runError
        : sourceWithoutFiles
          ? "choose source files before running"
          : connectionInstruction ?? "all required inputs connected · ready to run";

  const onNodesChange: OnNodesChange<PrototypeFlowNode> = React.useCallback(
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

  const onEdgesChange: OnEdgesChange<PrototypeFlowEdge> = React.useCallback(
    (changes) => {
      setEdges((current) => applyEdgeChanges(changes, current));
      if (changes.some((change) => change.type !== "select")) {
        clearWorkflowResults();
      }
    },
    [clearWorkflowResults],
  );

  const addWorkflowEdge = React.useCallback((
    connection: Connection,
    projection?: PrototypeFieldProjection,
  ) => {
    const source = decodeHandleId(connection.sourceHandle);
    const color = source
      ? ARTIFACT_TYPE_COLOR[source.artifactTypeId] ?? tokens.colorAccent
      : tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 4,
    };
    const edge: PrototypeFlowEdge = projection && source
      ? {
          ...connection,
          ...PROJECTION_EDGE_PRESENTATION,
          id: `edge-${edgeCounterRef.current++}`,
          animated: false,
          data: { projection: { path: [...projection.path] } },
          label: `${source.portName}.${projection.path.join(".")}`,
          style: edgeStyle,
        }
      : {
          ...connection,
          id: `edge-${edgeCounterRef.current++}`,
          animated: false,
          style: edgeStyle,
        };
    setEdges((current) => addEdge(edge, current));
    clearWorkflowResults();
  }, [clearWorkflowResults]);

  const isValidConnection = React.useCallback<
    IsValidConnection<PrototypeFlowEdge>
  >((connection) => {
    if (!projectionAwareConnectionIsValid(
      connection,
      registry?.artifact_types ?? [],
    )) {
      return false;
    }

    const target = decodeHandleId(connection.targetHandle);
    const targetNode = nodes.find((node) => node.id === connection.target);
    const input = targetNode?.data.spec.inputs.find(
      (port) => port.name === target?.portName,
    );
    if (!target || !input) return false;

    const connectionEdgeId = "id" in connection ? connection.id : null;
    const pendingNodeIds = [connection.target];
    const visitedNodeIds = new Set<string>();
    while (pendingNodeIds.length) {
      const nodeId = pendingNodeIds.pop();
      if (!nodeId || visitedNodeIds.has(nodeId)) continue;
      if (nodeId === connection.source) return false;
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
      edge.target === connection.target &&
      decodeHandleId(edge.targetHandle)?.portName === target.portName,
    );
  }, [edges, nodes, registry?.artifact_types]);

  const onConnect: OnConnect = React.useCallback((connection) => {
    if (!isValidConnection(connection)) return;

    if (connectionIsValid(connection)) {
      addWorkflowEdge(connection);
      return;
    }

    const candidates = projectionCandidatesForConnection(
      connection,
      registry?.artifact_types ?? [],
    );
    const source = decodeHandleId(connection.sourceHandle);
    const target = decodeHandleId(connection.targetHandle);
    const sourceNode = nodes.find((node) => node.id === connection.source);
    const targetNode = nodes.find((node) => node.id === connection.target);
    if (!source || !target || !sourceNode || !targetNode || !candidates.length) {
      return;
    }

    setPendingProjection({
      connection,
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
  }, [addWorkflowEdge, isValidConnection, nodes, registry?.artifact_types]);

  const addCatalogNode = React.useCallback((spec: PrototypeNodeSpec) => {
    const number = nodeCounterRef.current++;
    const id = `node-${number}`;
    const center = flow?.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }) ?? { x: 600, y: 280 };
    const data = createPrototypeNodeData(spec);
    data.onConfigChange = updateConfig;
    data.onFilesSelected = spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID
      ? handleFilesSelected
      : undefined;
    data.onRemoveSelection = removeSelection;
    setNodes((current) => [
      ...current.map((node) => ({ ...node, selected: false })),
      { id, type: PROTOTYPE_NODE_TYPE, position: { x: center.x - 140, y: center.y - 110 }, selected: true, data },
    ]);
    setLibraryOpen(false);
    setSearch("");
  }, [flow, handleFilesSelected, removeSelection, updateConfig]);

  const wireExample = React.useCallback(() => {
    if (!registry || !canWireExample || running) return;
    setEdges(exampleWorkflowEdges(nodes, registry.artifact_types));
    setPendingProjection(null);
    clearWorkflowResults();
  }, [canWireExample, clearWorkflowResults, nodes, registry, running]);

  const runWorkflow = async () => {
    if (runDisabled) return;
    setRunning(true);
    setRunError(null);
    setNodes((current) => current.map((node) => ({
      ...node,
      data: { ...node.data, run: null, execution: { status: "running" } },
    })));
    try {
      const runEdges = edges.flatMap<PrototypeRunEdgeInput>(
        (edge) => {
          const source = decodeHandleId(edge.sourceHandle);
          const target = decodeHandleId(edge.targetHandle);
          if (!source || !target) return [];
          const runEdge = {
            from_node: edge.source,
            from_port: source.portName,
            to_node: edge.target,
            to_port: target.portName,
          };
          if (!edge.data?.projection) return [runEdge];
          return [{
            ...runEdge,
            projection: { path: [...edge.data.projection.path] },
          }];
        },
      );
      const response = await runPrototypeGraph({
        nodes: nodes.map((node) => ({
          id: node.id,
          operator_id: node.data.spec.operator_id,
          config: serializePrototypeConfig(node.data),
        })),
        edges: runEdges,
      });
      const byNode = new Map(response.node_runs.map((run) => [run.node_id, run]));
      setNodes((current) => current.map((node) => {
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
      const message = runFailure instanceof Error ? runFailure.message : "Workflow run failed";
      setRunError(message);
      setNodes((current) => current.map((node) => ({
        ...node,
        data: { ...node.data, execution: { status: "failed", error: message } },
      })));
    } finally {
      setRunning(false);
    }
  };

  return (
    <main {...stylex.props(s.shell)}>
      <section {...stylex.props(s.canvas)} aria-label="Workflow canvas">
        <WorkflowCanvas
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onPaneReady={setFlow}
          onPaneClick={() => setLibraryOpen(false)}
          animateEdges={running}
        />
      </section>

      <div {...stylex.props(s.topBar)}>
        <div {...stylex.props(s.chrome, s.identity)}>
          <span {...stylex.props(s.brandMark)}><Workflow size={16} strokeWidth={2.2} /></span>
          <span {...stylex.props(s.identityCopy)}>
            <span {...stylex.props(s.brand)}>NOTARIUS</span>
            <span {...stylex.props(s.workflowName)}>Arithmetic field projection</span>
          </span>
        </div>

        <div {...stylex.props(s.chrome, s.actions)}>
          <button type="button" {...stylex.props(s.toolButton)} disabled={!registry} onClick={() => setLibraryOpen((open) => !open)}>
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
            disabled={runDisabled}
            title={!registry
              ? "Waiting for the live node registry"
              : missingOperators.length
                ? "The live registry is missing required arithmetic operators"
                : sourceWithoutFiles
                  ? "Choose at least one source image"
                  : connectionInstruction ?? undefined}
            {...stylex.props(s.toolButton, s.primaryButton)}
            onClick={() => void runWorkflow()}
          >
            {running ? <LoaderCircle size={13} {...stylex.props(s.spinner)} /> : <Play size={13} />}
            {running ? "Running…" : "Run workflow"}
          </button>
        </div>
      </div>

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
                <span {...stylex.props(s.libraryAccent)} style={{ backgroundColor: groupColor(spec.operator_id) }} />
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
        open={pendingProjection !== null}
        onOpenChange={(open) => {
          if (!open) setPendingProjection(null);
        }}
      >
        <DialogContent style={{ width: "430px" }}>
          <DialogHeader>
            <DialogTitle>Choose a result field</DialogTitle>
            <DialogDescription>
              This output can satisfy the input through a declared field projection.
            </DialogDescription>
          </DialogHeader>
          <DialogBody>
            {pendingProjection ? (
              <>
                <div {...stylex.props(s.projectionFlow)}>
                  <div {...stylex.props(s.projectionEndpoint)}>
                    <span {...stylex.props(s.projectionDirection)}>Source</span>
                    <span {...stylex.props(s.projectionEndpointName)}>
                      {pendingProjection.source.nodeTitle} · {pendingProjection.source.portName}
                    </span>
                    <span {...stylex.props(s.projectionEndpointType)}>
                      {pendingProjection.source.artifactType}
                    </span>
                  </div>
                  <span aria-hidden="true" {...stylex.props(s.projectionArrow)}>→</span>
                  <div {...stylex.props(s.projectionEndpoint)}>
                    <span {...stylex.props(s.projectionDirection)}>Target</span>
                    <span {...stylex.props(s.projectionEndpointName)}>
                      {pendingProjection.target.nodeTitle} · {pendingProjection.target.portName}
                    </span>
                    <span {...stylex.props(s.projectionEndpointType)}>
                      {pendingProjection.target.artifactType}
                    </span>
                  </div>
                </div>
                <p {...stylex.props(s.projectionPrompt)}>Route one field into this input:</p>
                <div {...stylex.props(s.projectionChoices)}>
                  {pendingProjection.candidates.map((candidate, index) => (
                    <button
                      key={`${candidate.title}-${candidate.path.join(".")}`}
                      type="button"
                      autoFocus={index === 0}
                      aria-label={`Use ${candidate.title} field from ${pendingProjection.source.nodeTitle}`}
                      {...stylex.props(s.projectionChoice)}
                      onClick={() => {
                        addWorkflowEdge(pendingProjection.connection, candidate);
                        setPendingProjection(null);
                      }}
                    >
                      <span {...stylex.props(s.projectionChoiceTitle)}>{candidate.title}</span>
                      <span {...stylex.props(s.projectionChoicePath)}>
                        {pendingProjection.source.portName}.{candidate.path.join(".")}
                      </span>
                    </button>
                  ))}
                </div>
                <div {...stylex.props(s.projectionActions)}>
                  <button
                    type="button"
                    {...stylex.props(s.projectionCancel)}
                    onClick={() => setPendingProjection(null)}
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
          registryError || missingOperators.length || runError ? s.statusDotError : null,
          !registryError && !missingOperators.length && !runError &&
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

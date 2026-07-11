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
  Monitor,
  Moon,
  Play,
  Plus,
  Search,
  Sun,
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
import {
  ARTIFACT_TYPE_COLOR,
  projectionEdgePresentation,
} from "@/components/canvas/nodes.css";
import { useTheme } from "@/components/theme";
import { pathsEqual } from "@/lib/output-port-projection";
import type { OutputPortTreatment } from "@/components/canvas/output-port-treatment";
import { defaultTreatments } from "@/components/canvas/output-port-treatment";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  createWorkflowNodeData,
  portMetaForPort,
  removeSelectionItem,
  replaceSelection,
  selectedSourceItems,
  serializeNodeConfig,
  type WorkflowEdge,
  type WorkflowNodeData,
} from "@/components/canvas/types";
import { useNodeRegistry } from "@/hooks/use-api";
import {
  fileToBase64,
  runGraph,
  uploadFile,
  type ArtifactTypeSpec,
  type FieldProjection,
  type NodeSpec,
  type RunEdgeInput,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

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

interface ProjectionEndpoint {
  nodeTitle: string;
  portName: string;
  artifactType: string;
}

interface PendingProjection {
  connection: Connection;
  candidates: FieldProjection[];
  source: ProjectionEndpoint;
  target: ProjectionEndpoint;
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
      ...projectionEdgePresentation(),
      id: `edge-${source.id}-${output.name}-${target.id}-${input.name}`,
      data: { projection: { path: [...projection.path] } },
      label: `${output.name}.${projection.path.join(".")}`,
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
  identity: { minHeight: "43px", gap: "10px", padding: "6px 9px 6px 7px" },
  brandMark: {
    width: "29px",
    height: "29px",
    display: "grid",
    placeItems: "center",
    borderRadius: "6px",
    backgroundColor: tokens.colorAccent,
    color: tokens.colorOnAccent,
  },
  identityCopy: { display: "grid", gap: "1px" },
  brand: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    fontWeight: 800,
    letterSpacing: "0.16em",
    lineHeight: 1.1,
  },
  workflowName: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeMd,
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

export default function Home() {
  const { data: registry, error: registryError } = useNodeRegistry();
  const { preference, cycleTheme } = useTheme();
  const [nodes, setNodes] = React.useState<WorkflowNode[]>([]);
  const [edges, setEdges] = React.useState<WorkflowEdge[]>([]);
  const [flow, setFlow] = React.useState<
    ReactFlowInstance<WorkflowNode, WorkflowEdge>
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

  const updateOutputTreatment = React.useCallback(
    (nodeId: string, portName: string, treatment: OutputPortTreatment) => {
      setNodes((current) =>
        current.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  outputTreatments: {
                    ...node.data.outputTreatments,
                    [portName]: treatment,
                  },
                },
              }
            : node,
        ),
      );
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
              ? removeSelectionItem(node.data, index)
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
        uploadFile(file.name, await fileToBase64(file)),
      ));
      setNodes((current) => current.map((node) => ({
        ...node,
        data: {
          ...(node.id === nodeId
            ? replaceSelection(node.data, selections)
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

  const attachNodeCallbacks = React.useCallback(
    (data: WorkflowNodeData): WorkflowNodeData => ({
      ...data,
      outputTreatments:
        data.outputTreatments ?? defaultTreatments(data.spec.outputs),
      onConfigChange: updateConfig,
      onFilesSelected:
        data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID
          ? handleFilesSelected
          : undefined,
      onRemoveSelection: removeSelection,
      onOutputTreatmentChange: updateOutputTreatment,
    }),
    [handleFilesSelected, removeSelection, updateConfig, updateOutputTreatment],
  );

  React.useEffect(() => {
    if (!registry) return;
    if (!initializedRef.current) {
      const initialNodes = workflowNodes(registry.nodes).map((node) => ({
        ...node,
        data: attachNodeCallbacks(node.data),
      }));
      setNodes(initialNodes);
      setEdges([]);
      initializedRef.current = true;
      return;
    }
    const byOperator = new Map(registry.nodes.map((spec) => [spec.operator_id, spec]));
    setNodes((current) => current.map((node) => {
      const spec = byOperator.get(node.data.spec.operator_id);
      if (!spec) return { ...node, data: attachNodeCallbacks(node.data) };
      return {
        ...node,
        data: attachNodeCallbacks({ ...node.data, spec }),
      };
    }));
  }, [attachNodeCallbacks, registry]);

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
    (node) => node.data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID && !selectedSourceItems(node.data).length,
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

  const addWorkflowEdge = React.useCallback((
    connection: Connection,
    projection?: FieldProjection,
  ) => {
    const source = decodeHandleId(connection.sourceHandle);
    const color = source
      ? ARTIFACT_TYPE_COLOR[source.artifactTypeId] ?? tokens.colorAccent
      : tokens.colorAccent;
    const edgeStyle = {
      stroke: color,
      strokeWidth: 2,
    };
    const edge: WorkflowEdge = projection && source
      ? {
          ...connection,
          ...projectionEdgePresentation(),
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
    IsValidConnection<WorkflowEdge>
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

    const artifactTypes = registry?.artifact_types ?? [];
    const candidates = projectionCandidatesForConnection(
      connection,
      artifactTypes,
    );
    const source = decodeHandleId(connection.sourceHandle);
    const sourceNode = nodes.find((node) => node.id === connection.source);

    if (source && sourceNode && candidates.length) {
      const treatment = sourceNode.data.outputTreatments[source.portName];
      if (treatment?.projectionPath.length) {
        const match = candidates.find((candidate) =>
          pathsEqual(candidate.path, treatment.projectionPath),
        );
        if (match) {
          addWorkflowEdge(connection, match);
          return;
        }
      }
    }

    const target = decodeHandleId(connection.targetHandle);
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

  const addCatalogNode = React.useCallback((spec: NodeSpec) => {
    const number = nodeCounterRef.current++;
    const id = `node-${number}`;
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
      const runEdges = edges.flatMap<RunEdgeInput>(
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
      const response = await runGraph({
        nodes: nodes.map((node) => ({
          id: node.id,
          operator_id: node.data.spec.operator_id,
          config: serializeNodeConfig(node.data),
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

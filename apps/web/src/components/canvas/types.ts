import type { Edge } from "@xyflow/react";

import type {
  ArtifactConversionInput,
  NodeSpec,
  NodeConfigInput,
  Port,
  RunEdgeCollectionMode,
  RunEdgeProjectionInput,
  RunNodeResult,
  SelectionItem,
} from "@/lib/api";

export type WorkflowEdgeProjection = RunEdgeProjectionInput;
export type WorkflowEdgeConversion = ArtifactConversionInput;

export interface WorkflowEdgeRoute {
  projection?: WorkflowEdgeProjection;
  conversion?: WorkflowEdgeConversion;
}

export interface WorkflowEdgeRouteOption extends WorkflowEdgeRoute {
  projectionTitle?: string;
  conversionTitle?: string;
}

export interface WorkflowEdgeRouteOffset {
  x: number;
  y: number;
}

export interface WorkflowEdgeUpdate {
  collectionMode?: RunEdgeCollectionMode;
  route?: WorkflowEdgeRoute;
}

export interface WorkflowEdgeData extends Record<string, unknown> {
  collectionMode: RunEdgeCollectionMode;
  projection?: WorkflowEdgeProjection;
  conversion?: WorkflowEdgeConversion;
  /** Visual routing adjustment from the edge's natural midpoint. */
  routeOffset?: WorkflowEdgeRouteOffset;
  sourcePortName?: string;
  conversionTitle?: string;
  routeOptions?: readonly WorkflowEdgeRouteOption[];
  allowedCollectionModes?: readonly RunEdgeCollectionMode[];
  onUpdate?: (edgeId: string, update: WorkflowEdgeUpdate) => void;
  onRouteOffsetChange?: (
    edgeId: string,
    offset: WorkflowEdgeRouteOffset,
  ) => void;
}

export type WorkflowEdge = Edge<WorkflowEdgeData>;

/** Metadata encoded into React Flow handle ids for typed connections. */
export interface PortMeta {
  portName: string;
  artifactTypeId: string;
  schemaVersion: number;
  shape: Port["shape"];
  direction: "input" | "output";
}

export type WorkflowNodeConfig = Record<string, unknown> & {
  connector_id?: string;
  selection?: SelectionItem[];
};

export type NodeExecutionStatus =
  | "idle"
  | "uploading"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped";

export interface NodeExecution {
  status: NodeExecutionStatus;
  error?: string;
}

export interface WorkflowNodeData extends Record<string, unknown> {
  spec: NodeSpec;
  /** Derived from incoming map edges; never persisted as node configuration. */
  mappedInputPort: string | null;
  config: WorkflowNodeConfig;
  run: RunNodeResult | null;
  execution: NodeExecution;
  onFilesSelected?: (nodeId: string, files: File[]) => void;
  onConfigChange?: (nodeId: string, name: string, value: unknown) => void;
  onRemoveNode?: (nodeId: string) => void;
  onRemoveSelection?: (nodeId: string, index: number) => void;
}

export const WORKFLOW_NODE_TYPE = "notariusWorkflowNode";
export const WORKFLOW_EDGE_TYPE = "notariusWorkflowEdge";
export const LOCAL_UPLOAD_OPERATOR_ID = "source.local_upload.images";

function schemaRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function defaultNodeConfig(
  spec: NodeSpec,
): WorkflowNodeConfig {
  const schema = schemaRecord(spec.config_schema);
  const properties = schemaRecord(schema?.properties);
  const config: WorkflowNodeConfig = {};

  if (properties) {
    for (const [name, propertyValue] of Object.entries(properties)) {
      const property = schemaRecord(propertyValue);
      if (property && "default" in property) {
        config[name] = property.default;
      }
    }
  }

  if (spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID) {
    config.connector_id = "local_upload";
    config.selection = [];
  }
  return config;
}

export function createWorkflowNodeData(
  spec: NodeSpec,
): WorkflowNodeData {
  return {
    spec,
    mappedInputPort: null,
    config: defaultNodeConfig(spec),
    run: null,
    execution: { status: "idle" },
  };
}

export function selectedSourceItems(
  data: WorkflowNodeData,
): SelectionItem[] {
  return Array.isArray(data.config.selection) ? data.config.selection : [];
}

export function isLocalUploadSource(data: WorkflowNodeData): boolean {
  return data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID;
}

export function serializeNodeConfig(
  data: WorkflowNodeData,
): NodeConfigInput {
  if (!isLocalUploadSource(data)) return data.config;

  return {
    ...data.config,
    connector_id: "local_upload",
    selection: selectedSourceItems(data).map((item, index) => ({
      ...item,
      order_index: index,
    })),
  };
}

export function appendSelection(
  data: WorkflowNodeData,
  items: readonly SelectionItem[],
): WorkflowNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      connector_id: "local_upload",
      selection: [...selectedSourceItems(data), ...items],
    },
  };
}

export function replaceSelection(
  data: WorkflowNodeData,
  items: readonly SelectionItem[],
): WorkflowNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      connector_id: "local_upload",
      selection: [...items],
    },
  };
}

export function removeSelectionItem(
  data: WorkflowNodeData,
  index: number,
): WorkflowNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      selection: selectedSourceItems(data).filter(
        (_, itemIndex) => itemIndex !== index,
      ),
    },
  };
}

export function updateNodeRun(
  data: WorkflowNodeData,
  run: RunNodeResult | null,
): WorkflowNodeData {
  return {
    ...data,
    run,
    execution: run
      ? { status: run.status, error: run.error ?? undefined }
      : data.execution,
  };
}

export function effectivePortShape(
  data: WorkflowNodeData,
  port: Port,
): Port["shape"] {
  if (!data.mappedInputPort) return port.shape;
  if (port.direction === "output") return "many";
  return data.mappedInputPort === port.name ? "many" : port.shape;
}

export function portMetaForPort(
  port: Port,
  shape: Port["shape"] = port.shape,
): PortMeta {
  return {
    portName: port.name,
    artifactTypeId: port.artifact_type.id,
    schemaVersion: port.artifact_type.schema_version,
    shape,
    direction: port.direction,
  };
}

export function portTypeLabel(port: Port): string {
  return `${port.artifact_type.id}@${port.artifact_type.schema_version}`;
}

export function portSummary(port: Port): string {
  const extras = [port.shape, port.variadic ? "variadic" : null].filter(
    Boolean,
  );

  return `${portTypeLabel(port)}${
    extras.length ? ` · ${extras.join(" · ")}` : ""
  }`;
}

export function portCountLabel(spec: NodeSpec): string {
  return `${spec.inputs.length} in · ${spec.outputs.length} out`;
}

export function groupLabel(group: string): string {
  return group.charAt(0).toUpperCase() + group.slice(1);
}

export function selectionSizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

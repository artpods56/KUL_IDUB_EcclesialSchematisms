import type { Edge } from "@xyflow/react";

import type {
  ArtifactConversionInput,
  ArtifactConversionPathInput,
  ArtifactTypeKey,
  ImageUploadItem,
  InputPlugInput,
  NodeSpec,
  Port,
  RunEdgeCollectionMode,
  RunEdgeProjectionInput,
  RunNodeInput,
  RunNodeResult,
} from "@/lib/api";
import {
  initialInputPlugs,
  type WorkflowInputPlug,
  type WorkflowInputPlugBinding,
} from "./input-plugs";
import type { SchemaBuilderField } from "./schema-builder";
import type { WorkflowNodeSecretStatuses } from "./node-secrets";

export type { WorkflowInputPlug, WorkflowInputPlugBinding } from "./input-plugs";

export type WorkflowEdgeProjection = RunEdgeProjectionInput;
export type WorkflowEdgeConversion = ArtifactConversionInput;
export type WorkflowEdgeConversionPath = readonly WorkflowEdgeConversion[];

export interface WorkflowEdgeRoute {
  projection?: WorkflowEdgeProjection;
  conversionPath: WorkflowEdgeConversionPath;
}

export interface WorkflowEdgeRouteOption extends WorkflowEdgeRoute {
  projectionTitle?: string;
  conversionTitles: readonly string[];
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
  conversionPath?: WorkflowEdgeConversionPath;
  /** Visual routing adjustment from the edge's natural midpoint. */
  routeOffset?: WorkflowEdgeRouteOffset;
  sourcePortName?: string;
  conversionTitles?: readonly string[];
  routeOptions?: readonly WorkflowEdgeRouteOption[];
  allowedCollectionModes?: readonly RunEdgeCollectionMode[];
  onUpdate?: (edgeId: string, update: WorkflowEdgeUpdate) => void;
  onRouteOffsetChange?: (
    edgeId: string,
    offset: WorkflowEdgeRouteOffset,
  ) => void;
}

export type WorkflowEdge = Edge<WorkflowEdgeData>;

export interface WorkflowEdgeTransport {
  collection_mode: RunEdgeCollectionMode;
  projection: RunEdgeProjectionInput | null;
  conversion_path: ArtifactConversionPathInput;
}

export function serializeWorkflowEdgeTransport(
  data: WorkflowEdgeData | undefined,
): WorkflowEdgeTransport {
  return {
    collection_mode: data?.collectionMode ?? "direct",
    projection: data?.projection
      ? { path: [...data.projection.path] }
      : null,
    conversion_path: (data?.conversionPath ?? []).map((conversion) => ({
      id: conversion.id,
      version: conversion.version,
    })),
  };
}

interface PortMetaBase {
  portName: string;
  shape: Port["shape"];
  direction: "input" | "output";
  plugId?: string;
}

/** Metadata encoded into React Flow handle ids for typed connections. */
export type PortMeta = PortMetaBase &
  (
    | {
        artifactTypeId: string;
        schemaVersion: number;
        artifactTypeVariable?: never;
      }
    | {
        artifactTypeId?: never;
        schemaVersion?: never;
        artifactTypeVariable: string;
      }
  );

export type WorkflowArtifactTypeBindings = Readonly<
  Record<string, ArtifactTypeKey>
>;

export interface WorkflowArtifactTypeBindingInput {
  variable: string;
  artifact_type: ArtifactTypeKey;
}

export type WorkflowNodeConfig = Record<string, unknown> & {
  uploads?: ImageUploadItem[];
};

export type NodeExecutionStatus =
  | "idle"
  | "uploading"
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "succeeded"
  | "failed"
  | "skipped";

export interface NodeExecution {
  status: NodeExecutionStatus;
  error?: string;
}

export interface WorkflowNodeData extends Record<string, unknown> {
  spec: NodeSpec;
  /** Persisted concrete choices for artifact type variables declared by ports. */
  artifactTypeBindings: WorkflowArtifactTypeBindings;
  /** Ordered, serializable input instances. Their ids remain stable on reorder. */
  inputPlugs: readonly WorkflowInputPlug[];
  /** Edge- and result-derived display data; never persisted. */
  inputPlugBindings: Readonly<Record<string, WorkflowInputPlugBinding>>;
  /** Derived from incoming map edges; never persisted as node configuration. */
  mappedInputPort: string | null;
  /** Server-reported write-only state; never persisted with the graph. */
  secretStatuses: WorkflowNodeSecretStatuses;
  /** Per-input match against its saved operator and declared config dependencies. */
  secretInputReadiness: Readonly<Record<string, boolean>>;
  /** Derived lifecycle scope for clearing unapplied write-only input values. */
  secretInputScope: string;
  config: WorkflowNodeConfig;
  run: RunNodeResult | null;
  execution: NodeExecution;
  onImagesSelected?: (nodeId: string, files: File[]) => void;
  onConfigChange?: (nodeId: string, name: string, value: unknown) => void;
  onRemoveNode?: (nodeId: string) => void;
  onRemoveImageUpload?: (nodeId: string, index: number) => void;
  onAddInputPlug?: (nodeId: string, portName: string) => void;
  onRemoveInputPlug?: (nodeId: string, plugId: string) => void;
  onReorderInputPlug?: (
    nodeId: string,
    portName: string,
    plugId: string,
    toIndex: number,
  ) => void;
  onSchemaBuilderFieldsChange?: (
    nodeId: string,
    fields: readonly SchemaBuilderField[],
    inputPlugs: readonly WorkflowInputPlug[],
  ) => void;
  onApplyNodeSecret?: (
    nodeId: string,
    name: string,
    value: string,
  ) => Promise<boolean>;
  onRemoveNodeSecret?: (
    nodeId: string,
    name: string,
  ) => Promise<boolean>;
  onResetArtifactTypeBinding?: (nodeId: string, variable: string) => void;
  onHandlesMeasured?: (
    nodeId: string,
    artifactTypeBindings: WorkflowArtifactTypeBindings,
  ) => void;
}

export const WORKFLOW_NODE_TYPE = "notariusWorkflowNode";
export const WORKFLOW_EDGE_TYPE = "notariusWorkflowEdge";
export const IMAGE_UPLOAD_OPERATOR_ID = "image.upload";

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

  if (spec.operator_id === IMAGE_UPLOAD_OPERATOR_ID) {
    config.uploads = [];
  }
  return config;
}

export function createWorkflowNodeData(
  spec: NodeSpec,
  savedInputPlugs?: readonly InputPlugInput[],
): WorkflowNodeData {
  return {
    spec,
    artifactTypeBindings: {},
    inputPlugs: savedInputPlugs
      ? savedInputPlugs.map((plug) => ({
          id: plug.id,
          portName: plug.port,
        }))
      : initialInputPlugs(spec),
    inputPlugBindings: {},
    mappedInputPort: null,
    secretStatuses: {},
    secretInputReadiness: {},
    secretInputScope: "unsaved:none",
    config: defaultNodeConfig(spec),
    run: null,
    execution: { status: "idle" },
  };
}

export function serializeRunNode(
  id: string,
  data: WorkflowNodeData,
): RunNodeInput {
  return {
    id,
    operator_id: data.spec.operator_id,
    operator_version: data.spec.operator_version,
    config: data.config,
    input_plugs: serializeInputPlugs(data),
    artifact_type_bindings: serializeArtifactTypeBindings(data),
  };
}

export function serializeInputPlugs(
  data: WorkflowNodeData,
): InputPlugInput[] {
  return data.inputPlugs.map((plug) => ({
    id: plug.id,
    port: plug.portName,
  }));
}

export function serializeArtifactTypeBindings(
  data: WorkflowNodeData,
): WorkflowArtifactTypeBindingInput[] {
  return Object.entries(data.artifactTypeBindings)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([variable, artifactType]) => ({
      variable,
      artifact_type: {
        id: artifactType.id,
        schema_version: artifactType.schema_version,
      },
    }));
}

export function declaredArtifactTypeVariables(
  spec: NodeSpec,
): readonly string[] {
  return [
    ...new Set(
      [...spec.inputs, ...spec.outputs].flatMap((port) => {
        const variable = portArtifactTypeVariable(port);
        return variable ? [variable] : [];
      }),
    ),
  ];
}

export function resetArtifactTypeBinding(
  data: WorkflowNodeData,
  variable: string,
  hasIncidentEdges: boolean,
): WorkflowNodeData {
  if (hasIncidentEdges || !(variable in data.artifactTypeBindings)) {
    return data;
  }

  const bindings = { ...data.artifactTypeBindings };
  delete bindings[variable];
  return {
    ...data,
    artifactTypeBindings: bindings,
    run: null,
    execution: { status: "idle" },
  };
}

export function bindArtifactTypeVariable(
  data: WorkflowNodeData,
  variable: string,
  artifactType: ArtifactTypeKey,
): WorkflowNodeData {
  if (!declaredArtifactTypeVariables(data.spec).includes(variable)) {
    throw new Error(
      `Cannot bind artifact type variable ${variable}: it is not declared by ${data.spec.operator_id}@${data.spec.operator_version}`,
    );
  }
  return {
    ...data,
    artifactTypeBindings: {
      ...data.artifactTypeBindings,
      [variable]: {
        id: artifactType.id,
        schema_version: artifactType.schema_version,
      },
    },
  };
}

export function imageUploads(data: WorkflowNodeData): ImageUploadItem[] {
  return Array.isArray(data.config.uploads) ? data.config.uploads : [];
}

export function replaceImageUploads(
  data: WorkflowNodeData,
  uploads: readonly ImageUploadItem[],
): WorkflowNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      uploads: [...uploads],
    },
  };
}

export function removeImageUpload(
  data: WorkflowNodeData,
  index: number,
): WorkflowNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      uploads: imageUploads(data).filter(
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

interface WorkflowNodeState {
  id: string;
  data: WorkflowNodeData;
}

interface WorkflowConnectionState {
  source: string;
  target: string;
}

export function invalidateWorkflowNodeRuns<NodeType extends WorkflowNodeState>(
  nodes: readonly NodeType[],
  edges: readonly WorkflowConnectionState[],
  changedTargetNodeIds: readonly string[],
): NodeType[] {
  const invalidatedNodeIds = new Set(changedTargetNodeIds);
  const pendingNodeIds = [...invalidatedNodeIds];

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

  return nodes.map((node) => {
    if (!invalidatedNodeIds.has(node.id)) return node;
    return {
      ...node,
      data: {
        ...node.data,
        run: null,
        execution: { status: "idle" },
      },
    };
  });
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
  plugId?: string,
  artifactTypeBindings: WorkflowArtifactTypeBindings = {},
): PortMeta {
  const artifactType = resolvedPortArtifactType(port, artifactTypeBindings);
  const base = {
    portName: port.name,
    shape,
    direction: port.direction,
    ...(plugId ? { plugId } : {}),
  };
  if (artifactType) {
    return {
      ...base,
      artifactTypeId: artifactType.id,
      schemaVersion: artifactType.schema_version,
    };
  }

  const variable = portArtifactTypeVariable(port);
  if (!variable) {
    throw new Error(
      `Cannot encode port ${port.name}: it has no artifact type or artifact type variable`,
    );
  }
  return {
    ...base,
    artifactTypeVariable: variable,
  };
}

export function portArtifactType(port: Port): ArtifactTypeKey | null {
  return port.artifact_type ?? null;
}

export function portArtifactTypeVariable(port: Port): string | null {
  return port.artifact_type_variable ?? null;
}

export function resolvedPortArtifactType(
  port: Port,
  artifactTypeBindings: WorkflowArtifactTypeBindings = {},
): ArtifactTypeKey | null {
  const artifactType = portArtifactType(port);
  if (artifactType) return artifactType;

  const variable = portArtifactTypeVariable(port);
  return variable ? artifactTypeBindings[variable] ?? null : null;
}

export function acceptedPortShapes(port: Port): readonly Port["shape"][] {
  return port.accepted_shapes?.length ? port.accepted_shapes : [port.shape];
}

export function portHasInstancePlugs(port: Port): boolean {
  return port.direction === "input" && port.instance_plugs === true;
}

export function portTypeLabel(
  port: Port,
  artifactTypeBindings: WorkflowArtifactTypeBindings = {},
): string {
  const artifactType = resolvedPortArtifactType(port, artifactTypeBindings);
  return artifactType
    ? `${artifactType.id}@${artifactType.schema_version}`
    : "Any artifact";
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

export function imageUploadSizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

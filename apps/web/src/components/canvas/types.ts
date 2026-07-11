import type { Edge } from "@xyflow/react";

import type {
  PrototypeNodeSpec,
  PrototypeNodeConfigInput,
  PrototypePort,
  PrototypeRunEdgeProjectionInput,
  PrototypeRunNodeResult,
  PrototypeSelectionItem,
} from "@/lib/api";

export type PrototypeEdgeProjection = PrototypeRunEdgeProjectionInput;

export interface PrototypeEdgeData extends Record<string, unknown> {
  projection?: PrototypeEdgeProjection;
}

export type PrototypeFlowEdge = Edge<PrototypeEdgeData>;

/** Metadata encoded into React Flow handle ids for typed connections. */
export interface PortMeta {
  portName: string;
  artifactTypeId: string;
  schemaVersion: number;
  shape: PrototypePort["shape"];
  direction: "input" | "output";
}

export type PrototypeNodeConfig = Record<string, unknown> & {
  connector_id?: string;
  selection?: PrototypeSelectionItem[];
};

export type PrototypeExecutionStatus =
  | "idle"
  | "uploading"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped";

export interface PrototypeNodeExecution {
  status: PrototypeExecutionStatus;
  error?: string;
}

export interface PrototypeNodeData extends Record<string, unknown> {
  spec: PrototypeNodeSpec;
  config: PrototypeNodeConfig;
  run: PrototypeRunNodeResult | null;
  execution: PrototypeNodeExecution;
  onFilesSelected?: (nodeId: string, files: File[]) => void;
  onConfigChange?: (nodeId: string, name: string, value: unknown) => void;
  onRemoveSelection?: (nodeId: string, index: number) => void;
}

export const PROTOTYPE_NODE_TYPE = "notariusPrototypeNode";
export const LOCAL_UPLOAD_OPERATOR_ID = "source.local_upload.images";

function schemaRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function defaultPrototypeConfig(
  spec: PrototypeNodeSpec,
): PrototypeNodeConfig {
  const schema = schemaRecord(spec.config_schema);
  const properties = schemaRecord(schema?.properties);
  const config: PrototypeNodeConfig = {};

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

export function createPrototypeNodeData(
  spec: PrototypeNodeSpec,
): PrototypeNodeData {
  return {
    spec,
    config: defaultPrototypeConfig(spec),
    run: null,
    execution: { status: "idle" },
  };
}

export function selectedPrototypeItems(
  data: PrototypeNodeData,
): PrototypeSelectionItem[] {
  return Array.isArray(data.config.selection) ? data.config.selection : [];
}

export function isLocalUploadSource(data: PrototypeNodeData): boolean {
  return data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID;
}

export function serializePrototypeConfig(
  data: PrototypeNodeData,
): PrototypeNodeConfigInput {
  if (!isLocalUploadSource(data)) return data.config;

  return {
    ...data.config,
    connector_id: "local_upload",
    selection: selectedPrototypeItems(data).map((item, index) => ({
      ...item,
      order_index: index,
    })),
  };
}

export function appendPrototypeSelection(
  data: PrototypeNodeData,
  items: readonly PrototypeSelectionItem[],
): PrototypeNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      connector_id: "local_upload",
      selection: [...selectedPrototypeItems(data), ...items],
    },
  };
}

export function replacePrototypeSelection(
  data: PrototypeNodeData,
  items: readonly PrototypeSelectionItem[],
): PrototypeNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      connector_id: "local_upload",
      selection: [...items],
    },
  };
}

export function removePrototypeSelectionItem(
  data: PrototypeNodeData,
  index: number,
): PrototypeNodeData {
  return {
    ...data,
    config: {
      ...data.config,
      selection: selectedPrototypeItems(data).filter(
        (_, itemIndex) => itemIndex !== index,
      ),
    },
  };
}

export function updatePrototypeRun(
  data: PrototypeNodeData,
  run: PrototypeRunNodeResult | null,
): PrototypeNodeData {
  return {
    ...data,
    run,
    execution: run
      ? { status: run.status, error: run.error ?? undefined }
      : data.execution,
  };
}

export function portMetaForPrototypePort(port: PrototypePort): PortMeta {
  return {
    portName: port.name,
    artifactTypeId: port.artifact_type.id,
    schemaVersion: port.artifact_type.schema_version,
    shape: port.shape,
    direction: port.direction,
  };
}

export function prototypePortTypeLabel(port: PrototypePort): string {
  return `${port.artifact_type.id}@${port.artifact_type.schema_version}`;
}

export function prototypePortSummary(port: PrototypePort): string {
  const extras = [port.shape, port.variadic ? "variadic" : null].filter(
    Boolean,
  );

  return `${prototypePortTypeLabel(port)}${
    extras.length ? ` · ${extras.join(" · ")}` : ""
  }`;
}

export function prototypePortCountLabel(spec: PrototypeNodeSpec): string {
  return `${spec.inputs.length} in · ${spec.outputs.length} out`;
}

export function prototypeGroupLabel(group: string): string {
  return group.charAt(0).toUpperCase() + group.slice(1);
}

export function prototypeSelectionSizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

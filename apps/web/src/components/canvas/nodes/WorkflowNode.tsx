"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  ExternalLink,
  LoaderCircle,
  Trash2,
  Upload,
} from "lucide-react";

import { artifactContentUrl, type Port } from "@/lib/api";
import { parseArtifactPayload } from "@/lib/artifact-payload";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { encodeHandleId, handleStyle } from "../handles";
import { ARTIFACT_TYPE_COLOR } from "../nodes.css";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  portMetaForPort,
  selectionSizeLabel,
  selectedSourceItems,
  type WorkflowNodeData,
} from "../types";
import {
  artifactTextForPort,
  OutputPortRow,
} from "@/components/canvas/output-port-row";
import {
  DEFAULT_PORT_TREATMENT,
} from "@/components/canvas/output-port-treatment";
import { artifactTypeSpecForKey } from "@/lib/output-port-projection";
import { useNodeRegistry } from "@/hooks/use-api";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

const ACCEPTED_IMAGE_TYPES =
  ".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,image/png,image/jpeg,image/webp,image/tiff,image/bmp";

interface SchemaField {
  name: string;
  title: string;
  description?: string;
  type: "string" | "integer" | "number" | "boolean";
  enumValues?: readonly (string | number)[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  required: boolean;
}

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function resolveSchema(
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
): Record<string, unknown> {
  const reference = schema.$ref;
  if (typeof reference !== "string" || !reference.startsWith("#/$defs/")) {
    return schema;
  }
  const name = reference.slice("#/$defs/".length);
  const definitions = record(root.$defs);
  return record(definitions?.[name]) ?? schema;
}

function schemaFields(rawSchema: unknown): SchemaField[] {
  const root = record(rawSchema);
  const properties = record(root?.properties);
  if (!root || !properties) return [];

  const required = new Set(
    Array.isArray(root.required)
      ? root.required.filter(
          (value): value is string => typeof value === "string",
        )
      : [],
  );

  return Object.entries(properties).flatMap(([name, rawProperty]) => {
    if (
      /api.?key|token|secret/i.test(name) ||
      name === "connector_id" ||
      name === "selection"
    ) {
      return [];
    }

    const propertyRecord = record(rawProperty);
    if (!propertyRecord) return [];
    const property = resolveSchema(propertyRecord, root);
    const enumValues = Array.isArray(property.enum)
      ? property.enum.filter(
          (value): value is string | number =>
            typeof value === "string" || typeof value === "number",
        )
      : undefined;
    const candidateType =
      typeof property.type === "string"
        ? property.type
        : enumValues?.every((value) => typeof value === "number")
          ? "number"
          : "string";
    if (
      candidateType !== "string" &&
      candidateType !== "integer" &&
      candidateType !== "number" &&
      candidateType !== "boolean"
    ) {
      return [];
    }

    return [{
      name,
      title:
        typeof property.title === "string"
          ? property.title
          : name.replaceAll("_", " "),
      description:
        typeof property.description === "string"
          ? property.description
          : undefined,
      type: candidateType,
      enumValues,
      minimum:
        typeof property.minimum === "number" ? property.minimum : undefined,
      maximum:
        typeof property.maximum === "number" ? property.maximum : undefined,
      minLength:
        typeof property.minLength === "number" ? property.minLength : undefined,
      maxLength:
        typeof property.maxLength === "number" ? property.maxLength : undefined,
      pattern: typeof property.pattern === "string" ? property.pattern : undefined,
      required: required.has(name),
    }];
  });
}

const s = stylex.create({
  shell: {
    position: "relative",
    width: "340px",
    overflow: "visible",
    borderWidth: 0,
    borderRadius: 0,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    boxSizing: "border-box",
  },
  selected: {
    boxShadow: tokens.shadowNodeSelected,
  },
  header: {
    minHeight: "40px",
    display: "flex",
    alignItems: "center",
    gap: tokens.space2,
    padding: "8px 10px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurface,
  },
  titleWrap: { minWidth: 0, flex: 1, display: "grid", gap: "2px" },
  title: {
    overflow: "hidden",
    color: tokens.colorText,
    fontSize: tokens.fontSizeMd,
    fontWeight: 600,
    lineHeight: 1.2,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  operator: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.2,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  ports: { backgroundColor: tokens.colorSurface },
  portRow: {
    position: "relative",
    minHeight: "36px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    paddingInline: "10px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  outputRow: { justifyContent: "flex-end", textAlign: "right" },
  portName: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  typeLabel: {
    marginLeft: "auto",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
  },
  outputType: { marginLeft: 0, marginRight: "auto" },
  shape: {
    paddingInline: "3px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: 0,
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    lineHeight: "14px",
  },
  required: { color: tokens.colorWarning, fontSize: tokens.fontSizeSm },
  body: { padding: "10px", backgroundColor: tokens.colorSurface },
  description: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1.45,
  },
  upload: {
    width: "100%",
    minHeight: "33px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorder,
      ":hover": tokens.colorBorderStrong,
    },
    borderRadius: 0,
    backgroundColor: {
      default: tokens.colorSurface,
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
    maxHeight: "154px",
    display: "grid",
    gap: "4px",
    marginTop: tokens.space2,
    overflowY: "auto",
  },
  fileRow: {
    minWidth: 0,
    display: "grid",
    gridTemplateColumns: "18px minmax(0,1fr) auto 22px",
    alignItems: "center",
    gap: "6px",
    padding: "5px 6px",
    borderRadius: 0,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurface,
  },
  fileIndex: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  fileName: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
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
    borderRadius: 0,
    backgroundColor: { default: "transparent", ":hover": tokens.colorDangerHover },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  moreFiles: {
    marginTop: "5px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
  },
  configList: { display: "grid", gap: "9px", marginTop: "10px" },
  field: { display: "grid", gap: "5px" },
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
    paddingInline: tokens.space2,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorder,
      ":focus": tokens.colorAccent,
    },
    borderRadius: 0,
    outline: "none",
    backgroundColor: tokens.colorSurface,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  checkRow: {
    minHeight: "30px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
  },
  check: { accentColor: tokens.colorAccent },
  resultList: {
    maxHeight: "150px",
    display: "grid",
    gap: "5px",
    marginTop: "9px",
    overflowY: "auto",
  },
  result: {
    minWidth: 0,
    display: "grid",
    gridTemplateColumns: "auto minmax(0,1fr) auto",
    alignItems: "baseline",
    gap: tokens.space2,
    padding: "7px 8px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: 0,
    backgroundColor: tokens.colorSurface,
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
  },
  resultPort: { flexShrink: 0, fontWeight: 700, textTransform: "capitalize" },
  resultValue: {
    minWidth: 0,
    overflow: "hidden",
    color: tokens.colorText,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  resultLink: {
    width: "20px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    borderRadius: 0,
    color: { default: tokens.colorMuted, ":hover": tokens.colorText },
  },
  error: {
    marginTop: "9px",
    overflow: "hidden",
    color: tokens.colorDanger,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
});

function portColor(artifactTypeId: string): string {
  return ARTIFACT_TYPE_COLOR[artifactTypeId] ?? tokens.colorAccent;
}

function artifactLabel(port: Port): string {
  const name = port.artifact_type.id.split(".").at(-1) ?? port.artifact_type.id;
  return port.shape === "many" ? `${name}[]` : name;
}

function PortRow({ port }: { port: Port }) {
  const input = port.direction === "input";
  const color = portColor(port.artifact_type.id);
  const contract = artifactLabel(port);
  const accessibleLabel = input
    ? `Input port ${port.name}, accepts ${contract}${port.required ? ", required" : ""}`
    : `Output port ${port.name}, provides ${contract}`;
  const handle = (
    <Handle
      type={input ? "target" : "source"}
      position={input ? Position.Left : Position.Right}
      id={encodeHandleId(portMetaForPort(port))}
      aria-label={accessibleLabel}
      title={input
        ? `${accessibleLabel}. Connect a compatible output here.`
        : `${accessibleLabel}. Drag to a compatible input.`}
      style={handleStyle("50%", color, port.variadic)}
    />
  );

  return (
    <div {...stylex.props(s.portRow, input ? null : s.outputRow)}>
      {input ? handle : null}
      {input ? (
        <>
          <span {...stylex.props(s.portName)}>{port.name}</span>
          {port.required ? <span {...stylex.props(s.required)}>*</span> : null}
          {port.shape === "many" ? <span {...stylex.props(s.shape)}>many</span> : null}
          <span {...stylex.props(s.typeLabel)}>{artifactLabel(port)}</span>
        </>
      ) : (
        <>
          <span {...stylex.props(s.typeLabel, s.outputType)}>{artifactLabel(port)}</span>
          {port.shape === "many" ? <span {...stylex.props(s.shape)}>many</span> : null}
          <span {...stylex.props(s.portName)}>{port.name}</span>
        </>
      )}
      {input ? null : handle}
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
            className="nodrag nowheel"
            checked={value === true}
            {...stylex.props(s.check)}
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
          className="nodrag nowheel"
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          {...stylex.props(s.input)}
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
      ) : (
        <input
          type={
            field.type === "number" || field.type === "integer"
              ? "number"
              : "text"
          }
          className="nodrag nowheel"
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          min={field.minimum}
          max={field.maximum}
          minLength={field.minLength}
          maxLength={field.maxLength}
          pattern={field.pattern}
          step={field.type === "integer" ? 1 : undefined}
          {...stylex.props(s.input)}
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

function RunResults({ data }: { data: WorkflowNodeData }) {
  const artifacts = (data.run?.outputs ?? []).flatMap((output) =>
    output.artifacts.map((artifact) => ({ port: output.port, artifact })),
  );
  if (!artifacts.length) return null;

  return (
    <div {...stylex.props(s.resultList)}>
      {artifacts.map(({ port, artifact }) => {
        const contentUrl = artifactContentUrl(artifact.content_url);
        const fallback = artifact.artifact_type.split(".").at(-1) ?? "artifact";
        return (
          <div key={`${port}-${artifact.artifact_id}`} {...stylex.props(s.result)}>
            <span {...stylex.props(s.resultPort)}>{port}</span>
            <span
              title={artifact.text ?? fallback}
              {...stylex.props(s.resultValue)}
            >
              {artifact.text ?? fallback}
            </span>
            {contentUrl ? (
              <a
                className="nodrag"
                href={contentUrl}
                target="_blank"
                rel="noreferrer"
                aria-label={`Open ${port} artifact`}
                title={`Open ${port} artifact`}
                {...stylex.props(s.resultLink)}
              >
                <ExternalLink size={10} />
              </a>
            ) : null}
          </div>
        );
      })}
    </div>
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
        className="nodrag"
        {...stylex.props(s.hiddenInput)}
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          if (files.length) data.onFilesSelected?.(id, files);
        }}
      />
      <button type="button" className="nodrag" {...stylex.props(s.upload)} onClick={() => inputRef.current?.click()}>
        {data.execution.status === "uploading" ? <LoaderCircle size={12} {...stylex.props(s.spinner)} /> : <Upload size={12} />}
        {data.execution.status === "uploading" ? "Uploading…" : items.length ? "Replace images" : "Choose images"}
      </button>
      {items.length ? (
        <div {...stylex.props(s.fileList)}>
          {items.map((item, index) => (
            <div key={`${item.external_uri}-${index}`} {...stylex.props(s.fileRow)}>
              <span {...stylex.props(s.fileIndex)}>{String(index + 1).padStart(2, "0")}</span>
              <span {...stylex.props(s.fileName)}>{item.display_name}</span>
              <span {...stylex.props(s.fileSize)}>{selectionSizeLabel(item.size_bytes)}</span>
              <button
                type="button"
                className="nodrag"
                aria-label={`Remove ${item.display_name}`}
                title={`Remove ${item.display_name}`}
                {...stylex.props(s.fileRemove)}
                onClick={() => data.onRemoveSelection?.(id, index)}
              >
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
      ) : <p {...stylex.props(s.moreFiles)}>PNG, JPEG, WebP, TIFF or BMP · ordered as selected</p>}
      <RunResults data={data} />
      {data.execution.error ? <div {...stylex.props(s.error)} title={data.execution.error}>{data.execution.error}</div> : null}
    </div>
  );
}

function GenericBody({
  id,
  data,
}: {
  id: string;
  data: WorkflowNodeData;
}) {
  const fields = schemaFields(data.spec.config_schema);

  return (
    <div {...stylex.props(s.body)}>
      <p {...stylex.props(s.description)}>{data.spec.description}</p>
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
      {data.execution.error ? <div {...stylex.props(s.error)} title={data.execution.error}>{data.execution.error}</div> : null}
    </div>
  );
}

function WorkflowNodeCard({ id, data, selected }: NodeProps<WorkflowNode>) {
  const { data: registry } = useNodeRegistry();
  const artifactTypes = registry?.artifact_types ?? [];

  return (
    <article {...stylex.props(s.shell, selected ? s.selected : null)}>
      <header {...stylex.props(s.header)}>
        <span {...stylex.props(s.titleWrap)}>
          <span {...stylex.props(s.title)} title={data.spec.title}>{data.spec.title}</span>
          <span {...stylex.props(s.operator)}>{data.spec.operator_id}</span>
        </span>
      </header>
      <div {...stylex.props(s.ports)}>
        {data.spec.inputs.map((port) => <PortRow key={`in-${port.name}`} port={port} />)}
      </div>
      {data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID ? (
        <SourceBody id={id} data={data} />
      ) : (
        <GenericBody id={id} data={data} />
      )}
      <div {...stylex.props(s.ports)}>
        {data.spec.outputs.map((port) => {
          const artifactSpec = artifactTypeSpecForKey(
            artifactTypes,
            port.artifact_type.id,
            port.artifact_type.schema_version,
          );
          const fieldProjections = artifactSpec?.field_projections ?? [];
          const artifactText = artifactTextForPort(data.run?.outputs, port.name);
          const hasPayload = Boolean(parseArtifactPayload(artifactText));
          const inspectable =
            fieldProjections.length > 0 || hasPayload;

          return (
            <OutputPortRow
              key={`out-${port.name}`}
              port={port}
              nodeTitle={data.spec.title}
              artifactText={artifactText}
              fieldProjections={fieldProjections}
              inspectable={inspectable}
              treatment={
                data.outputTreatments[port.name] ?? DEFAULT_PORT_TREATMENT
              }
              onTreatmentChange={(treatment) =>
                data.onOutputTreatmentChange?.(id, port.name, treatment)
              }
            />
          );
        })}
      </div>
    </article>
  );
}

export default WorkflowNodeCard;

"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  AlertCircle,
  Asterisk,
  Check,
  Circle,
  ExternalLink,
  FileDown,
  Hash,
  LoaderCircle,
  Rows3,
  ScanLine,
  Sigma,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";

import { prototypeContentUrl, type PrototypePort } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { encodeHandleId, handleStyle } from "../handles";
import { ARTIFACT_TYPE_COLOR } from "../nodes.css";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  PROTOTYPE_NODE_TYPE,
  portMetaForPrototypePort,
  prototypeSelectionSizeLabel,
  selectedPrototypeItems,
  type PrototypeNodeData,
} from "../types";

type PrototypeFlowNode = Node<PrototypeNodeData, typeof PROTOTYPE_NODE_TYPE>;

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
    width: "306px",
    overflow: "visible",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "7px",
    backgroundColor: tokens.colorSurface,
    boxShadow: "0 14px 30px rgba(0,0,0,0.32)",
    color: tokens.colorText,
    fontSize: "12px",
    transitionProperty: "border-color, box-shadow",
    transitionDuration: "150ms",
  },
  selected: {
    borderColor: tokens.colorAccent,
    boxShadow: "0 0 0 1px rgba(128,103,232,0.5), 0 18px 42px rgba(0,0,0,0.44)",
  },
  nodeNumber: {
    position: "absolute",
    top: "-18px",
    right: "7px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "8px",
  },
  header: {
    height: "39px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    paddingInline: "9px 7px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
    borderLeftWidth: 3,
    borderLeftStyle: "solid",
    borderRadius: "6px 6px 0 0",
    backgroundColor: tokens.colorSurfaceRaised,
  },
  headerIcon: {
    width: "20px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    flexShrink: 0,
    borderRadius: "4px",
    backgroundColor: "rgba(255,255,255,0.055)",
  },
  titleWrap: { minWidth: 0, flex: 1, display: "grid", gap: "1px" },
  title: {
    overflow: "hidden",
    color: "#eef0f2",
    fontSize: "11.5px",
    fontWeight: 720,
    letterSpacing: "-0.01em",
    lineHeight: 1.15,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  operator: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "7.5px",
    lineHeight: 1.15,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  ports: { backgroundColor: "#1e2023" },
  portRow: {
    position: "relative",
    minHeight: "34px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    paddingInline: "10px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: "rgba(255,255,255,0.045)",
  },
  outputRow: { justifyContent: "flex-end", textAlign: "right" },
  portName: {
    overflow: "hidden",
    color: "#e4e6e8",
    fontSize: "10.5px",
    fontWeight: 650,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  typeLabel: {
    marginLeft: "auto",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "7.5px",
    textOverflow: "ellipsis",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
  },
  outputType: { marginLeft: 0, marginRight: "auto" },
  shape: {
    paddingInline: "3px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "rgba(255,255,255,0.11)",
    borderRadius: "3px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "7px",
    lineHeight: "12px",
  },
  required: { color: tokens.colorWarning, fontSize: "9px" },
  body: { padding: "10px", backgroundColor: tokens.colorSurface },
  description: { color: tokens.colorMuted, fontSize: "9.5px", lineHeight: 1.45 },
  upload: {
    width: "100%",
    minHeight: "33px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: { default: "#464a50", ":hover": "#5a5f67" },
    borderRadius: "5px",
    backgroundColor: { default: "#2a2d31", ":hover": "#31353a" },
    color: "#e3e5e8",
    cursor: "pointer",
    fontSize: "10px",
    fontWeight: 650,
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
    marginTop: "8px",
    overflowY: "auto",
  },
  fileRow: {
    minWidth: 0,
    display: "grid",
    gridTemplateColumns: "18px minmax(0,1fr) auto 22px",
    alignItems: "center",
    gap: "6px",
    padding: "5px 6px",
    borderRadius: "4px",
    backgroundColor: "rgba(255,255,255,0.035)",
  },
  fileIndex: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "8px",
  },
  fileName: {
    overflow: "hidden",
    color: "#d4d7da",
    fontSize: "9px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  fileSize: { color: tokens.colorSubtle, fontSize: "7.5px" },
  fileRemove: {
    width: "22px",
    height: "22px",
    display: "grid",
    placeItems: "center",
    borderWidth: 0,
    borderRadius: "4px",
    backgroundColor: { default: "transparent", ":hover": "rgba(232,105,105,0.12)" },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  moreFiles: { marginTop: "5px", color: tokens.colorSubtle, fontSize: "8px" },
  configList: { display: "grid", gap: "9px", marginTop: "10px" },
  field: { display: "grid", gap: "5px" },
  fieldLabel: {
    display: "flex",
    alignItems: "center",
    gap: "3px",
    color: "#cfd2d6",
    fontSize: "9.5px",
    fontWeight: 650,
    textTransform: "capitalize",
  },
  fieldDescription: {
    color: tokens.colorSubtle,
    fontSize: "8.5px",
    lineHeight: 1.4,
  },
  input: {
    width: "100%",
    height: "31px",
    paddingInline: "8px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorderStrong,
      ":focus": tokens.colorAccent,
    },
    borderRadius: "5px",
    outline: "none",
    backgroundColor: "#191b1e",
    color: tokens.colorText,
    fontSize: "10px",
  },
  checkRow: {
    minHeight: "30px",
    display: "flex",
    alignItems: "center",
    gap: "7px",
    color: "#cfd2d6",
    fontSize: "9.5px",
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
    gap: "8px",
    padding: "7px 8px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "rgba(67,197,158,0.22)",
    borderRadius: "5px",
    backgroundColor: "rgba(67,197,158,0.055)",
    color: "#a8d8ca",
    fontSize: "8.5px",
  },
  resultPort: { flexShrink: 0, fontWeight: 700, textTransform: "capitalize" },
  resultValue: {
    minWidth: 0,
    overflow: "hidden",
    color: "#d2e8e1",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  resultLink: {
    width: "20px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    borderRadius: "4px",
    color: { default: "#a8d8ca", ":hover": "#d2e8e1" },
  },
  error: {
    marginTop: "9px",
    overflow: "hidden",
    color: "#efaaaa",
    fontSize: "8.5px",
    lineHeight: 1.4,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  footer: {
    minHeight: "28px",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    paddingInline: "9px",
    borderTopWidth: 1,
    borderTopStyle: "solid",
    borderTopColor: tokens.colorBorder,
    borderRadius: "0 0 6px 6px",
    backgroundColor: "#1b1d20",
    color: tokens.colorMuted,
    fontSize: "8.5px",
  },
  statusSuccess: { color: tokens.colorSuccess },
  statusError: { color: tokens.colorDanger },
  statusActive: { color: tokens.colorInfo },
  footerMeta: {
    marginLeft: "auto",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
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

function artifactLabel(port: PrototypePort): string {
  const name = port.artifact_type.id.split(".").at(-1) ?? port.artifact_type.id;
  return port.shape === "many" ? `${name}[]` : name;
}

function operatorAccent(operatorId: string): string {
  if (operatorId.startsWith("source.")) return "#43c59e";
  if (operatorId.startsWith("ocr.")) return "#9a7cf2";
  if (operatorId.startsWith("arithmetic.")) return "#a78bfa";
  if (operatorId.includes("csv") || operatorId.includes("export")) return "#f0a65a";
  return "#57a5ef";
}

function OperatorIcon({ operatorId }: { operatorId: string }) {
  if (operatorId.startsWith("source.")) return <Upload size={12} />;
  if (operatorId.startsWith("ocr.")) return <ScanLine size={12} />;
  if (operatorId === "arithmetic.number") return <Hash size={12} />;
  if (operatorId === "arithmetic.add_subtract") return <Sigma size={12} />;
  if (operatorId === "arithmetic.multiply") return <Asterisk size={12} />;
  if (operatorId.includes("extract")) return <Table2 size={12} />;
  if (operatorId.includes("merge")) return <Rows3 size={12} />;
  return <FileDown size={12} />;
}

function PortRow({ port }: { port: PrototypePort }) {
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
      id={encodeHandleId(portMetaForPrototypePort(port))}
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

function RunResults({ data }: { data: PrototypeNodeData }) {
  const artifacts = (data.run?.outputs ?? []).flatMap((output) =>
    output.artifacts.map((artifact) => ({ port: output.port, artifact })),
  );
  if (!artifacts.length) return null;

  return (
    <div {...stylex.props(s.resultList)}>
      {artifacts.map(({ port, artifact }) => {
        const contentUrl = prototypeContentUrl(artifact.content_url);
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

function SourceBody({ id, data }: { id: string; data: PrototypeNodeData }) {
  const items = selectedPrototypeItems(data);
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
              <span {...stylex.props(s.fileSize)}>{prototypeSelectionSizeLabel(item.size_bytes)}</span>
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

function GenericBody({ id, data }: { id: string; data: PrototypeNodeData }) {
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
      <RunResults data={data} />
      {data.execution.error ? <div {...stylex.props(s.error)} title={data.execution.error}>{data.execution.error}</div> : null}
    </div>
  );
}

function Status({ data }: { data: PrototypeNodeData }) {
  const status = data.execution.status;
  if (status === "uploading" || status === "running") {
    return <><LoaderCircle size={10} {...stylex.props(s.statusActive, s.spinner)} />{status === "uploading" ? "Uploading" : "Running"}</>;
  }
  if (status === "failed") {
    return <><AlertCircle size={10} {...stylex.props(s.statusError)} />Failed</>;
  }
  if (status === "succeeded") {
    return <><Check size={10} {...stylex.props(s.statusSuccess)} />Succeeded</>;
  }
  if (status === "skipped") {
    return <><Circle size={9} />Skipped</>;
  }
  return <><Circle size={9} />Ready</>;
}

function PrototypeNode({ id, data, selected }: NodeProps<PrototypeFlowNode>) {
  const accent = operatorAccent(data.spec.operator_id);
  return (
    <article {...stylex.props(s.shell, selected ? s.selected : null)}>
      <span {...stylex.props(s.nodeNumber)}>{id}</span>
      <header {...stylex.props(s.header)} style={{ borderLeftColor: accent }}>
        <span {...stylex.props(s.headerIcon)} style={{ color: accent }}><OperatorIcon operatorId={data.spec.operator_id} /></span>
        <span {...stylex.props(s.titleWrap)}>
          <span {...stylex.props(s.title)} title={data.spec.title}>{data.spec.title}</span>
          <span {...stylex.props(s.operator)}>{data.spec.operator_id}</span>
        </span>
      </header>
      <div {...stylex.props(s.ports)}>
        {data.spec.inputs.map((port) => <PortRow key={`in-${port.name}`} port={port} />)}
      </div>
      {data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID ? <SourceBody id={id} data={data} /> : <GenericBody id={id} data={data} />}
      <div {...stylex.props(s.ports)}>
        {data.spec.outputs.map((port) => <PortRow key={`out-${port.name}`} port={port} />)}
      </div>
      <footer {...stylex.props(s.footer)}>
        <Status data={data} />
        <span {...stylex.props(s.footerMeta)}>v{data.spec.operator_version}</span>
      </footer>
    </article>
  );
}

export default PrototypeNode;

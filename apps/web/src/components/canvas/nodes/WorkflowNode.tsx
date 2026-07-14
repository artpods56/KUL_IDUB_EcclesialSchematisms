"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";
import {
  Handle,
  Position,
  useUpdateNodeInternals,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { CircleHelp, LoaderCircle, Trash2, Upload, X } from "lucide-react";

import type { Port } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { schemaFields, type SchemaField } from "../config-schema";
import { encodeHandleId, handleStyle } from "../handles";
import { ARTIFACT_TYPE_COLOR } from "../nodes.css";
import {
  LOCAL_UPLOAD_OPERATOR_ID,
  WORKFLOW_NODE_TYPE,
  effectivePortShape,
  portMetaForPort,
  selectionSizeLabel,
  selectedSourceItems,
  type WorkflowNodeData,
} from "../types";
import { ArtifactsAppendix } from "./ArtifactsAppendix";
import { PortTypePopover } from "./type-inspector";

type WorkflowNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;

const ACCEPTED_IMAGE_TYPES =
  ".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,image/png,image/jpeg,image/webp,image/tiff,image/bmp";

const s = stylex.create({
  shell: {
    position: "relative",
    width: "300px",
    overflow: "visible",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    boxSizing: "border-box",
  },
  selected: {
    boxShadow: `${tokens.shadowNode}, 0 0 0 2px ${tokens.colorAccentBorder}`,
  },
  header: {
    display: "grid",
    gap: "2px",
    padding: "12px 16px 12px 12px",
  },
  titleRow: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "4px",
  },
  headerButton: {
    width: "22px",
    height: "22px",
    display: "grid",
    placeItems: "center",
    flexShrink: 0,
    borderWidth: 0,
    borderRadius: "9999px",
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorText },
    cursor: "pointer",
  },
  removeButton: {
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
  },
  title: {
    minWidth: 0,
    overflow: "hidden",
    marginLeft: "4px",
    color: tokens.colorText,
    fontSize: tokens.fontSizeMd,
    fontWeight: 650,
    letterSpacing: "-0.01em",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  operator: {
    overflow: "hidden",
    marginLeft: "56px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  helpPopup: {
    width: "280px",
    display: "grid",
    gap: "6px",
    padding: "11px 13px",
    borderRadius: "12px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
  helpTitle: { fontSize: tokens.fontSizeSm, fontWeight: 750 },
  helpDescription: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.5,
  },
  tabs: {
    display: "grid",
    gap: "5px",
    paddingBlock: "2px",
  },
  tabsOutput: {
    display: "grid",
    gap: "5px",
    paddingTop: "2px",
    paddingBottom: "14px",
  },
  tabRow: {
    position: "relative",
    display: "flex",
    minHeight: "28px",
    alignItems: "center",
  },
  tabRowOut: { justifyContent: "flex-end" },
  tab: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
    maxWidth: "calc(100% - 12px)",
    height: "24px",
    paddingInline: "14px 12px",
    borderWidth: 0,
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHoverStrong,
    },
    color: tokens.colorTextEmphasis,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
  },
  tabLabel: {
    minWidth: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  tabShape: { flexShrink: 0 },
  tabIn: { borderRadius: "0 9999px 9999px 0" },
  tabOut: {
    flexDirection: "row-reverse",
    paddingInline: "12px 14px",
    borderRadius: "9999px 0 0 9999px",
  },
  dot: {
    width: "6px",
    height: "6px",
    flexShrink: 0,
    borderRadius: "9999px",
  },
  body: {
    display: "grid",
    gap: "9px",
    padding: "0 16px 14px",
  },
  upload: {
    width: "100%",
    minHeight: "34px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    borderWidth: 0,
    borderRadius: "10px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
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
    maxHeight: "132px",
    display: "grid",
    gap: "5px",
    overflowY: "auto",
  },
  fileRow: {
    minWidth: 0,
    display: "grid",
    gridTemplateColumns: "18px minmax(0,1fr) auto 22px",
    alignItems: "center",
    gap: "6px",
    minHeight: "28px",
    paddingInline: "10px 4px",
    borderRadius: "8px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  fileIndex: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  fileName: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 550,
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
    borderRadius: "6px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  moreFiles: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  configList: { display: "grid", gap: "9px" },
  field: { display: "grid", gap: "4px" },
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
    paddingInline: "10px",
    borderWidth: 0,
    borderRadius: "8px",
    outline: {
      default: "none",
      ":focus": `2px solid ${tokens.colorAccentBorder}`,
    },
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  textarea: {
    height: "96px",
    paddingBlock: "8px",
    lineHeight: 1.45,
    resize: "none",
  },
  checkRow: {
    minHeight: "30px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 650,
  },
  check: { accentColor: tokens.colorAccent },
  required: { color: tokens.colorWarning, fontSize: tokens.fontSizeSm },
  error: {
    overflow: "hidden",
    color: tokens.colorDanger,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  emptyBody: {
    padding: "0 16px 14px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  spacer: { minHeight: "4px" },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
});

function nodeInteractionProps(props: ReturnType<typeof stylex.props>) {
  return {
    ...props,
    className: `nodrag nowheel${props.className ? ` ${props.className}` : ""}`,
  };
}

function PortTab({ port, shape }: { port: Port; shape: Port["shape"] }) {
  const input = port.direction === "input";
  const visibleName = port.title ?? port.name;
  const color = ARTIFACT_TYPE_COLOR[port.artifact_type.id] ?? tokens.colorAccent;
  const artifactContract = `${port.artifact_type.id}@${port.artifact_type.schema_version}`;
  const effectiveContract =
    shape === "many" ? `list[${artifactContract}]` : artifactContract;
  const accessibleLabel = input
    ? `Input port ${visibleName}, accepts ${effectiveContract}${port.required ? ", required" : ""}`
    : `Output port ${visibleName}, provides ${effectiveContract}`;

  return (
    <div {...stylex.props(s.tabRow, input ? null : s.tabRowOut)}>
      <PortTypePopover port={port} shape={shape}>
        <Popover.Trigger
          type="button"
          aria-label={`Inspect ${visibleName} type`}
          title={port.description ?? `Inspect ${visibleName} type`}
          {...nodeInteractionProps(
            stylex.props(s.tab, input ? s.tabIn : s.tabOut),
          )}
        >
          <span {...stylex.props(s.dot)} style={{ backgroundColor: color }} />
          <span {...stylex.props(s.tabLabel)}>{visibleName}</span>
          {input && port.required ? (
            <span {...stylex.props(s.required, s.tabShape)}>*</span>
          ) : null}
          {shape === "many" ? (
            <span {...stylex.props(s.tabShape)}>· many</span>
          ) : null}
        </Popover.Trigger>
      </PortTypePopover>
      <Handle
        type={input ? "target" : "source"}
        position={input ? Position.Left : Position.Right}
        id={encodeHandleId(portMetaForPort(port, input ? port.shape : shape))}
        aria-label={accessibleLabel}
        title={
          input
            ? `${accessibleLabel}. Connect a compatible output here.${port.description ? ` ${port.description}` : ""}`
            : `${accessibleLabel}. Drag to a compatible input.${port.description ? ` ${port.description}` : ""}`
        }
        style={handleStyle("50%", color, port.variadic)}
      />
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
            checked={value === true}
            {...nodeInteractionProps(stylex.props(s.check))}
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
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          {...nodeInteractionProps(stylex.props(s.input))}
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
      ) : field.type === "string" && field.format === "textarea" ? (
        <textarea
          value={typeof value === "string" ? value : ""}
          minLength={field.minLength}
          maxLength={field.maxLength}
          {...nodeInteractionProps(stylex.props(s.input, s.textarea))}
          onChange={(event) => onChange(event.currentTarget.value)}
        />
      ) : (
        <input
          type={
            field.type === "number" || field.type === "integer"
              ? "number"
              : "text"
          }
          value={
            typeof value === "string" || typeof value === "number" ? value : ""
          }
          min={field.minimum}
          max={field.maximum}
          minLength={field.minLength}
          maxLength={field.maxLength}
          pattern={field.pattern}
          step={field.type === "integer" ? 1 : undefined}
          {...nodeInteractionProps(stylex.props(s.input))}
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
        {...nodeInteractionProps(stylex.props(s.hiddenInput))}
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          if (files.length) data.onFilesSelected?.(id, files);
        }}
      />
      <button
        type="button"
        {...nodeInteractionProps(stylex.props(s.upload))}
        onClick={() => inputRef.current?.click()}
      >
        {data.execution.status === "uploading" ? (
          <LoaderCircle size={12} {...stylex.props(s.spinner)} />
        ) : (
          <Upload size={12} />
        )}
        {data.execution.status === "uploading"
          ? "Uploading…"
          : items.length
            ? "Replace images"
            : "Choose images"}
      </button>
      {items.length ? (
        <div {...nodeInteractionProps(stylex.props(s.fileList))}>
          {items.map((item, index) => (
            <div
              key={`${item.external_uri}-${index}`}
              {...stylex.props(s.fileRow)}
            >
              <span {...stylex.props(s.fileIndex)}>
                {String(index + 1).padStart(2, "0")}
              </span>
              <span {...stylex.props(s.fileName)}>{item.display_name}</span>
              <span {...stylex.props(s.fileSize)}>
                {selectionSizeLabel(item.size_bytes)}
              </span>
              <button
                type="button"
                aria-label={`Remove ${item.display_name}`}
                title={`Remove ${item.display_name}`}
                {...nodeInteractionProps(stylex.props(s.fileRemove))}
                onClick={() => data.onRemoveSelection?.(id, index)}
              >
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p {...stylex.props(s.moreFiles)}>
          PNG, JPEG, WebP, TIFF or BMP · ordered as selected
        </p>
      )}
      {data.execution.error ? (
        <div {...stylex.props(s.error)} title={data.execution.error}>
          {data.execution.error}
        </div>
      ) : null}
    </div>
  );
}

function GenericBody({ id, data }: { id: string; data: WorkflowNodeData }) {
  const fields = schemaFields(data.spec.config_schema);
  if (!fields.length && !data.execution.error) return null;

  return (
    <div {...stylex.props(s.body)}>
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
      {data.execution.error ? (
        <div {...stylex.props(s.error)} title={data.execution.error}>
          {data.execution.error}
        </div>
      ) : null}
    </div>
  );
}

function NodeHeader({ id, data }: { id: string; data: WorkflowNodeData }) {
  return (
    <header {...stylex.props(s.header)}>
      <span {...stylex.props(s.titleRow)}>
        <Popover.Root>
          <Popover.Trigger
            type="button"
            aria-label={`About ${data.spec.title}`}
            title={`About ${data.spec.title}`}
            {...nodeInteractionProps(stylex.props(s.headerButton))}
          >
            <CircleHelp size={13} />
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Positioner side="top" align="start" sideOffset={7}>
              <Popover.Popup
                {...nodeInteractionProps(stylex.props(s.helpPopup))}
              >
                <span {...stylex.props(s.helpTitle)}>{data.spec.title}</span>
                <span {...stylex.props(s.helpDescription)}>
                  {data.spec.description ||
                    "No description is available for this node."}
                </span>
              </Popover.Popup>
            </Popover.Positioner>
          </Popover.Portal>
        </Popover.Root>
        <button
          type="button"
          aria-label={`Remove ${data.spec.title}`}
          title={`Remove ${data.spec.title}`}
          {...nodeInteractionProps(
            stylex.props(s.headerButton, s.removeButton),
          )}
          onClick={() => data.onRemoveNode?.(id)}
        >
          <X size={13} />
        </button>
        <span {...stylex.props(s.title)} title={data.spec.title}>
          {data.spec.title}
        </span>
      </span>
      <span {...stylex.props(s.operator)} title={data.spec.description}>
        {data.spec.operator_id}@{data.spec.operator_version}
      </span>
    </header>
  );
}

function WorkflowNodeCard({ id, data, selected }: NodeProps<WorkflowNode>) {
  const fields = schemaFields(data.spec.config_schema);
  const isUpload = data.spec.operator_id === LOCAL_UPLOAD_OPERATOR_ID;
  const hasConfig = fields.length > 0;
  const producedArtifactCount = (data.run?.outputs ?? []).reduce(
    (count, output) => count + output.artifacts.length,
    0,
  );
  const updateNodeInternals = useUpdateNodeInternals();

  React.useEffect(() => {
    const frame = window.requestAnimationFrame(() => updateNodeInternals(id));
    return () => window.cancelAnimationFrame(frame);
  }, [
    data.mappedInputPort,
    data.spec.inputs.length,
    data.spec.outputs.length,
    fields.length,
    producedArtifactCount,
    id,
    updateNodeInternals,
  ]);

  return (
    <>
      <article {...stylex.props(s.shell, selected ? s.selected : null)}>
        <NodeHeader id={id} data={data} />
        {data.spec.inputs.length ? (
          <div {...stylex.props(s.tabs)}>
            {data.spec.inputs.map((port) => (
              <PortTab
                key={`in-${port.name}`}
                port={port}
                shape={effectivePortShape(data, port)}
              />
            ))}
          </div>
        ) : null}
        {isUpload ? (
          <SourceBody id={id} data={data} />
        ) : hasConfig || data.execution.error ? (
          <GenericBody id={id} data={data} />
        ) : (
          <div {...stylex.props(s.spacer)} aria-hidden />
        )}
        {data.spec.outputs.length ? (
          <div {...stylex.props(s.tabsOutput)}>
            {data.spec.outputs.map((port) => (
              <PortTab
                key={`out-${port.name}`}
                port={port}
                shape={effectivePortShape(data, port)}
              />
            ))}
          </div>
        ) : null}
        {!isUpload &&
        !hasConfig &&
        !data.execution.error &&
        !data.spec.inputs.length &&
        !data.spec.outputs.length ? (
          <p {...stylex.props(s.emptyBody)}>
            {data.spec.description || "No configuration for this operator."}
          </p>
        ) : null}
      </article>
      <ArtifactsAppendix data={data} />
    </>
  );
}

export default WorkflowNodeCard;

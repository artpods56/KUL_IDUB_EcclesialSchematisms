"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Handle, Position } from "@xyflow/react";
import { Popover } from "@base-ui/react/popover";
import { ChevronRight } from "lucide-react";

import type { FieldProjection, Port, RunNodeResult } from "@/lib/api";
import { parseArtifactPayload } from "@/lib/artifact-payload";
import {
  formatArtifactTypeKey,
  wireIntentForPath,
} from "@/lib/output-port-projection";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { encodeHandleId, handleStyle } from "@/components/canvas/handles";
import { ARTIFACT_TYPE_COLOR } from "@/components/canvas/nodes.css";
import { portMetaForPort } from "@/components/canvas/types";
import { ArtifactTree } from "./artifact-tree";
import {
  DEFAULT_PORT_TREATMENT,
  isConfigured,
  treatmentFromPath,
  treatmentFromProjection,
  treatmentLabel,
  type OutputPortTreatment,
} from "./output-port-treatment";

const s = stylex.create({
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
  portCell: {
    minWidth: 0,
    flex: 1,
    display: "flex",
    alignItems: "center",
    gap: "7px",
    padding: "4px 6px",
    margin: "-4px -6px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "transparent",
    backgroundColor: "transparent",
    color: "inherit",
    cursor: "pointer",
    textAlign: "inherit",
  },
  portCellActive: {
    borderColor: tokens.colorAccentBorder,
    backgroundColor: tokens.colorAccentSoft,
  },
  portCellConfigured: {
    borderColor: tokens.colorBorderStrong,
  },
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
  chevron: { color: tokens.colorSubtle, flexShrink: 0 },
  popup: {
    width: "min(380px, calc(100vw - 24px))",
    maxHeight: "min(520px, calc(100vh - 80px))",
    display: "grid",
    gridTemplateRows: "auto auto 1fr auto",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    fontSize: tokens.fontSizeSm,
    color: tokens.colorText,
    zIndex: 40,
  },
  header: {
    display: "grid",
    gap: "3px",
    padding: "10px 12px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  title: { fontSize: tokens.fontSizeMd, fontWeight: 700 },
  meta: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  section: {
    padding: "10px 12px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
    display: "grid",
    gap: "6px",
  },
  sectionTitle: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  treeBody: {
    overflowY: "auto",
    padding: "10px 12px",
    minHeight: "120px",
  },
  footer: {
    padding: "10px 12px",
    borderTopWidth: 1,
    borderTopStyle: "solid",
    borderTopColor: tokens.colorBorder,
    display: "grid",
    gap: "6px",
  },
  option: {
    width: "100%",
    display: "grid",
    gap: "2px",
    padding: "8px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurface,
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  optionActive: {
    borderColor: tokens.colorAccentBorder,
    backgroundColor: tokens.colorAccentSoft,
  },
  optionTitle: { fontWeight: 650, fontSize: tokens.fontSizeSm },
  optionDescription: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.4,
  },
  summary: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  badge: {
    display: "inline-flex",
    width: "fit-content",
    padding: "2px 6px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
});

function portColor(artifactTypeId: string): string {
  return ARTIFACT_TYPE_COLOR[artifactTypeId] ?? tokens.colorAccent;
}

function artifactLabel(port: Port): string {
  const name = port.artifact_type.id.split(".").at(-1) ?? port.artifact_type.id;
  return port.shape === "many" ? `${name}[]` : name;
}

function wireShapeLabel(
  port: Port,
  treatment: OutputPortTreatment,
): "ONE" | "MANY" {
  if (!treatment.projectionPath.length) {
    return port.shape === "many" ? "MANY" : "ONE";
  }
  return "ONE";
}

export function OutputPortRow({
  port,
  nodeTitle,
  artifactText,
  fieldProjections,
  inspectable,
  treatment,
  onTreatmentChange,
}: {
  port: Port;
  nodeTitle: string;
  artifactText?: string | null;
  fieldProjections: readonly FieldProjection[];
  inspectable: boolean;
  treatment: OutputPortTreatment;
  onTreatmentChange: (treatment: OutputPortTreatment) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const color = portColor(port.artifact_type.id);
  const contract = artifactLabel(port);
  const payload = parseArtifactPayload(artifactText);
  const selectedPath = [...treatment.projectionPath];
  const intent = selectedPath.length
    ? wireIntentForPath(port.name, selectedPath, fieldProjections)
    : null;
  const configured = isConfigured(treatment);
  const wireShape = wireShapeLabel(port, treatment);

  const openInspector = React.useCallback(() => {
    if (!inspectable) return;
    setOpen(true);
  }, [inspectable]);

  const handle = (
    <Handle
      type="source"
      position={Position.Right}
      id={encodeHandleId(portMetaForPort(port))}
      aria-label={`Output port ${port.name}, provides ${contract}`}
      title={`Output port ${port.name}. Drag to a compatible input.`}
      style={handleStyle("50%", color, port.variadic)}
    />
  );

  if (!inspectable) {
    return (
      <div {...stylex.props(s.portRow, s.outputRow)}>
        <button
          type="button"
          className="nodrag nowheel"
          disabled
          {...stylex.props(s.portCell)}
        >
          <span {...stylex.props(s.typeLabel, s.outputType)}>{contract}</span>
          {port.shape === "many" ? <span {...stylex.props(s.shape)}>many</span> : null}
          <span {...stylex.props(s.portName)}>{port.name}</span>
        </button>
        {handle}
      </div>
    );
  }

  const emitOptions: Array<{
    key: string;
    active: boolean;
    title: string;
    description: string;
    onSelect: () => void;
  }> = [
    {
      key: "whole",
      active: !selectedPath.length,
      title: "Whole artifact",
      description: `Emit ${formatArtifactTypeKey(port.artifact_type)} as declared on the port.`,
      onSelect: () => onTreatmentChange(DEFAULT_PORT_TREATMENT),
    },
    ...fieldProjections.map((projection) => ({
      key: projection.path.join("."),
      active:
        selectedPath.length === projection.path.length &&
        projection.path.every((segment, index) => segment === selectedPath[index]),
      title: projection.title,
      description: `Project ${projection.path.join(".")} → ${formatArtifactTypeKey(projection.target_artifact_type)}`,
      onSelect: () => onTreatmentChange(treatmentFromProjection(projection)),
    })),
  ];

  return (
    <div {...stylex.props(s.portRow, s.outputRow)}>
      <Popover.Root open={open} onOpenChange={setOpen} modal="trap-focus">
        <Popover.Trigger
          type="button"
          className="nodrag nowheel"
          nativeButton={false}
          title="Inspect output and choose projection before wiring"
          {...stylex.props(
            s.portCell,
            open ? s.portCellActive : null,
            configured ? s.portCellConfigured : null,
          )}
          onContextMenu={(event: React.MouseEvent) => {
            event.preventDefault();
            openInspector();
          }}
        >
          <span {...stylex.props(s.typeLabel, s.outputType)}>{contract}</span>
          {port.shape === "many" ? <span {...stylex.props(s.shape)}>many</span> : null}
          <span {...stylex.props(s.portName)}>
            {configured
              ? treatmentLabel(port.name, treatment, fieldProjections)
              : port.name}
          </span>
          <ChevronRight size={12} {...stylex.props(s.chevron)} />
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Positioner side="right" align="start" sideOffset={10}>
            <Popover.Popup
              className="nodrag nowheel"
              {...stylex.props(s.popup)}
              onClick={(event) => event.stopPropagation()}
            >
              <header {...stylex.props(s.header)}>
                <span {...stylex.props(s.title)}>{nodeTitle}</span>
                <span {...stylex.props(s.meta)}>
                  Output · {port.name} · {port.artifact_type.id}@
                  {port.artifact_type.schema_version}
                </span>
              </header>

              <div {...stylex.props(s.section)}>
                <span {...stylex.props(s.sectionTitle)}>Emit as</span>
                {emitOptions.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    {...stylex.props(
                      s.option,
                      option.active ? s.optionActive : null,
                    )}
                    onClick={option.onSelect}
                  >
                    <span {...stylex.props(s.optionTitle)}>{option.title}</span>
                    <span {...stylex.props(s.optionDescription)}>
                      {option.description}
                    </span>
                  </button>
                ))}
              </div>

              <div {...stylex.props(s.treeBody)}>
                <span {...stylex.props(s.sectionTitle)}>Inspect value</span>
                {payload ? (
                  <ArtifactTree
                    value={payload}
                    selectedPath={selectedPath}
                    onSelectPath={(path) =>
                      onTreatmentChange(treatmentFromPath(path))
                    }
                    defaultExpandedDepth={1}
                  />
                ) : (
                  <span {...stylex.props(s.summary)}>
                    Run the workflow to inspect payload values. Declared
                    projections above can still be chosen before wiring.
                  </span>
                )}
              </div>

              <footer {...stylex.props(s.footer)}>
                <span {...stylex.props(s.summary)}>
                  Edge would carry{" "}
                  <span {...stylex.props(s.badge)}>
                    {treatmentLabel(port.name, treatment, fieldProjections)}
                  </span>{" "}
                  as{" "}
                  <span {...stylex.props(s.badge)}>{wireShape}</span>
                </span>
                {intent?.declared ? (
                  <span {...stylex.props(s.summary)}>
                    Declared projection to {intent.targetType}.
                  </span>
                ) : intent ? (
                  <span {...stylex.props(s.summary)}>
                    No declared projection for this path — wiring may require a
                    backend contract.
                  </span>
                ) : (
                  <span {...stylex.props(s.summary)}>
                    Choose a declared projection or explore nested fields after
                    a run.
                  </span>
                )}
              </footer>
            </Popover.Popup>
          </Popover.Positioner>
        </Popover.Portal>
      </Popover.Root>
      {handle}
    </div>
  );
}

export function artifactTextForPort(
  outputs: RunNodeResult["outputs"] | undefined,
  portName: string,
): string | null | undefined {
  const output = outputs?.find((item) => item.port === portName);
  return output?.artifacts[0]?.text;
}

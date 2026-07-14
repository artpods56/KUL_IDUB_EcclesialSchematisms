"use client";

import * as stylex from "@stylexjs/stylex";
import {
  Handle,
  Position,
  type Node,
  type NodeProps,
} from "@xyflow/react";

import type { Port } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { encodeHandleId, handleStyle } from "../../handles";
import { ARTIFACT_TYPE_COLOR } from "../../nodes.css";
import {
  WORKFLOW_NODE_TYPE,
  portMetaForPort,
  type WorkflowNodeData,
} from "../../types";

type BandsTintNode = Node<WorkflowNodeData, typeof WORKFLOW_NODE_TYPE>;
type BandsTintProps = NodeProps<BandsTintNode>;

/**
 * Archived node exploration: input and output ports become full-width tinted
 * bands around a deliberately small node body. This component is not registered.
 */

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
  },
  selected: {
    boxShadow: `${tokens.shadowNode}, 0 0 0 2px ${tokens.colorAccentBorder}`,
  },
  band: {
    position: "relative",
    display: "grid",
    gap: "1px",
    padding: "6px 0",
  },
  bandTop: {
    borderTopLeftRadius: tokens.radiusLg,
    borderTopRightRadius: tokens.radiusLg,
  },
  bandBottom: {
    borderBottomLeftRadius: tokens.radiusLg,
    borderBottomRightRadius: tokens.radiusLg,
    paddingBottom: "8px",
  },
  bandRow: {
    position: "relative",
    minHeight: "28px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    paddingInline: "16px",
  },
  bandRowOut: { flexDirection: "row-reverse" },
  bandName: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 650,
  },
  bandType: {
    marginInline: "auto 0",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
    textTransform: "uppercase",
  },
  bandTypeOut: { marginInline: "0 auto" },
  header: {
    display: "grid",
    gap: "2px",
    padding: "12px 16px 8px",
  },
  title: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeMd,
    fontWeight: 650,
    letterSpacing: "-0.01em",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  operator: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  body: {
    minHeight: "42px",
    padding: "0 16px 14px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
});

function portColor(artifactTypeId: string): string {
  return ARTIFACT_TYPE_COLOR[artifactTypeId] ?? tokens.colorAccent;
}

function portTypeLabel(port: Port): string {
  const name = port.artifact_type.id.split(".").at(-1) ?? port.artifact_type.id;
  return port.shape === "many" ? `${name}[]` : name;
}

function PortBand({
  ports,
  placement,
}: {
  ports: readonly Port[];
  placement: "top" | "bottom";
}) {
  if (!ports.length) return null;

  const dominantColor = portColor(ports[0].artifact_type.id);

  return (
    <div
      {...stylex.props(s.band, placement === "top" ? s.bandTop : s.bandBottom)}
      style={{
        backgroundColor: `color-mix(in srgb, ${dominantColor} 9%, ${tokens.colorSurface})`,
      }}
    >
      {ports.map((port) => {
        const isInput = port.direction === "input";
        const color = portColor(port.artifact_type.id);

        return (
          <div
            key={port.name}
            {...stylex.props(s.bandRow, isInput ? null : s.bandRowOut)}
          >
            <Handle
              type={isInput ? "target" : "source"}
              position={isInput ? Position.Left : Position.Right}
              id={encodeHandleId(portMetaForPort(port))}
              style={handleStyle("50%", color, port.variadic)}
            />
            <span {...stylex.props(s.bandName)}>
              {port.title ?? port.name}
            </span>
            <span {...stylex.props(s.bandType, isInput ? null : s.bandTypeOut)}>
              {portTypeLabel(port)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function BandsTint({ data, selected }: BandsTintProps) {
  return (
    <article {...stylex.props(s.shell, selected ? s.selected : null)}>
      <PortBand ports={data.spec.inputs} placement="top" />
      <header {...stylex.props(s.header)}>
        <span {...stylex.props(s.title)} title={data.spec.title}>
          {data.spec.title}
        </span>
        <span {...stylex.props(s.operator)}>
          {data.spec.operator_id}@{data.spec.operator_version}
        </span>
      </header>
      <div {...stylex.props(s.body)}>
        {data.spec.description ||
          "No description is available for this node."}
      </div>
      <PortBand ports={data.spec.outputs} placement="bottom" />
    </article>
  );
}

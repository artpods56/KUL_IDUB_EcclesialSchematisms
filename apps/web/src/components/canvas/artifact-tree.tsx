"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";

import { summarizePayload } from "@/lib/artifact-payload";
import { tokens } from "@/lib/stylex/tokens.stylex";

const s = stylex.create({
  tree: {
    display: "grid",
    gap: "2px",
    fontSize: tokens.fontSizeXs,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  },
  row: {
    display: "grid",
    gridTemplateColumns: "14px minmax(0,1fr) auto",
    alignItems: "center",
    gap: "6px",
    minHeight: "24px",
    padding: "2px 4px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "transparent",
    backgroundColor: "transparent",
    color: tokens.colorText,
    cursor: "pointer",
    textAlign: "left",
  },
  rowSelected: {
    borderColor: tokens.colorAccentBorder,
    backgroundColor: tokens.colorAccentSoft,
  },
  toggle: {
    width: "14px",
    height: "14px",
    display: "grid",
    placeItems: "center",
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1,
  },
  key: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  preview: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    maxWidth: "140px",
  },
  children: {
    display: "grid",
    gap: "2px",
    paddingLeft: "12px",
    borderLeftWidth: 1,
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorBorder,
    marginLeft: "6px",
  },
});

export interface ArtifactTreeProps {
  value: unknown;
  pathPrefix?: string[];
  selectedPath: string[] | null;
  onSelectPath: (path: string[]) => void;
  defaultExpandedDepth?: number;
  depth?: number;
}

function pathsEqual(a: string[] | null, b: string[]): boolean {
  if (!a || a.length !== b.length) return false;
  return a.every((segment, index) => segment === b[index]);
}

function TreeNode({
  label,
  value,
  path,
  selectedPath,
  onSelectPath,
  defaultExpandedDepth,
  depth,
}: {
  label: string;
  value: unknown;
  path: string[];
  selectedPath: string[] | null;
  onSelectPath: (path: string[]) => void;
  defaultExpandedDepth: number;
  depth: number;
}) {
  const expandable =
    (Array.isArray(value) && value.length > 0) ||
    (typeof value === "object" && value !== null && !Array.isArray(value));
  const [expanded, setExpanded] = React.useState(depth < defaultExpandedDepth);
  const selected = pathsEqual(selectedPath, path);

  return (
    <div>
      <button
        type="button"
        className="nodrag nowheel"
        {...stylex.props(s.row, selected ? s.rowSelected : null)}
        onClick={() => onSelectPath(path)}
        onDoubleClick={(event) => {
          if (!expandable) return;
          event.preventDefault();
          setExpanded((open) => !open);
        }}
      >
        <span
          {...stylex.props(s.toggle)}
          onClick={(event) => {
            if (!expandable) return;
            event.stopPropagation();
            setExpanded((open) => !open);
          }}
        >
          {expandable ? (expanded ? "▾" : "▸") : "·"}
        </span>
        <span {...stylex.props(s.key)}>{label}</span>
        <span {...stylex.props(s.preview)}>{summarizePayload(value)}</span>
      </button>

      {expandable && expanded ? (
        <div {...stylex.props(s.children)}>
          {Array.isArray(value)
            ? value.map((item, index) => (
                <TreeNode
                  key={`${path.join(".")}-${index}`}
                  label={`[${index}]`}
                  value={item}
                  path={[...path, String(index)]}
                  selectedPath={selectedPath}
                  onSelectPath={onSelectPath}
                  defaultExpandedDepth={defaultExpandedDepth}
                  depth={depth + 1}
                />
              ))
            : Object.entries(value as Record<string, unknown>).map(
                ([key, child]) => (
                  <TreeNode
                    key={`${path.join(".")}-${key}`}
                    label={key}
                    value={child}
                    path={[...path, key]}
                    selectedPath={selectedPath}
                    onSelectPath={onSelectPath}
                    defaultExpandedDepth={defaultExpandedDepth}
                    depth={depth + 1}
                  />
                ),
              )}
        </div>
      ) : null}
    </div>
  );
}

export function ArtifactTree({
  value,
  pathPrefix = [],
  selectedPath,
  onSelectPath,
  defaultExpandedDepth = 1,
  depth = 0,
}: ArtifactTreeProps) {
  if (Array.isArray(value)) {
    return (
      <div {...stylex.props(s.tree)}>
        {value.map((item, index) => (
          <TreeNode
            key={`root-${index}`}
            label={`[${index}]`}
            value={item}
            path={[...pathPrefix, String(index)]}
            selectedPath={selectedPath}
            onSelectPath={onSelectPath}
            defaultExpandedDepth={defaultExpandedDepth}
            depth={depth}
          />
        ))}
      </div>
    );
  }

  if (typeof value === "object" && value !== null) {
    return (
      <div {...stylex.props(s.tree)}>
        {Object.entries(value as Record<string, unknown>).map(([key, child]) => (
          <TreeNode
            key={`root-${key}`}
            label={key}
            value={child}
            path={[...pathPrefix, key]}
            selectedPath={selectedPath}
            onSelectPath={onSelectPath}
            defaultExpandedDepth={defaultExpandedDepth}
            depth={depth}
          />
        ))}
      </div>
    );
  }

  return (
    <div {...stylex.props(s.preview)}>
      {summarizePayload(value)}
    </div>
  );
}

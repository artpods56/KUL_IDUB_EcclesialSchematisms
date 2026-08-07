"use client";

import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
  type EdgeProps,
} from "@xyflow/react";
import { ChevronDown, Link2, Plus, Trash2, X } from "lucide-react";

import { tokens } from "@/lib/stylex/tokens.stylex";
import type {
  ArtifactViewerInteractionEdge,
  CanvasEdge,
  CanvasNode,
} from "../artifact-viewer";
import type { ArtifactViewerEffect } from "../artifact-interactions";

const s = stylex.create({
  positioner: {
    position: "absolute",
    zIndex: 11,
    pointerEvents: "all",
  },
  label: {
    minHeight: "24px",
    display: "inline-flex",
    alignItems: "center",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "8px",
    backgroundColor: tokens.colorSurfaceRaised,
    boxShadow: tokens.shadowNode,
    color: tokens.colorMuted,
    fontSize: "9px",
    fontWeight: 700,
  },
  labelSelected: {
    borderColor: tokens.colorAccentBorder,
    boxShadow: tokens.shadowNodeSelected,
  },
  summary: {
    minHeight: "24px",
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    paddingInline: "8px",
    borderWidth: 0,
    borderRadius: "8px 0 0 8px",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
    },
    color: tokens.colorMuted,
    cursor: "pointer",
    fontSize: "9px",
    fontWeight: 700,
    whiteSpace: "nowrap",
  },
  removeButton: {
    width: "24px",
    height: "24px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderLeftWidth: 1,
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorBorder,
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
  },
  popup: {
    width: "min(520px, calc(100vw - 24px))",
    display: "grid",
    gap: "10px",
    padding: "11px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "7px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
  heading: {
    color: tokens.colorTextEmphasis,
    fontSize: "10px",
    fontWeight: 750,
  },
  mapping: {
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr) 16px minmax(0, 1fr) 28px",
    alignItems: "center",
    gap: "7px",
  },
  input: {
    minWidth: 0,
    height: "30px",
    paddingInline: "9px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "6px",
    backgroundColor: tokens.colorSurface,
    color: tokens.colorText,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
  },
  arrow: {
    color: tokens.colorSubtle,
    textAlign: "center",
  },
  iconButton: {
    width: "24px",
    height: "24px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderRadius: "6px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHover,
    },
    color: tokens.colorMuted,
    cursor: "pointer",
  },
  footer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
  },
  effects: {
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap",
    gap: "7px",
  },
  effect: {
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    color: tokens.colorMuted,
    fontSize: "9px",
    fontWeight: 650,
  },
});

const EFFECTS: readonly ArtifactViewerEffect[] = [
  "filter",
  "highlight",
  "focus",
];

export default function ArtifactViewerInteractionEdgeControl({
  id,
  data,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  selected,
}: EdgeProps<ArtifactViewerInteractionEdge>) {
  const { deleteElements } = useReactFlow<CanvasNode, CanvasEdge>();
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const binding = data?.binding;
  if (!binding) {
    return (
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        interactionWidth={24}
        style={style}
      />
    );
  }
  const sourceFieldListId = `${binding.id}-source-fields`;
  const targetFieldListId = `${binding.id}-target-fields`;

  const updateBinding = (
    update: Partial<
      Pick<typeof binding, "mappings" | "effects">
    >,
  ) => {
    data?.onBindingChange?.(binding.id, { ...binding, ...update });
  };

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        interactionWidth={24}
        style={{
          ...style,
          opacity: selected ? 0.95 : 0.72,
          strokeWidth: selected ? 2.5 : (style?.strokeWidth ?? 2),
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan nowheel"
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
          {...stylex.props(s.positioner)}
        >
          <div
            aria-label="Artifact viewer interaction"
            {...stylex.props(
              s.label,
              selected ? s.labelSelected : null,
            )}
          >
            <Popover.Root>
              <Popover.Trigger
                type="button"
                aria-label="Configure viewer interaction"
                title="Configure field mapping and effects"
                {...stylex.props(s.summary)}
              >
                <Link2 size={10} aria-hidden="true" />
                selection · {binding.effects.join(" + ")}
                <ChevronDown size={10} aria-hidden="true" />
              </Popover.Trigger>
              <Popover.Portal>
                <Popover.Positioner
                  side="bottom"
                  align="center"
                  sideOffset={7}
                >
                  <Popover.Popup
                    className="nodrag nopan nowheel"
                    {...stylex.props(s.popup)}
                  >
                    <span {...stylex.props(s.heading)}>Field mapping</span>
                    {binding.mappings.map((mapping, index) => (
                      <div key={index} {...stylex.props(s.mapping)}>
                        <input
                          aria-label={`Source field ${index + 1}`}
                          placeholder="source field"
                          list={data?.sourceFields?.length
                            ? sourceFieldListId
                            : undefined}
                          title={mapping.sourceField || "Source field"}
                          value={mapping.sourceField}
                          {...stylex.props(s.input)}
                          onChange={(event) => {
                            const mappings = binding.mappings.map(
                              (candidate, candidateIndex) =>
                                candidateIndex === index
                                  ? {
                                      ...candidate,
                                      sourceField: event.currentTarget.value,
                                    }
                                  : candidate,
                            );
                            updateBinding({ mappings });
                          }}
                        />
                        <span aria-hidden="true" {...stylex.props(s.arrow)}>→</span>
                        <input
                          aria-label={`Target field ${index + 1}`}
                          placeholder="target field"
                          list={data?.targetFields?.length
                            ? targetFieldListId
                            : undefined}
                          title={mapping.targetField || "Target field"}
                          value={mapping.targetField}
                          {...stylex.props(s.input)}
                          onChange={(event) => {
                            const mappings = binding.mappings.map(
                              (candidate, candidateIndex) =>
                                candidateIndex === index
                                  ? {
                                      ...candidate,
                                      targetField: event.currentTarget.value,
                                    }
                                  : candidate,
                            );
                            updateBinding({ mappings });
                          }}
                        />
                        <button
                          type="button"
                          aria-label={`Remove field mapping ${index + 1}`}
                          title="Remove mapping"
                          disabled={binding.mappings.length === 1}
                          {...stylex.props(s.iconButton)}
                          onClick={() =>
                            updateBinding({
                              mappings: binding.mappings.filter(
                                (_, candidateIndex) => candidateIndex !== index,
                              ),
                            })}
                        >
                          <Trash2 size={11} aria-hidden="true" />
                        </button>
                      </div>
                    ))}
                    <div {...stylex.props(s.footer)}>
                      <span {...stylex.props(s.effects)}>
                        {EFFECTS.map((effect) => (
                          <label key={effect} {...stylex.props(s.effect)}>
                            <input
                              type="checkbox"
                              checked={binding.effects.includes(effect)}
                              onChange={(event) => {
                                const effects = event.currentTarget.checked
                                  ? [...binding.effects, effect]
                                  : binding.effects.filter(
                                      (candidate) => candidate !== effect,
                                    );
                                if (effects.length) updateBinding({ effects });
                              }}
                            />
                            {effect}
                          </label>
                        ))}
                      </span>
                      <button
                        type="button"
                        aria-label="Add field mapping"
                        title="Add mapping"
                        disabled={binding.mappings.length >= 8}
                        {...stylex.props(s.iconButton)}
                        onClick={() =>
                          updateBinding({
                            mappings: [
                              ...binding.mappings,
                              { sourceField: "", targetField: "" },
                            ],
                          })}
                      >
                        <Plus size={12} aria-hidden="true" />
                      </button>
                    </div>
                    <datalist id={sourceFieldListId}>
                      {data?.sourceFields?.map((field) => (
                        <option
                          key={field.id}
                          value={field.id}
                          label={`${field.title} · ${field.valueType}`}
                        />
                      ))}
                    </datalist>
                    <datalist id={targetFieldListId}>
                      {data?.targetFields?.map((field) => (
                        <option
                          key={field.id}
                          value={field.id}
                          label={`${field.title} · ${field.valueType}`}
                        />
                      ))}
                    </datalist>
                  </Popover.Popup>
                </Popover.Positioner>
              </Popover.Portal>
            </Popover.Root>
            <button
              type="button"
              aria-label="Remove viewer interaction"
              title="Remove viewer interaction"
              {...stylex.props(s.removeButton)}
              onClick={(event) => {
                event.stopPropagation();
                void deleteElements({ edges: [{ id }] });
              }}
            >
              <X size={11} aria-hidden="true" />
            </button>
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

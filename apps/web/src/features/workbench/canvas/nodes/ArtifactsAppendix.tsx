"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import useSWR from "swr";

import { useNodeRegistry } from "@/hooks/use-api";
import {
  artifactContentUrl,
  type ArtifactTypeSpec,
  type RunPortOutput,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  resolvedAppendixHeight,
  resolvedNodeWidth,
  type WorkflowNodeLayout,
} from "../node-layout";
import type { WorkflowNodeData } from "../types";
import {
  META_ARTIFACT_RENDERER,
  PrettyValue,
  rendererFor,
} from "./artifact-renderers";
import { LayoutResizeHandle } from "./LayoutResizeHandle";
import { schemaTypeLabel } from "./type-inspector";

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const s = stylex.create({
  appendix: {
    position: "relative",
    display: "grid",
    gap: "12px",
    width: "300px",
    marginTop: "10px",
    padding: "11px 12px 12px",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  section: { display: "grid", gap: "8px" },
  headRow: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
  },
  headTitle: {
    color: tokens.colorMuted,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.06em",
    textTransform: "uppercase",
  },
  kindBadge: {
    padding: "1px 7px",
    borderRadius: "9999px",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "10px",
  },
  modeToggle: {
    marginLeft: "auto",
    display: "flex",
    gap: "2px",
    padding: "2px",
    borderRadius: "8px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  modeButton: {
    height: "20px",
    paddingInline: "9px",
    borderWidth: 0,
    borderRadius: "6px",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorSubtle,
    cursor: "pointer",
    fontSize: "10px",
    fontWeight: 700,
  },
  modeButtonActive: {
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNode,
    color: tokens.colorTextEmphasis,
  },
  fieldSelect: {
    width: "100%",
    height: "24px",
    paddingInline: "7px",
    borderWidth: 0,
    borderRadius: "7px",
    outline: "none",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorText,
    fontFamily: MONO,
    fontSize: "10px",
  },
  pager: {
    display: "flex",
    alignItems: "center",
    gap: "4px",
  },
  pageChip: {
    minWidth: "22px",
    height: "22px",
    paddingInline: "4px",
    flexShrink: 0,
    borderWidth: 0,
    borderRadius: "7px",
    backgroundColor: {
      default: tokens.colorSurfaceMuted,
      ":hover": tokens.colorHoverStrong,
    },
    color: tokens.colorMuted,
    cursor: "pointer",
    fontFamily: MONO,
    fontSize: "10px",
  },
  pageChipActive: {
    backgroundColor: tokens.colorAccentSoft,
    color: tokens.colorAccent,
    fontWeight: 700,
  },
  ellipsis: {
    paddingInline: "1px",
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "10px",
  },
  indexGroup: {
    marginLeft: "auto",
    display: "flex",
    alignItems: "center",
    gap: "4px",
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "10px",
  },
  indexInput: {
    width: "36px",
    height: "22px",
    paddingInline: "5px",
    borderWidth: 0,
    borderRadius: "7px",
    outline: "none",
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorText,
    fontFamily: MONO,
    fontSize: "10px",
    textAlign: "center",
    appearance: "textfield",
  },
  body: {
    maxHeight: "230px",
    overflowY: "auto",
    overflowX: "auto",
    padding: "9px 10px",
    borderRadius: "10px",
    backgroundColor: tokens.colorSurfaceMuted,
  },
  raw: {
    margin: 0,
    fontFamily: MONO,
    fontSize: "10px",
    lineHeight: 1.55,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  projectedList: { display: "grid", gap: "5px" },
  projectedRow: {
    display: "grid",
    gridTemplateColumns: "18px minmax(0, 1fr)",
    alignItems: "baseline",
    gap: "6px",
  },
  projectedIndex: {
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "10px",
  },
  projectionType: {
    color: tokens.colorAccent,
    fontFamily: MONO,
    fontSize: "10px",
  },
  notice: {
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1.45,
  },
  noticeError: { color: tokens.colorDanger },
});

function nodeInteractionProps(props: ReturnType<typeof stylex.props>) {
  return {
    ...props,
    className: `nodrag nowheel${props.className ? ` ${props.className}` : ""}`,
  };
}

interface FieldOption {
  name: string;
  type: string;
}

interface JsonPayloadRequest {
  artifactId: string;
  url: string;
}

type ArtifactPayloadKey = readonly ["artifact-json", JsonPayloadRequest];
type SequencePayloadKey = readonly [
  "artifact-json-sequence",
  string,
  readonly JsonPayloadRequest[],
];

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function schemaCandidates(
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
): Record<string, unknown>[] {
  let resolved = schema;
  const reference = resolved.$ref;
  if (typeof reference === "string" && reference.startsWith("#/$defs/")) {
    const definitions = record(root.$defs);
    resolved =
      record(definitions?.[reference.slice("#/$defs/".length)]) ?? resolved;
  }
  if (!Array.isArray(resolved.anyOf)) return [resolved];
  return resolved.anyOf.flatMap((candidate) => {
    const candidateSchema = record(candidate);
    return candidateSchema
      ? schemaCandidates(candidateSchema, root)
      : [];
  });
}

function projectionFieldType(
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
): string | null {
  const variants = schemaCandidates(schema, root).filter(
    (candidate) => candidate.type !== "null",
  );
  if (variants.length !== 1) return null;

  const variant = variants[0];
  if (
    variant.type === "string" ||
    variant.type === "integer" ||
    variant.type === "number" ||
    variant.type === "boolean"
  ) {
    return schemaTypeLabel(schema, root);
  }
  if (variant.type !== "array") return null;

  const items = record(variant.items);
  if (!items) return null;
  const itemVariants = schemaCandidates(items, root).filter(
    (candidate) => candidate.type !== "null",
  );
  if (
    itemVariants.length !== 1 ||
    !["string", "integer", "number", "boolean"].includes(
      String(itemVariants[0].type),
    )
  ) {
    return null;
  }
  return schemaTypeLabel(schema, root);
}

async function fetchJsonPayload(request: JsonPayloadRequest): Promise<unknown> {
  const response = await fetch(request.url, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(
      `Could not load JSON payload for artifact ${request.artifactId}: ${response.status} ${response.statusText}`,
    );
  }
  try {
    return await response.json();
  } catch (error) {
    throw new Error(
      `Could not parse JSON payload for artifact ${request.artifactId}`,
      { cause: error },
    );
  }
}

async function loadArtifactPayload([
  ,
  request,
]: ArtifactPayloadKey): Promise<unknown> {
  return fetchJsonPayload(request);
}

async function loadSequencePayloads([
  ,
  ,
  requests,
]: SequencePayloadKey): Promise<Record<string, unknown>> {
  const entries = await Promise.all(
    requests.map(async (request) => [
      request.artifactId,
      await fetchJsonPayload(request),
    ] as const),
  );
  return Object.fromEntries(entries);
}

function projectableFields(spec: ArtifactTypeSpec | undefined): FieldOption[] {
  const root = record(spec?.payload_schema);
  const properties = record(root?.properties);
  if (!root || !properties) return [];

  return Object.entries(properties).flatMap(([name, rawProperty]) => {
    const property = record(rawProperty);
    if (!property) return [];
    const type = projectionFieldType(property, root);
    return type ? [{ name, type }] : [];
  });
}

function pageWindow(count: number, current: number): (number | "gap")[] {
  if (count <= 7) {
    return Array.from({ length: count }, (_, index) => index + 1);
  }

  const pages = new Set([
    1,
    2,
    current - 1,
    current,
    current + 1,
    count - 1,
    count,
  ]);
  const sorted = [...pages]
    .filter((page) => page >= 1 && page <= count)
    .sort((left, right) => left - right);
  const result: (number | "gap")[] = [];
  let previous = 0;
  for (const page of sorted) {
    if (page - previous > 1) result.push("gap");
    result.push(page);
    previous = page;
  }
  return result;
}

function SequencePager({
  count,
  index,
  onChange,
}: {
  count: number;
  index: number;
  onChange: (index: number) => void;
}) {
  const clamp = (value: number) => Math.min(count - 1, Math.max(0, value));

  return (
    <div {...stylex.props(s.pager)}>
      {pageWindow(count, index + 1).map((page, position) =>
        page === "gap" ? (
          <span key={`gap-${position}`} {...stylex.props(s.ellipsis)}>
            …
          </span>
        ) : (
          <button
            key={page}
            type="button"
            aria-label={`Show item ${page}`}
            aria-current={page === index + 1 ? "page" : undefined}
            {...nodeInteractionProps(
              stylex.props(
                s.pageChip,
                page === index + 1 ? s.pageChipActive : null,
              ),
            )}
            onClick={() => onChange(page - 1)}
          >
            {page}
          </button>
        ),
      )}
      <label {...stylex.props(s.indexGroup)}>
        <input
          type="number"
          min={1}
          max={count}
          value={index + 1}
          aria-label="Item index"
          {...nodeInteractionProps(stylex.props(s.indexInput))}
          onChange={(event) => {
            const value = Number.parseInt(event.currentTarget.value, 10);
            if (!Number.isNaN(value)) onChange(clamp(value - 1));
          }}
        />
        / {count}
      </label>
    </div>
  );
}

export function ArtifactPortPreview({
  output,
  artifactTypes,
  previewHeight,
}: {
  output: RunPortOutput;
  artifactTypes: readonly ArtifactTypeSpec[];
  previewHeight: number;
}) {
  const artifacts = output.artifacts;
  const sequence = output.kind === "sequence";
  const [index, setIndex] = React.useState(0);
  const [modeChoice, setModeChoice] = React.useState<string | null>(null);
  const [field, setField] = React.useState("");

  const focusedIndex = Math.min(index, artifacts.length - 1);
  const active = artifacts[focusedIndex];
  const activeContentUrl = artifactContentUrl(active.content_url);
  const activePayloadKey: ArtifactPayloadKey | null =
    active.content_type === "application/json" && activeContentUrl
      ? [
          "artifact-json",
          { artifactId: active.artifact_id, url: activeContentUrl },
        ]
      : null;
  const {
    data: activePayload,
    error: activePayloadError,
    isLoading: activePayloadLoading,
  } = useSWR(activePayloadKey, loadArtifactPayload);

  const artifactType = artifactTypes.find(
    (candidate) =>
      candidate.key.id === active.artifact_type &&
      candidate.key.schema_version === active.schema_version,
  );
  const fields =
    sequence &&
    active.content_type === "application/json" &&
    activeContentUrl
      ? projectableFields(artifactType)
      : [];
  const selectedField = fields.find((option) => option.name === field);

  const projectionRequests = selectedField
    ? artifacts.flatMap((artifact) => {
        const url =
          artifact.content_type === "application/json"
            ? artifactContentUrl(artifact.content_url)
            : null;
        return url
          ? [{ artifactId: artifact.artifact_id, url }]
          : [];
      })
    : [];
  const projectionKey: SequencePayloadKey | null =
    selectedField && projectionRequests.length === artifacts.length
      ? ["artifact-json-sequence", output.port, projectionRequests]
      : null;
  const {
    data: projectionPayloads,
    error: projectionError,
    isLoading: projectionLoading,
  } = useSWR(projectionKey, loadSequencePayloads);

  const jsonPayloadMissing =
    active.content_type === "application/json" && !activeContentUrl;
  const jsonPayloadFailed =
    active.content_type === "application/json" && Boolean(activePayloadError);
  const renderer = jsonPayloadMissing || jsonPayloadFailed
    ? META_ARTIFACT_RENDERER
    : rendererFor(active, activePayload);
  const mode =
    modeChoice && renderer.modes.includes(modeChoice)
      ? modeChoice
      : renderer.modes[0];
  const projectionModes = ["pretty", "raw"] as const;
  const projectionMode =
    modeChoice && projectionModes.includes(modeChoice as "pretty" | "raw")
      ? modeChoice
      : projectionModes[0];
  const projectionMissingContent =
    Boolean(selectedField) && projectionRequests.length !== artifacts.length;
  const projectionFallback =
    Boolean(selectedField) && (Boolean(projectionError) || projectionMissingContent);
  const visibleModes = selectedField
    ? projectionFallback
      ? META_ARTIFACT_RENDERER.modes
      : projectionModes
    : activePayloadLoading
      ? []
      : renderer.modes;
  const projectedValues =
    selectedField && projectionPayloads
      ? artifacts.map(
          (artifact) =>
            record(projectionPayloads[artifact.artifact_id])?.[
              selectedField.name
            ],
        )
      : null;
  return (
    <section {...stylex.props(s.section)}>
      <div {...stylex.props(s.headRow)}>
        <span {...stylex.props(s.headTitle)}>{output.port}</span>
        <span {...stylex.props(s.kindBadge)}>
          {sequence ? `sequence · ${artifacts.length}` : "single"}
        </span>
        {visibleModes.length > 1 ? (
          <div
            role="tablist"
            aria-label="Artifact view mode"
            {...stylex.props(s.modeToggle)}
          >
            {visibleModes.map((option) => {
              const selectedMode = selectedField ? projectionMode : mode;
              return (
                <button
                  key={option}
                  type="button"
                  role="tab"
                  aria-selected={selectedMode === option}
                  {...nodeInteractionProps(
                    stylex.props(
                      s.modeButton,
                      selectedMode === option ? s.modeButtonActive : null,
                    ),
                  )}
                  onClick={() => setModeChoice(option)}
                >
                  {option}
                </button>
              );
            })}
          </div>
        ) : null}
      </div>
      {fields.length ? (
        <select
          aria-label="Project artifacts onto a field"
          value={field}
          {...nodeInteractionProps(stylex.props(s.fieldSelect))}
          onChange={(event) => setField(event.currentTarget.value)}
        >
          <option value="">whole objects</option>
          {fields.map((option) => (
            <option key={option.name} value={option.name}>
              map .{option.name} → list[{option.type}]
            </option>
          ))}
        </select>
      ) : null}
      {selectedField ? (
        <>
          <span {...stylex.props(s.projectionType)}>
            list[{selectedField.type}] · {artifacts.length} items
          </span>
          <div
            {...nodeInteractionProps(stylex.props(s.body))}
            style={{ maxHeight: previewHeight }}
          >
            {projectionLoading ? (
              <span {...stylex.props(s.notice)}>Loading projected values…</span>
            ) : projectionError || projectionMissingContent ? (
              <>
                <p
                  title={
                    projectionError instanceof Error
                      ? projectionError.message
                      : undefined
                  }
                  {...stylex.props(s.notice, s.noticeError)}
                >
                  Projection preview unavailable; showing artifact metadata.
                </p>
                <META_ARTIFACT_RENDERER.Component
                  artifact={active}
                  mode="meta"
                />
              </>
            ) : projectedValues ? (
              projectionMode === "raw" ? (
                <pre {...stylex.props(s.raw)}>
                  {JSON.stringify(projectedValues, null, 2)}
                </pre>
              ) : (
                <div {...stylex.props(s.projectedList)}>
                  {projectedValues.map((value, itemIndex) => (
                    <div key={itemIndex} {...stylex.props(s.projectedRow)}>
                      <span {...stylex.props(s.projectedIndex)}>
                        {String(itemIndex + 1).padStart(2, "0")}
                      </span>
                      <PrettyValue value={value ?? "—"} />
                    </div>
                  ))}
                </div>
              )
            ) : null}
          </div>
        </>
      ) : (
        <>
          {sequence ? (
            <SequencePager
              count={artifacts.length}
              index={focusedIndex}
              onChange={setIndex}
            />
          ) : null}
          <div
            {...nodeInteractionProps(stylex.props(s.body))}
            style={{ maxHeight: previewHeight }}
          >
            {activePayloadLoading ? (
              <p {...stylex.props(s.notice)}>Loading JSON preview…</p>
            ) : activePayloadError || jsonPayloadMissing ? (
              <p
                title={
                  activePayloadError instanceof Error
                    ? activePayloadError.message
                    : undefined
                }
                {...stylex.props(s.notice, s.noticeError)}
              >
                JSON preview unavailable; showing artifact metadata.
              </p>
            ) : null}
            {activePayloadLoading ? null : (
              <renderer.Component
                artifact={active}
                payload={activePayload}
                mode={mode}
              />
            )}
          </div>
        </>
      )}
    </section>
  );
}

export function ArtifactsAppendix({
  data,
  layout,
  onLayoutDraft,
  onLayoutCommit,
}: {
  data: WorkflowNodeData;
  layout: WorkflowNodeLayout | null;
  onLayoutDraft: (layout: WorkflowNodeLayout | null) => void;
  onLayoutCommit: (layout: WorkflowNodeLayout | null) => void;
}) {
  const { data: registry } = useNodeRegistry();
  const outputs = (data.run?.outputs ?? []).filter(
    (output) => output.artifacts.length > 0,
  );
  if (!outputs.length) return null;
  const width = resolvedNodeWidth(layout);
  const previewHeight = resolvedAppendixHeight(layout);

  return (
    <aside
      aria-label="Produced artifacts"
      {...nodeInteractionProps(stylex.props(s.appendix))}
      style={{ width }}
    >
      {outputs.map((output) => (
        <ArtifactPortPreview
          key={output.port}
          output={output}
          artifactTypes={registry?.artifact_types ?? []}
          previewHeight={previewHeight}
        />
      ))}
      <LayoutResizeHandle
        layout={layout}
        axes={["width", "appendixHeight"]}
        ariaLabel="Resize artifact preview"
        onDraft={onLayoutDraft}
        onCommit={onLayoutCommit}
      />
    </aside>
  );
}

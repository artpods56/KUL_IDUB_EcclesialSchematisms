"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";

import { useNodeRegistry } from "@/hooks/use-api";
import type { Port } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import { ARTIFACT_TYPE_COLOR } from "../nodes.css";
import {
  portArtifactTypeVariable,
  resolvedPortArtifactType,
  type WorkflowArtifactTypeBindings,
} from "../types";

/**
 * Port type inspector — a popover that renders the artifact payload schema
 * as a nested field tree, plus declared field projections.
 */

const s = stylex.create({
  popup: {
    width: "320px",
    overflow: "hidden",
    borderRadius: "12px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
  header: {
    display: "grid",
    gap: "3px",
    padding: "12px 14px 10px",
  },
  contract: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeSm,
    fontWeight: 700,
  },
  dot: {
    width: "7px",
    height: "7px",
    flexShrink: 0,
    borderRadius: "9999px",
  },
  description: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  section: {
    padding: "9px 14px 12px",
    borderTopWidth: 1,
    borderTopStyle: "solid",
    borderTopColor: tokens.colorDivider,
  },
  sectionTitle: {
    marginBottom: "6px",
    color: tokens.colorMuted,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  tree: {
    display: "grid",
    gap: "2px",
    maxHeight: "260px",
    overflowY: "auto",
  },
  row: {
    display: "flex",
    alignItems: "baseline",
    gap: "8px",
    minHeight: "20px",
  },
  fieldName: {
    color: tokens.colorTextEmphasis,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  required: { color: tokens.colorWarning },
  fieldType: {
    marginLeft: "auto",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
    whiteSpace: "nowrap",
  },
  empty: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  emptyError: { color: tokens.colorDanger },
  projection: {
    display: "flex",
    alignItems: "baseline",
    gap: "6px",
    minHeight: "20px",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  projectionPath: { color: tokens.colorTextEmphasis },
  projectionArrow: { color: tokens.colorSubtle },
  projectionTarget: { color: tokens.colorAccent },
});

type Schema = Record<string, unknown>;

function record(value: unknown): Schema | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Schema;
}

function dereferenceSchema(schema: Schema, root: Schema): Schema {
  let current = schema;
  const visited = new Set<string>();

  while (typeof current.$ref === "string" && !visited.has(current.$ref)) {
    const reference = current.$ref;
    visited.add(reference);
    if (!reference.startsWith("#/$defs/")) break;
    const definitions = record(root.$defs);
    const resolved = record(definitions?.[reference.slice("#/$defs/".length)]);
    if (!resolved) break;
    current = resolved;
  }
  return current;
}

function schemaVariants(schema: Schema, root: Schema): Schema[] {
  const resolved = dereferenceSchema(schema, root);
  if (!Array.isArray(resolved.anyOf)) return [resolved];

  const variants = resolved.anyOf.flatMap((candidate) => {
    const candidateSchema = record(candidate);
    return candidateSchema ? [dereferenceSchema(candidateSchema, root)] : [];
  });
  return variants.length ? variants : [resolved];
}

function directTypeLabel(schema: Schema, root: Schema): string {
  switch (schema.type) {
    case "string":
      return "str";
    case "integer":
      return "int";
    case "number":
      return "float";
    case "boolean":
      return "bool";
    case "null":
      return "None";
    case "array":
      return `list[${schemaTypeLabel(record(schema.items), root)}]`;
    case "object":
      return "object";
    default:
      return record(schema.properties) ? "object" : "any";
  }
}

/** Python-flavored type label for a JSON-schema fragment. */
export function schemaTypeLabel(
  schema: Schema | null,
  root: Schema | null = schema,
): string {
  if (!schema || !root) return "any";

  const labels = schemaVariants(schema, root).map((variant) =>
    directTypeLabel(variant, root),
  );
  return [...new Set(labels)].join(" | ");
}

interface FieldRow {
  depth: number;
  name: string;
  type: string;
  required: boolean;
}

function collectRows(schema: Schema, root: Schema, depth: number): FieldRow[] {
  if (depth > 8) return [];

  const resolved = schemaVariants(schema, root).find((variant) =>
    record(variant.properties),
  );
  const properties = record(resolved?.properties);
  if (!properties) return [];

  const required = new Set(
    Array.isArray(resolved?.required)
      ? resolved.required.filter((value): value is string =>
          typeof value === "string",
        )
      : [],
  );

  return Object.entries(properties).flatMap(([name, rawProperty]) => {
    const propertySchema = record(rawProperty);
    if (!propertySchema) return [];
    const variants = schemaVariants(propertySchema, root);

    const row: FieldRow = {
      depth,
      name,
      type: schemaTypeLabel(propertySchema, root),
      required: required.has(name),
    };
    const children: FieldRow[] = [];
    const objectVariant = variants.find((variant) =>
      record(variant.properties),
    );
    if (objectVariant) {
      children.push(...collectRows(objectVariant, root, depth + 1));
    }

    for (const variant of variants) {
      if (variant.type !== "array") continue;
      const items = record(variant.items);
      if (items) {
        children.push(...collectRows(items, root, depth + 1));
      }
    }

    return [row, ...children];
  });
}

export function SchemaTree({ schema }: { schema: Schema }) {
  const rows = collectRows(schema, schema, 0);
  if (!rows.length) {
    return (
      <p {...stylex.props(s.empty)}>
        No declared payload schema — this artifact carries opaque content.
      </p>
    );
  }

  return (
    <div
      className={`nodrag nowheel ${stylex.props(s.tree).className}`}
    >
      {rows.map((row, index) => (
        <div
          key={`${row.depth}-${row.name}-${index}`}
          {...stylex.props(s.row)}
          style={{ paddingLeft: row.depth * 14 }}
        >
          <span {...stylex.props(s.fieldName)}>
            {row.name}
            {row.required ? (
              <span {...stylex.props(s.required)}>*</span>
            ) : null}
          </span>
          <span {...stylex.props(s.fieldType)}>{row.type}</span>
        </div>
      ))}
    </div>
  );
}

export function PortTypePopover({
  port,
  shape,
  artifactTypeBindings = {},
  children,
}: {
  port: Port;
  shape: Port["shape"];
  artifactTypeBindings?: WorkflowArtifactTypeBindings;
  children: React.ReactNode;
}) {
  const {
    data: registry,
    error: registryError,
    isLoading: registryLoading,
  } = useNodeRegistry();
  const artifactType = resolvedPortArtifactType(port, artifactTypeBindings);
  const variable = portArtifactTypeVariable(port);
  const spec = artifactType
    ? registry?.artifact_types.find(
        (artifact) =>
          artifact.key.id === artifactType.id &&
          artifact.key.schema_version === artifactType.schema_version,
      )
    : undefined;
  const color = artifactType
    ? ARTIFACT_TYPE_COLOR[artifactType.id] ?? tokens.colorAccent
    : tokens.colorAccent;
  const contract = artifactType
    ? `${artifactType.id}@${artifactType.schema_version}`
    : "Any artifact";
  const payloadSchema = record(spec?.payload_schema) ?? {};
  const projections = spec?.field_projections ?? [];

  return (
    <Popover.Root>
      {children}
      <Popover.Portal>
        <Popover.Positioner
          side={port.direction === "input" ? "left" : "right"}
          align="start"
          sideOffset={10}
        >
          <Popover.Popup
            className={`nodrag nowheel ${stylex.props(s.popup).className}`}
          >
            <header {...stylex.props(s.header)}>
              <span {...stylex.props(s.contract)}>
                <span
                  {...stylex.props(s.dot)}
                  style={{ backgroundColor: color }}
                />
                {shape === "many" ? `list[${contract}]` : contract}
              </span>
              {port.description ? (
                <span {...stylex.props(s.description)}>
                  {port.description}
                </span>
              ) : null}
            </header>
            <section {...stylex.props(s.section)}>
              <div {...stylex.props(s.sectionTitle)}>
                {spec?.title ?? "Payload"}
              </div>
              {!artifactType ? (
                <p {...stylex.props(s.empty)}>
                  This generic port binds to a concrete artifact type when it is
                  connected{variable ? ` (${variable})` : ""}.
                </p>
              ) : registryLoading ? (
                <p {...stylex.props(s.empty)}>Loading payload schema…</p>
              ) : registryError ? (
                <p
                  title={
                    registryError instanceof Error
                      ? registryError.message
                      : undefined
                  }
                  {...stylex.props(s.empty, s.emptyError)}
                >
                  Payload schema unavailable.
                </p>
              ) : spec ? (
                <SchemaTree schema={payloadSchema} />
              ) : (
                <p {...stylex.props(s.empty)}>
                  This artifact type is not declared in the current registry.
                </p>
              )}
            </section>
            {projections.length ? (
              <section {...stylex.props(s.section)}>
                <div {...stylex.props(s.sectionTitle)}>Projectable fields</div>
                {projections.map((projection) => (
                  <div
                    key={projection.path.join(".")}
                    {...stylex.props(s.projection)}
                  >
                    <span {...stylex.props(s.projectionPath)}>
                      .{projection.path.join(".")}
                    </span>
                    <span {...stylex.props(s.projectionArrow)}>→</span>
                    <span {...stylex.props(s.projectionTarget)}>
                      {projection.target_artifact_type.id}
                    </span>
                  </div>
                ))}
              </section>
            ) : null}
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}

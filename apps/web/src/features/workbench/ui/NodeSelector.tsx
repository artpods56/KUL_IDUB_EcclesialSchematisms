"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Cable,
  ExternalLink,
  Plus,
  Search,
  Settings2,
  Workflow,
} from "lucide-react";

import {
  connectionRoutesFor,
  encodeHandleId,
  type ConnectionRoute,
} from "../canvas/handles";
import { schemaFields, type SchemaField } from "../canvas/config-schema";
import { artifactTypeColor } from "../canvas/nodes.css";
import {
  acceptedPortShapes,
  portArtifactType,
  portArtifactTypeVariable,
  portHasInstancePlugs,
  portMetaForPort,
} from "../canvas/types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import type {
  NodeRegistry,
  NodeSpec,
  Port,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  catalogNodesForGoal,
  catalogNodeSpecs,
  moduleReleaseSpecs,
  NODE_GOAL_CATEGORIES,
  type NodeGoalCategoryId,
} from "../model/node-catalog";

const MODULE_PLUGIN_SLUG = "graph.module";

/** A port-scoped Add node invocation using the same routes as canvas wiring. */
export type NodeSelectorCompatibilityContext =
  | {
      direction: "upstream";
      port: Port & { readonly direction: "input" };
    }
  | {
      direction: "downstream";
      port: Port & { readonly direction: "output" };
    };

export interface NodeSelectorProps {
  open: boolean;
  registry: NodeRegistry;
  activeGraphId: string | null;
  /** Limits results to nodes that can connect to the invoking port. */
  compatibility?: NodeSelectorCompatibilityContext;
  /** User-facing reason the registry could not be loaded. */
  errorMessage?: string | null;
  loading?: boolean;
  /** Keeps inspection available while disabling graph mutation for viewers. */
  canInsert?: boolean;
  insertDisabledReason?: string;
  /** Controlled-dialog opener, forwarded to Base UI's finalFocus contract. */
  returnFocusRef?: React.RefObject<HTMLElement | null>;
  onOpenChange: (open: boolean) => void;
  onAddNode: (spec: NodeSpec) => void;
  onRetry?: () => void;
  onOpenGraph?: (graphId: string) => void;
  onOpenWorkspaceLibrary?: () => void;
}

interface CompatibleNode {
  spec: NodeSpec;
  routeSummary: string;
  additionalRouteCount: number;
}

interface CompatiblePortPair {
  source: Port;
  target: Port;
  route: ConnectionRoute;
  routeCount: number;
}

function nodeKey(spec: NodeSpec): string {
  return `${spec.operator_id}@${spec.operator_version}`;
}

function pluginFor(
  registry: NodeRegistry,
  slug: string,
): NodeRegistry["plugins"][number] {
  const plugin = registry.plugins.find((candidate) => candidate.slug === slug);
  if (!plugin) {
    throw new Error(`Node registry is missing owner plugin "${slug}".`);
  }
  return plugin;
}

function nodeSearchText(
  spec: NodeSpec,
  plugin: NodeRegistry["plugins"][number],
): string {
  const fields = schemaFields(spec.config_schema);
  return [
    spec.title,
    spec.operator_id,
    spec.description,
    spec.plugin_slug,
    plugin.title,
    plugin.origin,
    ...spec.inputs.flatMap((port) => [
      port.name,
      port.title ?? "",
      port.description ?? "",
      portArtifactType(port)?.id ?? "any artifact generic",
      portArtifactTypeVariable(port) ?? "",
      port.instance_plugs ? "collect ordered input plugs" : "",
      ...(port.accepted_shapes ?? []).map((shape) =>
        shape === "many" ? "sequence" : "single",
      ),
    ]),
    ...spec.outputs.flatMap((port) => [
      port.name,
      port.title ?? "",
      port.description ?? "",
      portArtifactType(port)?.id ?? "any artifact generic",
      portArtifactTypeVariable(port) ?? "",
    ]),
    ...fields.flatMap((field) => [
      field.name,
      field.title,
      field.description ?? "",
      ...(field.enumValues?.map(String) ?? []),
    ]),
  ]
    .join(" ")
    .toLowerCase();
}

function shapesAreCompatible(source: Port, target: Port): boolean {
  const acceptedShapes = acceptedPortShapes(target);
  return (
    acceptedShapes.includes(source.shape) ||
    (!portHasInstancePlugs(target) &&
      source.shape === "many" &&
      acceptedShapes.includes("one"))
  );
}

function routeTitle(route: ConnectionRoute): string | null {
  const conversionTitle = route.conversionPath
    .map((conversion) => conversion.title)
    .join(" → ");
  if (route.kind === "projection") return route.projection.title;
  if (route.kind === "conversion") return conversionTitle;
  if (route.kind === "projection-conversion") {
    return `${route.projection.title} → ${conversionTitle}`;
  }
  return null;
}

function compatiblePortPairs(
  sourceSpec: NodeSpec,
  targetSpec: NodeSpec,
  registry: NodeRegistry,
): CompatiblePortPair[] {
  const pairs: CompatiblePortPair[] = [];
  for (const source of sourceSpec.outputs) {
    for (const target of targetSpec.inputs) {
      if (!shapesAreCompatible(source, target)) continue;
      const routes = connectionRoutesFor(
        {
          sourceHandle: encodeHandleId(portMetaForPort(source)),
          targetHandle: encodeHandleId(portMetaForPort(target)),
        },
        registry.artifact_types,
        registry.artifact_conversions,
      );
      const route = routes[0];
      if (!route) continue;
      pairs.push({ source, target, route, routeCount: routes.length });
    }
  }
  return pairs;
}

function portsCanConnect(
  source: Port,
  target: Port,
  registry: NodeRegistry,
): boolean {
  if (!shapesAreCompatible(source, target)) return false;
  return connectionRoutesFor(
    {
      sourceHandle: encodeHandleId(portMetaForPort(source)),
      targetHandle: encodeHandleId(portMetaForPort(target)),
    },
    registry.artifact_types,
    registry.artifact_conversions,
  ).length > 0;
}

function nodeMatchesCompatibility(
  spec: NodeSpec,
  compatibility: NodeSelectorCompatibilityContext,
  registry: NodeRegistry,
): boolean {
  if (compatibility.direction === "upstream") {
    return spec.outputs.some((output) =>
      portsCanConnect(output, compatibility.port, registry),
    );
  }
  return spec.inputs.some((input) =>
    portsCanConnect(compatibility.port, input, registry),
  );
}

function compatibleNodes(
  selected: NodeSpec,
  direction: "upstream" | "downstream",
  registry: NodeRegistry,
): CompatibleNode[] {
  return registry.nodes.flatMap((candidate) => {
    const pairs = direction === "upstream"
      ? compatiblePortPairs(candidate, selected, registry)
      : compatiblePortPairs(selected, candidate, registry);
    const first = pairs[0];
    if (!first) return [];

    const transport = portHasInstancePlugs(first.target)
      ? "direct to ordered input"
      : acceptedPortShapes(first.target).includes(first.source.shape)
        ? "direct"
        : "map each item";
    const transformation = routeTitle(first.route);
    const routeSummary = [
      `${first.source.title ?? first.source.name} → ${first.target.title ?? first.target.name}`,
      transport,
      transformation,
    ]
      .filter((value): value is string => Boolean(value))
      .join(" · ");
    const totalRouteCount = pairs.reduce(
      (count, pair) => count + pair.routeCount,
      0,
    );
    return [{
      spec: candidate,
      routeSummary,
      additionalRouteCount: totalRouteCount - 1,
    }];
  });
}

function artifactTitleFor(registry: NodeRegistry, port: Port): string {
  const artifactType = portArtifactType(port);
  if (!artifactType) return "Any artifact";
  return registry.artifact_types.find(
    (artifact) =>
      artifact.key.id === artifactType.id &&
      artifact.key.schema_version === artifactType.schema_version,
  )?.title ?? artifactType.id;
}

function fieldTypeLabel(field: SchemaField): string {
  if (field.enumValues?.length) return "choice";
  if (field.format === "textarea") return "multiline text";
  if (field.type === "number-tuple") {
    return `${field.items.length}-number tuple`;
  }
  if (field.type === "string-list") return "text list";
  return field.type;
}

function fieldConstraintLabel(field: SchemaField): string {
  const constraints = [field.required ? "required" : "optional"];
  if (field.type === "string-list") {
    if (field.minItems !== undefined && field.maxItems !== undefined) {
      constraints.push(`${field.minItems}–${field.maxItems} items`);
    } else if (field.minItems !== undefined) {
      constraints.push(`min ${field.minItems} items`);
    } else if (field.maxItems !== undefined) {
      constraints.push(`max ${field.maxItems} items`);
    }
    if (
      field.itemMinLength !== undefined &&
      field.itemMaxLength !== undefined
    ) {
      constraints.push(
        `${field.itemMinLength}–${field.itemMaxLength} characters per item`,
      );
    } else if (field.itemMinLength !== undefined) {
      constraints.push(`min ${field.itemMinLength} characters per item`);
    } else if (field.itemMaxLength !== undefined) {
      constraints.push(`max ${field.itemMaxLength} characters per item`);
    }
    return constraints.join(" · ");
  }
  if (field.minimum !== undefined && field.maximum !== undefined) {
    constraints.push(`${field.minimum}–${field.maximum}`);
  } else if (field.minimum !== undefined) {
    constraints.push(`min ${field.minimum}`);
  } else if (field.maximum !== undefined) {
    constraints.push(`max ${field.maximum}`);
  }
  if (field.minLength !== undefined && field.maxLength !== undefined) {
    constraints.push(`${field.minLength}–${field.maxLength} characters`);
  } else if (field.minLength !== undefined) {
    constraints.push(`min ${field.minLength} characters`);
  } else if (field.maxLength !== undefined) {
    constraints.push(`max ${field.maxLength} characters`);
  }
  return constraints.join(" · ");
}

const s = stylex.create({
  header: {
    display: "grid",
    gridTemplateColumns: {
      default: "minmax(220px, 0.55fr) minmax(360px, 1.45fr)",
      "@media (max-width: 720px)": "1fr",
    },
    alignItems: "center",
    gap: "18px",
    padding: {
      default: "18px 52px 16px 20px",
      "@media (max-width: 720px)": "16px 48px 14px 16px",
    },
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  heading: { minWidth: 0 },
  titleRow: { display: "flex", alignItems: "center", gap: "9px" },
  titleIcon: {
    width: "28px",
    height: "28px",
    display: "grid",
    placeItems: "center",
    borderRadius: tokens.radiusSm,
    backgroundColor: tokens.colorAccentSoft,
    color: tokens.colorAccent,
  },
  title: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeLg,
    fontWeight: 760,
  },
  description: {
    marginTop: "3px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeSm,
  },
  searchWrap: { position: "relative" },
  searchIcon: {
    position: "absolute",
    top: "11px",
    left: "12px",
    color: tokens.colorSubtle,
    pointerEvents: "none",
  },
  search: {
    width: "100%",
    height: "40px",
    padding: "0 12px 0 34px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorderStrong,
      ":focus": tokens.colorAccent,
    },
    borderRadius: tokens.radiusSm,
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "2px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  layout: {
    minHeight: 0,
    flex: 1,
    display: "grid",
    gridTemplateColumns: {
      default: "minmax(300px, 0.72fr) minmax(0, 1.28fr)",
      "@media (max-width: 900px)": "minmax(280px, 0.82fr) minmax(0, 1.18fr)",
      "@media (max-width: 720px)": "1fr",
    },
    gridTemplateRows: {
      default: "minmax(0, 1fr)",
      "@media (max-width: 720px)": "minmax(230px, 0.8fr) minmax(300px, 1.2fr)",
    },
  },
  originBadge: {
    minHeight: "17px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    paddingInline: "5px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "99px",
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorMuted,
    fontSize: "8px",
    fontWeight: 840,
    letterSpacing: "0.06em",
    lineHeight: 1,
    textTransform: "uppercase",
    whiteSpace: "nowrap",
  },
  originBadgeExternal: {
    borderColor: tokens.colorInfo,
    backgroundColor: "light-dark(rgba(74, 143, 212, 0.1), rgba(96, 165, 250, 0.1))",
    color: tokens.colorInfo,
  },
  nodePane: {
    minWidth: 0,
    minHeight: 0,
    display: "flex",
    flexDirection: "column",
    borderRightWidth: {
      default: 1,
      "@media (max-width: 720px)": 0,
    },
    borderRightStyle: "solid",
    borderRightColor: tokens.colorBorder,
    borderBottomWidth: {
      default: 0,
      "@media (max-width: 720px)": 1,
    },
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  nodePaneHeader: {
    display: "grid",
    gap: "9px",
    padding: "11px 12px 10px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  nodePaneTitle: {
    overflow: "hidden",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 730,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  resultCount: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    whiteSpace: "nowrap",
  },
  resultHeading: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
  },
  categoryToolbar: {
    display: "flex",
    gap: "4px",
    paddingBottom: "1px",
    overflowX: "auto",
  },
  categoryButton: {
    minHeight: "28px",
    flexShrink: 0,
    paddingInline: "8px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "transparent",
    borderRadius: tokens.radiusSm,
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "1px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorSubtle,
    cursor: "pointer",
    fontSize: tokens.fontSizeXs,
    fontWeight: 680,
    whiteSpace: "nowrap",
  },
  categoryButtonActive: {
    borderColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurfaceRaised,
    color: tokens.colorTextEmphasis,
  },
  compatibilityBanner: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "7px 12px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorDivider,
    backgroundColor: tokens.colorSurfaceMuted,
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
  },
  nodeList: { minHeight: 0, overflowY: "auto" },
  nodeButton: {
    width: "100%",
    minHeight: "73px",
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr) auto",
    alignItems: "start",
    gap: "10px",
    padding: "11px 12px",
    borderWidth: 0,
    borderLeftWidth: 2,
    borderLeftStyle: "solid",
    borderLeftColor: "transparent",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorDivider,
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorText,
    cursor: "pointer",
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "-3px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    textAlign: "left",
    transitionProperty: "background-color, border-color",
    transitionDuration: "120ms",
  },
  nodeButtonActive: {
    borderLeftColor: tokens.colorAccent,
    backgroundColor: tokens.colorAccentSoft,
  },
  nodeCopy: { minWidth: 0, display: "grid", gap: "4px" },
  nodeTitleRow: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  nodeTitle: {
    minWidth: 0,
    overflow: "hidden",
    fontSize: tokens.fontSizeSm,
    fontWeight: 730,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  nodeDescription: {
    display: "-webkit-box",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.35,
    WebkitBoxOrient: "vertical",
    WebkitLineClamp: 2,
  },
  nodePorts: {
    paddingTop: "2px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
    whiteSpace: "nowrap",
  },
  empty: {
    minHeight: "160px",
    display: "grid",
    placeItems: "center",
    alignContent: "center",
    gap: "10px",
    padding: "24px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1.5,
    textAlign: "center",
  },
  moduleDiagnostics: {
    display: "grid",
    gap: "8px",
    padding: "12px 14px 16px",
    borderTopWidth: 1,
    borderTopStyle: "solid",
    borderTopColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurfaceMuted,
  },
  moduleDiagnosticsNote: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  openGraphButton: {
    flexShrink: 0,
    minHeight: "24px",
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    paddingInline: "7px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "5px",
    backgroundColor: {
      default: tokens.colorSurface,
      ":hover": tokens.colorHover,
    },
    color: tokens.colorMuted,
    cursor: "pointer",
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "2px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    fontSize: "10px",
    fontWeight: 700,
  },
  inspectorOpenGraph: {
    marginTop: "8px",
    alignSelf: "flex-start",
  },
  resetButton: {
    minHeight: "29px",
    paddingInline: "10px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorMuted,
    cursor: "pointer",
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "2px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
  },
  inspector: {
    minWidth: 0,
    minHeight: 0,
    display: "flex",
    flexDirection: "column",
    backgroundColor: tokens.colorSurface,
  },
  inspectorScroll: { minHeight: 0, flex: 1, overflowY: "auto" },
  inspectorHeader: {
    padding: "20px 22px 18px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  inspectorProvenance: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
  },
  eyebrow: {
    color: tokens.colorAccent,
    fontSize: "10px",
    fontWeight: 820,
    letterSpacing: "0.11em",
    textTransform: "uppercase",
  },
  inspectorTitle: {
    marginTop: "5px",
    color: tokens.colorTextEmphasis,
    fontSize: "20px",
    fontWeight: 760,
    letterSpacing: "-0.015em",
    lineHeight: 1.2,
  },
  operatorId: {
    marginTop: "5px",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: tokens.fontSizeXs,
  },
  inspectorDescription: {
    maxWidth: "68ch",
    marginTop: "12px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1.55,
  },
  facts: {
    display: "flex",
    flexWrap: "wrap",
    gap: "5px 16px",
    marginTop: "15px",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
  },
  factStrong: { color: tokens.colorTextEmphasis, fontWeight: 750 },
  section: {
    padding: "18px 22px",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorBorder,
  },
  sectionTitleRow: {
    display: "flex",
    alignItems: "center",
    gap: "7px",
    marginBottom: "12px",
  },
  sectionIcon: { color: tokens.colorSubtle },
  sectionTitle: {
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 750,
  },
  compatibilityGrid: {
    display: "grid",
    gridTemplateColumns: {
      default: "repeat(2, minmax(0, 1fr))",
      "@media (max-width: 900px)": "1fr",
    },
    gap: "18px",
  },
  compatibilityHeading: {
    marginBottom: "5px",
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.09em",
    textTransform: "uppercase",
  },
  compatibilityList: { display: "grid" },
  compatibilityItem: {
    display: "grid",
    gap: "2px",
    padding: "8px 0",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorDivider,
  },
  compatibilityName: {
    overflow: "hidden",
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    fontWeight: 680,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  compatibilityMeta: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1.4,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  compatibilityEmpty: {
    padding: "8px 0",
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  portGrid: {
    display: "grid",
    gridTemplateColumns: {
      default: "repeat(2, minmax(0, 1fr))",
      "@media (max-width: 900px)": "1fr",
    },
    gap: "18px",
  },
  portColumnHeading: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    marginBottom: "5px",
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.09em",
    textTransform: "uppercase",
  },
  portList: { display: "grid" },
  portRow: {
    display: "grid",
    gridTemplateColumns: "7px minmax(0, 1fr)",
    gap: "9px",
    padding: "10px 0",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorDivider,
  },
  portDot: {
    width: "7px",
    height: "7px",
    marginTop: "5px",
    borderRadius: "99px",
  },
  portCopy: { minWidth: 0 },
  portTitle: {
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    fontWeight: 690,
  },
  portContract: {
    marginTop: "2px",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  portRules: {
    marginTop: "3px",
    color: tokens.colorSubtle,
    fontSize: "10px",
  },
  portDescription: {
    marginTop: "5px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  fieldList: { display: "grid" },
  fieldRow: {
    display: "grid",
    gridTemplateColumns: {
      default: "minmax(130px, 0.7fr) minmax(0, 1.3fr)",
      "@media (max-width: 900px)": "1fr",
    },
    gap: "8px 18px",
    padding: "11px 0",
    borderBottomWidth: 1,
    borderBottomStyle: "solid",
    borderBottomColor: tokens.colorDivider,
  },
  fieldIdentity: { minWidth: 0 },
  fieldTitle: { color: tokens.colorText, fontSize: tokens.fontSizeSm, fontWeight: 690 },
  fieldName: {
    marginTop: "2px",
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  fieldDetails: { minWidth: 0 },
  fieldMeta: { color: tokens.colorSubtle, fontSize: "10px" },
  fieldDescription: {
    marginTop: "4px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.45,
  },
  fieldChoices: {
    marginTop: "4px",
    overflowWrap: "anywhere",
    color: tokens.colorSubtle,
    fontSize: "10px",
  },
  inspectorFooter: {
    minHeight: "64px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "16px",
    padding: "10px 14px 10px 18px",
    borderTopWidth: 1,
    borderTopStyle: "solid",
    borderTopColor: tokens.colorBorder,
    backgroundColor: tokens.colorSurfaceRaised,
  },
  footerHint: {
    color: tokens.colorSubtle,
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.35,
  },
  addButton: {
    minHeight: "36px",
    maxWidth: "260px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    paddingInline: "13px",
    overflow: "hidden",
    borderWidth: 0,
    borderRadius: tokens.radiusSm,
    backgroundColor: { default: tokens.colorAccent, ":hover": tokens.colorAccentHover },
    color: tokens.colorOnAccent,
    cursor: "pointer",
    outlineColor: tokens.colorAccent,
    outlineStyle: "solid",
    outlineOffset: "2px",
    outlineWidth: { default: 0, ":focus-visible": "2px" },
    fontSize: tokens.fontSizeSm,
    fontWeight: 750,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  addButtonDisabled: {
    backgroundColor: tokens.colorAccentDisabled,
    color: tokens.colorTextDisabled,
    cursor: "not-allowed",
  },
});

interface CompatibilityListProps {
  title: string;
  matches: readonly CompatibleNode[];
  registry: NodeRegistry;
  emptyMessage: string;
}

function CompatibilityList({
  title,
  matches,
  registry,
  emptyMessage,
}: CompatibilityListProps) {
  return (
    <div>
      <h4 {...stylex.props(s.compatibilityHeading)}>{title}</h4>
      {matches.length ? (
        <div {...stylex.props(s.compatibilityList)}>
          {matches.map((match) => (
            <div key={nodeKey(match.spec)} {...stylex.props(s.compatibilityItem)}>
              <span {...stylex.props(s.compatibilityName)}>
                {match.spec.title}
              </span>
              <span {...stylex.props(s.compatibilityMeta)}>
                {pluginFor(registry, match.spec.plugin_slug).title} · {match.routeSummary}
                {match.additionalRouteCount > 0
                  ? ` · +${match.additionalRouteCount} route${match.additionalRouteCount === 1 ? "" : "s"}`
                  : ""}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p {...stylex.props(s.compatibilityEmpty)}>{emptyMessage}</p>
      )}
    </div>
  );
}

interface PortListProps {
  direction: "input" | "output";
  ports: readonly Port[];
  registry: NodeRegistry;
}

function PortList({ direction, ports, registry }: PortListProps) {
  const input = direction === "input";
  return (
    <div>
      <h4 {...stylex.props(s.portColumnHeading)}>
        {input ? <ArrowDownToLine size={12} /> : <ArrowUpFromLine size={12} />}
        {input ? "Inputs" : "Outputs"} · {ports.length}
      </h4>
      {ports.length ? (
        <div {...stylex.props(s.portList)}>
          {ports.map((port) => {
            const artifactType = portArtifactType(port);
            const variable = portArtifactTypeVariable(port);
            const contract = artifactType
              ? `${artifactType.id}@${artifactType.schema_version}`
              : variable ?? "generic";
            const acceptedShapeRule = acceptedPortShapes(port)
              .map((shape) =>
                shape === "many" ? "sequence" : "single value",
              )
              .join(" or ");
            const rules = [
              port.required ? "required" : "optional",
              acceptedShapeRule,
              portHasInstancePlugs(port)
                ? "ordered input plugs"
                : port.variadic
                  ? "multiple connections"
                  : null,
            ].filter(Boolean).join(" · ");
            return (
              <div key={`${direction}-${port.name}`} {...stylex.props(s.portRow)}>
                <span
                  aria-hidden="true"
                  {...stylex.props(s.portDot)}
                  style={{
                    backgroundColor: artifactType
                      ? artifactTypeColor(artifactType.id, tokens.colorAccent)
                      : tokens.colorAccent,
                  }}
                />
                <div {...stylex.props(s.portCopy)}>
                  <div {...stylex.props(s.portTitle)}>
                    {port.title ?? port.name}
                  </div>
                  <div {...stylex.props(s.portContract)}>
                    {artifactTitleFor(registry, port)} · {contract}
                  </div>
                  <div {...stylex.props(s.portRules)}>{rules}</div>
                  {port.description ? (
                    <p {...stylex.props(s.portDescription)}>{port.description}</p>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p {...stylex.props(s.compatibilityEmpty)}>
          {input ? "No inputs. This node can start a workflow." : "No outputs. This node finishes a branch."}
        </p>
      )}
    </div>
  );
}

export function NodeSelector({
  open,
  registry,
  activeGraphId,
  compatibility,
  errorMessage = null,
  loading = false,
  canInsert = true,
  insertDisabledReason = "You do not have permission to edit this graph.",
  returnFocusRef: providedReturnFocusRef,
  onOpenChange,
  onAddNode,
  onRetry,
  onOpenGraph,
  onOpenWorkspaceLibrary,
}: NodeSelectorProps) {
  const [query, setQuery] = React.useState("");
  const [category, setCategory] = React.useState<NodeGoalCategoryId>(
    compatibility ? "all" : "suggested",
  );
  const [selectedNodeKey, setSelectedNodeKey] = React.useState<string | null>(null);
  const [selectedRelease, setSelectedRelease] = React.useState<{
    moduleKey: string;
    releaseKey: string;
  } | null>(null);
  const resultRefs = React.useRef(new Map<string, HTMLButtonElement>());
  const categoryRefs = React.useRef(new Map<NodeGoalCategoryId, HTMLButtonElement>());
  const wasOpen = React.useRef(false);

  const catalogNodes = React.useMemo(
    () => catalogNodeSpecs(registry, activeGraphId),
    [activeGraphId, registry],
  );
  const compatibleCatalogNodes = React.useMemo(
    () => compatibility
      ? catalogNodes.filter((spec) =>
          nodeMatchesCompatibility(spec, compatibility, registry),
        )
      : catalogNodes,
    [catalogNodes, compatibility, registry],
  );
  const catalogRegistry = React.useMemo(
    () => ({ ...registry, nodes: catalogNodes }),
    [catalogNodes, registry],
  );
  const showingModules = category === "reuse";
  const activeEditingModule = React.useMemo(
    () =>
      activeGraphId
        ? registry.nodes.find(
            (spec) =>
              spec.module_graph_id === activeGraphId &&
              spec.catalog_visible !== false,
          ) ?? null
        : null,
    [activeGraphId, registry.nodes],
  );

  React.useEffect(() => {
    if (open && !wasOpen.current) {
      setQuery("");
      setCategory(compatibility ? "all" : "suggested");
      setSelectedNodeKey(null);
      setSelectedRelease(null);
    }
    wasOpen.current = open;
  }, [compatibility, open]);

  const normalizedQuery = query.trim().toLowerCase();
  const filteredNodes = React.useMemo(
    () => {
      if (loading || errorMessage) return [];
      return catalogNodesForGoal(compatibleCatalogNodes, category).filter(
        (spec) =>
          !normalizedQuery ||
          nodeSearchText(spec, pluginFor(registry, spec.plugin_slug)).includes(
            normalizedQuery,
          ),
      );
    },
    [category, compatibleCatalogNodes, errorMessage, loading, normalizedQuery, registry],
  );
  const listedSpec = filteredNodes.find(
    (spec) => nodeKey(spec) === selectedNodeKey,
  ) ?? filteredNodes[0] ?? null;
  const moduleReleases = React.useMemo(
    () =>
      listedSpec?.plugin_slug === MODULE_PLUGIN_SLUG
        ? moduleReleaseSpecs(
            registry,
            listedSpec.module_id,
            listedSpec.module_graph_id,
          )
        : [],
    [listedSpec, registry],
  );
  const selectedSpec = React.useMemo(() => {
    if (!listedSpec) return null;
    if (listedSpec.plugin_slug !== MODULE_PLUGIN_SLUG) return listedSpec;
    const moduleKey = listedSpec.module_id ?? listedSpec.module_graph_id;
    const selectedReleaseKey =
      moduleKey && selectedRelease?.moduleKey === moduleKey
        ? selectedRelease.releaseKey
        : null;
    return (
      moduleReleases.find((spec) => nodeKey(spec) === selectedReleaseKey) ??
      moduleReleases.find((spec) => spec.catalog_visible !== false) ??
      moduleReleases[0] ??
      listedSpec
    );
  }, [listedSpec, moduleReleases, selectedRelease]);
  const selectedFields = selectedSpec
    ? schemaFields(selectedSpec.config_schema)
    : [];
  const upstreamMatches = React.useMemo(
    () => selectedSpec
      ? compatibleNodes(selectedSpec, "upstream", catalogRegistry)
      : [],
    [catalogRegistry, selectedSpec],
  );
  const downstreamMatches = React.useMemo(
    () => selectedSpec
      ? compatibleNodes(selectedSpec, "downstream", catalogRegistry)
      : [],
    [catalogRegistry, selectedSpec],
  );
  const selectedPlugin = selectedSpec
    ? pluginFor(registry, selectedSpec.plugin_slug)
    : null;
  const activeCategoryTitle = NODE_GOAL_CATEGORIES.find(
    (candidate) => candidate.id === category,
  )?.title ?? "Nodes";
  const isModuleSelection = selectedPlugin?.origin === "module";
  const isDeprecatedModule = selectedSpec?.publication_state === "deprecated";
  const activeResultId = listedSpec
    ? `node-selector-result-${nodeKey(listedSpec)}`
    : undefined;
  const compatibilityPortTitle = compatibility
    ? compatibility.port.title ?? compatibility.port.name
    : null;
  const resultStatus = loading
    ? "Loading nodes…"
    : errorMessage
      ? "Nodes could not be loaded."
      : filteredNodes.length === 0
        ? "No nodes found."
        : `${filteredNodes.length} ${filteredNodes.length === 1 ? "node" : "nodes"}.`;

  const selectCategory = (nextCategory: NodeGoalCategoryId) => {
    setCategory(nextCategory);
    setSelectedNodeKey(null);
  };

  const focusCategoryAt = (index: number) => {
    const boundedIndex = Math.max(
      0,
      Math.min(index, NODE_GOAL_CATEGORIES.length - 1),
    );
    const nextCategory = NODE_GOAL_CATEGORIES[boundedIndex];
    if (!nextCategory) return;
    selectCategory(nextCategory.id);
    categoryRefs.current.get(nextCategory.id)?.focus();
  };

  const focusResultAt = (index: number) => {
    if (!filteredNodes.length) return;
    const boundedIndex = Math.max(0, Math.min(index, filteredNodes.length - 1));
    const nextSpec = filteredNodes[boundedIndex];
    if (!nextSpec) return;
    const key = nodeKey(nextSpec);
    setSelectedNodeKey(key);
    resultRefs.current.get(key)?.focus();
  };

  const insertSelected = () => {
    if (!selectedSpec || !canInsert) return;
    if (
      isDeprecatedModule &&
      !window.confirm(
        `Insert deprecated Module “${selectedSpec.title}”? New inserts are discouraged. Existing pinned calls keep working.`,
      )
    ) {
      return;
    }
    onAddNode(selectedSpec);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        aria-labelledby="node-selector-title"
        aria-describedby="node-selector-description"
        finalFocus={providedReturnFocusRef}
        style={{
          width: "min(1040px, calc(100vw - 40px))",
          maxWidth: "none",
          height: "min(720px, calc(100vh - 40px))",
          maxHeight: "none",
        }}
      >
        <div {...stylex.props(s.header)}>
          <div {...stylex.props(s.heading)}>
            <div {...stylex.props(s.titleRow)}>
              <span {...stylex.props(s.titleIcon)}>
                <Workflow size={15} />
              </span>
              <DialogTitle id="node-selector-title" {...stylex.props(s.title)}>
                Add node
              </DialogTitle>
            </div>
            <DialogDescription
              id="node-selector-description"
              {...stylex.props(s.description)}
            >
              Search, inspect the contract, then insert into the current graph.
            </DialogDescription>
          </div>
          <div {...stylex.props(s.searchWrap)}>
            <Search size={14} {...stylex.props(s.searchIcon)} />
            <input
              autoFocus
              aria-label="Search nodes"
              aria-autocomplete="list"
              aria-controls="node-selector-results"
              aria-activedescendant={activeResultId}
              value={query}
              placeholder="Search nodes, ports, types, or settings…"
              {...stylex.props(s.search)}
              onChange={(event) => {
                const nextQuery = event.currentTarget.value;
                setQuery(nextQuery);
                setCategory(
                  nextQuery.trim() || compatibility ? "all" : "suggested",
                );
                setSelectedNodeKey(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  focusResultAt(0);
                } else if (event.key === "Enter") {
                  event.preventDefault();
                  insertSelected();
                }
              }}
            />
          </div>
        </div>

        <div {...stylex.props(s.layout)}>
          <section aria-labelledby="node-selector-results-heading" {...stylex.props(s.nodePane)}>
            <header {...stylex.props(s.nodePaneHeader)}>
              <div {...stylex.props(s.resultHeading)}>
                <h3 id="node-selector-results-heading" {...stylex.props(s.nodePaneTitle)}>
                  {activeCategoryTitle}
                </h3>
                <span
                  role={errorMessage ? "alert" : "status"}
                  aria-live={errorMessage ? "assertive" : "polite"}
                  aria-atomic="true"
                  {...stylex.props(s.resultCount)}
                >
                  {resultStatus}
                </span>
              </div>
              <div
                role="toolbar"
                aria-label="Node categories"
                aria-orientation="horizontal"
                {...stylex.props(s.categoryToolbar)}
              >
                {NODE_GOAL_CATEGORIES.map((goal, index) => {
                  const active = category === goal.id;
                  const count = catalogNodesForGoal(
                    compatibleCatalogNodes,
                    goal.id,
                  ).length;
                  return (
                    <button
                      key={goal.id}
                      ref={(element) => {
                        if (element) categoryRefs.current.set(goal.id, element);
                        else categoryRefs.current.delete(goal.id);
                      }}
                      type="button"
                      tabIndex={active ? 0 : -1}
                      aria-label={`${goal.title}, ${count} ${count === 1 ? "node" : "nodes"}`}
                      aria-pressed={active}
                      {...stylex.props(
                        s.categoryButton,
                        active ? s.categoryButtonActive : null,
                      )}
                      onClick={() => selectCategory(goal.id)}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowRight") {
                          event.preventDefault();
                          focusCategoryAt(index + 1);
                        } else if (event.key === "ArrowLeft") {
                          event.preventDefault();
                          focusCategoryAt(index - 1);
                        } else if (event.key === "Home") {
                          event.preventDefault();
                          focusCategoryAt(0);
                        } else if (event.key === "End") {
                          event.preventDefault();
                          focusCategoryAt(NODE_GOAL_CATEGORIES.length - 1);
                        }
                      }}
                    >
                      {goal.title}
                    </button>
                  );
                })}
              </div>
            </header>
            {compatibility && compatibilityPortTitle ? (
              <div id="node-selector-compatibility" {...stylex.props(s.compatibilityBanner)}>
                <Cable size={12} aria-hidden="true" />
                Showing nodes that can connect {compatibility.direction === "upstream" ? "to" : "from"}{" "}
                <strong>{compatibilityPortTitle}</strong>.
              </div>
            ) : null}
            <div
              id="node-selector-results"
              role="listbox"
              aria-label="Node results"
              aria-busy={loading}
              aria-activedescendant={activeResultId}
              {...stylex.props(s.nodeList)}
            >
              {errorMessage ? (
                <div {...stylex.props(s.empty)}>
                  <span>Nodes couldn’t be loaded. {errorMessage}</span>
                  {onRetry ? (
                    <button type="button" {...stylex.props(s.resetButton)} onClick={onRetry}>
                      Try again
                    </button>
                  ) : null}
                </div>
              ) : loading ? (
                <div {...stylex.props(s.empty)}>Loading nodes…</div>
              ) : filteredNodes.length ? filteredNodes.map((spec, index) => {
                const owner = pluginFor(registry, spec.plugin_slug);
                const key = nodeKey(spec);
                const active = listedSpec
                  ? key === nodeKey(listedSpec)
                  : false;
                const moduleState = spec.publication_state ?? "published";
                return (
                  <button
                    key={key}
                    id={`node-selector-result-${key}`}
                    ref={(element) => {
                      if (element) resultRefs.current.set(key, element);
                      else resultRefs.current.delete(key);
                    }}
                    type="button"
                    role="option"
                    tabIndex={active ? 0 : -1}
                    aria-selected={active}
                    {...stylex.props(
                      s.nodeButton,
                      active ? s.nodeButtonActive : null,
                    )}
                    onClick={() => setSelectedNodeKey(key)}
                    onFocus={() => setSelectedNodeKey(key)}
                    onKeyDown={(event) => {
                      if (event.key === "ArrowDown") {
                        event.preventDefault();
                        focusResultAt(index + 1);
                      } else if (event.key === "ArrowUp") {
                        event.preventDefault();
                        focusResultAt(index - 1);
                      } else if (event.key === "Home") {
                        event.preventDefault();
                        focusResultAt(0);
                      } else if (event.key === "End") {
                        event.preventDefault();
                        focusResultAt(filteredNodes.length - 1);
                      } else if (event.key === "Enter") {
                        event.preventDefault();
                        insertSelected();
                      }
                    }}
                  >
                    <span {...stylex.props(s.nodeCopy)}>
                      <span {...stylex.props(s.nodeTitleRow)}>
                        <span {...stylex.props(s.nodeTitle)}>{spec.title}</span>
                        {owner.origin === "external" ? (
                          <span
                            {...stylex.props(
                              s.originBadge,
                              s.originBadgeExternal,
                            )}
                          >
                            External
                          </span>
                        ) : owner.origin === "module" ? (
                          <span {...stylex.props(s.originBadge)}>
                            Module · release {spec.module_graph_revision} · {moduleState}
                          </span>
                        ) : null}
                      </span>
                      <span {...stylex.props(s.nodeDescription)}>
                        {spec.description || "No description is available."}
                      </span>
                    </span>
                    <span {...stylex.props(s.nodePorts)}>
                      {spec.inputs.length} in · {spec.outputs.length} out
                    </span>
                  </button>
                );
              }) : (
                <div {...stylex.props(s.empty)}>
                  <span>
                    {compatibility
                      ? "No nodes match this port and the current search or category."
                      : showingModules
                        ? "No published Modules match the current search."
                        : "No nodes match the current search or category."}
                  </span>
                  {normalizedQuery || category !== "suggested" ? (
                    <button
                      type="button"
                      {...stylex.props(s.resetButton)}
                      onClick={() => {
                        setQuery("");
                        selectCategory(compatibility ? "all" : "suggested");
                      }}
                    >
                      Reset search and category
                    </button>
                  ) : null}
                </div>
              )}
            </div>
            {showingModules && !loading && !errorMessage ? (
              <div
                aria-label="Workspace library notes"
                {...stylex.props(s.moduleDiagnostics)}
              >
                {activeEditingModule ? (
                  <p {...stylex.props(s.moduleDiagnosticsNote)}>
                    “{activeEditingModule.title}” is hidden here because it is
                    the graph currently being edited.
                  </p>
                ) : null}
                {filteredNodes.length === 0 && !normalizedQuery ? (
                  <p {...stylex.props(s.moduleDiagnosticsNote)}>
                    {compatibility
                      ? "No published Modules in this workspace can connect to this port."
                      : "No published Modules in this workspace yet. Open a source graph, declare Module Input/Output boundaries, then Publish release."}
                    {onOpenWorkspaceLibrary ? (
                      <>
                        {" "}
                        <button
                          type="button"
                          {...stylex.props(s.resetButton)}
                          onClick={onOpenWorkspaceLibrary}
                        >
                          Open workspace library
                        </button>
                      </>
                    ) : null}
                  </p>
                ) : null}
              </div>
            ) : null}
          </section>

          <aside
            aria-label="Node information"
            aria-labelledby={selectedSpec ? "node-selector-inspector-title" : undefined}
            {...stylex.props(s.inspector)}
          >
            {selectedSpec && selectedPlugin ? (
              <>
                <div
                  key={nodeKey(selectedSpec)}
                  className={`${stylex.props(s.inspectorScroll).className} ns-node-detail`}
                >
                  <header {...stylex.props(s.inspectorHeader)}>
                    <div {...stylex.props(s.inspectorProvenance)}>
                      <div {...stylex.props(s.eyebrow)}>
                        {isModuleSelection
                          ? `Module · release ${selectedSpec.module_graph_revision}`
                          : selectedPlugin.title}
                      </div>
                      <span
                        {...stylex.props(
                          s.originBadge,
                          selectedPlugin.origin === "external"
                            ? s.originBadgeExternal
                            : null,
                        )}
                      >
                        {selectedPlugin.origin === "external"
                          ? "External"
                          : selectedPlugin.origin === "module"
                            ? selectedSpec.publication_state ?? "published"
                            : "Built-in"}
                      </span>
                    </div>
                    <h3 id="node-selector-inspector-title" {...stylex.props(s.inspectorTitle)}>
                      {selectedSpec.title}
                    </h3>
                    <div {...stylex.props(s.operatorId)}>
                      {typeof selectedSpec.module_graph_revision === "number"
                        ? `Module contract · release ${selectedSpec.module_graph_revision}`
                        : `${selectedSpec.operator_id}@${selectedSpec.operator_version}`}
                    </div>
                    {isModuleSelection && moduleReleases.length > 1 ? (
                      <label {...stylex.props(s.operatorId)}>
                        Release{" "}
                        <select
                          aria-label="Module release"
                          value={nodeKey(selectedSpec)}
                          onChange={(event) => {
                            const moduleKey =
                              listedSpec.module_id ?? listedSpec.module_graph_id;
                            if (!moduleKey) return;
                            setSelectedRelease({
                              moduleKey,
                              releaseKey: event.currentTarget.value,
                            });
                          }}
                        >
                          {moduleReleases.map((release) => (
                            <option key={nodeKey(release)} value={nodeKey(release)}>
                              Release {release.module_graph_revision}
                              {release.is_current_library_release
                                ? " (current)"
                                : ""}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : null}
                    {isDeprecatedModule ? (
                      <p {...stylex.props(s.moduleDiagnosticsNote)}>
                        This Module is deprecated. New inserts are discouraged;
                        existing pins keep working.
                      </p>
                    ) : null}
                    {selectedSpec.module_graph_id && onOpenGraph ? (
                      <button
                        type="button"
                        title="Open the saved graph that defines this module"
                        {...stylex.props(
                          s.openGraphButton,
                          s.inspectorOpenGraph,
                        )}
                        onClick={() =>
                          onOpenGraph(selectedSpec.module_graph_id!)
                        }
                      >
                        <ExternalLink size={10} />
                        Open source graph
                      </button>
                    ) : null}
                    <p {...stylex.props(s.inspectorDescription)}>
                      {selectedSpec.description || "No description is available for this node."}
                    </p>
                    <div {...stylex.props(s.facts)}>
                      <span><strong {...stylex.props(s.factStrong)}>{selectedSpec.inputs.length}</strong> inputs</span>
                      <span><strong {...stylex.props(s.factStrong)}>{selectedSpec.outputs.length}</strong> outputs</span>
                      <span><strong {...stylex.props(s.factStrong)}>{selectedFields.length}</strong> editable settings</span>
                    </div>
                  </header>

                  <section {...stylex.props(s.section)}>
                    <div {...stylex.props(s.sectionTitleRow)}>
                      <Cable size={13} {...stylex.props(s.sectionIcon)} />
                      <h3 {...stylex.props(s.sectionTitle)}>Works with</h3>
                    </div>
                    <div {...stylex.props(s.compatibilityGrid)}>
                      <CompatibilityList
                        title="Can receive from"
                        matches={upstreamMatches}
                        registry={registry}
                        emptyMessage={selectedSpec.inputs.length === 0
                          ? "No upstream node is needed; this node starts a workflow."
                          : "No registered node currently provides a compatible output."}
                      />
                      <CompatibilityList
                        title="Can connect to"
                        matches={downstreamMatches}
                        registry={registry}
                        emptyMessage={selectedSpec.outputs.length === 0
                          ? "No outputs are declared; this node finishes a branch."
                          : "No registered node currently accepts these outputs."}
                      />
                    </div>
                  </section>

                  <section {...stylex.props(s.section)}>
                    <div {...stylex.props(s.sectionTitleRow)}>
                      <Workflow size={13} {...stylex.props(s.sectionIcon)} />
                      <h3 {...stylex.props(s.sectionTitle)}>
                        {isModuleSelection ? "Module contract" : "Ports"}
                      </h3>
                    </div>
                    <div {...stylex.props(s.portGrid)}>
                      <PortList
                        direction="input"
                        ports={selectedSpec.inputs}
                        registry={registry}
                      />
                      <PortList
                        direction="output"
                        ports={selectedSpec.outputs}
                        registry={registry}
                      />
                    </div>
                  </section>

                  <section {...stylex.props(s.section)}>
                    <div {...stylex.props(s.sectionTitleRow)}>
                      <Settings2 size={13} {...stylex.props(s.sectionIcon)} />
                      <h3 {...stylex.props(s.sectionTitle)}>Configuration</h3>
                    </div>
                    {selectedFields.length ? (
                      <div {...stylex.props(s.fieldList)}>
                        {selectedFields.map((field) => (
                          <div key={field.name} {...stylex.props(s.fieldRow)}>
                            <div {...stylex.props(s.fieldIdentity)}>
                              <div {...stylex.props(s.fieldTitle)}>{field.title}</div>
                              <div {...stylex.props(s.fieldName)}>{field.name}</div>
                            </div>
                            <div {...stylex.props(s.fieldDetails)}>
                              <div {...stylex.props(s.fieldMeta)}>
                                {fieldTypeLabel(field)} · {fieldConstraintLabel(field)}
                              </div>
                              {field.description ? (
                                <p {...stylex.props(s.fieldDescription)}>{field.description}</p>
                              ) : null}
                              {field.enumValues?.length ? (
                                <p {...stylex.props(s.fieldChoices)}>
                                  Choices: {field.enumValues.map(String).join(", ")}
                                </p>
                              ) : null}
                              {field.pattern ? (
                                <p {...stylex.props(s.fieldChoices)}>Pattern: {field.pattern}</p>
                              ) : null}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p {...stylex.props(s.compatibilityEmpty)}>
                        No editable scalar settings are declared. Upload or custom controls, when available, appear on the node after it is added.
                      </p>
                    )}
                  </section>
                </div>

                <footer {...stylex.props(s.inspectorFooter)}>
                  <span id="node-selector-insert-hint" {...stylex.props(s.footerHint)}>
                    {!canInsert
                      ? insertDisabledReason
                      : isModuleSelection
                        ? "Inserts a Module call pinned to the selected immutable release."
                        : "Added at the center of the current canvas view."}
                  </span>
                  <button
                    type="button"
                    disabled={!canInsert}
                    aria-describedby="node-selector-insert-hint"
                    title={
                      !canInsert
                        ? insertDisabledReason
                        : isModuleSelection
                        ? `Insert module call for ${selectedSpec.title}`
                        : `Add ${selectedSpec.title} to the workflow`
                    }
                    {...stylex.props(
                      s.addButton,
                      !canInsert ? s.addButtonDisabled : null,
                    )}
                    onClick={insertSelected}
                  >
                    <Plus size={14} />{" "}
                    {isModuleSelection
                      ? "Insert module call"
                      : `Add ${selectedSpec.title}`}
                  </button>
                </footer>
              </>
            ) : (
              <div {...stylex.props(s.empty)}>
                {loading
                  ? "Loading node details…"
                  : errorMessage
                    ? "Node details are unavailable until the list loads."
                    : "Select a node to inspect its contract."}
              </div>
            )}
          </aside>
        </div>
      </DialogContent>
    </Dialog>
  );
}

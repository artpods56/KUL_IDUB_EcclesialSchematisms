"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Cable,
  Package,
  Plus,
  Search,
  Settings2,
  Workflow,
} from "lucide-react";

import {
  connectionRoutesFor,
  encodeHandleId,
  type ConnectionRoute,
} from "@/components/canvas/handles";
import { schemaFields, type SchemaField } from "@/components/canvas/config-schema";
import { ARTIFACT_TYPE_COLOR } from "@/components/canvas/nodes.css";
import {
  acceptedPortShapes,
  portArtifactType,
  portArtifactTypeVariable,
  portHasInstancePlugs,
  portMetaForPort,
} from "@/components/canvas/types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import type { NodeRegistry, NodeSpec, Port } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import {
  catalogNodeSpecs,
  catalogPluginSections,
} from "./node-catalog";

interface NodeSelectorProps {
  open: boolean;
  registry: NodeRegistry;
  activeGraphId: string | null;
  onOpenChange: (open: boolean) => void;
  onAddNode: (spec: NodeSpec) => void;
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
  return field.type;
}

function fieldConstraintLabel(field: SchemaField): string {
  const constraints = [field.required ? "required" : "optional"];
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
      default: "minmax(0, 1fr) minmax(280px, 390px)",
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
    height: "36px",
    padding: "0 12px 0 34px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: {
      default: tokens.colorBorderStrong,
      ":focus": tokens.colorAccent,
    },
    borderRadius: tokens.radiusSm,
    outline: "none",
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
  },
  layout: {
    minHeight: 0,
    flex: 1,
    display: "grid",
    gridTemplateColumns: {
      default: "180px 300px minmax(0, 1fr)",
      "@media (max-width: 900px)": "155px 260px minmax(0, 1fr)",
      "@media (max-width: 720px)": "1fr",
    },
    gridTemplateRows: {
      default: "minmax(0, 1fr)",
      "@media (max-width: 720px)": "auto minmax(180px, 0.65fr) minmax(280px, 1.35fr)",
    },
  },
  pluginPane: {
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
    backgroundColor: tokens.colorSurfaceMuted,
  },
  paneLabel: {
    padding: "14px 14px 8px",
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
  },
  pluginList: {
    minHeight: 0,
    display: "flex",
    flexDirection: {
      default: "column",
      "@media (max-width: 720px)": "row",
    },
    gap: "2px",
    alignItems: {
      default: "stretch",
      "@media (max-width: 720px)": "center",
    },
    padding: {
      default: "0 8px 12px",
      "@media (max-width: 720px)": "0 10px 10px",
    },
    overflowX: {
      default: "hidden",
      "@media (max-width: 720px)": "auto",
    },
    overflowY: {
      default: "auto",
      "@media (max-width: 720px)": "hidden",
    },
  },
  pluginSection: {
    minWidth: 0,
    display: "flex",
    flexDirection: {
      default: "column",
      "@media (max-width: 720px)": "row",
    },
    alignItems: {
      default: "stretch",
      "@media (max-width: 720px)": "center",
    },
    flexShrink: 0,
    gap: "2px",
    marginTop: {
      default: "5px",
      "@media (max-width: 720px)": 0,
    },
    paddingTop: {
      default: "5px",
      "@media (max-width: 720px)": 0,
    },
    paddingLeft: {
      default: 0,
      "@media (max-width: 720px)": "7px",
    },
    borderTopWidth: {
      default: 1,
      "@media (max-width: 720px)": 0,
    },
    borderTopStyle: "solid",
    borderTopColor: tokens.colorDivider,
    borderLeftWidth: {
      default: 0,
      "@media (max-width: 720px)": 1,
    },
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorDivider,
  },
  pluginSectionLabel: {
    padding: {
      default: "5px 8px 3px",
      "@media (max-width: 720px)": "0 5px 0 0",
    },
    color: tokens.colorSubtle,
    fontSize: "9px",
    fontWeight: 820,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
  },
  pluginSectionEmpty: {
    padding: {
      default: "5px 8px 7px",
      "@media (max-width: 720px)": "0 8px 0 0",
    },
    color: tokens.colorSubtle,
    fontSize: "10px",
    whiteSpace: "nowrap",
  },
  pluginButton: {
    position: "relative",
    minWidth: 0,
    minHeight: "38px",
    display: "grid",
    gridTemplateColumns: "16px minmax(0, 1fr) auto",
    alignItems: "center",
    gap: "7px",
    padding: "7px 8px",
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: { default: "transparent", ":hover": tokens.colorHover },
    color: tokens.colorMuted,
    cursor: "pointer",
    flexShrink: 0,
    textAlign: "left",
    whiteSpace: "nowrap",
    transitionProperty: "background-color, color",
    transitionDuration: "120ms",
  },
  pluginButtonActive: {
    backgroundColor: tokens.colorAccentSoft,
    color: tokens.colorTextEmphasis,
  },
  pluginIcon: { color: tokens.colorSubtle },
  pluginIconActive: { color: tokens.colorAccent },
  pluginName: {
    overflow: "hidden",
    fontSize: tokens.fontSizeSm,
    fontWeight: 680,
    textOverflow: "ellipsis",
  },
  pluginMeta: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
  },
  pluginCount: {
    color: tokens.colorSubtle,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: "10px",
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
    minHeight: "51px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    padding: "10px 13px",
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
    fontSize: tokens.fontSizeSm,
    fontWeight: 750,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
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
                    backgroundColor:
                      (artifactType
                        ? ARTIFACT_TYPE_COLOR[artifactType.id]
                        : undefined) ?? tokens.colorAccent,
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
  onOpenChange,
  onAddNode,
}: NodeSelectorProps) {
  const [query, setQuery] = React.useState("");
  const [pluginFilter, setPluginFilter] = React.useState<string | null>(null);
  const [selectedNodeKey, setSelectedNodeKey] = React.useState<string | null>(null);

  const pluginSections = catalogPluginSections(registry);
  const catalogNodes = React.useMemo(
    () => catalogNodeSpecs(registry, activeGraphId),
    [activeGraphId, registry],
  );
  const catalogRegistry = React.useMemo(
    () => ({ ...registry, nodes: catalogNodes }),
    [catalogNodes, registry],
  );

  const normalizedQuery = query.trim().toLowerCase();
  const filteredNodes = React.useMemo(
    () => catalogNodes.filter((spec) => {
      if (pluginFilter && spec.plugin_slug !== pluginFilter) return false;
      if (!normalizedQuery) return true;
      return nodeSearchText(spec, pluginFor(registry, spec.plugin_slug))
        .includes(normalizedQuery);
    }),
    [catalogNodes, normalizedQuery, pluginFilter, registry],
  );
  const selectedSpec = filteredNodes.find(
    (spec) => nodeKey(spec) === selectedNodeKey,
  ) ?? filteredNodes[0] ?? null;
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
  const activePluginTitle = pluginFilter
    ? pluginFor(registry, pluginFilter).title
    : "All nodes";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        aria-label="Node catalog"
        style={{
          width: "min(1120px, calc(100vw - 40px))",
          maxWidth: "none",
          height: "min(760px, calc(100vh - 40px))",
          maxHeight: "none",
        }}
      >
        <div {...stylex.props(s.header)}>
          <div {...stylex.props(s.heading)}>
            <div {...stylex.props(s.titleRow)}>
              <span {...stylex.props(s.titleIcon)}>
                <Workflow size={15} />
              </span>
              <DialogTitle {...stylex.props(s.title)}>Node catalog</DialogTitle>
            </div>
            <DialogDescription {...stylex.props(s.description)}>
              Browse built-in nodes, saved graph modules, and registered external
              plugins, then inspect contracts before adding a node.
            </DialogDescription>
          </div>
          <div {...stylex.props(s.searchWrap)}>
            <Search size={14} {...stylex.props(s.searchIcon)} />
            <input
              autoFocus
              aria-label="Search node catalog"
              value={query}
              placeholder="Search nodes, ports, types, or settings…"
              {...stylex.props(s.search)}
              onChange={(event) => setQuery(event.currentTarget.value)}
            />
          </div>
        </div>

        <div {...stylex.props(s.layout)}>
          <nav aria-label="Node catalog groups" {...stylex.props(s.pluginPane)}>
            <div {...stylex.props(s.paneLabel)}>Catalog</div>
            <div {...stylex.props(s.pluginList)}>
              <button
                type="button"
                aria-pressed={pluginFilter === null}
                {...stylex.props(
                  s.pluginButton,
                  pluginFilter === null ? s.pluginButtonActive : null,
                )}
                onClick={() => setPluginFilter(null)}
              >
                <Workflow
                  size={14}
                  {...stylex.props(
                    s.pluginIcon,
                    pluginFilter === null ? s.pluginIconActive : null,
                  )}
                />
                <span {...stylex.props(s.pluginName)}>All nodes</span>
                <span {...stylex.props(s.pluginCount)}>{catalogNodes.length}</span>
              </button>
              {pluginSections.map((section) => (
                <section
                  key={section.origin}
                  aria-labelledby={`node-catalog-${section.origin}`}
                  {...stylex.props(s.pluginSection)}
                >
                  <h3
                    id={`node-catalog-${section.origin}`}
                    {...stylex.props(s.pluginSectionLabel)}
                  >
                    {section.title}
                  </h3>
                  {section.plugins.length ? section.plugins.map((plugin) => {
                    const active = pluginFilter === plugin.slug;
                    const count = catalogNodes.filter(
                      (spec) => spec.plugin_slug === plugin.slug,
                    ).length;
                    return (
                      <button
                        key={plugin.slug}
                        type="button"
                        aria-pressed={active}
                        {...stylex.props(
                          s.pluginButton,
                          active ? s.pluginButtonActive : null,
                        )}
                        onClick={() => setPluginFilter(plugin.slug)}
                      >
                        <Package
                          size={14}
                          {...stylex.props(
                            s.pluginIcon,
                            active ? s.pluginIconActive : null,
                          )}
                        />
                        <span {...stylex.props(s.pluginName)}>{plugin.title}</span>
                        <span {...stylex.props(s.pluginMeta)}>
                          {plugin.origin === "external" ? (
                            <span
                              {...stylex.props(
                                s.originBadge,
                                s.originBadgeExternal,
                              )}
                            >
                              External
                            </span>
                          ) : plugin.origin === "module" ? (
                            <span {...stylex.props(s.originBadge)}>
                              Module
                            </span>
                          ) : null}
                          <span {...stylex.props(s.pluginCount)}>{count}</span>
                        </span>
                      </button>
                    );
                  }) : (
                    <span {...stylex.props(s.pluginSectionEmpty)}>
                      None registered
                    </span>
                  )}
                </section>
              ))}
            </div>
          </nav>

          <section aria-label="Nodes" {...stylex.props(s.nodePane)}>
            <header {...stylex.props(s.nodePaneHeader)}>
              <h3 {...stylex.props(s.nodePaneTitle)}>{activePluginTitle}</h3>
              <span {...stylex.props(s.resultCount)}>
                {filteredNodes.length} {filteredNodes.length === 1 ? "node" : "nodes"}
              </span>
            </header>
            <div {...stylex.props(s.nodeList)}>
              {filteredNodes.length ? filteredNodes.map((spec) => {
                const owner = pluginFor(registry, spec.plugin_slug);
                const active = selectedSpec
                  ? nodeKey(spec) === nodeKey(selectedSpec)
                  : false;
                return (
                  <button
                    key={nodeKey(spec)}
                    type="button"
                    aria-pressed={active}
                    {...stylex.props(
                      s.nodeButton,
                      active ? s.nodeButtonActive : null,
                    )}
                    onClick={() => setSelectedNodeKey(nodeKey(spec))}
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
                            Module · r{spec.module_graph_revision}
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
                  <span>No nodes match the current category and search.</span>
                  <button
                    type="button"
                    {...stylex.props(s.resetButton)}
                    onClick={() => {
                      setQuery("");
                      setPluginFilter(null);
                    }}
                  >
                    Reset filters
                  </button>
                </div>
              )}
            </div>
          </section>

          <aside aria-label="Node information" {...stylex.props(s.inspector)}>
            {selectedSpec && selectedPlugin ? (
              <>
                <div
                  key={nodeKey(selectedSpec)}
                  className={`${stylex.props(s.inspectorScroll).className} ns-node-detail`}
                >
                  <header {...stylex.props(s.inspectorHeader)}>
                    <div {...stylex.props(s.inspectorProvenance)}>
                      <div {...stylex.props(s.eyebrow)}>
                        {selectedPlugin.title}
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
                            ? `Module · r${selectedSpec.module_graph_revision}`
                            : "Built-in"}
                      </span>
                    </div>
                    <h3 {...stylex.props(s.inspectorTitle)}>{selectedSpec.title}</h3>
                    <div {...stylex.props(s.operatorId)}>
                      {typeof selectedSpec.module_graph_revision === "number"
                        ? `Saved graph module · revision ${selectedSpec.module_graph_revision}`
                        : `${selectedSpec.operator_id}@${selectedSpec.operator_version}`}
                    </div>
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
                      <h3 {...stylex.props(s.sectionTitle)}>Ports</h3>
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
                  <span {...stylex.props(s.footerHint)}>
                    Added at the center of the current canvas view.
                  </span>
                  <button
                    type="button"
                    title={`Add ${selectedSpec.title} to the workflow`}
                    {...stylex.props(s.addButton)}
                    onClick={() => onAddNode(selectedSpec)}
                  >
                    <Plus size={14} /> Add {selectedSpec.title}
                  </button>
                </footer>
              </>
            ) : (
              <div {...stylex.props(s.empty)}>
                Select a node to inspect its contract.
              </div>
            )}
          </aside>
        </div>
      </DialogContent>
    </Dialog>
  );
}

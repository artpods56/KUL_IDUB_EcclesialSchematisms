import type { Edge, Node, XYPosition } from "@xyflow/react";

import {
  clampNodeLayout,
  type WorkflowNodeLayout,
} from "./node-layout";
import {
  WORKFLOW_NODE_TYPE,
  type WorkflowEdge,
  type WorkflowNodeData,
} from "./types";
import type {
  ArtifactInteractionField,
  ArtifactKeySelection,
  ArtifactViewerBinding,
  ArtifactViewerIncomingBinding,
} from "./artifact-interactions";

export const ARTIFACT_VIEWER_NODE_TYPE = "notariusArtifactViewerNode";
export const ARTIFACT_VIEWER_EDGE_TYPE = "notariusArtifactViewerEdge";
export const ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE =
  "notariusArtifactViewerInteractionEdge";
export const ARTIFACT_VIEWER_INPUT_HANDLE = "artifact-viewer-input";
export const ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE =
  "artifact-viewer-interaction-input";
export const ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE =
  "artifact-viewer-interaction-output";

export interface ArtifactViewerNodeData extends Record<string, unknown> {
  layout: WorkflowNodeLayout | null;
  mode: string | null;
  outgoingFields?: string[];
  selection?: ArtifactKeySelection;
  incomingBindings?: ArtifactViewerIncomingBinding[];
  fields?: ArtifactInteractionField[];
  onLayoutChange?: (
    nodeId: string,
    layout: WorkflowNodeLayout | null,
  ) => void;
  onModeChange?: (nodeId: string, mode: string) => void;
  onSelectionChange?: (
    nodeId: string,
    selection: ArtifactKeySelection,
  ) => void;
  onFieldsChange?: (
    nodeId: string,
    fields: ArtifactInteractionField[],
  ) => void;
  onRemoveNode?: (nodeId: string) => void;
}

export type ArtifactViewerNode = Node<
  ArtifactViewerNodeData,
  typeof ARTIFACT_VIEWER_NODE_TYPE
>;

export interface ArtifactViewerEdgeData extends Record<string, unknown> {
  sourcePortName: string;
}

export type ArtifactViewerEdge = Edge<
  ArtifactViewerEdgeData,
  typeof ARTIFACT_VIEWER_EDGE_TYPE
>;

export interface ArtifactViewerInteractionEdgeData
  extends Record<string, unknown> {
  binding: ArtifactViewerBinding;
  sourceFields?: ArtifactInteractionField[];
  targetFields?: ArtifactInteractionField[];
  onBindingChange?: (
    bindingId: string,
    binding: ArtifactViewerBinding,
  ) => void;
}

export type ArtifactViewerInteractionEdge = Edge<
  ArtifactViewerInteractionEdgeData,
  typeof ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE
>;

export type CanvasWorkflowNode = Node<
  WorkflowNodeData,
  typeof WORKFLOW_NODE_TYPE
>;
export type CanvasNode = CanvasWorkflowNode | ArtifactViewerNode;
export type CanvasEdge =
  | WorkflowEdge
  | ArtifactViewerEdge
  | ArtifactViewerInteractionEdge;

export interface ArtifactViewerCanvasState {
  graphId: string | null;
  nodes: ArtifactViewerNode[];
  edges: ArtifactViewerEdge[];
  bindings: ArtifactViewerBinding[];
}

interface PersistedArtifactViewer {
  id: string;
  position: XYPosition;
  layout: WorkflowNodeLayout | null;
  mode: string | null;
}

interface PersistedArtifactViewerLink {
  id: string;
  sourceNodeId: string;
  sourcePortName: string;
  targetViewerId: string;
}

interface ArtifactViewerDocumentV2 {
  schemaVersion: 2;
  viewers: PersistedArtifactViewer[];
  links: PersistedArtifactViewerLink[];
  bindings: ArtifactViewerBinding[];
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function finitePosition(value: unknown): XYPosition | null {
  const candidate = record(value);
  return candidate &&
      typeof candidate.x === "number" &&
      Number.isFinite(candidate.x) &&
      typeof candidate.y === "number" &&
      Number.isFinite(candidate.y)
    ? { x: candidate.x, y: candidate.y }
    : null;
}

function persistedLayout(value: unknown): WorkflowNodeLayout | null {
  const candidate = record(value);
  if (!candidate) return null;
  return clampNodeLayout({
    width: typeof candidate.width === "number" ? candidate.width : undefined,
    bodyHeight:
      typeof candidate.bodyHeight === "number"
        ? candidate.bodyHeight
        : undefined,
    appendixHeight:
      typeof candidate.appendixHeight === "number"
        ? candidate.appendixHeight
        : undefined,
  });
}

export function artifactViewerStorageKey(
  workspaceSlug: string,
  graphId: string,
): string {
  return `ns-workbench-presentation:v1:${encodeURIComponent(workspaceSlug)}:${graphId}`;
}

export function serializeArtifactViewerDocument(
  nodes: readonly ArtifactViewerNode[],
  edges: readonly ArtifactViewerEdge[],
  bindings: readonly ArtifactViewerBinding[],
): string {
  const document: ArtifactViewerDocumentV2 = {
    schemaVersion: 2,
    viewers: nodes.map((node) => ({
      id: node.id,
      position: { x: node.position.x, y: node.position.y },
      layout: clampNodeLayout(node.data.layout),
      mode: node.data.mode,
    })),
    links: edges.map((edge) => ({
      id: edge.id,
      sourceNodeId: edge.source,
      sourcePortName: edge.data?.sourcePortName ?? "",
      targetViewerId: edge.target,
    })),
    bindings: bindings.map((binding) => ({
      id: binding.id,
      sourceViewerId: binding.sourceViewerId,
      targetViewerId: binding.targetViewerId,
      mappings: binding.mappings.map((mapping) => ({
        sourceField: mapping.sourceField,
        targetField: mapping.targetField,
      })),
      effects: [...binding.effects],
      emptySelection: binding.emptySelection,
    })),
  };
  return JSON.stringify(document);
}

export function hydrateArtifactViewerDocument(
  serialized: string,
  graphId: string,
): ArtifactViewerCanvasState | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    return null;
  }
  const document = record(parsed);
  if (
    (document?.schemaVersion !== 1 && document?.schemaVersion !== 2) ||
    !Array.isArray(document.viewers) ||
    !Array.isArray(document.links) ||
    (document.schemaVersion === 2 && !Array.isArray(document.bindings))
  ) {
    return null;
  }

  const nodes: ArtifactViewerNode[] = [];
  const nodeIds = new Set<string>();
  for (const value of document.viewers) {
    const viewer = record(value);
    const position = finitePosition(viewer?.position);
    if (
      !viewer ||
      typeof viewer.id !== "string" ||
      !viewer.id.startsWith("artifact-viewer-") ||
      nodeIds.has(viewer.id) ||
      !position ||
      (viewer.mode !== null && typeof viewer.mode !== "string")
    ) {
      continue;
    }
    nodeIds.add(viewer.id);
    nodes.push({
      id: viewer.id,
      type: ARTIFACT_VIEWER_NODE_TYPE,
      position,
      data: {
        layout: persistedLayout(viewer.layout),
        mode: viewer.mode,
      },
    });
  }

  const edges: ArtifactViewerEdge[] = [];
  const edgeIds = new Set<string>();
  const connectedViewerIds = new Set<string>();
  for (const value of document.links) {
    const link = record(value);
    if (
      !link ||
      typeof link.id !== "string" ||
      !link.id.startsWith("artifact-viewer-edge-") ||
      edgeIds.has(link.id) ||
      typeof link.sourceNodeId !== "string" ||
      !link.sourceNodeId ||
      typeof link.sourcePortName !== "string" ||
      !link.sourcePortName ||
      typeof link.targetViewerId !== "string" ||
      !nodeIds.has(link.targetViewerId) ||
      connectedViewerIds.has(link.targetViewerId)
    ) {
      continue;
    }
    edgeIds.add(link.id);
    connectedViewerIds.add(link.targetViewerId);
    edges.push({
      id: link.id,
      type: ARTIFACT_VIEWER_EDGE_TYPE,
      source: link.sourceNodeId,
      target: link.targetViewerId,
      targetHandle: ARTIFACT_VIEWER_INPUT_HANDLE,
      data: { sourcePortName: link.sourcePortName },
    });
  }

  const bindings: ArtifactViewerBinding[] = [];
  const bindingIds = new Set<string>();
  if (document.schemaVersion === 2) {
    for (const value of document.bindings as unknown[]) {
      const binding = record(value);
      if (
        !binding ||
        typeof binding.id !== "string" ||
        !binding.id.startsWith("artifact-viewer-binding-") ||
        bindingIds.has(binding.id) ||
        typeof binding.sourceViewerId !== "string" ||
        !nodeIds.has(binding.sourceViewerId) ||
        typeof binding.targetViewerId !== "string" ||
        !nodeIds.has(binding.targetViewerId) ||
        binding.sourceViewerId === binding.targetViewerId ||
        !Array.isArray(binding.mappings) ||
        binding.mappings.length > 8 ||
        !Array.isArray(binding.effects) ||
        binding.effects.length === 0 ||
        binding.effects.length > 3 ||
        binding.emptySelection !== "show_all"
      ) {
        continue;
      }
      const mappings = binding.mappings.flatMap((value) => {
        const mapping = record(value);
        return mapping &&
            typeof mapping.sourceField === "string" &&
            mapping.sourceField.length <= 255 &&
            typeof mapping.targetField === "string" &&
            mapping.targetField.length <= 255
          ? [{
              sourceField: mapping.sourceField,
              targetField: mapping.targetField,
            }]
          : [];
      });
      const validEffects = binding.effects.every(
        (effect) =>
          effect === "filter" ||
          effect === "highlight" ||
          effect === "focus",
      );
      if (mappings.length !== binding.mappings.length || !validEffects) {
        continue;
      }
      const effects = [...new Set(binding.effects)] as ArtifactViewerBinding[
        "effects"
      ];
      bindingIds.add(binding.id);
      bindings.push({
        id: binding.id,
        sourceViewerId: binding.sourceViewerId,
        targetViewerId: binding.targetViewerId,
        mappings,
        effects,
        emptySelection: "show_all",
      });
    }
  }

  return { graphId, nodes, edges, bindings };
}

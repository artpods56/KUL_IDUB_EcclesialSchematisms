export const DEFAULT_NODE_WIDTH = 300;
export const DEFAULT_BODY_HEIGHT = 96;
export const DEFAULT_APPENDIX_HEIGHT = 230;

/** Floors keep node chrome (ports, textarea, appendix) usable. */
export const NODE_WIDTH_MIN = 260;
export const BODY_HEIGHT_MIN = 96;
export const APPENDIX_HEIGHT_MIN = 120;

/**
 * Shared ceiling for width / body / appendix. Anchored to the common
 * browser/GPU max texture dimension; larger layers can fail to composite.
 */
export const LAYOUT_DIMENSION_MAX = 16_384;

export interface WorkflowNodeLayout {
  width?: number;
  bodyHeight?: number;
  appendixHeight?: number;
}

export interface SavedNodeLayout {
  width?: number | null;
  body_height?: number | null;
  appendix_height?: number | null;
}

function clamp(
  value: number | undefined,
  min: number,
  max: number,
): number | undefined {
  if (value === undefined || !Number.isFinite(value)) return undefined;
  return Math.min(max, Math.max(min, value));
}

/** Clamp layout dimensions; returns null when nothing remains set. */
export function clampNodeLayout(
  layout: WorkflowNodeLayout | null | undefined,
): WorkflowNodeLayout | null {
  if (!layout) return null;
  const next: WorkflowNodeLayout = {};
  const width = clamp(layout.width, NODE_WIDTH_MIN, LAYOUT_DIMENSION_MAX);
  const bodyHeight = clamp(
    layout.bodyHeight,
    BODY_HEIGHT_MIN,
    LAYOUT_DIMENSION_MAX,
  );
  const appendixHeight = clamp(
    layout.appendixHeight,
    APPENDIX_HEIGHT_MIN,
    LAYOUT_DIMENSION_MAX,
  );
  if (width !== undefined) next.width = width;
  if (bodyHeight !== undefined) next.bodyHeight = bodyHeight;
  if (appendixHeight !== undefined) next.appendixHeight = appendixHeight;
  return next.width !== undefined ||
    next.bodyHeight !== undefined ||
    next.appendixHeight !== undefined
    ? next
    : null;
}

export function mergeNodeLayout(
  current: WorkflowNodeLayout | null | undefined,
  patch: WorkflowNodeLayout,
): WorkflowNodeLayout | null {
  return clampNodeLayout({
    width: patch.width ?? current?.width,
    bodyHeight: patch.bodyHeight ?? current?.bodyHeight,
    appendixHeight: patch.appendixHeight ?? current?.appendixHeight,
  });
}

export function serializeNodeLayout(
  layout: WorkflowNodeLayout | null | undefined,
): SavedNodeLayout | null {
  const clamped = clampNodeLayout(layout);
  if (!clamped) return null;
  return {
    width: clamped.width ?? null,
    body_height: clamped.bodyHeight ?? null,
    appendix_height: clamped.appendixHeight ?? null,
  };
}

export function hydrateNodeLayout(
  layout: SavedNodeLayout | null | undefined,
): WorkflowNodeLayout | null {
  if (!layout) return null;
  return clampNodeLayout({
    width: layout.width ?? undefined,
    bodyHeight: layout.body_height ?? undefined,
    appendixHeight: layout.appendix_height ?? undefined,
  });
}

export function resolvedNodeWidth(
  layout: WorkflowNodeLayout | null | undefined,
): number {
  return layout?.width ?? DEFAULT_NODE_WIDTH;
}

export function resolvedBodyHeight(
  layout: WorkflowNodeLayout | null | undefined,
): number | null {
  return layout?.bodyHeight ?? null;
}

export function resolvedAppendixHeight(
  layout: WorkflowNodeLayout | null | undefined,
): number {
  return layout?.appendixHeight ?? DEFAULT_APPENDIX_HEIGHT;
}

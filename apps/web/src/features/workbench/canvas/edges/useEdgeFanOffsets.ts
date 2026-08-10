"use client";

import type { InternalNode, Node } from "@xyflow/react";
import { Position, useStore } from "@xyflow/react";

import {
  edgeFanOffsetsById,
  fanSortAxis,
  type EdgeFanEndpoint,
  type EdgeFanOffsets,
  type FanSortAxis,
} from "./edge-path";

interface Point {
  x: number;
  y: number;
}

function handlePoint(
  node: InternalNode<Node> | undefined,
  handleId: string | null | undefined,
  handleType: "source" | "target",
): Point | undefined {
  if (!node) return undefined;
  const origin = node.internals.positionAbsolute;
  const handles = node.internals.handleBounds?.[handleType];
  const handle = handleId
    ? handles?.find((candidate) => candidate.id === handleId)
    : undefined;
  if (handle) {
    return {
      x: origin.x + handle.x + handle.width / 2,
      y: origin.y + handle.y + handle.height / 2,
    };
  }
  return {
    x: origin.x + (node.measured.width ?? 0) / 2,
    y: origin.y + (node.measured.height ?? 0) / 2,
  };
}

function orderPointsForEdges(
  edges: readonly EdgeFanEndpoint[],
  nodeLookup: ReadonlyMap<string, InternalNode<Node>>,
  far: "source" | "target",
): Map<string, Point> {
  const points = new Map<string, Point>();
  for (const edge of edges) {
    const point =
      far === "target"
        ? handlePoint(nodeLookup.get(edge.target), edge.targetHandle, "target")
        : handlePoint(nodeLookup.get(edge.source), edge.sourceHandle, "source");
    if (point) points.set(edge.id, point);
  }
  return points;
}

interface EdgeFanStoreState {
  edges: readonly EdgeFanEndpoint[];
  nodeLookup: ReadonlyMap<string, InternalNode<Node>>;
}

interface EdgeFanStateCache {
  sourceOrderPoints: ReadonlyMap<string, Point>;
  targetOrderPoints: ReadonlyMap<string, Point>;
  offsetsByAxes: Map<string, ReadonlyMap<string, EdgeFanOffsets>>;
}

const fanStateCache = new WeakMap<object, EdgeFanStateCache>();

function fanOffsetsForState(
  state: EdgeFanStoreState,
  sourceAxis: FanSortAxis,
  targetAxis: FanSortAxis,
): ReadonlyMap<string, EdgeFanOffsets> {
  let cached = fanStateCache.get(state);
  if (!cached) {
    cached = {
      sourceOrderPoints: orderPointsForEdges(
        state.edges,
        state.nodeLookup,
        "target",
      ),
      targetOrderPoints: orderPointsForEdges(
        state.edges,
        state.nodeLookup,
        "source",
      ),
      offsetsByAxes: new Map(),
    };
    fanStateCache.set(state, cached);
  }

  const axesKey = `${sourceAxis}:${targetAxis}`;
  let offsets = cached.offsetsByAxes.get(axesKey);
  if (!offsets) {
    offsets = edgeFanOffsetsById(state.edges, {
      sourceOrderPoints: cached.sourceOrderPoints,
      targetOrderPoints: cached.targetOrderPoints,
      sourceAxis,
      targetAxis,
    });
    cached.offsetsByAxes.set(axesKey, offsets);
  }
  return offsets;
}

/** Live fan offsets; sibling order tracks the far-end handle as nodes move. */
export function useEdgeFanOffsets(
  edgeId: string,
  sourcePosition: Position = Position.Right,
  targetPosition: Position = Position.Left,
): EdgeFanOffsets {
  const sourceAxis = fanSortAxis(sourcePosition);
  const targetAxis = fanSortAxis(targetPosition);
  return useStore(
    (state) =>
      fanOffsetsForState(state, sourceAxis, targetAxis).get(edgeId) ?? {
        source: 0,
        target: 0,
      },
    (left, right) =>
      left.source === right.source && left.target === right.target,
  );
}

"use client";

import type { InternalNode, Node } from "@xyflow/react";
import { Position, useStore } from "@xyflow/react";

import {
  edgeFanOffsets,
  fanSortAxis,
  type EdgeFanEndpoint,
  type EdgeFanOffsets,
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
  nodeLookup: Map<string, InternalNode<Node>>,
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

/** Live fan offsets; sibling order tracks the far-end handle as nodes move. */
export function useEdgeFanOffsets(
  edgeId: string,
  sourcePosition: Position = Position.Right,
  targetPosition: Position = Position.Left,
): EdgeFanOffsets {
  return useStore(
    (state) =>
      edgeFanOffsets(state.edges, edgeId, {
        // Source-side fan: order by where each cable is going.
        sourceOrderPoints: orderPointsForEdges(
          state.edges,
          state.nodeLookup,
          "target",
        ),
        // Target-side fan: order by where each cable is coming from.
        targetOrderPoints: orderPointsForEdges(
          state.edges,
          state.nodeLookup,
          "source",
        ),
        sourceAxis: fanSortAxis(sourcePosition),
        targetAxis: fanSortAxis(targetPosition),
      }),
    (left, right) =>
      left.source === right.source && left.target === right.target,
  );
}

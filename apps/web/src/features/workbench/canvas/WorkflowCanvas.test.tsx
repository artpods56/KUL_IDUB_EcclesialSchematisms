// @vitest-environment jsdom

import * as React from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const flowMocks = vi.hoisted(() => ({
  props: null as Record<string, unknown> | null,
}));

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({
    children,
    ...props
  }: {
    children?: React.ReactNode;
    [key: string]: unknown;
  }) => {
    flowMocks.props = props;
    return <div>{children}</div>;
  },
  Background: () => null,
  BackgroundVariant: { Lines: "lines" },
  Controls: () => null,
  addEdge: vi.fn(),
  applyEdgeChanges: vi.fn(),
  applyNodeChanges: vi.fn(),
}));

vi.mock("@/components/theme", () => ({
  useTheme: () => ({ resolved: "light" }),
}));

vi.mock("./handles", () => ({
  connectionIsValid: vi.fn(() => true),
}));
vi.mock("./edges/ArtifactViewerEdge", () => ({ default: () => null }));
vi.mock("./edges/ArtifactViewerInteractionEdge", () => ({ default: () => null }));
vi.mock("./edges/WorkflowEdge", () => ({ default: () => null }));
vi.mock("./nodes/AnnotationNode", () => ({ default: () => null }));
vi.mock("./nodes/ArtifactViewerNode", () => ({ default: () => null }));
vi.mock("./nodes/WorkflowNode", () => ({ default: () => null }));

import { WorkflowCanvas } from "./WorkflowCanvas";

const roots: Root[] = [];

afterEach(() => {
  React.act(() => {
    for (const root of roots.splice(0)) root.unmount();
  });
  flowMocks.props = null;
});

describe("WorkflowCanvas", () => {
  it("passes visible-element rendering through to React Flow", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    roots.push(root);

    React.act(() => {
      root.render(
        <WorkflowCanvas
          nodes={[]}
          edges={[]}
          onNodesChange={() => undefined}
          onEdgesChange={() => undefined}
          onConnect={() => undefined}
          onlyRenderVisibleElements
        />,
      );
    });

    expect(flowMocks.props?.onlyRenderVisibleElements).toBe(true);
  });
});

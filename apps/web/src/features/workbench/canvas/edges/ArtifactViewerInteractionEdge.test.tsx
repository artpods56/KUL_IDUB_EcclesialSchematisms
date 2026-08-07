// @vitest-environment jsdom

import * as React from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const flowMocks = vi.hoisted(() => ({
  deleteElements: vi.fn(),
}));

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

vi.mock("@xyflow/react", () => ({
  BaseEdge: () => <svg data-testid="base-edge" />,
  EdgeLabelRenderer: ({ children }: { children: React.ReactNode }) => children,
  getBezierPath: () => ["M 0 0 C 0 0 100 100 100 100", 50, 50],
  useReactFlow: () => ({ deleteElements: flowMocks.deleteElements }),
}));

import {
  ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE,
  ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE,
  ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE,
  type ArtifactViewerInteractionEdge,
} from "../artifact-viewer";
import ArtifactViewerInteractionEdgeControl from "./ArtifactViewerInteractionEdge";

const roots: Root[] = [];
const containers: HTMLElement[] = [];

afterEach(() => {
  React.act(() => {
    for (const root of roots.splice(0)) root.unmount();
  });
  for (const container of containers.splice(0)) container.remove();
  flowMocks.deleteElements.mockReset();
});

describe("ArtifactViewerInteractionEdge", () => {
  it("authors field mappings and effects from discovered viewer fields", () => {
    const onBindingChange = vi.fn();
    const edge: ArtifactViewerInteractionEdge = {
      id: "artifact-viewer-binding-1",
      type: ARTIFACT_VIEWER_INTERACTION_EDGE_TYPE,
      source: "artifact-viewer-table",
      sourceHandle: ARTIFACT_VIEWER_INTERACTION_OUTPUT_HANDLE,
      target: "artifact-viewer-map",
      targetHandle: ARTIFACT_VIEWER_INTERACTION_INPUT_HANDLE,
      data: {
        binding: {
          id: "artifact-viewer-binding-1",
          sourceViewerId: "artifact-viewer-table",
          targetViewerId: "artifact-viewer-map",
          mappings: [{ sourceField: "", targetField: "" }],
          effects: ["highlight", "focus"],
          emptySelection: "show_all",
        },
        sourceFields: [{
          id: "normalized_name",
          title: "Normalized name",
          valueType: "text",
        }],
        targetFields: [{
          id: "transliteration",
          title: "Transliteration",
          valueType: "text",
        }],
        onBindingChange,
      },
    };
    const container = document.createElement("div");
    document.body.append(container);
    containers.push(container);
    const root = createRoot(container);
    roots.push(root);
    React.act(() => {
      root.render(
        <ArtifactViewerInteractionEdgeControl
          {...({
            id: edge.id,
            data: edge.data,
            sourceX: 0,
            sourceY: 0,
            targetX: 100,
            targetY: 100,
            sourcePosition: "right",
            targetPosition: "left",
            selected: false,
          } as React.ComponentProps<
            typeof ArtifactViewerInteractionEdgeControl
          >)}
        />,
      );
    });

    expect(
      document.body.querySelector('[aria-label="Source field 1"]'),
    ).toBeNull();
    React.act(() => {
      container.querySelector<HTMLButtonElement>(
        '[aria-label="Configure viewer interaction"]',
      )?.click();
    });

    expect(
      document.body.querySelector('option[value="normalized_name"]')
        ?.getAttribute("label"),
    ).toBe("Normalized name · text");
    expect(
      document.body.querySelector('option[value="transliteration"]')
        ?.getAttribute("label"),
    ).toBe("Transliteration · text");

    const sourceInput = document.body.querySelector<HTMLInputElement>(
      '[aria-label="Source field 1"]',
    );
    React.act(() => {
      if (!sourceInput) return;
      const valueSetter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )?.set;
      valueSetter?.call(sourceInput, "normalized_name");
      sourceInput.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(onBindingChange).toHaveBeenLastCalledWith(
      "artifact-viewer-binding-1",
      expect.objectContaining({
        mappings: [{
          sourceField: "normalized_name",
          targetField: "",
        }],
      }),
    );

    const filter = [...document.body.querySelectorAll("label")].find(
      (label) => label.textContent?.trim() === "filter",
    )?.querySelector<HTMLInputElement>("input");
    React.act(() => filter?.click());
    expect(onBindingChange).toHaveBeenLastCalledWith(
      "artifact-viewer-binding-1",
      expect.objectContaining({
        effects: ["highlight", "focus", "filter"],
      }),
    );
  });

  it("removes the persisted binding through the canvas edge contract", () => {
    const container = document.createElement("div");
    document.body.append(container);
    containers.push(container);
    const root = createRoot(container);
    roots.push(root);
    React.act(() => {
      root.render(
        <ArtifactViewerInteractionEdgeControl
          {...({
            id: "artifact-viewer-binding-1",
            data: {
              binding: {
                id: "artifact-viewer-binding-1",
                sourceViewerId: "artifact-viewer-table",
                targetViewerId: "artifact-viewer-map",
                mappings: [{ sourceField: "id", targetField: "id" }],
                effects: ["highlight"],
                emptySelection: "show_all",
              },
            },
            sourceX: 0,
            sourceY: 0,
            targetX: 100,
            targetY: 100,
            sourcePosition: "right",
            targetPosition: "left",
            selected: false,
          } as React.ComponentProps<
            typeof ArtifactViewerInteractionEdgeControl
          >)}
        />,
      );
    });

    React.act(() => {
      container.querySelector<HTMLButtonElement>(
        '[aria-label="Remove viewer interaction"]',
      )?.click();
    });
    expect(flowMocks.deleteElements).toHaveBeenCalledWith({
      edges: [{ id: "artifact-viewer-binding-1" }],
    });
  });
});

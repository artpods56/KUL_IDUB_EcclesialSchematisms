// @vitest-environment jsdom

import * as React from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

import type { NodeSpec } from "@/lib/api";
import { createWorkflowNodeData } from "../types";
import { VectorLayerStyleBody } from "./VectorLayerStyleBody";

function vectorLayerSpec(): NodeSpec {
  return {
    operator_id: "gis.map.vector_layer",
    operator_version: 1,
    plugin_slug: "external.gis",
    origin: "builtin",
    title: "Vector map layer",
    description: "Styles one feature collection.",
    catalog_visible: true,
    runnable: true,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function choose(select: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLSelectElement.prototype,
    "value",
  )?.set;
  setter?.call(select, value);
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("VectorLayerStyleBody", () => {
  it("authors categorized styling as structured vector-layer config", () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    const onConfigChange = vi.fn();
    const data = {
      ...createWorkflowNodeData(vectorLayerSpec()),
      onConfigChange,
    };

    React.act(() => {
      root.render(<VectorLayerStyleBody id="vector-layer" data={data} />);
    });
    const mode = container.querySelector<HTMLSelectElement>(
      'select[aria-label="Feature style mode"]',
    );
    expect(mode).not.toBeNull();

    React.act(() => choose(mode!, "categorized_points"));

    expect(onConfigChange).toHaveBeenCalledWith(
      "vector-layer",
      "style",
      expect.objectContaining({
        kind: "categorized_points",
        category_property: "type",
        categories: [
          expect.objectContaining({
            id: "category_1",
            values: [1],
          }),
        ],
      }),
    );

    React.act(() => root.unmount());
    container.remove();
  });
});

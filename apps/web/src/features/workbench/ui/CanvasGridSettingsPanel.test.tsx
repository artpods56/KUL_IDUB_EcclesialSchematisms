// @vitest-environment jsdom

import * as React from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const settingsMocks = vi.hoisted(() => ({
  patchSettings: vi.fn(),
}));

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

vi.mock("../canvas/canvas-grid-settings", () => ({
  useCanvasGridSettings: () => ({
    settings: {
      enabled: true,
      showBackground: true,
      onlyRenderVisibleElements: false,
      snapPosition: true,
      snapSize: true,
      snapWhileDragging: false,
      snapWhileResizing: true,
      allowWorkflowCornerResize: false,
      cellSize: 50,
    },
    patchSettings: settingsMocks.patchSettings,
    resetSettings: vi.fn(),
    bypassSnap: false,
    panelOpen: true,
    setPanelOpen: vi.fn(),
  }),
}));

import { CanvasGridSettingsPanel } from "./CanvasGridSettingsPanel";

const roots: Root[] = [];

afterEach(() => {
  React.act(() => {
    for (const root of roots.splice(0)) root.unmount();
  });
  settingsMocks.patchSettings.mockReset();
});

describe("CanvasGridSettingsPanel", () => {
  it("exposes the offscreen rendering optimization as an opt-in switch", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    roots.push(root);

    React.act(() => {
      root.render(
        <CanvasGridSettingsPanel
          selectedCount={0}
          onSnapSelection={() => undefined}
        />,
      );
    });

    const toggle = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Render visible elements only"]',
    );
    expect(toggle?.getAttribute("aria-checked")).toBe("false");
    expect(container.textContent).toContain(
      "Temporary table and map view state can reset",
    );

    React.act(() => {
      toggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(settingsMocks.patchSettings).toHaveBeenCalledWith({
      onlyRenderVisibleElements: true,
    });
  });
});

// @vitest-environment jsdom

import * as React from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

const xyflowMocks = vi.hoisted(() => ({
  updateNodeInternals: vi.fn(),
}));

import type { NodeSpec } from "@/lib/api";
import {
  compatibilityHandleId,
  createWorkflowNodeData,
} from "../types";
import WorkflowNodeCard from "./WorkflowNode";

vi.mock("@xyflow/react", () => ({
  Handle: ({
    id,
    isConnectable,
  }: {
    id: string;
    isConnectable: boolean;
  }) => (
    <span
      data-testid="compatibility-handle"
      data-handle-id={id}
      data-connectable={String(isConnectable)}
    />
  ),
  Position: { Left: "left", Right: "right" },
  useEdges: () => [],
  useNodeConnections: () => [],
  useUpdateNodeInternals: () => xyflowMocks.updateNodeInternals,
}));

vi.mock("./LayoutResizeHandle", () => ({
  LayoutResizeHandle: () => null,
}));

vi.mock("./type-inspector", () => ({
  PortTypePopover: ({ children }: { children: React.ReactNode }) => children,
}));

function unavailableSpec(): NodeSpec {
  return {
    operator_id: "legacy.operator",
    operator_version: 4,
    plugin_slug: "unavailable",
    title: "legacy.operator",
    description: "Unavailable operator",
    catalog_visible: false,
    config_schema: {},
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function boundsSpec(): NodeSpec {
  return {
    operator_id: "gis.map.wms_layer",
    operator_version: 1,
    plugin_slug: "gis",
    title: "Remote WMS map layer",
    description: "Adds a remote WMS layer.",
    catalog_visible: true,
    config_schema: {
      type: "object",
      properties: {
        bounds: {
          type: "array",
          title: "Bounds",
          description:
            "WGS84 bounds ordered as west longitude, south latitude, east longitude, north latitude.",
          prefixItems: [
            {
              type: "number",
              title: "West longitude",
              minimum: -180,
              maximum: 180,
            },
            {
              type: "number",
              title: "South latitude",
              minimum: -90,
              maximum: 90,
            },
            {
              type: "number",
              title: "East longitude",
              minimum: -180,
              maximum: 180,
            },
            {
              type: "number",
              title: "North latitude",
              minimum: -90,
              maximum: 90,
            },
          ],
          minItems: 4,
          maxItems: 4,
        },
      },
      required: ["bounds"],
    },
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function fuzzyMatchSpec(): NodeSpec {
  return {
    operator_id: "table.fuzzy_match",
    operator_version: 1,
    plugin_slug: "builtin.table",
    title: "Fuzzy match tables",
    description: "Ranks candidate records.",
    catalog_visible: true,
    config_schema: {
      type: "object",
      properties: {
        right_alias_columns: {
          type: "array",
          title: "Right Alias Columns",
          description: "Additional normalized candidate-name columns.",
          items: { type: "string", minLength: 1 },
          maxItems: 8,
        },
      },
    },
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function artifactQuerySpec(): NodeSpec {
  return {
    operator_id: "sql.artifacts.query",
    operator_version: 1,
    plugin_slug: "external.sql",
    title: "Query artifact tables",
    description: "Runs read-only queries over table artifacts.",
    catalog_visible: true,
    config_schema: {
      type: "object",
      properties: { relations: { type: "array" } },
      required: ["relations"],
    },
    input_schema: {},
    output_schema: {},
    inputs: [
      {
        name: "statements",
        title: "Statements",
        description: null,
        direction: "input",
        artifact_type: { id: "sql.statement", schema_version: 1 },
        artifact_type_variable: null,
        shape: "one",
        accepted_shapes: ["one"],
        instance_plugs: true,
        variadic: true,
        required: true,
      },
      {
        name: "relations",
        title: "Relations",
        description: null,
        direction: "input",
        artifact_type: { id: "table.data", schema_version: 1 },
        artifact_type_variable: null,
        shape: "one",
        accepted_shapes: ["one"],
        instance_plugs: true,
        variadic: true,
        required: true,
      },
    ],
    outputs: [],
  };
}

function rawSqlStatementSpec(): NodeSpec {
  return {
    operator_id: "sql.statement.raw",
    operator_version: 1,
    plugin_slug: "external.sql",
    title: "Raw SQL statement",
    description: "Builds a parameterized SQL statement.",
    catalog_visible: true,
    config_schema: {
      type: "object",
      properties: {
        sql: {
          type: "string",
          title: "Sql",
          description:
            "SQL statement using canonical named :parameter placeholders.",
          format: "textarea",
          contentMediaType: "application/sql",
        },
      },
      required: ["sql"],
    },
    input_schema: {},
    output_schema: {},
    inputs: [],
    outputs: [],
  };
}

function enterInputValue(input: HTMLInputElement, value: string): void {
  const valueSetter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  if (!valueSetter) throw new Error("HTML input value setter is unavailable");
  valueSetter.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("WorkflowNode compatibility rendering", () => {
  it("renders an unsupported node with inert historical handles and removal", () => {
    const removeNode = vi.fn();
    const input = { portName: "request", plugId: "plug-1" };
    const output = { portName: "result" };
    const data = createWorkflowNodeData(unavailableSpec());
    data.config = { preserved: true };
    data.compatibility = {
      status: "unsupported",
      issues: ["Operator legacy.operator@4 is unavailable."],
      inputs: [input],
      outputs: [output],
      persistedNode: {
        id: "legacy-node",
        operator_id: "legacy.operator",
        operator_version: 4,
        config: { preserved: true },
        position: { x: 10, y: 20 },
        input_plugs: [{ id: "plug-1", port: "request" }],
        artifact_type_bindings: [],
      },
    };
    data.onRemoveNode = removeNode;

    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "legacy-node",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    expect(
      container.querySelector(
        '[aria-label="legacy.operator unsupported node"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("unsupported");
    expect(container.textContent).toContain(
      "Operator legacy.operator@4 is unavailable.",
    );
    expect(container.textContent).toContain('"preserved": true');

    const handles = [
      ...container.querySelectorAll<HTMLElement>(
        '[data-testid="compatibility-handle"]',
      ),
    ];
    expect(handles).toHaveLength(2);
    expect(handles.map((handle) => handle.dataset.handleId)).toEqual([
      compatibilityHandleId("input", input),
      compatibilityHandleId("output", output),
    ]);
    expect(
      handles.every((handle) => handle.dataset.connectable === "false"),
    ).toBe(true);

    const removeButton = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Remove legacy.operator"]',
    );
    expect(removeButton).not.toBeNull();
    React.act(() => removeButton?.click());
    expect(removeNode).toHaveBeenCalledWith("legacy-node");
    React.act(() => root.unmount());
  });
});

describe("WorkflowNode fixed numeric tuple fields", () => {
  it("renders coordinate inputs and emits a complete number array", () => {
    const onConfigChange = vi.fn();
    const data = createWorkflowNodeData(boundsSpec());
    data.onConfigChange = onConfigChange;

    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "wms-layer",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    const labels = [
      "Bounds: West longitude",
      "Bounds: South latitude",
      "Bounds: East longitude",
      "Bounds: North latitude",
    ];
    const inputs = labels.map((label) => {
      const input = container.querySelector<HTMLInputElement>(
        `input[aria-label="${label}"]`,
      );
      expect(input).not.toBeNull();
      return input!;
    });

    expect(inputs.map((input) => input.step)).toEqual([
      "any",
      "any",
      "any",
      "any",
    ]);
    expect(inputs.map((input) => [input.min, input.max])).toEqual([
      ["-180", "180"],
      ["-90", "90"],
      ["-180", "180"],
      ["-90", "90"],
    ]);

    ["181", "49.97", "19.82", "50.03"].forEach((value, index) => {
      React.act(() => enterInputValue(inputs[index]!, value));
    });

    expect(onConfigChange.mock.calls.map(([, , value]) => value)).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
    React.act(() => enterInputValue(inputs[0]!, "19.75"));

    expect(onConfigChange).toHaveBeenLastCalledWith(
      "wms-layer",
      "bounds",
      [19.75, 49.97, 19.82, 50.03],
    );
    expect(
      onConfigChange.mock.calls.every(
        ([, , configValue]) => typeof configValue !== "string",
      ),
    ).toBe(true);
    React.act(() => root.unmount());
  });
});

describe("WorkflowNode string-list fields", () => {
  it("edits, appends, and removes string values through the node form", () => {
    const onConfigChange = vi.fn();
    const data = createWorkflowNodeData(fuzzyMatchSpec());
    data.config = {
      right_alias_columns: ["candidate_current_name_normalized"],
    };
    data.onConfigChange = onConfigChange;

    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "fuzzy-match",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    const input = container.querySelector<HTMLInputElement>(
      'input[aria-label="Right Alias Columns item 1"]',
    );
    expect(input?.value).toBe("candidate_current_name_normalized");
    React.act(() => enterInputValue(input!, "alternate_name"));
    expect(onConfigChange).toHaveBeenLastCalledWith(
      "fuzzy-match",
      "right_alias_columns",
      ["alternate_name"],
    );

    React.act(() => {
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="Add Right Alias Columns item"]',
      )?.click();
    });
    expect(onConfigChange).toHaveBeenLastCalledWith(
      "fuzzy-match",
      "right_alias_columns",
      ["candidate_current_name_normalized", ""],
    );

    React.act(() => {
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="Remove Right Alias Columns item 1"]',
      )?.click();
    });
    expect(onConfigChange).toHaveBeenLastCalledWith(
      "fuzzy-match",
      "right_alias_columns",
      [],
    );
    React.act(() => root.unmount());
  });
});

describe("WorkflowNode multiline fields", () => {
  it("treats a saved body height as a minimum so the textarea cannot escape the node", () => {
    const data = createWorkflowNodeData(rawSqlStatementSpec());
    data.config = { sql: "select *\nfrom parcels" };
    data.layout = { bodyHeight: 96 };

    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "sql-statement",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    const textarea = container.querySelector("textarea");
    expect(textarea?.value).toBe("select *\nfrom parcels");
    const body = textarea?.parentElement?.parentElement?.parentElement;
    expect(body?.style.minHeight).toBe("96px");
    expect(body?.style.height).toBe("");

    React.act(() => root.unmount());
  });
});

describe("WorkflowNode artifact table query relations", () => {
  it("keeps SQL statements as ordinary plugs and edits named table relations", () => {
    const onRelationsChange = vi.fn();
    const data = createWorkflowNodeData(artifactQuerySpec());
    data.onArtifactQueryRelationsChange = onRelationsChange;
    const relationPlug = data.inputPlugs.find(
      (plug) => plug.portName === "relations",
    );

    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "artifact-query",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    expect(container.textContent).toContain("Statements");
    expect(container.textContent).toContain("Relations");
    const aliasInput = container.querySelector<HTMLInputElement>(
      'input[aria-label="Relation 1 SQL alias"]',
    );
    expect(aliasInput?.value).toBe("relation_1");
    expect(
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="Remove relation 1"]',
      )?.disabled,
    ).toBe(true);

    React.act(() => enterInputValue(aliasInput!, "parcels"));
    expect(onRelationsChange).toHaveBeenLastCalledWith(
      "artifact-query",
      [{ id: relationPlug?.id, alias: "parcels" }],
      expect.arrayContaining([
        { id: relationPlug?.id, portName: "relations" },
      ]),
    );

    const addRelationButton = [
      ...container.querySelectorAll<HTMLButtonElement>("button"),
    ].find((button) => button.textContent?.includes("Add relation"));
    expect(addRelationButton).toBeDefined();
    React.act(() => addRelationButton?.click());
    const addedRelations = onRelationsChange.mock.calls.at(-1)?.[1];
    expect(addedRelations).toHaveLength(2);
    expect(addedRelations[0]).toEqual({
      id: relationPlug?.id,
      alias: "relation_1",
    });
    expect(addedRelations[1].alias).toBe("relation_2");
    React.act(() => root.unmount());
  });
});

describe("WorkflowNode execution progress", () => {
  it("integrates the selected execution appendix without remeasuring every event", () => {
    const data = createWorkflowNodeData(boundsSpec());
    const container = document.createElement("div");
    const root = createRoot(container);
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "module-1",
            data,
            selected: false,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });
    xyflowMocks.updateNodeInternals.mockClear();

    const materializedRun: NonNullable<typeof data.run> = {
      node_id: "module-1",
      status: "succeeded",
      error: null,
      outputs: [{
        port: "result",
        kind: "single",
        value: {
          artifact_id: "artifact-1",
          artifact_type: "json.object",
          schema_version: 1,
        },
        artifacts: [{
          artifact_id: "artifact-1",
          artifact_type: "json.object",
          schema_version: 1,
          content_type: "application/json",
          text: '{"ready":true}',
        }],
      }],
    };
    const withProgress: typeof data = {
      ...data,
      progress: {
        omittedCount: 1,
        entries: [
          {
            sequence: 1,
            sourceNodePath: [],
            invocationIndex: null,
            invocationPath: [],
            message: "Queued",
            current: null,
            total: null,
          },
          {
            sequence: 3,
            sourceNodePath: ["branch-a", "inner-1"],
            invocationIndex: 3,
            invocationPath: [2, 1],
            message: "<script>Preparing the payload</script>",
            current: 2,
            total: 5,
          },
          {
            sequence: 2,
            sourceNodePath: ["branch-b"],
            invocationIndex: 0,
            invocationPath: [],
            message: "Uploading",
            current: null,
            total: null,
          },
        ],
      },
    };
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "module-1",
            data: withProgress,
            selected: true,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });

    const appendix = container.querySelector<HTMLElement>(
      '[aria-label="Remote WMS map layer execution appendix"]',
    );
    expect(appendix).not.toBeNull();
    expect(appendix?.classList).toContain("nodrag");
    expect(appendix?.classList).toContain("nowheel");
    expect(
      [...appendix!.querySelectorAll('[role="tab"]')].map(
        (tab) => tab.textContent,
      ),
    ).toEqual(["Events 4", "History 0"]);
    expect(appendix?.textContent).toContain("1 earlier update omitted");
    expect(appendix?.textContent).toContain(
      "branch-a › inner-1 · items 3 › 2",
    );
    expect(appendix?.textContent).toContain("2 / 5");
    expect(appendix?.textContent).toContain(
      "<script>Preparing the payload</script>",
    );
    expect(appendix?.querySelector("script")).toBeNull();
    expect(appendix?.textContent).not.toContain("Uploading");
    expect(appendix?.textContent).not.toContain("Queued");
    expect(
      appendix?.querySelector<HTMLButtonElement>(
        'button[aria-label="Show 2 earlier events"]',
      )?.textContent,
    ).toBe("+2");
    expect(xyflowMocks.updateNodeInternals).toHaveBeenCalledOnce();

    xyflowMocks.updateNodeInternals.mockClear();
    const withMaterialization: typeof data = {
      ...withProgress,
      run: materializedRun,
    };
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "module-1",
            data: withMaterialization,
            selected: true,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });
    expect(container.textContent).not.toContain("Produced artifacts");
    expect(xyflowMocks.updateNodeInternals).toHaveBeenCalledOnce();

    xyflowMocks.updateNodeInternals.mockClear();
    const withMoreProgress: typeof data = {
      ...withMaterialization,
      progress: {
        ...withProgress.progress!,
        entries: [
          ...withProgress.progress!.entries,
          {
            sequence: 4,
            sourceNodePath: ["branch-c"],
            invocationIndex: 1,
            invocationPath: [],
            message: "Completed",
            current: null,
            total: null,
          },
        ],
      },
    };
    React.act(() => {
      root.render(
        <WorkflowNodeCard
          {...({
            id: "module-1",
            data: withMoreProgress,
            selected: true,
          } as React.ComponentProps<typeof WorkflowNodeCard>)}
        />,
      );
    });
    expect(xyflowMocks.updateNodeInternals).not.toHaveBeenCalled();
    expect(appendix?.textContent).toContain("branch-c · item 2 · Completed");
    expect(appendix?.textContent).not.toContain("Preparing the payload");

    const discloseEarlierEvents = appendix?.querySelector<HTMLButtonElement>(
      'button[aria-label="Show 3 earlier events"]',
    );
    expect(discloseEarlierEvents?.textContent).toBe("+3");
    React.act(() => discloseEarlierEvents?.click());

    const earlierEvents = appendix?.querySelector<HTMLOListElement>(
      'ol[aria-label="Earlier events"]',
    );
    expect(
      [...(earlierEvents?.querySelectorAll("li") ?? [])].map(
        (event) => event.textContent,
      ),
    ).toEqual([
      "branch-a › inner-1 · items 3 › 2 · <script>Preparing the payload</script> · 2 / 5",
      "branch-b · item 1 · Uploading",
      "Remote WMS map layer · Queued",
    ]);
    expect(earlierEvents?.querySelector("script")).toBeNull();
    React.act(() => root.unmount());
  });
});

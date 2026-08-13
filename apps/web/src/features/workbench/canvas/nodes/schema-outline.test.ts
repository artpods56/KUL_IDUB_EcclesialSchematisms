import { describe, expect, it } from "vitest";

import {
  findOutlineNode,
  outlineCrumbLabel,
  schemaOutline,
  schemaTitle,
  schemaTypeLabel,
} from "./schema-outline";

const LAYER_SCHEMA = {
  title: "GeoMapLayer",
  type: "object",
  required: ["title", "source"],
  properties: {
    title: { type: "string" },
    visible: { type: "boolean" },
    source: {
      oneOf: [
        {
          title: "GeoWmsSource",
          type: "object",
          properties: {
            kind: { const: "wms", type: "string" },
            url: { type: "string" },
            format: {
              type: "string",
              enum: ["image/png", "image/jpeg"],
            },
          },
        },
        {
          title: "GeoRasterArtifactSource",
          type: "object",
          properties: {
            kind: { const: "raster_scan", type: "string" },
            artifact: {
              type: "object",
              title: "ArtifactRef",
              properties: { artifact_id: { type: "string" } },
            },
          },
        },
      ],
    },
  },
};

describe("schemaOutline", () => {
  const outline = schemaOutline(LAYER_SCHEMA);

  it("keeps top-level fields and labels unions from kind", () => {
    expect(outline.map((node) => node.name)).toEqual([
      "title",
      "visible",
      "source",
    ]);
    expect(outline.find((node) => node.name === "title")?.required).toBe(true);
    expect(outline.find((node) => node.name === "visible")?.required).toBe(
      false,
    );
    const source = outline.find((node) => node.name === "source");
    expect(source?.typeLabel).toBe("wms | raster_scan");
    expect(source?.children.map((node) => node.name)).toEqual([
      "as wms",
      "as raster_scan",
    ]);
  });

  it("labels enums as their values instead of str", () => {
    const format = findOutlineNode(outline, "root/source/as:wms/format");
    expect(format?.typeLabel).toBe("image/png | image/jpeg");
  });

  it("strips as-prefix from branch crumbs", () => {
    const wms = findOutlineNode(outline, "root/source/as:wms");
    expect(wms).toBeDefined();
    expect(outlineCrumbLabel(wms!)).toBe("wms");
    expect(outlineCrumbLabel(outline[0]!)).toBe("title");
  });

  it("uses the schema title as the drill root", () => {
    expect(schemaTitle(LAYER_SCHEMA)).toBe("GeoMapLayer");
    expect(schemaTitle({})).toBe("Payload");
  });
});

describe("schemaTypeLabel", () => {
  it("returns any for missing schema", () => {
    expect(schemaTypeLabel(null)).toBe("any");
  });

  it("joins optional scalars", () => {
    expect(
      schemaTypeLabel({
        anyOf: [{ type: "string" }, { type: "null" }],
      }),
    ).toBe("str | None");
  });
});

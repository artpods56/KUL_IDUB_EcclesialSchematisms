import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@stylexjs/stylex", () => ({
  create: <Styles,>(styles: Styles) => styles,
  props: () => ({}),
}));

import type { ArtifactSummary } from "@/lib/api";
import {
  formatJsonSchemaPayload,
  markdownPayload,
  rendererFor,
} from "./artifact-renderers";

const JSON_SCHEMA_ARTIFACT: ArtifactSummary = {
  artifact_id: "schema-artifact",
  artifact_type: "json.schema",
  schema_version: 1,
  content_type: "application/json",
};

const MARKDOWN_ARTIFACT: ArtifactSummary = {
  artifact_id: "markdown-artifact",
  artifact_type: "text.markdown",
  schema_version: 1,
  content_type: "application/json",
};

const MAP_ARTIFACT: ArtifactSummary = {
  artifact_id: "map-artifact",
  artifact_type: "geo.map_document",
  schema_version: 1,
  content_type: "application/geo+json",
};

describe("GIS map artifact rendering", () => {
  it("selects the map renderer and exposes ordered layer toggles", () => {
    const payload = {
      layers: [
        {
          id: "layer-1",
          title: "cities.geojson",
          color: "#2563eb",
          visible: true,
          feature_collection: { type: "FeatureCollection", features: [] },
        },
        {
          id: "layer-2",
          title: "offices.geojson",
          color: "#dc2626",
          visible: true,
          feature_collection: { type: "FeatureCollection", features: [] },
        },
      ],
      bounds: [-0.12, 48.85, 13.4, 52.52],
    };
    const renderer = rendererFor(MAP_ARTIFACT, payload);
    expect(renderer.id).toBe("geo-map");
    expect(renderer.modes).toEqual(["map", "raw"]);

    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: MAP_ARTIFACT,
        payload,
        mode: "map",
      }),
    );
    expect(markup.indexOf("cities.geojson")).toBeLessThan(
      markup.indexOf("offices.geojson"),
    );
    expect(markup).toContain('aria-label="Interactive map"');
    expect(markup).toContain('type="checkbox"');
  });

  it("falls back to raw JSON for an invalid map payload", () => {
    const renderer = rendererFor(MAP_ARTIFACT, { layers: [] });
    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: MAP_ARTIFACT,
        payload: { layers: [] },
        mode: "map",
      }),
    );
    expect(markup).toContain("layers");
  });
});

describe("JSON Schema artifact rendering", () => {
  it("unwraps and indents the schema value for the pretty view", () => {
    const payload = {
      value:
        '{"type":"object","properties":{"invoice_id":{"type":"string"}}}',
    };

    expect(formatJsonSchemaPayload(payload)).toBe(
      [
        "{",
        '  "type": "object",',
        '  "properties": {',
        '    "invoice_id": {',
        '      "type": "string"',
        "    }",
        "  }",
        "}",
      ].join("\n"),
    );

    const renderer = rendererFor(JSON_SCHEMA_ARTIFACT, payload);
    expect(renderer.id).toBe("json-schema");
    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: JSON_SCHEMA_ARTIFACT,
        payload,
        mode: "pretty",
      }),
    );
    expect(markup).toContain("invoice_id");
    expect(markup).not.toContain("&quot;{\\&quot;type");
  });

  it("falls back safely for malformed or non-object schema payloads", () => {
    expect(formatJsonSchemaPayload({ value: "not-json" })).toBeNull();
    expect(formatJsonSchemaPayload({ value: "[]" })).toBeNull();
    expect(formatJsonSchemaPayload({ value: "true" })).toBeNull();
    expect(formatJsonSchemaPayload({})).toBeNull();
    expect(formatJsonSchemaPayload({ value: 42 })).toBeNull();
  });

  it("does not reinterpret JSON-shaped text artifacts", () => {
    const textArtifact: ArtifactSummary = {
      ...JSON_SCHEMA_ARTIFACT,
      artifact_id: "text-artifact",
      artifact_type: "scalar.text",
    };

    expect(
      rendererFor(textArtifact, { value: '{"type":"object"}' }).id,
    ).toBe("json");
  });

  it("keeps raw mode on the stored payload envelope", () => {
    const payload = { value: '{"type":"object"}' };
    const renderer = rendererFor(JSON_SCHEMA_ARTIFACT, payload);
    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: JSON_SCHEMA_ARTIFACT,
        payload,
        mode: "raw",
      }),
    );

    expect(markup).toContain("value");
    expect(markup).toContain("\\&quot;type\\&quot;");
  });
});

describe("Markdown artifact rendering", () => {
  it("selects the nominal renderer before generic JSON and renders GFM", () => {
    const payload = {
      markdown: [
        "# Extracted content",
        "",
        "A **useful** result.",
        "",
        "- first",
        "- second",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| title | Example |",
      ].join("\n"),
    };

    expect(markdownPayload(payload)).toEqual(payload);
    const renderer = rendererFor(MARKDOWN_ARTIFACT, payload);
    expect(renderer.id).toBe("markdown");
    expect(renderer.modes).toEqual(["preview", "raw"]);

    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: MARKDOWN_ARTIFACT,
        payload,
        mode: "preview",
      }),
    );

    expect(markup).toContain("<h1");
    expect(markup).toContain(">Extracted content</h1>");
    expect(markup).toContain("<strong>useful</strong>");
    expect(markup).toContain("<ul>");
    expect(markup).toContain("<table>");
    expect(markup).not.toContain("markdown<!-- -->");
  });

  it("shows the Markdown source itself in raw mode", () => {
    const payload = { markdown: "# Source\n\n`inline`" };
    const renderer = rendererFor(MARKDOWN_ARTIFACT, payload);
    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: MARKDOWN_ARTIFACT,
        payload,
        mode: "raw",
      }),
    );

    expect(markup).toContain("# Source");
    expect(markup).toContain("`inline`");
    expect(markup).not.toContain("&quot;markdown&quot;");
    expect(markup).not.toContain("<h1>");
  });

  it("does not interpret raw HTML or unsafe link protocols", () => {
    const payload = {
      markdown: [
        "# Safe heading",
        "",
        '<script>alert("unsafe")</script>',
        "",
        '<img src="x" onerror="alert(1)">',
        "",
        "[unsafe link](javascript:alert(1))",
        "",
        "[unsafe data link](data:image/svg+xml;base64,PHN2Zy8+)",
        "",
        "[safe link](https://example.com)",
        "",
        "![tracking pixel](https://tracker.example/pixel.gif)",
        "",
        "![inline image](data:image/png;base64,iVBORw0KGgo=)",
      ].join("\n"),
    };
    const renderer = rendererFor(MARKDOWN_ARTIFACT, payload);
    const markup = renderToStaticMarkup(
      createElement(renderer.Component, {
        artifact: MARKDOWN_ARTIFACT,
        payload,
        mode: "preview",
      }),
    );

    expect(markup).toContain("<h1");
    expect(markup).toContain(">Safe heading</h1>");
    expect(markup).not.toContain("<script");
    expect(markup).toContain("&lt;script&gt;");
    expect(markup).toContain("&lt;img src=&quot;x&quot; onerror=");
    expect(markup).not.toContain("javascript:");
    expect(markup).not.toContain("data:image");
    expect(markup).toContain('href="https://example.com"');
    expect(markup).toContain('rel="noreferrer noopener"');
    expect(markup).not.toContain("<img");
    expect(markup).toContain("Image: tracking pixel");
    expect(markup).toContain('href="https://tracker.example/pixel.gif"');
  });

  it("falls back safely when the payload does not satisfy the contract", () => {
    expect(markdownPayload(undefined)).toBeNull();
    expect(markdownPayload({ markdown: 42 })).toBeNull();
    expect(markdownPayload({ value: "# wrong envelope" })).toBeNull();

    const renderer = rendererFor(MARKDOWN_ARTIFACT, { markdown: 42 });
    expect(() =>
      renderToStaticMarkup(
        createElement(renderer.Component, {
          artifact: MARKDOWN_ARTIFACT,
          payload: { markdown: 42 },
          mode: "preview",
        }),
      ),
    ).not.toThrow();
  });
});

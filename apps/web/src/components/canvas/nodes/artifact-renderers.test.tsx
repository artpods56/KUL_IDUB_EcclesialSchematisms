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
  rendererFor,
} from "./artifact-renderers";

const JSON_SCHEMA_ARTIFACT: ArtifactSummary = {
  artifact_id: "schema-artifact",
  artifact_type: "json.schema",
  schema_version: 1,
  content_type: "application/json",
};

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

"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";

import { artifactContentUrl, type ArtifactSummary } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const s = stylex.create({
  raw: {
    margin: 0,
    fontFamily: MONO,
    fontSize: "10px",
    lineHeight: 1.55,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  prettyGrid: { display: "grid", gap: "6px" },
  prettyRow: {
    display: "grid",
    gridTemplateColumns: "94px minmax(0, 1fr)",
    alignItems: "baseline",
    gap: "8px",
  },
  prettyKey: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontFamily: MONO,
    fontSize: "10px",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  prettyText: {
    fontSize: tokens.fontSizeXs,
    lineHeight: 1.5,
    wordBreak: "break-word",
  },
  prettyNumber: {
    color: tokens.colorAccent,
    fontFamily: MONO,
    fontSize: tokens.fontSizeXs,
  },
  chips: { display: "flex", flexWrap: "wrap", gap: "4px" },
  valueChip: {
    padding: "1px 7px",
    borderRadius: "9999px",
    backgroundColor: tokens.colorSurface,
    fontSize: "10px",
    fontWeight: 600,
  },
  nestedGroup: {
    display: "grid",
    gap: "5px",
    marginTop: "2px",
    paddingLeft: "9px",
    borderLeftWidth: 2,
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorDivider,
  },
  image: {
    display: "block",
    width: "100%",
    borderRadius: "8px",
    backgroundColor: tokens.colorSurface,
  },
});

export interface ArtifactRenderProps {
  artifact: ArtifactSummary;
  payload?: unknown;
  mode: string;
}

export interface ArtifactRendererSpec {
  id: string;
  modes: readonly string[];
  matches(artifact: ArtifactSummary, payload?: unknown): boolean;
  Component: React.ComponentType<ArtifactRenderProps>;
}

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function PrettyValue({ value }: { value: unknown }) {
  if (typeof value === "string") {
    return <span {...stylex.props(s.prettyText)}>{value}</span>;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return <span {...stylex.props(s.prettyNumber)}>{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    if (value.every((item) => record(item) === null)) {
      return (
        <span {...stylex.props(s.chips)}>
          {value.map((item, index) => (
            <span key={index} {...stylex.props(s.valueChip)}>
              {typeof item === "string" ? item : JSON.stringify(item)}
            </span>
          ))}
        </span>
      );
    }
    return (
      <span {...stylex.props(s.nestedGroup)}>
        {value.map((item, index) => (
          <PrettyValue key={index} value={item} />
        ))}
      </span>
    );
  }
  const object = record(value);
  if (object) {
    return (
      <span {...stylex.props(s.prettyGrid)}>
        {Object.entries(object).map(([key, entry]) => (
          <span key={key} {...stylex.props(s.prettyRow)}>
            <span {...stylex.props(s.prettyKey)} title={key}>
              {key}
            </span>
            <PrettyValue value={entry} />
          </span>
        ))}
      </span>
    );
  }
  return <span {...stylex.props(s.prettyText)}>—</span>;
}

function artifactMeta(artifact: ArtifactSummary): Record<string, unknown> {
  return {
    type: `${artifact.artifact_type}@${artifact.schema_version}`,
    content_type: artifact.content_type,
    ...(artifact.byte_size != null ? { byte_size: artifact.byte_size } : {}),
    ...(artifact.text ? { text: artifact.text } : {}),
    artifact_id: artifact.artifact_id,
  };
}

const imageRenderer: ArtifactRendererSpec = {
  id: "image",
  modes: ["preview", "meta"],
  matches: (artifact) =>
    artifact.content_type.startsWith("image/") && Boolean(artifact.content_url),
  Component: ({ artifact, mode }) => {
    if (mode === "meta") {
      return <PrettyValue value={artifactMeta(artifact)} />;
    }
    const url =
      artifactContentUrl(artifact.content_url) ?? artifact.content_url ?? "";
    return (
      /* eslint-disable-next-line @next/next/no-img-element -- artifact URLs are dynamic */
      <img
        src={url}
        alt={artifact.text ?? artifact.artifact_type}
        {...stylex.props(s.image)}
      />
    );
  },
};

const jsonRenderer: ArtifactRendererSpec = {
  id: "json",
  modes: ["pretty", "raw"],
  matches: (artifact, payload) =>
    payload !== undefined || artifact.content_type === "application/json",
  Component: ({ artifact, payload, mode }) => {
    const value = payload === undefined ? artifactMeta(artifact) : payload;
    if (mode === "raw") {
      return <pre {...stylex.props(s.raw)}>{JSON.stringify(value, null, 2)}</pre>;
    }
    return <PrettyValue value={value} />;
  },
};

export const META_ARTIFACT_RENDERER: ArtifactRendererSpec = {
  id: "meta",
  modes: ["meta"],
  matches: () => true,
  Component: ({ artifact }) => <PrettyValue value={artifactMeta(artifact)} />,
};

export const ARTIFACT_RENDERERS: readonly ArtifactRendererSpec[] = [
  imageRenderer,
  jsonRenderer,
  META_ARTIFACT_RENDERER,
];

export function rendererFor(
  artifact: ArtifactSummary,
  payload?: unknown,
): ArtifactRendererSpec {
  return (
    ARTIFACT_RENDERERS.find((renderer) => renderer.matches(artifact, payload)) ??
    META_ARTIFACT_RENDERER
  );
}

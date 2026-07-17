"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import Markdown, {
  type MarkdownToJSX,
  sanitizer as sanitizeMarkdownUrl,
} from "markdown-to-jsx";

import { artifactContentUrl, type ArtifactSummary } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const s = stylex.create({
  jsonCode: {
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
  markdown: {
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    lineHeight: 1.65,
    overflowWrap: "anywhere",
  },
  markdownHeading1: {
    marginTop: "2px",
    marginBottom: "9px",
    color: tokens.colorTextEmphasis,
    fontSize: "15px",
    fontWeight: 700,
    lineHeight: 1.3,
  },
  markdownHeading2: {
    marginTop: "14px",
    marginBottom: "7px",
    color: tokens.colorTextEmphasis,
    fontSize: "13px",
    fontWeight: 700,
    lineHeight: 1.35,
  },
  markdownHeading3: {
    marginTop: "12px",
    marginBottom: "6px",
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeSm,
    fontWeight: 700,
    lineHeight: 1.4,
  },
  markdownParagraph: {
    marginTop: 0,
    marginBottom: "9px",
  },
  markdownList: {
    marginTop: 0,
    marginBottom: "9px",
    paddingLeft: "18px",
  },
  markdownListItem: { marginBottom: "3px" },
  markdownBlockquote: {
    marginTop: "9px",
    marginRight: 0,
    marginBottom: "9px",
    marginLeft: 0,
    paddingLeft: "10px",
    borderLeftWidth: 2,
    borderLeftStyle: "solid",
    borderLeftColor: tokens.colorAccentBorder,
    color: tokens.colorMuted,
  },
  markdownCode: {
    fontFamily: MONO,
    fontSize: "10px",
  },
  markdownInlineCode: {
    paddingTop: "1px",
    paddingRight: "4px",
    paddingBottom: "1px",
    paddingLeft: "4px",
    borderRadius: "4px",
    backgroundColor: tokens.colorSurfaceSunken,
  },
  markdownPre: {
    marginTop: "9px",
    marginBottom: "9px",
    padding: "9px 10px",
    overflowX: "auto",
    borderRadius: "6px",
    backgroundColor: tokens.colorSurfaceSunken,
    lineHeight: 1.55,
    whiteSpace: "pre",
  },
  markdownLink: {
    color: tokens.colorAccent,
    textDecorationLine: "underline",
    textUnderlineOffset: "2px",
  },
  markdownRule: {
    height: 1,
    marginTop: "12px",
    marginBottom: "12px",
    borderWidth: 0,
    backgroundColor: tokens.colorDivider,
  },
  markdownTable: {
    display: "block",
    width: "100%",
    marginTop: "9px",
    marginBottom: "9px",
    overflowX: "auto",
    borderCollapse: "collapse",
    fontSize: tokens.fontSizeXs,
  },
  markdownTableCell: {
    padding: "5px 7px",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorDivider,
    textAlign: "left",
    verticalAlign: "top",
  },
  markdownTableHeader: {
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorTextEmphasis,
    fontWeight: 700,
  },
  markdownImageReference: {
    display: "flex",
    alignItems: "baseline",
    flexWrap: "wrap",
    gap: "6px",
    marginTop: "9px",
    marginBottom: "9px",
    padding: "6px 8px",
    borderRadius: "6px",
    backgroundColor: tokens.colorSurfaceSunken,
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
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

export function formatJsonSchemaPayload(payload: unknown): string | null {
  const schemaText = record(payload)?.value;
  if (typeof schemaText !== "string") return null;

  try {
    const schema: unknown = JSON.parse(schemaText);
    if (record(schema) === null) return null;
    return JSON.stringify(schema, null, 2);
  } catch {
    return null;
  }
}

export interface MarkdownArtifactPayload {
  markdown: string;
}

export function markdownPayload(
  payload: unknown,
): MarkdownArtifactPayload | null {
  const markdown = record(payload)?.markdown;
  return typeof markdown === "string" ? { markdown } : null;
}

function safeMarkdownUrl(value: string | undefined): string | null {
  if (!value) return null;
  const sanitized = sanitizeMarkdownUrl(value);
  if (!sanitized) return null;
  const scheme = /^([a-z][a-z\d+.-]*):/i.exec(sanitized.trim())?.[1];
  if (!scheme) return sanitized;
  return ["http", "https", "mailto"].includes(scheme.toLowerCase())
    ? sanitized
    : null;
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

const jsonSchemaRenderer: ArtifactRendererSpec = {
  id: "json-schema",
  modes: ["pretty", "raw"],
  matches: (artifact) =>
    artifact.artifact_type === "json.schema" && artifact.schema_version === 1,
  Component: ({ artifact, payload, mode }) => {
    const value = payload === undefined ? artifactMeta(artifact) : payload;
    if (mode === "pretty") {
      const formattedSchema = formatJsonSchemaPayload(value);
      if (formattedSchema !== null) {
        return <pre {...stylex.props(s.jsonCode)}>{formattedSchema}</pre>;
      }
      return <PrettyValue value={value} />;
    }
    return (
      <pre {...stylex.props(s.jsonCode)}>{JSON.stringify(value, null, 2)}</pre>
    );
  },
};

function MarkdownCode({
  children,
  className,
  ...props
}: React.ComponentPropsWithoutRef<"code">) {
  const block = Boolean(className) || String(children).includes("\n");
  return (
    <code
      {...props}
      className={className}
      {...stylex.props(
        s.markdownCode,
        block ? null : s.markdownInlineCode,
      )}
    >
      {children}
    </code>
  );
}

function MarkdownLink({
  children,
  href,
  ...props
}: React.ComponentPropsWithoutRef<"a">) {
  const safeHref = safeMarkdownUrl(href);
  return (
    <a
      {...props}
      href={safeHref ?? undefined}
      target="_blank"
      rel="noreferrer noopener"
      {...stylex.props(s.markdownLink)}
    >
      {children}
    </a>
  );
}

function MarkdownImageReference({
  alt,
  src,
  title,
}: React.ComponentPropsWithoutRef<"img">) {
  const safeSource = safeMarkdownUrl(typeof src === "string" ? src : undefined);
  return (
    <span {...stylex.props(s.markdownImageReference)}>
      <span>Image: {alt || "untitled"}</span>
      {safeSource ? (
        <a
          href={safeSource}
          title={title}
          target="_blank"
          rel="noreferrer noopener"
          {...stylex.props(s.markdownLink)}
        >
          open source
        </a>
      ) : null}
    </span>
  );
}

const markdownOptions: MarkdownToJSX.Options = {
  disableParsingRawHTML: true,
  enforceAtxHeadings: true,
  sanitizer: (value) => safeMarkdownUrl(value),
  wrapper: React.Fragment,
  overrides: {
    h1: { component: "h1", props: stylex.props(s.markdownHeading1) },
    h2: { component: "h2", props: stylex.props(s.markdownHeading2) },
    h3: { component: "h3", props: stylex.props(s.markdownHeading3) },
    h4: { component: "h4", props: stylex.props(s.markdownHeading3) },
    h5: { component: "h5", props: stylex.props(s.markdownHeading3) },
    h6: { component: "h6", props: stylex.props(s.markdownHeading3) },
    p: { component: "p", props: stylex.props(s.markdownParagraph) },
    ul: { component: "ul", props: stylex.props(s.markdownList) },
    ol: { component: "ol", props: stylex.props(s.markdownList) },
    li: { component: "li", props: stylex.props(s.markdownListItem) },
    blockquote: {
      component: "blockquote",
      props: stylex.props(s.markdownBlockquote),
    },
    code: MarkdownCode,
    pre: { component: "pre", props: stylex.props(s.markdownPre) },
    a: MarkdownLink,
    hr: { component: "hr", props: stylex.props(s.markdownRule) },
    table: { component: "table", props: stylex.props(s.markdownTable) },
    th: {
      component: "th",
      props: stylex.props(s.markdownTableCell, s.markdownTableHeader),
    },
    td: { component: "td", props: stylex.props(s.markdownTableCell) },
    img: MarkdownImageReference,
  },
};

const markdownRenderer: ArtifactRendererSpec = {
  id: "markdown",
  modes: ["preview", "raw"],
  matches: (artifact) =>
    artifact.artifact_type === "text.markdown" && artifact.schema_version === 1,
  Component: ({ artifact, payload, mode }) => {
    const markdown = markdownPayload(payload)?.markdown ?? artifact.text;
    if (markdown === undefined || markdown === null) {
      return <PrettyValue value={payload ?? artifactMeta(artifact)} />;
    }
    if (mode === "raw") {
      return <pre {...stylex.props(s.jsonCode)}>{markdown}</pre>;
    }
    return (
      <div {...stylex.props(s.markdown)}>
        <Markdown options={markdownOptions}>{markdown}</Markdown>
      </div>
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
      return (
        <pre {...stylex.props(s.jsonCode)}>{JSON.stringify(value, null, 2)}</pre>
      );
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
  jsonSchemaRenderer,
  markdownRenderer,
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

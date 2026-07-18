"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import Markdown, {
  type MarkdownToJSX,
  sanitizer as sanitizeMarkdownUrl,
} from "markdown-to-jsx";
import maplibregl, { type GeoJSONSourceSpecification } from "maplibre-gl";

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
  mapShell: { display: "grid", gap: "7px" },
  map: {
    width: "100%",
    height: "220px",
    overflow: "hidden",
    borderRadius: "8px",
    backgroundColor: tokens.colorSurfaceSunken,
  },
  mapLegend: { display: "flex", flexWrap: "wrap", gap: "5px" },
  mapLayerToggle: {
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    padding: "3px 6px",
    borderRadius: "6px",
    backgroundColor: tokens.colorSurfaceSunken,
    fontSize: "10px",
  },
  mapSwatch: {
    width: "8px",
    height: "8px",
    borderRadius: "9999px",
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

interface GeoMapLayerPayload {
  id: string;
  title: string;
  color: string;
  visible: boolean;
  feature_collection: {
    type: "FeatureCollection";
    features: Record<string, unknown>[];
  };
}

interface GeoMapPayload {
  layers: GeoMapLayerPayload[];
  bounds: [number, number, number, number] | null;
}

function geoMapPayload(payload: unknown): GeoMapPayload | null {
  const value = record(payload);
  if (!value || !Array.isArray(value.layers)) return null;
  const layers = value.layers.flatMap((candidate) => {
    const layer = record(candidate);
    const collection = record(layer?.feature_collection);
    if (
      !layer ||
      typeof layer.id !== "string" ||
      typeof layer.title !== "string" ||
      typeof layer.color !== "string" ||
      typeof layer.visible !== "boolean" ||
      collection?.type !== "FeatureCollection" ||
      !Array.isArray(collection.features)
    ) return [];
    return [{
      id: layer.id,
      title: layer.title,
      color: layer.color,
      visible: layer.visible,
      feature_collection: {
        type: "FeatureCollection" as const,
        features: collection.features.filter(
          (feature): feature is Record<string, unknown> => record(feature) !== null,
        ),
      },
    }];
  });
  const bounds = Array.isArray(value.bounds) &&
      value.bounds.length === 4 &&
      value.bounds.every((coordinate) => typeof coordinate === "number")
    ? value.bounds as [number, number, number, number]
    : null;
  return layers.length ? { layers, bounds } : null;
}

function GeoMapPreview({ value }: { value: GeoMapPayload }) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const mapRef = React.useRef<maplibregl.Map | null>(null);

  React.useEffect(() => {
    if (!containerRef.current || typeof WebGLRenderingContext === "undefined") {
      return;
    }
    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [0, 20],
      zoom: 1,
      style: {
        version: 8,
        sources: {
          basemap: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "basemap", type: "raster", source: "basemap" }],
      },
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.on("load", () => {
      for (const layer of value.layers) {
        const sourceId = `geo-source-${layer.id}`;
        const visibility = layer.visible ? "visible" : "none";
        map.addSource(sourceId, {
          type: "geojson",
          data: layer.feature_collection as unknown as GeoJSONSourceSpecification["data"],
        });
        map.addLayer({
          id: `${layer.id}-fill`,
          type: "fill",
          source: sourceId,
          paint: { "fill-color": layer.color, "fill-opacity": 0.28 },
          layout: { visibility },
          filter: ["==", ["geometry-type"], "Polygon"],
        });
        map.addLayer({
          id: `${layer.id}-line`,
          type: "line",
          source: sourceId,
          paint: { "line-color": layer.color, "line-width": 2 },
          layout: { visibility },
          filter: ["in", ["geometry-type"], ["literal", ["LineString", "Polygon"]]],
        });
        map.addLayer({
          id: `${layer.id}-point`,
          type: "circle",
          source: sourceId,
          paint: {
            "circle-color": layer.color,
            "circle-radius": 5,
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 1,
          },
          layout: { visibility },
          filter: ["==", ["geometry-type"], "Point"],
        });
      }
      if (value.bounds) {
        map.fitBounds(
          [[value.bounds[0], value.bounds[1]], [value.bounds[2], value.bounds[3]]],
          { padding: 28, maxZoom: 14, duration: 0 },
        );
      }
    });
    map.on("click", (event) => {
      const layerIds = value.layers.flatMap((layer) => [
        `${layer.id}-fill`,
        `${layer.id}-line`,
        `${layer.id}-point`,
      ]).filter((id) => map.getLayer(id));
      const feature = map.queryRenderedFeatures(event.point, { layers: layerIds })[0];
      if (!feature) return;
      const content = document.createElement("pre");
      content.textContent = JSON.stringify(feature.properties ?? {}, null, 2);
      new maplibregl.Popup({ maxWidth: "320px" })
        .setLngLat(event.lngLat)
        .setDOMContent(content)
        .addTo(map);
    });
    return () => {
      mapRef.current = null;
      map.remove();
    };
  }, [value]);

  return (
    <div {...stylex.props(s.mapShell)}>
      <div ref={containerRef} aria-label="Interactive map" {...stylex.props(s.map)} />
      <div {...stylex.props(s.mapLegend)}>
        {value.layers.map((layer) => (
          <label key={layer.id} {...stylex.props(s.mapLayerToggle)}>
            <input
              type="checkbox"
              defaultChecked={layer.visible}
              onChange={(event) => {
                for (const suffix of ["fill", "line", "point"]) {
                  const id = `${layer.id}-${suffix}`;
                  if (mapRef.current?.getLayer(id)) {
                    mapRef.current.setLayoutProperty(
                      id,
                      "visibility",
                      event.currentTarget.checked ? "visible" : "none",
                    );
                  }
                }
              }}
            />
            <span style={{ backgroundColor: layer.color }} {...stylex.props(s.mapSwatch)} />
            {layer.title}
          </label>
        ))}
      </div>
    </div>
  );
}

const geoMapRenderer: ArtifactRendererSpec = {
  id: "geo-map",
  modes: ["map", "raw"],
  matches: (artifact) =>
    artifact.artifact_type === "geo.map_document" && artifact.schema_version === 1,
  Component: ({ artifact, payload, mode }) => {
    const value = geoMapPayload(payload);
    if (mode === "map" && value) return <GeoMapPreview value={value} />;
    return (
      <pre {...stylex.props(s.jsonCode)}>
        {JSON.stringify(payload ?? artifactMeta(artifact), null, 2)}
      </pre>
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
  geoMapRenderer,
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

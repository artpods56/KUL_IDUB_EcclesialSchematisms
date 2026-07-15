import type {
  ArtifactConversionInput,
  ArtifactConversionSpec,
  ArtifactTypeSpec,
  FieldProjection,
} from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";
import type { PortMeta } from "./types";

/**
 * Encode port type information into the React Flow handle id so that
 * `isValidConnection` can enforce typed artifact flow without extra
 * lookups.
 */
export function encodeHandleId(port: PortMeta): string {
  return [
    port.portName,
    port.artifactTypeId,
    port.schemaVersion,
    port.shape,
    port.direction,
  ].join("::");
}

export interface DecodedHandle {
  portName: string;
  artifactTypeId: string;
  schemaVersion: number;
  shape: PortMeta["shape"];
  direction: PortMeta["direction"];
}

export function decodeHandleId(
  id: string | null | undefined,
): DecodedHandle | null {
  if (!id) return null;
  const p = id.split("::");
  if (p.length !== 5) return null;
  const schemaVersion = Number(p[2]);
  const shape = p[3];
  const direction = p[4];
  if (!Number.isInteger(schemaVersion)) return null;
  if (shape !== "one" && shape !== "many") return null;
  if (direction !== "input" && direction !== "output") return null;
  return {
    portName: p[0],
    artifactTypeId: p[1],
    schemaVersion,
    shape,
    direction,
  };
}

/** A connection is valid when the output and input share an artifact contract. */
export function connectionIsValid(connection: {
  sourceHandle?: string | null;
  targetHandle?: string | null;
}): boolean {
  const s = decodeHandleId(connection.sourceHandle);
  const t = decodeHandleId(connection.targetHandle);
  if (!s || !t) return false;
  return (
    s.direction === "output" &&
    t.direction === "input" &&
    s.artifactTypeId === t.artifactTypeId &&
    s.schemaVersion === t.schemaVersion &&
    s.shape === t.shape
  );
}

/** Artifact identity compatibility without collection-shape handling. */
export function connectionArtifactContractIsValid(connection: {
  sourceHandle?: string | null;
  targetHandle?: string | null;
}): boolean {
  const source = decodeHandleId(connection.sourceHandle);
  const target = decodeHandleId(connection.targetHandle);
  if (!source || !target) return false;
  return (
    source.direction === "output" &&
    target.direction === "input" &&
    source.artifactTypeId === target.artifactTypeId &&
    source.schemaVersion === target.schemaVersion
  );
}

/** Declared source-field projections that can satisfy a typed connection. */
export function projectionCandidatesForConnection(
  connection: {
    sourceHandle?: string | null;
    targetHandle?: string | null;
  },
  artifactTypes: readonly ArtifactTypeSpec[],
): FieldProjection[] {
  const source = decodeHandleId(connection.sourceHandle);
  const target = decodeHandleId(connection.targetHandle);
  if (
    !source ||
    !target ||
    source.direction !== "output" ||
    target.direction !== "input" ||
    (source.artifactTypeId === target.artifactTypeId &&
      source.schemaVersion === target.schemaVersion)
  ) {
    return [];
  }

  const sourceArtifact = artifactTypes.find(
    (artifact) =>
      artifact.key.id === source.artifactTypeId &&
      artifact.key.schema_version === source.schemaVersion,
  );
  if (!sourceArtifact?.field_projections) return [];

  return sourceArtifact.field_projections.filter(
    (projection) =>
      projection.target_artifact_type.id === target.artifactTypeId &&
      projection.target_artifact_type.schema_version === target.schemaVersion,
  );
}

export type ConnectionRoute =
  | { kind: "exact" }
  | { kind: "projection"; projection: FieldProjection }
  | { kind: "conversion"; conversion: ArtifactConversionSpec }
  | {
      kind: "projection-conversion";
      projection: FieldProjection;
      conversion: ArtifactConversionSpec;
    };

export interface ConnectionRouteSelection {
  projection?: { path: readonly string[] };
  conversion?: ArtifactConversionInput;
}

function artifactTypeMatches(
  artifactType: { id: string; schema_version: number },
  id: string,
  schemaVersion: number,
): boolean {
  return artifactType.id === id && artifactType.schema_version === schemaVersion;
}

function conversionMatches(
  conversion: ArtifactConversionSpec,
  sourceId: string,
  sourceSchemaVersion: number,
  targetId: string,
  targetSchemaVersion: number,
): boolean {
  return (
    artifactTypeMatches(
      conversion.source_artifact_type,
      sourceId,
      sourceSchemaVersion,
    ) &&
    artifactTypeMatches(
      conversion.target_artifact_type,
      targetId,
      targetSchemaVersion,
    )
  );
}

/** All single-step routes, plus projection-then-conversion, between two ports. */
export function connectionRoutesFor(
  connection: {
    sourceHandle?: string | null;
    targetHandle?: string | null;
  },
  artifactTypes: readonly ArtifactTypeSpec[],
  conversions: readonly ArtifactConversionSpec[],
): ConnectionRoute[] {
  const source = decodeHandleId(connection.sourceHandle);
  const target = decodeHandleId(connection.targetHandle);
  if (
    !source ||
    !target ||
    source.direction !== "output" ||
    target.direction !== "input"
  ) {
    return [];
  }

  if (
    source.artifactTypeId === target.artifactTypeId &&
    source.schemaVersion === target.schemaVersion
  ) {
    return [{ kind: "exact" }];
  }

  const sourceArtifact = artifactTypes.find(
    (artifact) =>
      artifact.key.id === source.artifactTypeId &&
      artifact.key.schema_version === source.schemaVersion,
  );
  const routes: ConnectionRoute[] = [];

  for (const projection of sourceArtifact?.field_projections ?? []) {
    if (
      artifactTypeMatches(
        projection.target_artifact_type,
        target.artifactTypeId,
        target.schemaVersion,
      )
    ) {
      routes.push({ kind: "projection", projection });
    }
  }

  for (const conversion of conversions) {
    if (
      conversionMatches(
        conversion,
        source.artifactTypeId,
        source.schemaVersion,
        target.artifactTypeId,
        target.schemaVersion,
      )
    ) {
      routes.push({ kind: "conversion", conversion });
    }
  }

  for (const projection of sourceArtifact?.field_projections ?? []) {
    for (const conversion of conversions) {
      if (
        conversionMatches(
          conversion,
          projection.target_artifact_type.id,
          projection.target_artifact_type.schema_version,
          target.artifactTypeId,
          target.schemaVersion,
        )
      ) {
        routes.push({
          kind: "projection-conversion",
          projection,
          conversion,
        });
      }
    }
  }

  return routes;
}

export function connectionRouteSelection(
  route: ConnectionRoute,
): ConnectionRouteSelection {
  const projection =
    route.kind === "projection" || route.kind === "projection-conversion"
      ? { path: [...route.projection.path] }
      : undefined;
  const conversion =
    route.kind === "conversion" || route.kind === "projection-conversion"
      ? {
          id: route.conversion.key.id,
          version: route.conversion.key.version,
        }
      : undefined;
  return { projection, conversion };
}

export function connectionRouteMatchesSelection(
  route: ConnectionRoute,
  selection: ConnectionRouteSelection,
): boolean {
  const candidate = connectionRouteSelection(route);
  const projectionMatches = candidate.projection
    ? Boolean(
        selection.projection &&
          candidate.projection.path.length === selection.projection.path.length &&
          candidate.projection.path.every(
            (segment, index) => segment === selection.projection?.path[index],
          ),
      )
    : selection.projection === undefined;
  const conversionMatchesSelection = candidate.conversion
    ? Boolean(
        selection.conversion &&
          candidate.conversion.id === selection.conversion.id &&
          candidate.conversion.version === selection.conversion.version,
      )
    : selection.conversion === undefined;
  return projectionMatches && conversionMatchesSelection;
}

export type CSSProperties = Record<string, string | number>;

export function handleStyle(
  top: number | string,
  color: string,
  variadic = false,
): CSSProperties {
  return {
    top: typeof top === "number" ? `${top}px` : top,
    width: "30px",
    height: "30px",
    borderRadius: "9999px",
    background: variadic
      ? `radial-gradient(circle, ${tokens.colorSurface} 0 3px, ${color} 4px 6px, ${tokens.colorSurface} 7px 8px, transparent 9px)`
      : `radial-gradient(circle, ${color} 0 5px, ${tokens.colorSurface} 6px 7px, transparent 8px)`,
    border: "none",
    boxShadow: "none",
    cursor: "crosshair",
    touchAction: "none",
  };
}

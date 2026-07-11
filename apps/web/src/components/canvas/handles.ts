import type {
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
    source.shape !== target.shape ||
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

export function projectionAwareConnectionIsValid(
  connection: {
    sourceHandle?: string | null;
    targetHandle?: string | null;
  },
  artifactTypes: readonly ArtifactTypeSpec[],
): boolean {
  return (
    connectionIsValid(connection) ||
    projectionCandidatesForConnection(connection, artifactTypes).length > 0
  );
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

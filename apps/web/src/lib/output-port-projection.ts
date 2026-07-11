import type { ArtifactTypeSpec, FieldProjection } from "@/lib/api";

export function pathsEqual(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return (
    left.length === right.length &&
    left.every((segment, index) => segment === right[index])
  );
}

export function artifactTypeSpecForKey(
  artifactTypes: readonly ArtifactTypeSpec[],
  artifactTypeId: string,
  schemaVersion: number,
): ArtifactTypeSpec | undefined {
  return artifactTypes.find(
    (spec) =>
      spec.key.id === artifactTypeId &&
      spec.key.schema_version === schemaVersion,
  );
}

export function projectionPathLabel(
  port: string,
  path: readonly string[],
): string {
  if (!path.length) return port;
  return `${port}.${path.join(".")}`;
}

export function formatArtifactTypeKey(key: {
  id: string;
  schema_version: number;
}): string {
  return `${key.id}@${key.schema_version}`;
}

export function findProjectionForPath(
  path: readonly string[],
  fieldProjections: readonly FieldProjection[],
): FieldProjection | undefined {
  return fieldProjections.find((projection) =>
    pathsEqual(projection.path, path),
  );
}

export interface PortWireIntent {
  path: readonly string[];
  label: string;
  targetType: string;
  title: string;
  declared: boolean;
}

export function wireIntentForPath(
  port: string,
  path: readonly string[],
  fieldProjections: readonly FieldProjection[],
): PortWireIntent | null {
  if (!path.length) return null;

  const projection = findProjectionForPath(path, fieldProjections);
  const label = projectionPathLabel(port, path);
  if (projection) {
    return {
      path,
      label,
      targetType: formatArtifactTypeKey(projection.target_artifact_type),
      title: projection.title,
      declared: true,
    };
  }

  return {
    path,
    label,
    targetType: "undeclared",
    title: label,
    declared: false,
  };
}

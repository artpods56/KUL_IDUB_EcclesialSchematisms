import type { FieldProjection, Port } from "@/lib/api";
import {
  findProjectionForPath,
  projectionPathLabel,
} from "@/lib/output-port-projection";

/** How an output port emits data when wired to a downstream node. */
export interface OutputPortTreatment {
  /** Empty path = emit the whole port artifact unchanged. */
  projectionPath: readonly string[];
}

export const DEFAULT_PORT_TREATMENT: OutputPortTreatment = {
  projectionPath: [],
};

export function isConfigured(treatment: OutputPortTreatment): boolean {
  return treatment.projectionPath.length > 0;
}

export function treatmentLabel(
  port: string,
  treatment: OutputPortTreatment,
  fieldProjections: readonly FieldProjection[] = [],
): string {
  if (!treatment.projectionPath.length) return port;
  const projection = findProjectionForPath(
    treatment.projectionPath,
    fieldProjections,
  );
  if (projection) {
    return `${port}.${projection.path.join(".")}`;
  }
  return projectionPathLabel(port, treatment.projectionPath);
}

export function treatmentFromProjection(
  projection: FieldProjection,
): OutputPortTreatment {
  return { projectionPath: [...projection.path] };
}

export function treatmentFromPath(path: readonly string[]): OutputPortTreatment {
  if (!path.length) return DEFAULT_PORT_TREATMENT;
  return { projectionPath: [...path] };
}

export function defaultTreatments(
  outputs: readonly Port[],
): Record<string, OutputPortTreatment> {
  return Object.fromEntries(
    outputs.map((port) => [port.name, DEFAULT_PORT_TREATMENT]),
  );
}

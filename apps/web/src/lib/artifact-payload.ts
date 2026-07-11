export function parseArtifactPayload(
  text: string | null | undefined,
): unknown | null {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export function summarizePayload(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record.parish === "string") return record.parish;
    return `${Object.keys(record).length} fields`;
  }
  return typeof value;
}

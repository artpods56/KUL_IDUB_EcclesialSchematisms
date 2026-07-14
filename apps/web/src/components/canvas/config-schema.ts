export interface SchemaField {
  name: string;
  title: string;
  description?: string;
  type: "string" | "integer" | "number" | "boolean";
  enumValues?: readonly (string | number)[];
  format?: "textarea";
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  required: boolean;
}

function record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function resolveSchema(
  schema: Record<string, unknown>,
  root: Record<string, unknown>,
): Record<string, unknown> {
  const reference = schema.$ref;
  if (typeof reference !== "string" || !reference.startsWith("#/$defs/")) {
    return schema;
  }
  const name = reference.slice("#/$defs/".length);
  const definitions = record(root.$defs);
  return record(definitions?.[name]) ?? schema;
}

/** Editable scalar fields from a node's `config_schema`. */
export function schemaFields(rawSchema: unknown): SchemaField[] {
  const root = record(rawSchema);
  const properties = record(root?.properties);
  if (!root || !properties) return [];

  const required = new Set(
    Array.isArray(root.required)
      ? root.required.filter(
          (value): value is string => typeof value === "string",
        )
      : [],
  );

  return Object.entries(properties).flatMap(([name, rawProperty]) => {
    if (
      /api.?key|token|secret/i.test(name) ||
      name === "connector_id" ||
      name === "selection"
    ) {
      return [];
    }

    const propertyRecord = record(rawProperty);
    if (!propertyRecord) return [];
    const property = resolveSchema(propertyRecord, root);
    const enumValues = Array.isArray(property.enum)
      ? property.enum.filter(
          (value): value is string | number =>
            typeof value === "string" || typeof value === "number",
        )
      : undefined;
    const candidateType =
      typeof property.type === "string"
        ? property.type
        : enumValues?.every((value) => typeof value === "number")
          ? "number"
          : "string";
    if (
      candidateType !== "string" &&
      candidateType !== "integer" &&
      candidateType !== "number" &&
      candidateType !== "boolean"
    ) {
      return [];
    }

    return [{
      name,
      title:
        typeof property.title === "string"
          ? property.title
          : name.replaceAll("_", " "),
      description:
        typeof property.description === "string"
          ? property.description
          : undefined,
      type: candidateType,
      enumValues,
      format: property.format === "textarea" ? "textarea" : undefined,
      minimum:
        typeof property.minimum === "number" ? property.minimum : undefined,
      maximum:
        typeof property.maximum === "number" ? property.maximum : undefined,
      minLength:
        typeof property.minLength === "number" ? property.minLength : undefined,
      maxLength:
        typeof property.maxLength === "number" ? property.maxLength : undefined,
      pattern: typeof property.pattern === "string" ? property.pattern : undefined,
      required: required.has(name),
    }];
  });
}

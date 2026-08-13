import type { NodeRegistry, NodeSpec, Port } from "@/lib/api";
import type { SchemaField } from "@/features/workbench/canvas/config-schema";

function port(
  name: string,
  direction: Port["direction"],
  artifactTypeId: string,
  options: Partial<Port> = {},
): Port {
  return {
    name,
    title: options.title ?? name,
    description: options.description ?? null,
    direction,
    artifact_type: { id: artifactTypeId, schema_version: 1 },
    artifact_type_variable: null,
    shape: options.shape ?? "one",
    accepted_shapes: options.accepted_shapes ?? [options.shape ?? "one"],
    instance_plugs: false,
    variadic: false,
    required: options.required ?? true,
  };
}

function artifact(
  id: string,
  title: string,
  field_projections: NodeRegistry["artifact_types"][number]["field_projections"] = [],
): NodeRegistry["artifact_types"][number] {
  return {
    key: { id, schema_version: 1 },
    title,
    payload_schema: {},
    field_projections,
  };
}

const messages = port("messages", "input", "prompt.message", {
  title: "messages",
  description: "Ordered prompt messages.",
  shape: "many",
  accepted_shapes: ["many"],
  required: true,
});

const jsonSchema = port("json_schema", "input", "json.schema", {
  title: "JSON Schema",
  description: "Optional schema for a structured completion.",
  required: false,
});

export const COMPLETION_PORT = port("completion", "output", "llm.completion", {
  title: "completion",
  description: "Completion content and safe provider metadata.",
  shape: "many",
  accepted_shapes: ["many"],
});

export const CHAT_COMPLETION_SPEC: NodeSpec = {
  operator_id: "llm.openai_compatible.chat_completion",
  operator_version: 1,
  plugin_slug: "external.llm",
  title: "OpenAI-compatible Chat Completion",
  description:
    "Calls a saved graph's OpenAI-compatible Chat Completions endpoint.",
  config_schema: {},
  input_schema: {},
  output_schema: {},
  inputs: [messages, jsonSchema],
  outputs: [COMPLETION_PORT],
  catalog_visible: true,
};

export const CHAT_COMPLETION_FIELDS: readonly SchemaField[] = [
  {
    name: "base_url",
    title: "Base URL",
    type: "string",
    required: true,
    nullable: false,
  },
  {
    name: "model",
    title: "Model",
    type: "string",
    required: true,
    nullable: false,
  },
  {
    name: "temperature",
    title: "Temperature",
    type: "number",
    required: true,
    nullable: false,
  },
  {
    name: "max_completion_tokens",
    title: "Max completion tokens",
    type: "integer",
    required: true,
    nullable: false,
  },
];

export const CHAT_COMPLETION_REGISTRY: NodeRegistry = {
  plugins: [{ slug: "external.llm", title: "LLM", origin: "external" }],
  artifact_types: [
    artifact("prompt.message", "Prompt message"),
    artifact("json.schema", "JSON Schema"),
    artifact("llm.completion", "LLM completion", [
      {
        path: ["content"],
        title: "Content",
        target_artifact_type: { id: "scalar.text", schema_version: 1 },
      },
      {
        path: ["model"],
        title: "Model",
        target_artifact_type: { id: "scalar.text", schema_version: 1 },
      },
      {
        path: ["message_count"],
        title: "Message count",
        target_artifact_type: { id: "scalar.integer", schema_version: 1 },
      },
      {
        path: ["protocol"],
        title: "Protocol",
        target_artifact_type: { id: "scalar.text", schema_version: 1 },
      },
    ]),
    artifact("scalar.text", "Text"),
    artifact("scalar.integer", "Integer"),
  ],
  artifact_conversions: [],
  nodes: [CHAT_COMPLETION_SPEC],
};

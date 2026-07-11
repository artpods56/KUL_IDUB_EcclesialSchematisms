from notarius_core.application.workflows import NodeSpecRegistry
from notarius_core.domain.models import ExecutionMode, NodeSpec, PortSpec

DEBUG_EMIT_TEXT_OPERATOR_ID = "debug.emit_text"
DEBUG_EMIT_TEXT_OPERATOR_VERSION = "1.0.0"
OCR_EXTRACT_PAGES_OPERATOR_ID = "ocr.extract_pages"
OCR_EXTRACT_PAGES_OPERATOR_VERSION = "1.0.0"
OCR_EXTRACT_PAGE_OPERATOR_ID = "ocr.extract_page"
OCR_EXTRACT_PAGE_OPERATOR_VERSION = "1.0.0"
OCR_COLLECT_PAGES_OPERATOR_ID = "ocr.collect_pages"
OCR_COLLECT_PAGES_OPERATOR_VERSION = "1.0.0"
OCR_COMPARE_PAGES_OPERATOR_ID = "ocr.compare_pages"
OCR_COMPARE_PAGES_OPERATOR_VERSION = "1.0.0"
OCR_SELECT_PAGES_OPERATOR_ID = "ocr.select_pages"
OCR_SELECT_PAGES_OPERATOR_VERSION = "1.0.0"
CONTEXT_STATIC_DEFINE_OPERATOR_ID = "context.static.define"
CONTEXT_STATIC_DEFINE_OPERATOR_VERSION = "1.0.0"
PROMPT_TEMPLATE_DEFINE_OPERATOR_ID = "prompt.template.define"
PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION = "1.0.0"
EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID = "extraction.schema.define"
EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION = "1.0.0"
MODEL_BINDING_DEFINE_OPERATOR_ID = "model.binding.define"
MODEL_BINDING_DEFINE_OPERATOR_VERSION = "1.0.0"
INPUT_POLICY_DEFINE_OPERATOR_ID = "input.policy.define"
INPUT_POLICY_DEFINE_OPERATOR_VERSION = "1.0.0"
CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID = "extraction.contextual_structured"
CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION = "1.0.0"
EXPORT_DATASET_OPERATOR_ID = "export.dataset"
EXPORT_DATASET_OPERATOR_VERSION = "1.0.0"
SCHEMA_VALIDATION_OPERATOR_ID = "validation.schema"
SCHEMA_VALIDATION_OPERATOR_VERSION = "1.0.0"

DEBUG_TEXT_OUTPUT = PortSpec(
    name="text",
    artifact_type="debug.text",
    schema_version=1,
    description="Debug text artifact emitted from node configuration.",
)

DEBUG_EMIT_TEXT_SPEC = NodeSpec(
    id=DEBUG_EMIT_TEXT_OPERATOR_ID,
    version=DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    inputs=(),
    outputs=(DEBUG_TEXT_OUTPUT,),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "Text copied into the emitted debug artifact metadata.",
            },
            "payload_ref": {
                "type": "string",
                "minLength": 1,
                "description": "Optional payload reference for the emitted artifact.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    display_name="Emit Text",
    description="Emits a debug text artifact from workflow node configuration.",
)


OCR_EXTRACT_PAGES_SPEC = NodeSpec(
    id=OCR_EXTRACT_PAGES_OPERATOR_ID,
    version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
            sequence=True,
            description="Ordered source page-image artifacts.",
        ),
    ),
    outputs=(
        PortSpec(
            name="ocr_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Ordered OCR page-result artifacts.",
        ),
        PortSpec(
            name="ocr_document",
            artifact_type="ocr.document_result",
            schema_version=1,
            description="Document-level aggregate OCR result built from the ordered page sequence.",
        ),
    ),
    execution_mode=ExecutionMode.MAP,
    config_schema={
        "type": "object",
        "properties": {
            "engine": {
                "type": "string",
                "minLength": 1,
                "default": "local.text",
                "description": "Registered OCR engine binding used by the worker.",
            },
            "language_hints": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Optional language hints passed to OCR engines that support them.",
            },
            "engine_config": {
                "type": "object",
                "description": "Provider-specific OCR engine settings.",
            },
        },
        "additionalProperties": False,
    },
    display_name="OCR Extract Pages",
    description="Runs an OCR engine over an ordered page-image sequence.",
)


OCR_EXTRACT_PAGE_SPEC = NodeSpec(
    id=OCR_EXTRACT_PAGE_OPERATOR_ID,
    version=OCR_EXTRACT_PAGE_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
            sequence=True,
            description=(
                "Ordered source page-image artifacts. Concrete-map planning binds "
                "one item to each node run."
            ),
        ),
    ),
    outputs=(
        PortSpec(
            name="ocr_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="OCR page-result refs produced by this concrete map item.",
        ),
    ),
    execution_mode=ExecutionMode.MAP,
    config_schema=OCR_EXTRACT_PAGES_SPEC.config_schema,
    display_name="OCR Extract Page",
    description=(
        "Runs an OCR engine for one concrete page item while preserving the same "
        "config contract as OCR Extract Pages."
    ),
)


OCR_COLLECT_PAGES_SPEC = NodeSpec(
    id=OCR_COLLECT_PAGES_OPERATOR_ID,
    version=OCR_COLLECT_PAGES_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="ocr_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Ordered OCR page-result artifacts to collect.",
        ),
    ),
    outputs=(
        PortSpec(
            name="ocr_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="The collected ordered OCR page-result sequence.",
        ),
        PortSpec(
            name="ocr_document",
            artifact_type="ocr.document_result",
            schema_version=1,
            description="Document-level aggregate OCR result built from page results.",
        ),
    ),
    execution_mode=ExecutionMode.REDUCE,
    config_schema={
        "type": "object",
        "additionalProperties": False,
    },
    display_name="OCR Collect Pages",
    description="Collects concrete OCR page results into sequence and document artifacts.",
)


OCR_COMPARE_PAGES_SPEC = NodeSpec(
    id=OCR_COMPARE_PAGES_OPERATOR_ID,
    version=OCR_COMPARE_PAGES_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="candidate_a_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="First ordered OCR page-result sequence to compare.",
        ),
        PortSpec(
            name="candidate_b_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Second ordered OCR page-result sequence to compare.",
        ),
    ),
    outputs=(
        PortSpec(
            name="comparison_pages",
            artifact_type="ocr.comparison_result",
            schema_version=1,
            sequence=True,
            description="Ordered page-level OCR comparison artifacts.",
        ),
        PortSpec(
            name="metrics",
            artifact_type="evaluation.metrics",
            schema_version=1,
            description="Aggregate OCR comparison metrics.",
        ),
    ),
    execution_mode=ExecutionMode.REDUCE,
    config_schema={
        "type": "object",
        "properties": {
            "candidate_a_label": {
                "type": "string",
                "minLength": 1,
                "default": "candidate_a",
                "description": "Human-readable label for the first OCR sequence.",
            },
            "candidate_b_label": {
                "type": "string",
                "minLength": 1,
                "default": "candidate_b",
                "description": "Human-readable label for the second OCR sequence.",
            },
        },
        "additionalProperties": False,
    },
    display_name="Compare OCR Pages",
    description="Compares two ordered OCR page-result sequences page by page.",
)


OCR_SELECT_PAGES_SPEC = NodeSpec(
    id=OCR_SELECT_PAGES_OPERATOR_ID,
    version=OCR_SELECT_PAGES_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="candidate_a_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="First ordered OCR page-result sequence available for selection.",
        ),
        PortSpec(
            name="candidate_b_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Second ordered OCR page-result sequence available for selection.",
        ),
        PortSpec(
            name="comparison_pages",
            artifact_type="ocr.comparison_result",
            schema_version=1,
            sequence=True,
            required=False,
            description="Optional page-level comparison sequence used as selection evidence.",
        ),
    ),
    outputs=(
        PortSpec(
            name="selected_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Ordered OCR page-result sequence selected for downstream nodes.",
        ),
    ),
    execution_mode=ExecutionMode.REDUCE,
    config_schema={
        "type": "object",
        "properties": {
            "selected_candidate": {
                "type": "string",
                "enum": ["candidate_a", "candidate_b"],
                "default": "candidate_a",
                "description": "Candidate OCR sequence selected for downstream use.",
            },
            "decision_note": {
                "type": "string",
                "minLength": 1,
                "description": "Optional human-readable reason for the selection.",
            },
        },
        "additionalProperties": False,
    },
    display_name="Select OCR Pages",
    description="Selects one OCR page-result sequence after comparison.",
)


CONTEXT_STATIC_DEFINE_SPEC = NodeSpec(
    id=CONTEXT_STATIC_DEFINE_OPERATOR_ID,
    version=CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
    inputs=(),
    outputs=(
        PortSpec(
            name="context",
            artifact_type="context.bundle",
            schema_version=1,
            description="Static user-authored context bundle artifact.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Human-readable context bundle name.",
            },
            "context": {
                "type": "object",
                "description": "Serializable context object attached to downstream input assembly.",
            },
            "applies_to": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Optional input names, artifact types, or channels covered by this context.",
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "description": "Optional context description.",
            },
        },
        "required": ["name", "context"],
        "additionalProperties": False,
    },
    display_name="Static Context",
    description="Materializes user-authored static context as a typed artifact.",
)


PROMPT_TEMPLATE_DEFINE_SPEC = NodeSpec(
    id=PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
    version=PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
    inputs=(),
    outputs=(
        PortSpec(
            name="template",
            artifact_type="prompt.template",
            schema_version=1,
            description="Reusable prompt or instruction template artifact.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Human-readable prompt template name.",
            },
            "template": {
                "type": "string",
                "minLength": 1,
                "description": "Template text rendered by downstream model nodes.",
            },
            "template_format": {
                "type": "string",
                "enum": ["jinja2", "plain_text", "markdown"],
                "default": "jinja2",
                "description": "Template syntax expected by downstream renderers.",
            },
            "variables": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Declared variable names expected by the template.",
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "description": "Optional template description.",
            },
        },
        "required": ["name", "template"],
        "additionalProperties": False,
    },
    display_name="Prompt Template",
    description="Materializes a prompt or instruction template as a typed artifact.",
)


EXTRACTION_SCHEMA_DEFINE_SPEC = NodeSpec(
    id=EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
    version=EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
    inputs=(),
    outputs=(
        PortSpec(
            name="schema",
            artifact_type="extraction.schema",
            schema_version=1,
            description="Structured output schema artifact for extraction or validation.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Human-readable schema name.",
            },
            "json_schema": {
                "type": "object",
                "description": "JSON Schema describing the desired output structure.",
            },
            "schema_format": {
                "type": "string",
                "enum": ["json_schema"],
                "default": "json_schema",
                "description": "Schema format consumed by downstream extraction nodes.",
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "description": "Optional schema description.",
            },
        },
        "required": ["name", "json_schema"],
        "additionalProperties": False,
    },
    display_name="Extraction Schema",
    description="Materializes a structured output schema as a typed artifact.",
)


MODEL_BINDING_DEFINE_SPEC = NodeSpec(
    id=MODEL_BINDING_DEFINE_OPERATOR_ID,
    version=MODEL_BINDING_DEFINE_OPERATOR_VERSION,
    inputs=(),
    outputs=(
        PortSpec(
            name="binding",
            artifact_type="model.binding",
            schema_version=1,
            description="Model/provider binding artifact for downstream runtime nodes.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "minLength": 1,
                "description": "Provider or local runtime identifier.",
            },
            "model": {
                "type": "string",
                "minLength": 1,
                "description": "Provider model or local model identifier.",
            },
            "parameters": {
                "type": "object",
                "description": "Non-secret model parameters such as temperature or max output size.",
            },
            "capabilities": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Declared model capabilities such as text, vision, or structured_output.",
            },
            "credential_ref": {
                "type": "string",
                "minLength": 1,
                "description": "Reference to a secret managed outside the workflow artifact graph.",
            },
            "endpoint_ref": {
                "type": "string",
                "minLength": 1,
                "description": "Optional endpoint or runtime reference.",
            },
        },
        "required": ["provider", "model"],
        "additionalProperties": False,
    },
    display_name="Model Binding",
    description="Materializes a provider/model binding as a typed artifact.",
)


INPUT_POLICY_DEFINE_SPEC = NodeSpec(
    id=INPUT_POLICY_DEFINE_OPERATOR_ID,
    version=INPUT_POLICY_DEFINE_OPERATOR_VERSION,
    inputs=(),
    outputs=(
        PortSpec(
            name="policy",
            artifact_type="input.policy",
            schema_version=1,
            description="Input assembly, context, or history policy artifact.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Human-readable policy name.",
            },
            "policy_type": {
                "type": "string",
                "enum": ["stateless", "accumulating", "sliding_window", "custom"],
                "description": "Input/context policy family consumed by runtime nodes.",
            },
            "settings": {
                "type": "object",
                "description": "Policy-specific settings such as window size or pruning flags.",
            },
            "applies_to": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Optional input names, artifact types, or context channels covered by the policy.",
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "description": "Optional policy description.",
            },
        },
        "required": ["name", "policy_type"],
        "additionalProperties": False,
    },
    display_name="Input Policy",
    description="Materializes an input assembly or context policy as a typed artifact.",
)


CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC = NodeSpec(
    id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
    version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="text",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
            description="Ordered OCR or text-bearing page-result sequence.",
        ),
        PortSpec(
            name="schema",
            artifact_type="extraction.schema",
            schema_version=1,
            description="Structured output schema artifact.",
        ),
        PortSpec(
            name="template",
            artifact_type="prompt.template",
            schema_version=1,
            description="Prompt template artifact used to render model inputs.",
        ),
        PortSpec(
            name="binding",
            artifact_type="model.binding",
            schema_version=1,
            description="Provider/model binding artifact.",
        ),
        PortSpec(
            name="policy",
            artifact_type="input.policy",
            schema_version=1,
            description="Input assembly and context policy artifact.",
        ),
        PortSpec(
            name="context",
            artifact_type="context.bundle",
            schema_version=1,
            required=False,
            description="Optional static context bundle attached to model inputs.",
        ),
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
            sequence=True,
            required=False,
            description="Optional ordered source page-image sequence.",
        ),
    ),
    outputs=(
        PortSpec(
            name="page_results",
            artifact_type="extraction.record_result",
            schema_version=1,
            sequence=True,
            description="Ordered page-level structured extraction results.",
        ),
        PortSpec(
            name="document_result",
            artifact_type="extraction.document_result",
            schema_version=1,
            description="Document-level aggregate extraction artifact.",
        ),
        PortSpec(
            name="model_inputs",
            artifact_type="model.input",
            schema_version=1,
            sequence=True,
            description="Ordered rendered model input artifacts.",
        ),
        PortSpec(
            name="model_responses",
            artifact_type="model.response",
            schema_version=1,
            sequence=True,
            description="Ordered raw model response artifacts.",
        ),
    ),
    execution_mode=ExecutionMode.STATEFUL_SEQUENCE,
    config_schema={
        "type": "object",
        "properties": {
            "result_key": {
                "type": "string",
                "minLength": 1,
                "default": "record",
                "description": "Field name used by local engines when wrapping extracted data.",
            }
        },
        "additionalProperties": False,
    },
    display_name="Contextual Structured Extraction",
    description=(
        "Runs stateful structured extraction over ordered source text with "
        "explicit prompt, schema, model binding, and input policy artifacts."
    ),
)


EXPORT_DATASET_SPEC = NodeSpec(
    id=EXPORT_DATASET_OPERATOR_ID,
    version=EXPORT_DATASET_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="document",
            artifact_type="extraction.document_result",
            schema_version=1,
            description="Document-level structured extraction artifact to export.",
        ),
    ),
    outputs=(
        PortSpec(
            name="dataset",
            artifact_type="export.dataset",
            schema_version=1,
            description="Downloadable dataset export artifact.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "jsonl", "csv"],
                "default": "json",
                "description": "Export payload format.",
            },
            "filename": {
                "type": "string",
                "minLength": 1,
                "description": "Optional export filename stored in artifact metadata.",
            },
        },
        "additionalProperties": False,
    },
    display_name="Export Dataset",
    description="Exports document-level records as JSON, JSONL, or CSV.",
)


SCHEMA_VALIDATION_SPEC = NodeSpec(
    id=SCHEMA_VALIDATION_OPERATOR_ID,
    version=SCHEMA_VALIDATION_OPERATOR_VERSION,
    inputs=(
        PortSpec(
            name="document",
            artifact_type="extraction.document_result",
            schema_version=1,
            description="Document-level extraction artifact whose records will be validated.",
        ),
        PortSpec(
            name="schema",
            artifact_type="extraction.schema",
            schema_version=1,
            description="JSON Schema artifact used to validate each extracted record.",
        ),
    ),
    outputs=(
        PortSpec(
            name="validation",
            artifact_type="validation.result",
            schema_version=1,
            description="Schema validation result with per-record validation errors.",
        ),
        PortSpec(
            name="metrics",
            artifact_type="evaluation.metrics",
            schema_version=1,
            description="Normalized schema-validation metrics for experiment comparison.",
        ),
    ),
    execution_mode=ExecutionMode.SINGLE,
    config_schema={
        "type": "object",
        "additionalProperties": False,
    },
    display_name="Schema Validation",
    description=(
        "Validates document-level extraction records against an extraction schema "
        "and emits auditable validation results plus normalized metrics."
    ),
)


def builtin_node_specs() -> NodeSpecRegistry:
    return {
        (
            CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC.id,
            CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC.version,
        ): CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC,
        (CONTEXT_STATIC_DEFINE_SPEC.id, CONTEXT_STATIC_DEFINE_SPEC.version): (
            CONTEXT_STATIC_DEFINE_SPEC
        ),
        (DEBUG_EMIT_TEXT_SPEC.id, DEBUG_EMIT_TEXT_SPEC.version): DEBUG_EMIT_TEXT_SPEC,
        (EXPORT_DATASET_SPEC.id, EXPORT_DATASET_SPEC.version): EXPORT_DATASET_SPEC,
        (EXTRACTION_SCHEMA_DEFINE_SPEC.id, EXTRACTION_SCHEMA_DEFINE_SPEC.version): (
            EXTRACTION_SCHEMA_DEFINE_SPEC
        ),
        (INPUT_POLICY_DEFINE_SPEC.id, INPUT_POLICY_DEFINE_SPEC.version): (
            INPUT_POLICY_DEFINE_SPEC
        ),
        (MODEL_BINDING_DEFINE_SPEC.id, MODEL_BINDING_DEFINE_SPEC.version): (
            MODEL_BINDING_DEFINE_SPEC
        ),
        (OCR_COLLECT_PAGES_SPEC.id, OCR_COLLECT_PAGES_SPEC.version): (
            OCR_COLLECT_PAGES_SPEC
        ),
        (OCR_COMPARE_PAGES_SPEC.id, OCR_COMPARE_PAGES_SPEC.version): OCR_COMPARE_PAGES_SPEC,
        (OCR_EXTRACT_PAGE_SPEC.id, OCR_EXTRACT_PAGE_SPEC.version): OCR_EXTRACT_PAGE_SPEC,
        (OCR_EXTRACT_PAGES_SPEC.id, OCR_EXTRACT_PAGES_SPEC.version): OCR_EXTRACT_PAGES_SPEC,
        (OCR_SELECT_PAGES_SPEC.id, OCR_SELECT_PAGES_SPEC.version): OCR_SELECT_PAGES_SPEC,
        (PROMPT_TEMPLATE_DEFINE_SPEC.id, PROMPT_TEMPLATE_DEFINE_SPEC.version): (
            PROMPT_TEMPLATE_DEFINE_SPEC
        ),
        (SCHEMA_VALIDATION_SPEC.id, SCHEMA_VALIDATION_SPEC.version): (
            SCHEMA_VALIDATION_SPEC
        ),
    }

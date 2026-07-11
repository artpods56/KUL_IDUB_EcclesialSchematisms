from dataclasses import dataclass
from enum import StrEnum

from notarius_core.application.operators import (
    CONTEXT_STATIC_DEFINE_OPERATOR_ID,
    CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    EXPORT_DATASET_OPERATOR_ID,
    EXPORT_DATASET_OPERATOR_VERSION,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
    INPUT_POLICY_DEFINE_OPERATOR_ID,
    INPUT_POLICY_DEFINE_OPERATOR_VERSION,
    MODEL_BINDING_DEFINE_OPERATOR_ID,
    MODEL_BINDING_DEFINE_OPERATOR_VERSION,
    OCR_COLLECT_PAGES_OPERATOR_ID,
    OCR_COLLECT_PAGES_OPERATOR_VERSION,
    OCR_COMPARE_PAGES_OPERATOR_ID,
    OCR_COMPARE_PAGES_OPERATOR_VERSION,
    OCR_EXTRACT_PAGE_OPERATOR_ID,
    OCR_EXTRACT_PAGE_OPERATOR_VERSION,
    OCR_EXTRACT_PAGES_OPERATOR_ID,
    OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    OCR_SELECT_PAGES_OPERATOR_ID,
    OCR_SELECT_PAGES_OPERATOR_VERSION,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
)
from notarius_core.application.workflows.launcher import (
    CONCRETE_MAP_EXECUTION_PLANNING,
)
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    JsonObject,
    JsonValue,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
)


class WorkflowTemplateId(StrEnum):
    OCR_PAGES = "ocr-pages"
    CONTEXTUAL_EXTRACTION = "contextual-extraction"
    OCR_COMPARE_CONTEXTUAL_EXTRACTION = "ocr-compare-contextual-extraction"


@dataclass(frozen=True, slots=True)
class WorkflowTemplate:
    id: WorkflowTemplateId
    version: str
    display_name: str
    description: str
    config_schema: JsonObject


DEFAULT_CONTEXTUAL_EXTRACTION_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "page_number": {"type": "integer"},
    },
    "required": ["text", "page_number"],
    "additionalProperties": False,
}

DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE = (
    "Extract the requested structured record from page {{ CURRENT_PAGE_NUMBER }}.\n"
    "Current OCR text:\n{{ CURRENT_PAGE_TEXT }}\n"
    "Previous record: {{ PREVIOUS_RECORD }}\n"
    "Static context: {{ STATIC_CONTEXT }}"
)

PAGE_SEQUENCE_INPUT = PortSpec(
    name="pages",
    artifact_type="source.page_image",
    schema_version=1,
    sequence=True,
)

OCR_PAGES_TEMPLATE = WorkflowTemplate(
    id=WorkflowTemplateId.OCR_PAGES,
    version="1.0.0",
    display_name="OCR Pages",
    description="Run one OCR engine over an ordered source.page_image sequence.",
    config_schema={
        "type": "object",
        "properties": {
            "ocr": {
                "type": "object",
                "description": "Configuration for the ocr.extract_pages node.",
            },
            "execution_planning": {
                "type": "string",
                "enum": [CONCRETE_MAP_EXECUTION_PLANNING],
                "description": (
                    "Optional concrete-map planning mode that runs one OCR node "
                    "per page and collects the page artifacts."
                ),
            },
        },
        "additionalProperties": False,
    },
)

CONTEXTUAL_EXTRACTION_TEMPLATE = WorkflowTemplate(
    id=WorkflowTemplateId.CONTEXTUAL_EXTRACTION,
    version="1.0.0",
    display_name="OCR + Contextual Extraction",
    description=(
        "Run OCR, materialize prompt/schema/model/policy/static-context artifacts, "
        "then execute contextual structured extraction over the page sequence."
    ),
    config_schema={
        "type": "object",
        "properties": {
            "ocr": {"type": "object"},
            "prompt": {"type": "object"},
            "schema": {"type": "object"},
            "model": {"type": "object"},
            "policy": {"type": "object"},
            "context": {"type": "object"},
            "extraction": {"type": "object"},
            "export": {"type": "object"},
        },
        "additionalProperties": False,
    },
)

OCR_COMPARE_CONTEXTUAL_EXTRACTION_TEMPLATE = WorkflowTemplate(
    id=WorkflowTemplateId.OCR_COMPARE_CONTEXTUAL_EXTRACTION,
    version="1.0.0",
    display_name="OCR Compare + Contextual Extraction",
    description=(
        "Run two OCR branches, compare and select one OCR stream, then execute "
        "contextual structured extraction and export a dataset artifact."
    ),
    config_schema={
        "type": "object",
        "properties": {
            "ocr_a": {"type": "object"},
            "ocr_b": {"type": "object"},
            "compare": {"type": "object"},
            "select": {"type": "object"},
            "prompt": {"type": "object"},
            "schema": {"type": "object"},
            "model": {"type": "object"},
            "policy": {"type": "object"},
            "context": {"type": "object"},
            "extraction": {"type": "object"},
            "export": {"type": "object"},
        },
        "additionalProperties": False,
    },
)

WORKFLOW_TEMPLATES = {
    WorkflowTemplateId.OCR_PAGES: OCR_PAGES_TEMPLATE,
    WorkflowTemplateId.CONTEXTUAL_EXTRACTION: CONTEXTUAL_EXTRACTION_TEMPLATE,
    WorkflowTemplateId.OCR_COMPARE_CONTEXTUAL_EXTRACTION: (
        OCR_COMPARE_CONTEXTUAL_EXTRACTION_TEMPLATE
    ),
}


def list_workflow_templates() -> tuple[WorkflowTemplate, ...]:
    return tuple(WORKFLOW_TEMPLATES.values())


def build_workflow_definition_from_template(
    template_id: WorkflowTemplateId,
    config: JsonObject,
    name: str | None = None,
    description: str | None = None,
    metadata: JsonObject | None = None,
) -> WorkflowDefinition:
    _reject_sensitive_template_config(config)
    if template_id == WorkflowTemplateId.OCR_PAGES:
        return _build_ocr_pages_definition(config, name, description, metadata)
    if template_id == WorkflowTemplateId.CONTEXTUAL_EXTRACTION:
        return _build_contextual_extraction_definition(
            config,
            name,
            description,
            metadata,
        )
    if template_id == WorkflowTemplateId.OCR_COMPARE_CONTEXTUAL_EXTRACTION:
        return _build_ocr_compare_contextual_extraction_definition(
            config,
            name,
            description,
            metadata,
        )
    raise ValidationError(f"Unsupported workflow template: {template_id}")


def workflow_template(template_id: WorkflowTemplateId) -> WorkflowTemplate:
    try:
        return WORKFLOW_TEMPLATES[template_id]
    except KeyError as exc:
        raise ValidationError(f"Unsupported workflow template: {template_id}") from exc


def workflow_template_id(value: str) -> WorkflowTemplateId:
    try:
        return WorkflowTemplateId(value)
    except ValueError as exc:
        raise ValidationError(f"Unsupported workflow template: {value}") from exc


def _build_ocr_pages_definition(
    config: JsonObject,
    name: str | None,
    description: str | None,
    metadata: JsonObject | None,
) -> WorkflowDefinition:
    ocr_config = _object_config(config, "ocr")
    execution_planning = _execution_planning_config(config)
    workflow_name = name or _optional_string_config(config, "name") or "OCR pages"
    workflow_description = (
        description
        or _optional_string_config(config, "description")
        or OCR_PAGES_TEMPLATE.description
    )
    workflow_metadata = _template_metadata(OCR_PAGES_TEMPLATE, metadata)
    if execution_planning == CONCRETE_MAP_EXECUTION_PLANNING:
        workflow_metadata["execution_planning"] = CONCRETE_MAP_EXECUTION_PLANNING
        return WorkflowDefinition(
            name=workflow_name,
            description=workflow_description,
            nodes=[
                WorkflowNode(
                    id="ocr",
                    operator_id=OCR_EXTRACT_PAGE_OPERATOR_ID,
                    operator_version=OCR_EXTRACT_PAGE_OPERATOR_VERSION,
                    config=_ocr_node_config(ocr_config),
                ),
                WorkflowNode(
                    id="collect",
                    operator_id=OCR_COLLECT_PAGES_OPERATOR_ID,
                    operator_version=OCR_COLLECT_PAGES_OPERATOR_VERSION,
                    config={},
                ),
            ],
            edges=[
                WorkflowEdge(
                    from_node_id="ocr",
                    from_port="ocr_pages",
                    to_node_id="collect",
                    to_port="ocr_pages",
                ),
            ],
            declared_inputs=[PAGE_SEQUENCE_INPUT],
            metadata=workflow_metadata,
        )

    return WorkflowDefinition(
        name=workflow_name,
        description=workflow_description,
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
                operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
                config=_ocr_node_config(ocr_config),
            )
        ],
        declared_inputs=[PAGE_SEQUENCE_INPUT],
        metadata=workflow_metadata,
    )


def _build_contextual_extraction_definition(
    config: JsonObject,
    name: str | None,
    description: str | None,
    metadata: JsonObject | None,
) -> WorkflowDefinition:
    workflow_name = (
        name
        or _optional_string_config(config, "name")
        or "OCR + contextual extraction"
    )
    workflow_description = (
        description
        or _optional_string_config(config, "description")
        or CONTEXTUAL_EXTRACTION_TEMPLATE.description
    )
    return WorkflowDefinition(
        name=workflow_name,
        description=workflow_description,
        nodes=[
            WorkflowNode(
                id="prompt",
                operator_id=PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                operator_version=PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
                config=_prompt_node_config(_object_config(config, "prompt")),
            ),
            WorkflowNode(
                id="context",
                operator_id=CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                operator_version=CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
                config=_context_node_config(_object_config(config, "context")),
            ),
            WorkflowNode(
                id="schema",
                operator_id=EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                operator_version=EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
                config=_schema_node_config(_object_config(config, "schema")),
            ),
            WorkflowNode(
                id="model",
                operator_id=MODEL_BINDING_DEFINE_OPERATOR_ID,
                operator_version=MODEL_BINDING_DEFINE_OPERATOR_VERSION,
                config=_model_node_config(_object_config(config, "model")),
            ),
            WorkflowNode(
                id="policy",
                operator_id=INPUT_POLICY_DEFINE_OPERATOR_ID,
                operator_version=INPUT_POLICY_DEFINE_OPERATOR_VERSION,
                config=_policy_node_config(_object_config(config, "policy")),
            ),
            WorkflowNode(
                id="ocr",
                operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
                operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
                config=_ocr_node_config(_object_config(config, "ocr")),
            ),
            WorkflowNode(
                id="extract",
                operator_id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
                operator_version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
                config=_extraction_node_config(_object_config(config, "extraction")),
            ),
            WorkflowNode(
                id="export",
                operator_id=EXPORT_DATASET_OPERATOR_ID,
                operator_version=EXPORT_DATASET_OPERATOR_VERSION,
                config=_export_node_config(_object_config(config, "export")),
            ),
        ],
        edges=[
            WorkflowEdge(
                from_node_id="ocr",
                from_port="ocr_pages",
                to_node_id="extract",
                to_port="text",
            ),
            WorkflowEdge(
                from_node_id="prompt",
                from_port="template",
                to_node_id="extract",
                to_port="template",
            ),
            WorkflowEdge(
                from_node_id="context",
                from_port="context",
                to_node_id="extract",
                to_port="context",
            ),
            WorkflowEdge(
                from_node_id="schema",
                from_port="schema",
                to_node_id="extract",
                to_port="schema",
            ),
            WorkflowEdge(
                from_node_id="model",
                from_port="binding",
                to_node_id="extract",
                to_port="binding",
            ),
            WorkflowEdge(
                from_node_id="policy",
                from_port="policy",
                to_node_id="extract",
                to_port="policy",
            ),
            WorkflowEdge(
                from_node_id="extract",
                from_port="document_result",
                to_node_id="export",
                to_port="document",
            ),
        ],
        declared_inputs=[PAGE_SEQUENCE_INPUT],
        metadata=_template_metadata(CONTEXTUAL_EXTRACTION_TEMPLATE, metadata),
    )


def _build_ocr_compare_contextual_extraction_definition(
    config: JsonObject,
    name: str | None,
    description: str | None,
    metadata: JsonObject | None,
) -> WorkflowDefinition:
    workflow_name = (
        name
        or _optional_string_config(config, "name")
        or "OCR compare + contextual extraction"
    )
    workflow_description = (
        description
        or _optional_string_config(config, "description")
        or OCR_COMPARE_CONTEXTUAL_EXTRACTION_TEMPLATE.description
    )
    return WorkflowDefinition(
        name=workflow_name,
        description=workflow_description,
        nodes=[
            WorkflowNode(
                id="prompt",
                operator_id=PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                operator_version=PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
                config=_prompt_node_config(_object_config(config, "prompt")),
            ),
            WorkflowNode(
                id="context",
                operator_id=CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                operator_version=CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
                config=_context_node_config(_object_config(config, "context")),
            ),
            WorkflowNode(
                id="schema",
                operator_id=EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                operator_version=EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
                config=_schema_node_config(_object_config(config, "schema")),
            ),
            WorkflowNode(
                id="model",
                operator_id=MODEL_BINDING_DEFINE_OPERATOR_ID,
                operator_version=MODEL_BINDING_DEFINE_OPERATOR_VERSION,
                config=_model_node_config(_object_config(config, "model")),
            ),
            WorkflowNode(
                id="policy",
                operator_id=INPUT_POLICY_DEFINE_OPERATOR_ID,
                operator_version=INPUT_POLICY_DEFINE_OPERATOR_VERSION,
                config=_policy_node_config(_object_config(config, "policy")),
            ),
            WorkflowNode(
                id="ocr_a",
                operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
                operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
                config=_ocr_node_config(_object_config(config, "ocr_a")),
            ),
            WorkflowNode(
                id="ocr_b",
                operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
                operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
                config=_ocr_node_config(_object_config(config, "ocr_b")),
            ),
            WorkflowNode(
                id="compare",
                operator_id=OCR_COMPARE_PAGES_OPERATOR_ID,
                operator_version=OCR_COMPARE_PAGES_OPERATOR_VERSION,
                config=_compare_node_config(_object_config(config, "compare")),
            ),
            WorkflowNode(
                id="select",
                operator_id=OCR_SELECT_PAGES_OPERATOR_ID,
                operator_version=OCR_SELECT_PAGES_OPERATOR_VERSION,
                config=_select_node_config(_object_config(config, "select")),
            ),
            WorkflowNode(
                id="extract",
                operator_id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
                operator_version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
                config=_extraction_node_config(_object_config(config, "extraction")),
            ),
            WorkflowNode(
                id="export",
                operator_id=EXPORT_DATASET_OPERATOR_ID,
                operator_version=EXPORT_DATASET_OPERATOR_VERSION,
                config=_export_node_config(_object_config(config, "export")),
            ),
        ],
        edges=[
            WorkflowEdge(
                from_node_id="ocr_a",
                from_port="ocr_pages",
                to_node_id="compare",
                to_port="candidate_a_pages",
            ),
            WorkflowEdge(
                from_node_id="ocr_b",
                from_port="ocr_pages",
                to_node_id="compare",
                to_port="candidate_b_pages",
            ),
            WorkflowEdge(
                from_node_id="ocr_a",
                from_port="ocr_pages",
                to_node_id="select",
                to_port="candidate_a_pages",
            ),
            WorkflowEdge(
                from_node_id="ocr_b",
                from_port="ocr_pages",
                to_node_id="select",
                to_port="candidate_b_pages",
            ),
            WorkflowEdge(
                from_node_id="compare",
                from_port="comparison_pages",
                to_node_id="select",
                to_port="comparison_pages",
            ),
            WorkflowEdge(
                from_node_id="select",
                from_port="selected_pages",
                to_node_id="extract",
                to_port="text",
            ),
            WorkflowEdge(
                from_node_id="prompt",
                from_port="template",
                to_node_id="extract",
                to_port="template",
            ),
            WorkflowEdge(
                from_node_id="context",
                from_port="context",
                to_node_id="extract",
                to_port="context",
            ),
            WorkflowEdge(
                from_node_id="schema",
                from_port="schema",
                to_node_id="extract",
                to_port="schema",
            ),
            WorkflowEdge(
                from_node_id="model",
                from_port="binding",
                to_node_id="extract",
                to_port="binding",
            ),
            WorkflowEdge(
                from_node_id="policy",
                from_port="policy",
                to_node_id="extract",
                to_port="policy",
            ),
            WorkflowEdge(
                from_node_id="extract",
                from_port="document_result",
                to_node_id="export",
                to_port="document",
            ),
        ],
        declared_inputs=[PAGE_SEQUENCE_INPUT],
        metadata=_template_metadata(OCR_COMPARE_CONTEXTUAL_EXTRACTION_TEMPLATE, metadata),
    )


def _ocr_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "engine": _string_config(config, "engine", "local.text"),
            "language_hints": _string_list_config(config, "language_hints"),
            "engine_config": _object_config(config, "engine_config"),
        }
    )


def _compare_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "candidate_a_label": _string_config(
                config,
                "candidate_a_label",
                "candidate_a",
            ),
            "candidate_b_label": _string_config(
                config,
                "candidate_b_label",
                "candidate_b",
            ),
        }
    )


def _select_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "selected_candidate": _string_config(
                config,
                "selected_candidate",
                "candidate_a",
            ),
            "decision_note": _optional_string_config(config, "decision_note"),
        }
    )


def _prompt_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "name": _string_config(config, "name", "Default extraction prompt"),
            "template": _string_config(
                config,
                "template",
                DEFAULT_CONTEXTUAL_PROMPT_TEMPLATE,
            ),
            "template_format": _string_config(config, "template_format", "jinja2"),
            "variables": _string_list_config(
                config,
                "variables",
                [
                    "CURRENT_PAGE_NUMBER",
                    "CURRENT_PAGE_TEXT",
                    "PREVIOUS_RECORD",
                    "STATIC_CONTEXT",
                ],
            ),
            "description": _optional_string_config(config, "description"),
        }
    )


def _schema_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "name": _string_config(config, "name", "Default page record schema"),
            "json_schema": _object_config(
                config,
                "json_schema",
                DEFAULT_CONTEXTUAL_EXTRACTION_SCHEMA,
            ),
            "schema_format": _string_config(config, "schema_format", "json_schema"),
            "description": _optional_string_config(config, "description"),
        }
    )


def _model_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "provider": _string_config(config, "provider", "local.echo"),
            "model": _string_config(config, "model", "local.echo"),
            "parameters": _object_config(config, "parameters"),
            "capabilities": _string_list_config(
                config,
                "capabilities",
                ["structured_output"],
            ),
            "credential_ref": _optional_string_config(config, "credential_ref"),
            "endpoint_ref": _optional_string_config(config, "endpoint_ref"),
        }
    )


def _policy_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "name": _string_config(config, "name", "Default input policy"),
            "policy_type": _string_config(config, "policy_type", "accumulating"),
            "settings": _object_config(config, "settings"),
            "applies_to": _string_list_config(config, "applies_to", ["text", "pages"]),
            "description": _optional_string_config(config, "description"),
        }
    )


def _context_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "name": _string_config(config, "name", "Static context"),
            "context": _object_config(config, "context"),
            "applies_to": _string_list_config(config, "applies_to", ["text", "pages"]),
            "description": _optional_string_config(config, "description"),
        }
    )


def _extraction_node_config(config: JsonObject) -> JsonObject:
    return {"result_key": _string_config(config, "result_key", "record")}


def _export_node_config(config: JsonObject) -> JsonObject:
    return _omit_none(
        {
            "format": _string_config(config, "format", "json"),
            "filename": _optional_string_config(config, "filename"),
        }
    )


def _template_metadata(
    template: WorkflowTemplate,
    metadata: JsonObject | None,
) -> JsonObject:
    effective_metadata = dict(metadata or {})
    effective_metadata["template_id"] = template.id.value
    effective_metadata["template_version"] = template.version
    return effective_metadata


def _execution_planning_config(config: JsonObject) -> str | None:
    value = config.get("execution_planning")
    if value is None:
        return None
    if value != CONCRETE_MAP_EXECUTION_PLANNING:
        raise ValidationError(
            "Workflow template config field 'execution_planning' must be "
            f"{CONCRETE_MAP_EXECUTION_PLANNING!r}"
        )
    return value


def _object_config(
    config: JsonObject,
    key: str,
    default: JsonObject | None = None,
) -> JsonObject:
    value = config.get(key)
    if value is None:
        return dict(default or {})
    if not isinstance(value, dict):
        raise ValidationError(f"Workflow template config field {key!r} must be object")
    return dict(value)


def _string_config(config: JsonObject, key: str, default: str) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or value == "":
        raise ValidationError(
            f"Workflow template config field {key!r} must be a non-empty string"
        )
    return value


def _optional_string_config(config: JsonObject, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValidationError(
            f"Workflow template config field {key!r} must be a non-empty string"
        )
    return value


def _string_list_config(
    config: JsonObject,
    key: str,
    default: list[str] | None = None,
) -> list[str]:
    value = config.get(key, default or [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValidationError(
            f"Workflow template config field {key!r} must be a list of strings"
        )
    return list(value)


def _omit_none(config: dict[str, JsonValue | None]) -> JsonObject:
    return {key: value for key, value in config.items() if value is not None}


def _reject_sensitive_template_config(value: object) -> None:
    sensitive_keys = {
        "api_key",
        "access_key",
        "access_token",
        "bearer_token",
        "client_secret",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in sensitive_keys:
                raise ValidationError(
                    f"Workflow template config refuses sensitive field {key!r}; "
                    "use credential_ref or an *_env_var parameter instead"
                )
            _reject_sensitive_template_config(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_template_config(item)

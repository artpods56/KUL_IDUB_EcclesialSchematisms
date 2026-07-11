from notarius_core.application.operators import (
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC,
    DEBUG_EMIT_TEXT_OPERATOR_ID,
    DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    DEBUG_EMIT_TEXT_SPEC,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
    EXTRACTION_SCHEMA_DEFINE_SPEC,
    INPUT_POLICY_DEFINE_OPERATOR_ID,
    INPUT_POLICY_DEFINE_OPERATOR_VERSION,
    INPUT_POLICY_DEFINE_SPEC,
    MODEL_BINDING_DEFINE_OPERATOR_ID,
    MODEL_BINDING_DEFINE_OPERATOR_VERSION,
    MODEL_BINDING_DEFINE_SPEC,
    OCR_COLLECT_PAGES_OPERATOR_ID,
    OCR_COLLECT_PAGES_OPERATOR_VERSION,
    OCR_COLLECT_PAGES_SPEC,
    OCR_COMPARE_PAGES_OPERATOR_ID,
    OCR_COMPARE_PAGES_OPERATOR_VERSION,
    OCR_COMPARE_PAGES_SPEC,
    OCR_EXTRACT_PAGE_OPERATOR_ID,
    OCR_EXTRACT_PAGE_OPERATOR_VERSION,
    OCR_EXTRACT_PAGE_SPEC,
    OCR_EXTRACT_PAGES_OPERATOR_ID,
    OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    OCR_EXTRACT_PAGES_SPEC,
    OCR_SELECT_PAGES_OPERATOR_ID,
    OCR_SELECT_PAGES_OPERATOR_VERSION,
    OCR_SELECT_PAGES_SPEC,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
    PROMPT_TEMPLATE_DEFINE_SPEC,
    SCHEMA_VALIDATION_OPERATOR_ID,
    SCHEMA_VALIDATION_OPERATOR_VERSION,
    SCHEMA_VALIDATION_SPEC,
    builtin_node_specs,
)
from notarius_core.domain.models import ExecutionMode


def test_builtin_node_specs_registers_debug_emit_text_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(DEBUG_EMIT_TEXT_OPERATOR_ID, DEBUG_EMIT_TEXT_OPERATOR_VERSION)]

    assert spec is DEBUG_EMIT_TEXT_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert spec.inputs == ()
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "text"
    assert spec.outputs[0].artifact_type == "debug.text"
    assert spec.config_schema["required"] == ["text"]
    assert spec.config_schema["properties"]["text"]["minLength"] == 1
    assert spec.config_schema["properties"]["payload_ref"]["minLength"] == 1


def test_builtin_node_specs_registers_contextual_structured_extraction_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[
        (
            CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
            CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
        )
    ]

    assert spec is CONTEXTUAL_STRUCTURED_EXTRACTION_SPEC
    assert spec.execution_mode == ExecutionMode.STATEFUL_SEQUENCE
    assert [(port.name, port.artifact_type, port.sequence) for port in spec.inputs] == [
        ("text", "ocr.page_result", True),
        ("schema", "extraction.schema", False),
        ("template", "prompt.template", False),
        ("binding", "model.binding", False),
        ("policy", "input.policy", False),
        ("context", "context.bundle", False),
        ("pages", "source.page_image", True),
    ]
    assert spec.inputs[-2].required is False
    assert spec.inputs[-1].required is False
    assert [(port.name, port.artifact_type, port.sequence) for port in spec.outputs] == [
        ("page_results", "extraction.record_result", True),
        ("document_result", "extraction.document_result", False),
        ("model_inputs", "model.input", True),
        ("model_responses", "model.response", True),
    ]


def test_builtin_node_specs_registers_prompt_template_define_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[
        (PROMPT_TEMPLATE_DEFINE_OPERATOR_ID, PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION)
    ]

    assert spec is PROMPT_TEMPLATE_DEFINE_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert spec.inputs == ()
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "template"
    assert spec.outputs[0].artifact_type == "prompt.template"
    assert spec.config_schema["required"] == ["name", "template"]
    assert spec.config_schema["properties"]["template_format"]["enum"] == [
        "jinja2",
        "plain_text",
        "markdown",
    ]


def test_builtin_node_specs_registers_extraction_schema_define_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[
        (EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID, EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION)
    ]

    assert spec is EXTRACTION_SCHEMA_DEFINE_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert spec.inputs == ()
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "schema"
    assert spec.outputs[0].artifact_type == "extraction.schema"
    assert spec.config_schema["required"] == ["name", "json_schema"]
    assert spec.config_schema["properties"]["schema_format"]["enum"] == ["json_schema"]


def test_builtin_node_specs_registers_model_binding_define_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(MODEL_BINDING_DEFINE_OPERATOR_ID, MODEL_BINDING_DEFINE_OPERATOR_VERSION)]

    assert spec is MODEL_BINDING_DEFINE_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert spec.inputs == ()
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "binding"
    assert spec.outputs[0].artifact_type == "model.binding"
    assert spec.config_schema["required"] == ["provider", "model"]


def test_builtin_node_specs_registers_input_policy_define_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(INPUT_POLICY_DEFINE_OPERATOR_ID, INPUT_POLICY_DEFINE_OPERATOR_VERSION)]

    assert spec is INPUT_POLICY_DEFINE_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert spec.inputs == ()
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "policy"
    assert spec.outputs[0].artifact_type == "input.policy"
    assert spec.config_schema["required"] == ["name", "policy_type"]
    assert spec.config_schema["properties"]["policy_type"]["enum"] == [
        "stateless",
        "accumulating",
        "sliding_window",
        "custom",
    ]


def test_builtin_node_specs_registers_ocr_extract_pages_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(OCR_EXTRACT_PAGES_OPERATOR_ID, OCR_EXTRACT_PAGES_OPERATOR_VERSION)]

    assert spec is OCR_EXTRACT_PAGES_SPEC
    assert spec.execution_mode == ExecutionMode.MAP
    assert len(spec.inputs) == 1
    assert spec.inputs[0].name == "pages"
    assert spec.inputs[0].artifact_type == "source.page_image"
    assert spec.inputs[0].sequence is True
    assert len(spec.outputs) == 2
    assert spec.outputs[0].name == "ocr_pages"
    assert spec.outputs[0].artifact_type == "ocr.page_result"
    assert spec.outputs[0].sequence is True
    assert spec.outputs[1].name == "ocr_document"
    assert spec.outputs[1].artifact_type == "ocr.document_result"
    assert spec.outputs[1].sequence is False
    assert spec.config_schema["properties"]["engine"]["default"] == "local.text"
    assert spec.config_schema["properties"]["engine_config"]["type"] == "object"


def test_builtin_node_specs_registers_ocr_extract_page_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(OCR_EXTRACT_PAGE_OPERATOR_ID, OCR_EXTRACT_PAGE_OPERATOR_VERSION)]

    assert spec is OCR_EXTRACT_PAGE_SPEC
    assert spec.execution_mode == ExecutionMode.MAP
    assert len(spec.inputs) == 1
    assert spec.inputs[0].name == "pages"
    assert spec.inputs[0].artifact_type == "source.page_image"
    assert spec.inputs[0].sequence is True
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "ocr_pages"
    assert spec.outputs[0].artifact_type == "ocr.page_result"
    assert spec.outputs[0].sequence is True


def test_builtin_node_specs_registers_ocr_collect_pages_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(OCR_COLLECT_PAGES_OPERATOR_ID, OCR_COLLECT_PAGES_OPERATOR_VERSION)]

    assert spec is OCR_COLLECT_PAGES_SPEC
    assert spec.execution_mode == ExecutionMode.REDUCE
    assert len(spec.inputs) == 1
    assert spec.inputs[0].name == "ocr_pages"
    assert spec.inputs[0].artifact_type == "ocr.page_result"
    assert spec.inputs[0].sequence is True
    assert len(spec.outputs) == 2
    assert spec.outputs[0].name == "ocr_pages"
    assert spec.outputs[0].artifact_type == "ocr.page_result"
    assert spec.outputs[0].sequence is True
    assert spec.outputs[1].name == "ocr_document"
    assert spec.outputs[1].artifact_type == "ocr.document_result"
    assert spec.outputs[1].sequence is False


def test_builtin_node_specs_registers_ocr_compare_pages_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(OCR_COMPARE_PAGES_OPERATOR_ID, OCR_COMPARE_PAGES_OPERATOR_VERSION)]

    assert spec is OCR_COMPARE_PAGES_SPEC
    assert spec.execution_mode == ExecutionMode.REDUCE
    assert len(spec.inputs) == 2
    assert spec.inputs[0].name == "candidate_a_pages"
    assert spec.inputs[0].artifact_type == "ocr.page_result"
    assert spec.inputs[0].sequence is True
    assert spec.inputs[1].name == "candidate_b_pages"
    assert spec.inputs[1].artifact_type == "ocr.page_result"
    assert spec.inputs[1].sequence is True
    assert len(spec.outputs) == 2
    assert spec.outputs[0].name == "comparison_pages"
    assert spec.outputs[0].artifact_type == "ocr.comparison_result"
    assert spec.outputs[0].sequence is True
    assert spec.outputs[1].name == "metrics"
    assert spec.outputs[1].artifact_type == "evaluation.metrics"
    assert spec.outputs[1].sequence is False


def test_builtin_node_specs_registers_ocr_select_pages_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(OCR_SELECT_PAGES_OPERATOR_ID, OCR_SELECT_PAGES_OPERATOR_VERSION)]

    assert spec is OCR_SELECT_PAGES_SPEC
    assert spec.execution_mode == ExecutionMode.REDUCE
    assert len(spec.inputs) == 3
    assert spec.inputs[0].name == "candidate_a_pages"
    assert spec.inputs[0].artifact_type == "ocr.page_result"
    assert spec.inputs[0].sequence is True
    assert spec.inputs[1].name == "candidate_b_pages"
    assert spec.inputs[1].artifact_type == "ocr.page_result"
    assert spec.inputs[1].sequence is True
    assert spec.inputs[2].name == "comparison_pages"
    assert spec.inputs[2].artifact_type == "ocr.comparison_result"
    assert spec.inputs[2].sequence is True
    assert spec.inputs[2].required is False
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "selected_pages"
    assert spec.outputs[0].artifact_type == "ocr.page_result"
    assert spec.outputs[0].sequence is True
    assert spec.config_schema["properties"]["selected_candidate"]["enum"] == [
        "candidate_a",
        "candidate_b",
    ]


def test_builtin_node_specs_registers_schema_validation_operator() -> None:
    specs = builtin_node_specs()

    spec = specs[(SCHEMA_VALIDATION_OPERATOR_ID, SCHEMA_VALIDATION_OPERATOR_VERSION)]

    assert spec is SCHEMA_VALIDATION_SPEC
    assert spec.execution_mode == ExecutionMode.SINGLE
    assert [(port.name, port.artifact_type, port.sequence) for port in spec.inputs] == [
        ("document", "extraction.document_result", False),
        ("schema", "extraction.schema", False),
    ]
    assert [(port.name, port.artifact_type, port.sequence) for port in spec.outputs] == [
        ("validation", "validation.result", False),
        ("metrics", "evaluation.metrics", False),
    ]

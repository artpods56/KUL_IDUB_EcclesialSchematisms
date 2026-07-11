import pytest

from notarius_core.application.workflows.templates import (
    WorkflowTemplateId,
    build_workflow_definition_from_template,
    list_workflow_templates,
)
from notarius_core.domain.errors import ValidationError


def test_workflow_template_catalog_lists_backend_launchable_templates() -> None:
    templates = list_workflow_templates()

    assert [template.id for template in templates] == [
        WorkflowTemplateId.OCR_PAGES,
        WorkflowTemplateId.CONTEXTUAL_EXTRACTION,
        WorkflowTemplateId.OCR_COMPARE_CONTEXTUAL_EXTRACTION,
    ]
    assert templates[0].version == "1.0.0"
    assert templates[1].config_schema["type"] == "object"


def test_contextual_extraction_template_builds_typed_artifact_graph() -> None:
    definition = build_workflow_definition_from_template(
        WorkflowTemplateId.CONTEXTUAL_EXTRACTION,
        {
            "ocr": {"engine": "local.text"},
            "context": {
                "name": "Corpus context",
                "context": {"corpus": "schematism"},
            },
            "policy": {"policy_type": "sliding_window", "settings": {"size": 2}},
        },
        name="Schematism extraction",
        metadata={"source": "unit-test"},
    )

    nodes_by_id = {node.id: node for node in definition.nodes}
    edges = {
        (edge.from_node_id, edge.from_port, edge.to_node_id, edge.to_port)
        for edge in definition.edges
    }

    assert definition.name == "Schematism extraction"
    assert definition.metadata == {
        "source": "unit-test",
        "template_id": "contextual-extraction",
        "template_version": "1.0.0",
    }
    assert [port.name for port in definition.declared_inputs] == ["pages"]
    assert definition.declared_inputs[0].artifact_type == "source.page_image"
    assert set(nodes_by_id) == {
        "prompt",
        "context",
        "schema",
        "model",
        "policy",
        "ocr",
        "extract",
        "export",
    }
    assert nodes_by_id["ocr"].config["engine"] == "local.text"
    assert nodes_by_id["context"].config["context"] == {"corpus": "schematism"}
    assert nodes_by_id["policy"].config["policy_type"] == "sliding_window"
    assert nodes_by_id["export"].config["format"] == "json"
    assert edges == {
        ("ocr", "ocr_pages", "extract", "text"),
        ("prompt", "template", "extract", "template"),
        ("context", "context", "extract", "context"),
        ("schema", "schema", "extract", "schema"),
        ("model", "binding", "extract", "binding"),
        ("policy", "policy", "extract", "policy"),
        ("extract", "document_result", "export", "document"),
    }


def test_ocr_pages_template_rejects_sensitive_persisted_config() -> None:
    with pytest.raises(ValidationError, match="credential_ref"):
        build_workflow_definition_from_template(
            WorkflowTemplateId.OCR_PAGES,
            {"ocr": {"engine_config": {"api_key": "do-not-store"}}},
        )


def test_ocr_pages_template_can_build_concrete_map_graph() -> None:
    definition = build_workflow_definition_from_template(
        WorkflowTemplateId.OCR_PAGES,
        {
            "ocr": {"engine": "local.tesseract"},
            "execution_planning": "concrete_map",
        },
    )

    nodes_by_id = {node.id: node for node in definition.nodes}
    edges = {
        (edge.from_node_id, edge.from_port, edge.to_node_id, edge.to_port)
        for edge in definition.edges
    }

    assert definition.metadata == {
        "template_id": "ocr-pages",
        "template_version": "1.0.0",
        "execution_planning": "concrete_map",
    }
    assert set(nodes_by_id) == {"ocr", "collect"}
    assert nodes_by_id["ocr"].operator_id == "ocr.extract_page"
    assert nodes_by_id["ocr"].config["engine"] == "local.tesseract"
    assert nodes_by_id["collect"].operator_id == "ocr.collect_pages"
    assert edges == {("ocr", "ocr_pages", "collect", "ocr_pages")}


def test_ocr_compare_contextual_extraction_template_builds_full_workflow() -> None:
    definition = build_workflow_definition_from_template(
        WorkflowTemplateId.OCR_COMPARE_CONTEXTUAL_EXTRACTION,
        {
            "ocr_a": {"engine": "local.text"},
            "ocr_b": {"engine": "mistral.ocr"},
            "compare": {
                "candidate_a_label": "Local",
                "candidate_b_label": "Mistral",
            },
            "select": {
                "selected_candidate": "candidate_b",
                "decision_note": "Mistral preserved tables",
            },
        },
        name="Compare OCR then extract",
    )

    nodes_by_id = {node.id: node for node in definition.nodes}
    edges = {
        (edge.from_node_id, edge.from_port, edge.to_node_id, edge.to_port)
        for edge in definition.edges
    }

    assert definition.name == "Compare OCR then extract"
    assert definition.metadata == {
        "template_id": "ocr-compare-contextual-extraction",
        "template_version": "1.0.0",
    }
    assert [port.name for port in definition.declared_inputs] == ["pages"]
    assert set(nodes_by_id) == {
        "prompt",
        "context",
        "schema",
        "model",
        "policy",
        "ocr_a",
        "ocr_b",
        "compare",
        "select",
        "extract",
        "export",
    }
    assert nodes_by_id["ocr_a"].config["engine"] == "local.text"
    assert nodes_by_id["ocr_b"].config["engine"] == "mistral.ocr"
    assert nodes_by_id["compare"].config == {
        "candidate_a_label": "Local",
        "candidate_b_label": "Mistral",
    }
    assert nodes_by_id["select"].config == {
        "selected_candidate": "candidate_b",
        "decision_note": "Mistral preserved tables",
    }
    assert edges == {
        ("ocr_a", "ocr_pages", "compare", "candidate_a_pages"),
        ("ocr_b", "ocr_pages", "compare", "candidate_b_pages"),
        ("ocr_a", "ocr_pages", "select", "candidate_a_pages"),
        ("ocr_b", "ocr_pages", "select", "candidate_b_pages"),
        ("compare", "comparison_pages", "select", "comparison_pages"),
        ("select", "selected_pages", "extract", "text"),
        ("prompt", "template", "extract", "template"),
        ("context", "context", "extract", "context"),
        ("schema", "schema", "extract", "schema"),
        ("model", "binding", "extract", "binding"),
        ("policy", "policy", "extract", "policy"),
        ("extract", "document_result", "export", "document"),
    }

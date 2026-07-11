import asyncio
import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_core.domain.models import (
    Artifact,
    ArtifactSequence,
    ExecutionMode,
    Experiment,
    ExperimentVariant,
    InputAssemblyTrace,
    InvocationTrace,
    NodeRun,
    NodeSpec,
    OutboxMessage,
    PortSpec,
    WorkflowRun,
)
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)
from notarius_messaging.contracts import ErrorContext, RunEventType, WorkflowRunEvent
from notarius_messaging.outbox import (
    artifact_created_event_outbox_message,
    dlq_node_run_execute_outbox_message,
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)
from notarius_messaging.subjects import (
    NODE_RUN_CANCELLED_EVENT_SUBJECT,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    NODE_RUN_QUEUED_EVENT_SUBJECT,
    WORKFLOW_RUN_CANCELLED_EVENT_SUBJECT,
    WORKFLOW_RUN_QUEUED_EVENT_SUBJECT,
)
from notarius_storage import LocalArtifactPayloadStorage


PAGE_INPUT = PortSpec(
    name="pages",
    artifact_type="source.page_image",
    schema_version=1,
    sequence=True,
)
PAGE_INPUT_PAYLOAD = {
    "name": "pages",
    "artifact_type": "source.page_image",
    "schema_version": 1,
    "sequence": True,
    "required": True,
    "description": None,
}
OCR_OUTPUT = PortSpec(
    name="ocr_pages",
    artifact_type="ocr.page_result",
    schema_version=1,
    sequence=True,
)
DATASET_INPUT = PortSpec(
    name="records",
    artifact_type="ocr.page_result",
    schema_version=1,
    sequence=True,
)
DATASET_OUTPUT = PortSpec(
    name="dataset",
    artifact_type="export.dataset",
    schema_version=1,
)

OCR_SPEC = NodeSpec(
    id="test.ocr",
    version="1.0.0",
    execution_mode=ExecutionMode.MAP,
    inputs=(PAGE_INPUT,),
    outputs=(OCR_OUTPUT,),
)
EXPORT_SPEC = NodeSpec(
    id="test.export",
    version="1.0.0",
    execution_mode=ExecutionMode.REDUCE,
    inputs=(DATASET_INPUT,),
    outputs=(DATASET_OUTPUT,),
)
CONFIG_SPEC = NodeSpec(
    id="test.config",
    version="1.0.0",
    execution_mode=ExecutionMode.SINGLE,
    inputs=(),
    outputs=(),
)
VALID_NODE_SPEC_REGISTRY = {
    (OCR_SPEC.id, OCR_SPEC.version): OCR_SPEC,
    (EXPORT_SPEC.id, EXPORT_SPEC.version): EXPORT_SPEC,
    (CONFIG_SPEC.id, CONFIG_SPEC.version): CONFIG_SPEC,
}


def _artifact_ref_payload(
    artifact_type: str,
    content_hash: str,
) -> dict[str, object]:
    return {
        "artifact_id": str(uuid4()),
        "artifact_type": artifact_type,
        "schema_version": 1,
        "content_hash": content_hash,
    }


@pytest.fixture
def platform_client(tmp_path) -> Iterator[TestClient]:
    store = InMemoryDataStore()
    storage = LocalArtifactPayloadStorage(tmp_path / "artifacts")
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    app.dependency_overrides[api_deps.create_uow_factory] = lambda: (
        lambda: InMemoryUnitOfWork(store)
    )
    app.dependency_overrides[api_deps.get_artifact_payload_storage] = lambda: storage
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)
        app.dependency_overrides.pop(api_deps.create_uow_factory, None)
        app.dependency_overrides.pop(api_deps.get_artifact_payload_storage, None)
        app.dependency_overrides.pop(_node_spec_registry_dependency(), None)


def test_platform_workflow_run_node_run_and_artifact_routes() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        definition_response = client.post(
            "/v1/workflows",
            json={
                "name": "Compare OCR",
                "description": "Compare OCR providers before extraction",
                "nodes": [],
                "declared_inputs": [],
            },
        )

        assert definition_response.status_code == 201
        definition = definition_response.json()
        assert definition["name"] == "Compare OCR"

        version = client.post(
            f"/v1/workflows/{definition['id']}/versions",
            json={"change_note": "Initial OCR comparison graph"},
        ).json()
        run = client.post(
            "/v1/workflow-runs",
            json={"workflow_version_id": version["id"]},
        ).json()
        node_run = client.post(
            f"/v1/workflow-runs/{run['id']}/node-runs",
            json={
                "workflow_node_id": "ocr_a",
                "operator_id": "ocr.mistral",
                "operator_version": "1.0.0",
            },
        ).json()
        artifact = client.post(
            "/v1/artifacts",
            json={
                "artifact_type": "ocr.page_result",
                "schema_version": 1,
                "workflow_run_id": run["id"],
                "producer_node_run_id": node_run["id"],
                "producer_operator_id": "ocr.mistral",
                "producer_operator_version": "1.0.0",
                "payload_ref": "s3://notarius/runs/one/ocr/page-1.json",
                "content_hash": "abc123",
            },
        ).json()

        listed_versions = client.get(
            f"/v1/workflows/{definition['id']}/versions"
        ).json()
        listed_runs = client.get(f"/v1/workflow-versions/{version['id']}/runs").json()
        listed_node_runs = client.get(f"/v1/workflow-runs/{run['id']}/node-runs").json()
        listed_artifacts = client.get(f"/v1/workflow-runs/{run['id']}/artifacts").json()

        assert version["version_number"] == 1
        assert run["status"] == "queued"
        assert node_run["workflow_node_id"] == "ocr_a"
        assert artifact["artifact_type"] == "ocr.page_result"
        assert [item["id"] for item in listed_versions] == [version["id"]]
        assert [item["id"] for item in listed_runs] == [run["id"]]
        assert [item["id"] for item in listed_node_runs] == [node_run["id"]]
        assert [item["id"] for item in listed_artifacts] == [artifact["id"]]
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)


def test_list_node_specs_returns_builtin_registry(platform_client: TestClient) -> None:
    response = platform_client.get("/v1/node-specs")

    assert response.status_code == 200
    specs = response.json()
    assert [spec["id"] for spec in specs] == [
        "context.static.define",
        "debug.emit_text",
        "export.dataset",
        "extraction.contextual_structured",
        "extraction.schema.define",
        "input.policy.define",
        "model.binding.define",
        "ocr.collect_pages",
        "ocr.compare_pages",
        "ocr.extract_page",
        "ocr.extract_pages",
        "ocr.select_pages",
        "prompt.template.define",
        "validation.schema",
    ]
    specs_by_id = {spec["id"]: spec for spec in specs}
    assert specs_by_id["debug.emit_text"]["version"] == "1.0.0"
    assert specs_by_id["debug.emit_text"]["execution_mode"] == "single"
    assert specs_by_id["debug.emit_text"]["outputs"][0]["artifact_type"] == (
        "debug.text"
    )
    assert specs_by_id["debug.emit_text"]["config_schema"]["required"] == ["text"]
    assert specs_by_id["context.static.define"]["outputs"][0]["artifact_type"] == (
        "context.bundle"
    )
    assert specs_by_id["export.dataset"]["inputs"][0]["artifact_type"] == (
        "extraction.document_result"
    )
    assert specs_by_id["export.dataset"]["outputs"][0]["artifact_type"] == (
        "export.dataset"
    )
    assert specs_by_id["extraction.contextual_structured"]["execution_mode"] == (
        "stateful_sequence"
    )
    assert specs_by_id["extraction.contextual_structured"]["inputs"][0][
        "artifact_type"
    ] == "ocr.page_result"
    assert specs_by_id["extraction.contextual_structured"]["inputs"][0][
        "sequence"
    ] is True
    assert specs_by_id["extraction.contextual_structured"]["inputs"][-1][
        "required"
    ] is False
    assert [
        input_spec["artifact_type"]
        for input_spec in specs_by_id["extraction.contextual_structured"]["inputs"]
    ] == [
        "ocr.page_result",
        "extraction.schema",
        "prompt.template",
        "model.binding",
        "input.policy",
        "context.bundle",
        "source.page_image",
    ]
    assert specs_by_id["extraction.contextual_structured"]["outputs"][0][
        "artifact_type"
    ] == "extraction.record_result"
    assert [
        output_spec["artifact_type"]
        for output_spec in specs_by_id["ocr.extract_pages"]["outputs"]
    ] == ["ocr.page_result", "ocr.document_result"]
    assert specs_by_id["ocr.extract_pages"]["outputs"][0]["sequence"] is True
    assert specs_by_id["ocr.extract_pages"]["outputs"][1]["sequence"] is False
    assert specs_by_id["extraction.contextual_structured"]["outputs"][0][
        "sequence"
    ] is True
    assert specs_by_id["prompt.template.define"]["outputs"][0]["artifact_type"] == (
        "prompt.template"
    )
    assert specs_by_id["extraction.schema.define"]["outputs"][0]["artifact_type"] == (
        "extraction.schema"
    )
    assert specs_by_id["model.binding.define"]["outputs"][0]["artifact_type"] == (
        "model.binding"
    )
    assert specs_by_id["input.policy.define"]["outputs"][0]["artifact_type"] == (
        "input.policy"
    )
    assert specs_by_id["ocr.compare_pages"]["execution_mode"] == "reduce"
    assert specs_by_id["ocr.compare_pages"]["inputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.compare_pages"]["inputs"][0]["sequence"] is True
    assert specs_by_id["ocr.compare_pages"]["outputs"][0]["artifact_type"] == (
        "ocr.comparison_result"
    )
    assert specs_by_id["ocr.compare_pages"]["outputs"][0]["sequence"] is True
    assert specs_by_id["ocr.collect_pages"]["execution_mode"] == "reduce"
    assert specs_by_id["ocr.collect_pages"]["inputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.collect_pages"]["outputs"][1]["artifact_type"] == (
        "ocr.document_result"
    )
    assert specs_by_id["ocr.extract_page"]["execution_mode"] == "map"
    assert specs_by_id["ocr.extract_page"]["inputs"][0]["artifact_type"] == (
        "source.page_image"
    )
    assert specs_by_id["ocr.extract_page"]["outputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.extract_pages"]["execution_mode"] == "map"
    assert specs_by_id["ocr.extract_pages"]["inputs"][0]["artifact_type"] == (
        "source.page_image"
    )
    assert specs_by_id["ocr.extract_pages"]["inputs"][0]["sequence"] is True
    assert specs_by_id["ocr.extract_pages"]["outputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.extract_pages"]["outputs"][0]["sequence"] is True
    assert specs_by_id["ocr.select_pages"]["execution_mode"] == "reduce"
    assert specs_by_id["ocr.select_pages"]["inputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.select_pages"]["inputs"][0]["sequence"] is True
    assert specs_by_id["ocr.select_pages"]["inputs"][2]["artifact_type"] == (
        "ocr.comparison_result"
    )
    assert specs_by_id["ocr.select_pages"]["inputs"][2]["required"] is False
    assert specs_by_id["ocr.select_pages"]["outputs"][0]["artifact_type"] == (
        "ocr.page_result"
    )
    assert specs_by_id["ocr.select_pages"]["outputs"][0]["sequence"] is True


def test_list_artifact_types_returns_contracts_from_builtin_node_specs(
    platform_client: TestClient,
) -> None:
    response = platform_client.get("/v1/artifact-types")

    assert response.status_code == 200
    artifact_types = response.json()
    artifact_types_by_key = {
        (artifact_type["artifact_type"], artifact_type["schema_version"]): artifact_type
        for artifact_type in artifact_types
    }
    assert ("ocr.page_result", 1) in artifact_types_by_key
    assert ("ocr.document_result", 1) in artifact_types_by_key
    assert ("export.dataset", 1) in artifact_types_by_key

    ocr_page_contract = artifact_types_by_key[("ocr.page_result", 1)]
    assert ocr_page_contract["sequence"] is True
    assert {
        (port_use["operator_id"], port_use["port_name"], port_use["sequence"])
        for port_use in ocr_page_contract["produced_by"]
    } == {
        ("ocr.collect_pages", "ocr_pages", True),
        ("ocr.extract_page", "ocr_pages", True),
        ("ocr.extract_pages", "ocr_pages", True),
        ("ocr.select_pages", "selected_pages", True),
    }
    assert {
        (port_use["operator_id"], port_use["port_name"], port_use["sequence"])
        for port_use in ocr_page_contract["consumed_by"]
    } == {
        ("extraction.contextual_structured", "text", True),
        ("ocr.collect_pages", "ocr_pages", True),
        ("ocr.compare_pages", "candidate_a_pages", True),
        ("ocr.compare_pages", "candidate_b_pages", True),
        ("ocr.select_pages", "candidate_a_pages", True),
        ("ocr.select_pages", "candidate_b_pages", True),
    }

    ocr_document_contract = artifact_types_by_key[("ocr.document_result", 1)]
    assert ocr_document_contract["sequence"] is False
    assert ocr_document_contract["consumed_by"] == []
    assert [
        (port_use["operator_id"], port_use["port_name"])
        for port_use in ocr_document_contract["produced_by"]
    ] == [
        ("ocr.collect_pages", "ocr_document"),
        ("ocr.extract_pages", "ocr_document"),
    ]

    export_contract = artifact_types_by_key[("export.dataset", 1)]
    assert export_contract["sequence"] is False
    assert export_contract["consumed_by"] == []
    assert [
        (port_use["operator_id"], port_use["port_name"])
        for port_use in export_contract["produced_by"]
    ] == [("export.dataset", "dataset")]


def test_artifact_payload_schema_routes_return_known_json_contracts(
    platform_client: TestClient,
) -> None:
    response = platform_client.get("/v1/artifact-payload-schemas")

    assert response.status_code == 200
    schemas = response.json()
    schemas_by_key = {
        (schema["artifact_type"], schema["schema_version"]): schema
        for schema in schemas
    }
    assert ("ocr.page_result", 1) in schemas_by_key
    assert ("ocr.request_trace", 1) in schemas_by_key
    assert ("extraction.document_result", 1) in schemas_by_key
    assert ("export.dataset", 1) in schemas_by_key
    assert ("validation.result", 1) in schemas_by_key

    ocr_page_schema = schemas_by_key[("ocr.page_result", 1)]
    assert ocr_page_schema["content_type"] == "application/json"
    assert ocr_page_schema["json_schema"]["properties"]["page_number"]["type"] == (
        "integer"
    )
    assert ocr_page_schema["json_schema"]["properties"]["text"]["type"] == "string"
    assert ocr_page_schema["json_schema"]["properties"]["tokens"]["type"] == "array"

    export_schema_response = platform_client.get(
        "/v1/artifact-payload-schemas/export.dataset/1"
    )
    assert export_schema_response.status_code == 200
    export_schema = export_schema_response.json()
    assert export_schema["artifact_type"] == "export.dataset"
    assert export_schema["json_schema"]["properties"]["records"]["type"] == "array"
    validation_schema = schemas_by_key[("validation.result", 1)]
    assert validation_schema["json_schema"]["properties"]["valid"]["type"] == "boolean"
    assert validation_schema["json_schema"]["properties"]["errors"]["type"] == "array"


def test_artifact_payload_schema_route_returns_404_for_unknown_contract(
    platform_client: TestClient,
) -> None:
    response = platform_client.get("/v1/artifact-payload-schemas/missing.type/1")

    assert response.status_code == 404
    assert "ArtifactPayloadSchema not found: missing.type@v1" == response.json()[
        "detail"
    ]


def test_artifact_sequence_routes_and_workflow_run_sequence_input(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry({(OCR_SPEC.id, OCR_SPEC.version): OCR_SPEC})
    seed_version = _create_workflow_version(
        platform_client,
        {"name": "Seed artifacts", "nodes": []},
    )
    seed_run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": seed_version["id"]},
    ).json()
    first_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "workflow_run_id": seed_run["id"],
            "payload_ref": "memory://source/page-1.png",
            "content_hash": "page-one-hash",
        },
    ).json()
    second_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "workflow_run_id": seed_run["id"],
            "payload_ref": "memory://source/page-2.png",
            "content_hash": "page-two-hash",
        },
    ).json()
    item_refs = [
        {
            "artifact_id": first_artifact["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": "page-one-hash",
        },
        {
            "artifact_id": second_artifact["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": "page-two-hash",
        },
    ]

    sequence_response = platform_client.post(
        "/v1/artifact-sequences",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "item_refs": item_refs,
            "index_key": "page_number",
            "metadata": {"source_id": "unit-test"},
        },
    )

    assert sequence_response.status_code == 201
    sequence = sequence_response.json()
    assert sequence["item_refs"] == item_refs
    assert sequence["index_key"] == "page_number"
    listed_sequences = platform_client.get(
        "/v1/artifact-sequences",
        params={"artifact_type": "source.page_image"},
    ).json()
    fetched_sequence = platform_client.get(
        f"/v1/artifact-sequences/{sequence['id']}"
    ).json()
    assert [item["id"] for item in listed_sequences] == [sequence["id"]]
    assert fetched_sequence == sequence

    workflow_version = _create_workflow_version(
        platform_client,
        {
            "name": "OCR from page sequence",
            "nodes": [
                {
                    "id": "ocr",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                }
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )
    run_response = platform_client.post(
        "/v1/workflow-runs",
        json={
            "workflow_version_id": workflow_version["id"],
            "input_artifact_sequence_refs": [
                {
                    "sequence_id": sequence["id"],
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                }
            ],
        },
    )

    assert run_response.status_code == 201
    run = run_response.json()
    node_runs = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/node-runs"
    ).json()
    assert run["input_artifact_sequence_refs"] == [
        {
            "sequence_id": sequence["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
        }
    ]
    assert node_runs[0]["input_artifact_refs"] == {
        "pages": {
            "sequence_id": sequence["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
        }
    }


def test_workflow_template_launch_creates_version_run_and_node_runs(
    platform_client: TestClient,
) -> None:
    first_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "payload_ref": "memory://source/page-1.png",
        },
    ).json()
    second_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "payload_ref": "memory://source/page-2.png",
        },
    ).json()
    sequence_response = platform_client.post(
        "/v1/artifact-sequences",
        json={
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "item_refs": [
                {
                    "artifact_id": first_artifact["id"],
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                },
                {
                    "artifact_id": second_artifact["id"],
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                },
            ],
        },
    )
    assert sequence_response.status_code == 201
    sequence = sequence_response.json()

    templates_response = platform_client.get("/v1/workflow-templates")
    launch_response = platform_client.post(
        "/v1/workflow-templates/ocr-pages/launch",
        json={
            "name": "Template OCR launch",
            "config": {"ocr": {"engine": "local.text"}},
            "input_artifact_sequence_refs": [
                {
                    "sequence_id": sequence["id"],
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                }
            ],
            "metadata": {"source": "unit-test"},
        },
    )

    assert templates_response.status_code == 200
    assert [template["id"] for template in templates_response.json()] == [
        "ocr-pages",
        "contextual-extraction",
        "ocr-compare-contextual-extraction",
    ]
    assert launch_response.status_code == 201
    launch = launch_response.json()
    workflow = launch["workflow_definition"]
    version = launch["workflow_version"]
    run = launch["workflow_run"]
    assert launch["template"]["id"] == "ocr-pages"
    assert workflow["name"] == "Template OCR launch"
    assert workflow["metadata"] == {
        "source": "unit-test",
        "template_id": "ocr-pages",
        "template_version": "1.0.0",
    }
    assert workflow["nodes"][0]["operator_id"] == "ocr.extract_pages"
    assert workflow["nodes"][0]["config"]["engine"] == "local.text"
    assert version["workflow_definition_id"] == workflow["id"]
    assert version["version_number"] == 1
    assert run["workflow_version_id"] == version["id"]
    assert run["status"] == "queued"
    assert run["input_artifact_sequence_refs"] == [
        {
            "sequence_id": sequence["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
        }
    ]
    assert len(launch["queued_node_run_ids"]) == 1

    node_runs = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/node-runs"
    ).json()
    assert [node_run["id"] for node_run in node_runs] == launch["queued_node_run_ids"]
    assert node_runs[0]["workflow_node_id"] == "ocr"
    assert node_runs[0]["status"] == "queued"
    assert node_runs[0]["input_artifact_refs"] == {
        "pages": {
            "sequence_id": sequence["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
        }
    }


def test_workflow_template_materialize_creates_version_without_run(
    platform_client: TestClient,
) -> None:
    response = platform_client.post(
        "/v1/workflow-templates/ocr-pages/materialize",
        json={
            "name": "Experiment template",
            "config": {"ocr": {"engine": "local.text"}},
            "metadata": {"source": "unit-test"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    workflow = body["workflow_definition"]
    version = body["workflow_version"]
    assert body["template"]["id"] == "ocr-pages"
    assert workflow["name"] == "Experiment template"
    assert workflow["metadata"] == {
        "source": "unit-test",
        "template_id": "ocr-pages",
        "template_version": "1.0.0",
    }
    assert version["workflow_definition_id"] == workflow["id"]
    assert version["version_number"] == 1

    runs_response = platform_client.get(
        f"/v1/workflow-versions/{version['id']}/runs"
    )
    assert runs_response.status_code == 200
    assert runs_response.json() == []


def test_workflow_template_launch_rejects_sensitive_config(
    platform_client: TestClient,
) -> None:
    response = platform_client.post(
        "/v1/workflow-templates/contextual-extraction/launch",
        json={
            "config": {
                "model": {
                    "provider": "openai-compatible",
                    "model": "gpt-test",
                    "parameters": {"api_key": "do-not-store"},
                }
            }
        },
    )

    assert response.status_code == 422
    assert "credential_ref" in response.json()["detail"]


def test_create_json_artifact_stores_payload_and_artifact(
    platform_client: TestClient,
) -> None:
    payload = {
        "name": "Entity schema",
        "json_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    }

    response = platform_client.post(
        "/v1/artifacts/json",
        json={
            "artifact_type": "extraction.schema",
            "schema_version": 1,
            "payload": payload,
            "metadata": {"source": "unit-test"},
        },
    )

    assert response.status_code == 201
    artifact = response.json()
    expected_payload = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    payload_response = platform_client.get(f"/v1/artifacts/{artifact['id']}/payload")

    assert artifact["artifact_type"] == "extraction.schema"
    assert artifact["payload_ref"].startswith("artifact://script-artifacts/")
    assert artifact["content_hash"] == hashlib.sha256(expected_payload).hexdigest()
    assert artifact["metadata"] == {
        "source": "unit-test",
        "content_type": "application/json",
        "byte_size": len(expected_payload),
    }
    assert payload_response.status_code == 200
    assert payload_response.headers["content-type"].startswith("application/json")
    assert payload_response.json() == payload


def test_artifact_inspection_embeds_payload_and_lineage(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {"name": "Artifact inspection workflow", "nodes": []},
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    node_run = platform_client.post(
        f"/v1/workflow-runs/{run['id']}/node-runs",
        json={
            "workflow_node_id": "export",
            "operator_id": "export.dataset",
            "operator_version": "1.0.0",
        },
    ).json()
    input_artifact = platform_client.post(
        "/v1/artifacts/json",
        json={
            "artifact_type": "extraction.document_result",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "payload": {"records": [{"name": "Alpha"}]},
        },
    ).json()
    output_payload = {"records": [{"name": "Alpha", "page": 1}]}
    output_artifact = platform_client.post(
        "/v1/artifacts/json",
        json={
            "artifact_type": "export.dataset",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "producer_node_run_id": node_run["id"],
            "producer_operator_id": "export.dataset",
            "producer_operator_version": "1.0.0",
            "input_artifact_ids": [input_artifact["id"]],
            "payload": output_payload,
        },
    ).json()

    metadata_response = platform_client.get(
        f"/v1/artifacts/{output_artifact['id']}/inspect"
    )
    inspection_response = platform_client.get(
        f"/v1/artifacts/{output_artifact['id']}/inspect",
        params={"include_payload": "true", "include_lineage": "true"},
    )

    assert metadata_response.status_code == 200
    metadata_only = metadata_response.json()
    assert metadata_only["artifact"]["id"] == output_artifact["id"]
    assert metadata_only["payload"] is None
    assert metadata_only["lineage"] is None

    assert inspection_response.status_code == 200
    inspection = inspection_response.json()
    assert inspection["artifact"]["id"] == output_artifact["id"]
    assert inspection["payload"]["content_type"] == "application/json"
    assert inspection["payload"]["json_payload"] == output_payload
    assert inspection["lineage"]["root_artifact"]["id"] == output_artifact["id"]
    assert {artifact["id"] for artifact in inspection["lineage"]["artifacts"]} == {
        input_artifact["id"],
        output_artifact["id"],
    }
    assert _artifact_graph_edge_keys(inspection["lineage"]) == {
        (
            "artifact_input",
            "artifact",
            input_artifact["id"],
            "artifact",
            output_artifact["id"],
            None,
        ),
        (
            "node_output",
            "node_run",
            node_run["id"],
            "artifact",
            output_artifact["id"],
            None,
        ),
    }


def test_workflow_run_outputs_filters_and_embeds_json_payload(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {"name": "Output bundle seed", "nodes": []},
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    payload = {
        "records": [
            {"name": "Alpha", "page_number": 1},
            {"name": "Beta", "page_number": 2},
        ]
    }
    artifact = platform_client.post(
        "/v1/artifacts/json",
        json={
            "workflow_run_id": run["id"],
            "artifact_type": "extraction.document_result",
            "schema_version": 1,
            "payload": payload,
            "metadata": {"source": "unit-test"},
        },
    ).json()
    platform_client.post(
        "/v1/artifacts/json",
        json={
            "workflow_run_id": run["id"],
            "artifact_type": "debug.text",
            "schema_version": 1,
            "payload": {"text": "not selected"},
        },
    )

    response = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/outputs",
        params={
            "artifact_type": "extraction.document_result",
            "include_payloads": "true",
        },
    )

    assert response.status_code == 200
    bundle = response.json()
    assert bundle["workflow_run"]["id"] == run["id"]
    assert bundle["artifact_sequences"] == []
    assert bundle["traces"] == []
    assert len(bundle["artifacts"]) == 1
    output = bundle["artifacts"][0]
    assert output["artifact"]["id"] == artifact["id"]
    assert output["artifact"]["artifact_type"] == "extraction.document_result"
    assert output["payload"]["content_type"] == "application/json"
    assert output["payload"]["byte_size"] == artifact["metadata"]["byte_size"]
    assert output["payload"]["json_payload"] == payload
    assert output["payload"]["text"] is None
    assert output["payload"]["error"] is None


def test_workflow_run_outputs_embeds_jsonl_payload_as_text(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {"name": "JSONL output bundle seed", "nodes": []},
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    payload = b'{"name":"Alpha"}\n{"name":"Beta"}\n'
    artifact = platform_client.post(
        "/v1/artifacts/upload",
        data={
            "workflow_run_id": run["id"],
            "artifact_type": "export.dataset",
            "schema_version": "1",
            "metadata_json": '{"format":"jsonl"}',
        },
        files={
            "file": (
                "records.jsonl",
                payload,
                "application/x-ndjson",
            )
        },
    ).json()

    response = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/outputs",
        params={
            "artifact_type": "export.dataset",
            "include_payloads": "true",
        },
    )

    assert response.status_code == 200
    output = response.json()["artifacts"][0]
    assert output["artifact"]["id"] == artifact["id"]
    assert output["payload"]["content_type"] == "application/x-ndjson"
    assert output["payload"]["byte_size"] == len(payload)
    assert output["payload"]["json_payload"] is None
    assert output["payload"]["text"] == payload.decode("utf-8")
    assert output["payload"]["error"] is None


def test_workflow_run_outputs_reports_payload_load_errors(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {"name": "Missing payload output", "nodes": []},
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "workflow_run_id": run["id"],
            "artifact_type": "ocr.page_result",
            "schema_version": 1,
            "payload_ref": "artifact://missing-bucket/missing-payload.json",
            "metadata": {
                "content_type": "application/json",
                "byte_size": 12,
            },
        },
    ).json()

    response = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/outputs",
        params={
            "artifact_type": "ocr.page_result",
            "include_payloads": "true",
        },
    )

    assert response.status_code == 200
    output = response.json()["artifacts"][0]
    assert output["artifact"]["id"] == artifact["id"]
    assert output["payload"]["content_type"] == "application/json"
    assert output["payload"]["byte_size"] == 12
    assert output["payload"]["json_payload"] is None
    assert "missing-payload.json" in output["payload"]["error"]


def test_workflow_run_outputs_include_matching_artifact_sequences() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    app.dependency_overrides[api_deps.create_uow_factory] = lambda: (
        lambda: InMemoryUnitOfWork(store)
    )
    try:
        client = TestClient(app)
        workflow_run_id = asyncio.run(_seed_sequence_outputs(store))

        all_response = client.get(f"/v1/workflow-runs/{workflow_run_id}/outputs")
        filtered_response = client.get(
            f"/v1/workflow-runs/{workflow_run_id}/outputs",
            params={"artifact_type": "ocr.page_result"},
        )
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)
        app.dependency_overrides.pop(api_deps.create_uow_factory, None)

    assert all_response.status_code == 200
    all_bundle = all_response.json()
    all_sequences = all_bundle["artifact_sequences"]
    assert [sequence["artifact_type"] for sequence in all_sequences] == [
        "ocr.page_result"
    ]
    assert [item["artifact_id"] for item in all_sequences[0]["item_refs"]] == [
        artifact["artifact"]["id"]
        for artifact in all_bundle["artifacts"]
        if artifact["artifact"]["artifact_type"] == "ocr.page_result"
    ]

    assert filtered_response.status_code == 200
    filtered_bundle = filtered_response.json()
    filtered_artifact_types = [
        artifact["artifact"]["artifact_type"]
        for artifact in filtered_bundle["artifacts"]
    ]
    assert filtered_artifact_types == [
        "ocr.page_result",
        "ocr.page_result",
    ]
    assert len(filtered_bundle["artifact_sequences"]) == 1
    assert filtered_bundle["artifact_sequences"][0]["artifact_type"] == (
        "ocr.page_result"
    )


def test_experiment_outputs_include_variant_artifact_sequences() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    app.dependency_overrides[api_deps.create_uow_factory] = lambda: (
        lambda: InMemoryUnitOfWork(store)
    )
    try:
        client = TestClient(app)
        experiment_id = asyncio.run(_seed_experiment_sequence_outputs(store))

        response = client.get(
            f"/v1/experiments/{experiment_id}/outputs",
            params={"artifact_type": "ocr.page_result"},
        )
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)
        app.dependency_overrides.pop(api_deps.create_uow_factory, None)

    assert response.status_code == 200
    body = response.json()
    assert body["experiment"]["id"] == str(experiment_id)
    variant_bundle = body["variants"][0]
    assert variant_bundle["variant_key"] == "variant-0001"
    output_artifact_types = [
        artifact["artifact"]["artifact_type"]
        for artifact in variant_bundle["output_bundle"]["artifacts"]
    ]
    assert output_artifact_types == [
        "ocr.page_result",
        "ocr.page_result",
    ]
    assert len(variant_bundle["output_bundle"]["artifact_sequences"]) == 1
    assert variant_bundle["output_bundle"]["artifact_sequences"][0]["artifact_type"] == (
        "ocr.page_result"
    )


def test_upload_artifact_payload_stores_binary_payload(
    platform_client: TestClient,
) -> None:
    content = b"binary fixture"

    response = platform_client.post(
        "/v1/artifacts/upload",
        data={
            "artifact_type": "source.page_image",
            "schema_version": "1",
            "metadata_json": json.dumps({"page_number": 1}),
        },
        files={
            "file": ("page 1.bin", content, "application/octet-stream"),
        },
    )

    assert response.status_code == 201
    artifact = response.json()
    payload_response = platform_client.get(f"/v1/artifacts/{artifact['id']}/payload")

    assert artifact["artifact_type"] == "source.page_image"
    assert artifact["payload_ref"].startswith("artifact://script-artifacts/")
    assert artifact["content_hash"] == hashlib.sha256(content).hexdigest()
    assert artifact["metadata"] == {
        "page_number": 1,
        "filename": "page-1.bin",
        "content_type": "application/octet-stream",
        "byte_size": len(content),
    }
    assert payload_response.status_code == 200
    assert payload_response.content == content


def test_node_run_trace_routes_return_persisted_traces() -> None:
    store = InMemoryDataStore()
    node_run_id = asyncio.run(_seed_node_run_traces(store))
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        missing_node_run_id = uuid4()

        input_response = client.get(
            f"/v1/node-runs/{node_run_id}/input-assembly-traces"
        )
        invocation_response = client.get(
            f"/v1/node-runs/{node_run_id}/invocation-traces"
        )
        missing_input_response = client.get(
            f"/v1/node-runs/{missing_node_run_id}/input-assembly-traces"
        )
        missing_invocation_response = client.get(
            f"/v1/node-runs/{missing_node_run_id}/invocation-traces"
        )

        assert input_response.status_code == 200
        assert invocation_response.status_code == 200
        assert missing_input_response.status_code == 404
        assert missing_invocation_response.status_code == 404
        assert "NodeRun" in missing_input_response.json()["detail"]
        assert "NodeRun" in missing_invocation_response.json()["detail"]
        input_traces = input_response.json()
        invocation_traces = invocation_response.json()
        assert len(input_traces) == 2
        assert len(invocation_traces) == 2
        assert [trace["metadata"]["order"] for trace in input_traces] == [
            "first",
            "second",
        ]
        assert [trace["metadata"]["order"] for trace in invocation_traces] == [
            "first",
            "second",
        ]
        assert input_traces[0]["node_run_id"] == str(node_run_id)
        assert input_traces[0]["selected_inputs"]["pages"][0]["artifact_type"] == (
            "source.page_image"
        )
        assert input_traces[0]["selected_inputs"]["selected_page"][
            "artifact_type"
        ] == "ocr.page_result"
        assert input_traces[0]["omitted_inputs"] == {"history": "disabled"}
        assert input_traces[0]["policies"] == {"history": "none"}
        assert input_traces[0]["metadata"]["source"] == "unit-test"
        assert invocation_traces[0]["invocation_type"] == "ocr.select_pages"
        assert invocation_traces[0]["input_artifact_refs"][0]["artifact_type"] == (
            "source.page_image"
        )
        assert invocation_traces[0]["output_artifact_refs"][0]["artifact_type"] == (
            "ocr.page_result"
        )
        assert invocation_traces[0]["provider"] == "local"
        assert invocation_traces[0]["model"] == "configured_selection"
        assert invocation_traces[0]["request_ref"] == "artifact://requests/request.json"
        assert invocation_traces[0]["response_ref"] == (
            "artifact://responses/response.json"
        )
        assert invocation_traces[0]["runtime"] == {"page_count": 1}
        assert invocation_traces[0]["metadata"]["output_sequence_id"] == "sequence-1"
        assert invocation_traces[0]["error"] is None
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)


def test_create_workflow_run_rejects_invalid_builtin_node_config(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Invalid debug emit",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {},
                }
            ],
        },
    )

    response = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    )

    assert response.status_code == 422
    assert "node config" in response.json()["detail"]


def test_validate_workflow_definition_returns_execution_order_without_persisting(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry(VALID_NODE_SPEC_REGISTRY)

    response = platform_client.post(
        "/v1/workflows/validate",
        json={
            "name": "OCR export",
            "nodes": [
                {
                    "id": "export",
                    "operator_id": "test.export",
                    "operator_version": "1.0.0",
                },
                {
                    "id": "ocr",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                },
            ],
            "edges": [
                {
                    "from_node_id": "ocr",
                    "from_port": "ocr_pages",
                    "to_node_id": "export",
                    "to_port": "records",
                }
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["node_count"] == 2
    assert body["edge_count"] == 1
    assert body["execution_order"] == ["ocr", "export"]
    execution_plan = body["execution_plan"]
    assert execution_plan["workflow_version_id"]
    assert execution_plan["workflow_run_id"]
    assert execution_plan["execution_order"] == ["ocr", "export"]
    assert len(execution_plan["root_node_run_ids"]) == 1
    assert len(execution_plan["leaf_node_run_ids"]) == 1
    plan_nodes = execution_plan["nodes"]
    assert [
        (node["workflow_node_id"], node["execution_index"], node["execution_mode"])
        for node in plan_nodes
    ] == [
        ("ocr", 0, "map"),
        ("export", 1, "reduce"),
    ]
    assert plan_nodes[0]["root"] is True
    assert plan_nodes[0]["leaf"] is False
    assert plan_nodes[0]["input_ports"] == [PAGE_INPUT_PAYLOAD]
    assert plan_nodes[0]["output_ports"][0]["name"] == "ocr_pages"
    assert plan_nodes[1]["root"] is False
    assert plan_nodes[1]["leaf"] is True
    assert plan_nodes[1]["upstream_workflow_node_ids"] == ["ocr"]
    assert plan_nodes[1]["input_ports"][0]["name"] == "records"
    assert platform_client.get("/v1/workflows").json() == []


def test_validate_workflow_definition_returns_compile_errors_without_persisting(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry({})

    response = platform_client.post(
        "/v1/workflows/validate",
        json={
            "name": "Unknown operator",
            "nodes": [
                {
                    "id": "ocr",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                }
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["node_count"] == 1
    assert body["edge_count"] == 0
    assert body["execution_order"] == []
    assert body["execution_plan"] is None
    assert "Unknown operator" in body["errors"][0]
    assert platform_client.get("/v1/workflows").json() == []


def test_create_workflow_run_compiles_and_persists_node_runs(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry(VALID_NODE_SPEC_REGISTRY)
    version = _create_workflow_version(
        platform_client,
        {
            "name": "OCR export",
            "nodes": [
                {
                    "id": "export",
                    "operator_id": "test.export",
                    "operator_version": "1.0.0",
                },
                {
                    "id": "ocr",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                    "config": {"batch_size": 8},
                },
            ],
            "edges": [
                {
                    "from_node_id": "ocr",
                    "from_port": "ocr_pages",
                    "to_node_id": "export",
                    "to_port": "records",
                }
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )
    page_refs = [
        _artifact_ref_payload("source.page_image", "page-one-hash"),
        _artifact_ref_payload("source.page_image", "page-two-hash"),
    ]

    response = platform_client.post(
        "/v1/workflow-runs",
        json={
            "workflow_version_id": version["id"],
            "input_artifact_refs": page_refs,
        },
    )

    assert response.status_code == 201
    run = response.json()
    listed_runs = platform_client.get(
        f"/v1/workflow-versions/{version['id']}/runs"
    ).json()
    listed_node_runs = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/node-runs"
    ).json()

    assert [listed_run["id"] for listed_run in listed_runs] == [run["id"]]
    assert [node_run["workflow_node_id"] for node_run in listed_node_runs] == [
        "ocr",
        "export",
    ]
    assert [node_run["workflow_run_id"] for node_run in listed_node_runs] == [
        run["id"],
        run["id"],
    ]
    assert listed_node_runs[0]["operator_id"] == "test.ocr"
    assert listed_node_runs[0]["metadata"]["workflow_version_id"] == version["id"]
    assert listed_node_runs[0]["metadata"]["workflow_node_config"] == {"batch_size": 8}
    assert listed_node_runs[0]["metadata"]["execution_index"] == 0
    assert listed_node_runs[0]["metadata"]["execution_mode"] == "map"
    assert listed_node_runs[0]["metadata"]["expected_input_ports"] == [
        PAGE_INPUT_PAYLOAD
    ]
    assert listed_node_runs[0]["input_artifact_refs"] == {"pages": page_refs}
    assert listed_node_runs[1]["operator_id"] == "test.export"
    assert listed_node_runs[1]["metadata"]["execution_index"] == 1
    assert listed_node_runs[1]["metadata"]["execution_mode"] == "reduce"
    assert listed_node_runs[1]["metadata"]["upstream_node_run_ids"] == [
        listed_node_runs[0]["id"]
    ]
    assert listed_node_runs[1]["input_artifact_refs"] == {}

    plan_response = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/execution-plan"
    )
    assert plan_response.status_code == 200
    execution_plan = plan_response.json()
    assert execution_plan["workflow_version_id"] == version["id"]
    assert execution_plan["workflow_run_id"] == run["id"]
    assert execution_plan["execution_order"] == ["ocr", "export"]
    assert execution_plan["root_node_run_ids"] == [listed_node_runs[0]["id"]]
    assert execution_plan["leaf_node_run_ids"] == [listed_node_runs[1]["id"]]
    plan_nodes = execution_plan["nodes"]
    assert plan_nodes[0]["workflow_node_id"] == "ocr"
    assert plan_nodes[0]["root"] is True
    assert plan_nodes[0]["leaf"] is False
    assert plan_nodes[0]["input_artifact_refs"] == {"pages": page_refs}
    assert plan_nodes[0]["input_ports"] == [PAGE_INPUT_PAYLOAD]
    assert plan_nodes[1]["workflow_node_id"] == "export"
    assert plan_nodes[1]["root"] is False
    assert plan_nodes[1]["leaf"] is True
    assert plan_nodes[1]["upstream_node_run_ids"] == [listed_node_runs[0]["id"]]
    assert plan_nodes[1]["upstream_workflow_node_ids"] == ["ocr"]
    assert plan_nodes[1]["input_ports"][0]["artifact_type"] == "ocr.page_result"


def test_execute_next_node_run_processes_builtin_debug_workflow(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Script-driven debug workflow",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "script smoke output"},
                }
            ],
        },
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    node_runs = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/node-runs"
    ).json()

    execution_response = platform_client.post("/v1/node-runs/next/execute")
    empty_execution_response = platform_client.post("/v1/node-runs/next/execute")

    assert execution_response.status_code == 200
    execution = execution_response.json()
    assert execution["requested_node_run_id"] is None
    assert execution["processed_node_run_id"] == node_runs[0]["id"]
    assert execution["error"] is None
    assert execution["node_run"]["status"] == "succeeded"
    assert execution["node_run"]["output_artifact_refs"]["text"]["artifact_type"] == (
        "debug.text"
    )

    assert empty_execution_response.status_code == 200
    assert empty_execution_response.json() == {
        "requested_node_run_id": None,
        "processed_node_run_id": None,
        "node_run": None,
        "error": None,
    }

    completed_run = platform_client.get(f"/v1/workflow-runs/{run['id']}").json()
    artifacts = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/artifacts"
    ).json()
    traces = platform_client.get(
        f"/v1/node-runs/{node_runs[0]['id']}/invocation-traces"
    ).json()

    assert completed_run["status"] == "succeeded"
    assert completed_run["output_artifact_refs"] == [
        execution["node_run"]["output_artifact_refs"]["text"]
    ]
    assert artifacts[0]["artifact_type"] == "debug.text"
    assert artifacts[0]["metadata"] == {"text": "script smoke output"}
    assert traces[0]["invocation_type"] == "debug.emit_text"


def test_workflow_run_artifact_graph_exposes_node_output_edges(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Graph debug workflow",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "graph text"},
                }
            ],
        },
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()

    execution_response = platform_client.post(
        f"/v1/workflow-runs/{run['id']}/execute",
        json={"max_node_runs": 100},
    )
    graph_response = platform_client.get(
        f"/v1/workflow-runs/{run['id']}/artifact-graph"
    )

    assert execution_response.status_code == 200
    assert graph_response.status_code == 200
    graph = graph_response.json()
    artifact = graph["artifacts"][0]
    node_run = graph["node_runs"][0]
    assert graph["workflow_run"]["id"] == run["id"]
    assert graph["root_artifact"] is None
    assert artifact["artifact_type"] == "debug.text"
    assert node_run["workflow_node_id"] == "emit"
    assert _artifact_graph_edge_keys(graph) == {
        (
            "node_output",
            "node_run",
            node_run["id"],
            "artifact",
            artifact["id"],
            "text",
        )
    }


def test_artifact_lineage_exposes_input_artifact_edges(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Manual lineage workflow",
            "nodes": [],
        },
    )
    run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    node_run = platform_client.post(
        f"/v1/workflow-runs/{run['id']}/node-runs",
        json={
            "workflow_node_id": "manual_export",
            "operator_id": "export.dataset",
            "operator_version": "1.0.0",
        },
    ).json()
    input_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "extraction.document_result",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "payload_ref": "memory://input/document.json",
        },
    ).json()
    output_artifact = platform_client.post(
        "/v1/artifacts",
        json={
            "artifact_type": "export.dataset",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "producer_node_run_id": node_run["id"],
            "producer_operator_id": "export.dataset",
            "producer_operator_version": "1.0.0",
            "input_artifact_ids": [input_artifact["id"]],
            "payload_ref": "memory://output/dataset.json",
        },
    ).json()

    lineage_response = platform_client.get(
        f"/v1/artifacts/{output_artifact['id']}/lineage"
    )

    assert lineage_response.status_code == 200
    lineage = lineage_response.json()
    assert lineage["workflow_run"]["id"] == run["id"]
    assert lineage["root_artifact"]["id"] == output_artifact["id"]
    assert {artifact["id"] for artifact in lineage["artifacts"]} == {
        input_artifact["id"],
        output_artifact["id"],
    }
    assert [node["id"] for node in lineage["node_runs"]] == [node_run["id"]]
    assert _artifact_graph_edge_keys(lineage) == {
        (
            "artifact_input",
            "artifact",
            input_artifact["id"],
            "artifact",
            output_artifact["id"],
            None,
        ),
        (
            "node_output",
            "node_run",
            node_run["id"],
            "artifact",
            output_artifact["id"],
            None,
        ),
    }


def test_execute_workflow_run_processes_only_requested_run(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Scoped script execution workflow",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "scoped execution"},
                }
            ],
        },
    )
    first_run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    second_run = platform_client.post(
        "/v1/workflow-runs",
        json={"workflow_version_id": version["id"]},
    ).json()
    first_node_run = platform_client.get(
        f"/v1/workflow-runs/{first_run['id']}/node-runs"
    ).json()[0]
    second_node_run = platform_client.get(
        f"/v1/workflow-runs/{second_run['id']}/node-runs"
    ).json()[0]

    response = platform_client.post(
        f"/v1/workflow-runs/{first_run['id']}/execute",
        json={"max_node_runs": 10},
    )

    assert response.status_code == 200
    execution = response.json()
    updated_first_run = platform_client.get(
        f"/v1/workflow-runs/{first_run['id']}"
    ).json()
    summary = platform_client.get(
        f"/v1/workflow-runs/{first_run['id']}/summary"
    ).json()
    outputs = platform_client.get(
        f"/v1/workflow-runs/{first_run['id']}/outputs",
        params={"artifact_type": "debug.text", "include_traces": "true"},
    ).json()
    updated_second_run = platform_client.get(
        f"/v1/workflow-runs/{second_run['id']}"
    ).json()
    updated_second_node_run = platform_client.get(
        f"/v1/node-runs/{second_node_run['id']}"
    ).json()

    assert execution["workflow_run_id"] == first_run["id"]
    assert execution["workflow_run"]["status"] == "succeeded"
    assert execution["processed_node_run_ids"] == [first_node_run["id"]]
    assert execution["errors"] == []
    assert updated_first_run["status"] == "succeeded"
    assert summary["workflow_run"]["id"] == first_run["id"]
    assert summary["workflow_run"]["status"] == "succeeded"
    assert summary["node_run_status_counts"] == {"succeeded": 1}
    assert summary["artifact_counts"] == {"debug.text": 1}
    assert [node_run["id"] for node_run in summary["node_runs"]] == [
        first_node_run["id"]
    ]
    assert summary["artifacts"][0]["artifact_type"] == "debug.text"
    assert summary["errors"] == []
    assert outputs["workflow_run"]["id"] == first_run["id"]
    assert len(outputs["artifacts"]) == 1
    assert outputs["artifacts"][0]["artifact"]["artifact_type"] == "debug.text"
    assert len(outputs["traces"]) == 1
    assert outputs["traces"][0]["node_run"]["id"] == first_node_run["id"]
    assert outputs["traces"][0]["input_assembly_traces"] == []
    assert outputs["traces"][0]["invocation_traces"][0]["invocation_type"] == (
        "debug.emit_text"
    )
    assert updated_second_run["status"] == "queued"
    assert updated_second_node_run["status"] == "queued"


def test_workflow_run_events_return_normalized_timeline() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    other_workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="extract",
        operator_id="test.extract",
        operator_version="1.0.0",
    )
    artifact = Artifact(
        artifact_type="debug.text",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="artifact://debug/output.json",
    )
    base_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    workflow_message = workflow_run_event_outbox_message(
        workflow_run,
        RunEventType.RUNNING,
    )
    _set_outbox_event_time(workflow_message, "occurred_at", base_time)
    node_message = node_run_event_outbox_message(
        node_run,
        RunEventType.FAILED_PERMANENT,
        ErrorContext(
            operation="execute_node_run",
            error_code="handler_failed",
            error_message="Synthetic failure",
            retryable=False,
        ),
    )
    _set_outbox_event_time(node_message, "occurred_at", base_time.replace(minute=1))
    artifact_message = artifact_created_event_outbox_message(artifact)
    _set_outbox_event_time(
        artifact_message,
        "occurred_at",
        base_time.replace(minute=2),
    )
    artifact_message.mark_published()
    dlq_message = dlq_node_run_execute_outbox_message(
        original_subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        original_message_id=str(uuid4()),
        consumer_name="local-node-run-outbox-drainer",
        failure=ErrorContext(
            operation="drain_node_run_outbox_message",
            error_code="node_run_failed_permanent",
            error_message="Synthetic permanent failure",
            retryable=False,
        ),
        attempt_count=2,
        workflow_run_id=workflow_run.id,
        node_run_id=node_run.id,
    )
    _set_outbox_event_time(dlq_message, "failed_at", base_time.replace(minute=3))
    command_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    other_message = workflow_run_event_outbox_message(
        other_workflow_run,
        RunEventType.RUNNING,
    )
    _set_outbox_event_time(other_message, "occurred_at", base_time.replace(minute=4))
    asyncio.run(
        _seed_workflow_run_events(
            store,
            [workflow_run, other_workflow_run],
            [
                dlq_message,
                artifact_message,
                command_message,
                node_message,
                other_message,
                workflow_message,
            ],
        )
    )
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        response = client.get(f"/v1/workflow-runs/{workflow_run.id}/events")
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert response.status_code == 200
    timeline = response.json()
    assert timeline["workflow_run"]["id"] == str(workflow_run.id)
    events = timeline["events"]
    assert [event["event_kind"] for event in events] == [
        "workflow_run",
        "node_run",
        "artifact",
        "dead_letter",
    ]
    assert [event["event_type"] for event in events] == [
        "running",
        "failed_permanent",
        "created",
        "dead_letter",
    ]
    assert events[1]["node_run_id"] == str(node_run.id)
    assert events[1]["error"]["error_code"] == "handler_failed"
    assert events[2]["artifact_id"] == str(artifact.id)
    assert events[2]["artifact_type"] == "debug.text"
    assert events[2]["outbox_status"] == "published"
    assert events[3]["details"]["consumer_name"] == "local-node-run-outbox-drainer"
    assert events[3]["details"]["attempt_count"] == 2
    assert events[3]["error"]["error_code"] == "node_run_failed_permanent"


def test_workflow_run_events_include_malformed_outbox_payloads() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    base_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    malformed_message = OutboxMessage(
        subject="events.workflow_run.running",
        message_type=WorkflowRunEvent.__name__,
        payload={
            "workflow_run_id": str(workflow_run.id),
            "event_type": "not-a-run-event",
            "occurred_at": base_time.isoformat(),
        },
        created_at=base_time,
    )
    asyncio.run(
        _seed_workflow_run_events(
            store,
            [workflow_run],
            [malformed_message],
        )
    )
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        response = client.get(f"/v1/workflow-runs/{workflow_run.id}/events")
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    event = events[0]
    assert event["event_kind"] == "malformed_outbox"
    assert event["event_type"] == "malformed"
    assert event["workflow_run_id"] == str(workflow_run.id)
    assert event["outbox_message_id"] == str(malformed_message.id)
    assert event["error"]["error_code"] == "malformed_outbox_payload"
    assert event["details"]["payload_keys"] == [
        "event_type",
        "occurred_at",
        "workflow_run_id",
    ]
    assert event["details"]["validation_errors"] != []


def test_retry_node_run_requeues_retryable_failure_through_outbox() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Retryable provider workflow",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "retry target"},
                    }
                ],
            },
        )
        run = client.post(
            "/v1/workflow-runs",
            json={"workflow_version_id": version["id"]},
        ).json()
        node_run = client.get(f"/v1/workflow-runs/{run['id']}/node-runs").json()[0]
        asyncio.run(
            _mark_node_run_failed(
                store,
                UUID(node_run["id"]),
                "Provider rate limit",
                retryable=True,
            )
        )

        response = client.post(f"/v1/node-runs/{node_run['id']}/retry")
        conflict_response = client.post(f"/v1/node-runs/{node_run['id']}/retry")
        pending_messages = asyncio.run(_pending_outbox_messages(store))
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert response.status_code == 200
    body = response.json()
    assert body["workflow_run"]["id"] == run["id"]
    assert body["workflow_run"]["status"] == "queued"
    assert body["workflow_run"]["error"] is None
    assert body["node_run"]["id"] == node_run["id"]
    assert body["node_run"]["status"] == "queued"
    assert body["node_run"]["attempt_count"] == 1
    assert body["node_run"]["error"] is None
    assert [message.subject for message in pending_messages].count(
        WORKFLOW_RUN_QUEUED_EVENT_SUBJECT
    ) == 2
    assert [message.subject for message in pending_messages].count(
        NODE_RUN_QUEUED_EVENT_SUBJECT
    ) == 2
    assert [message.subject for message in pending_messages].count(
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    ) == 2
    retry_outbox_message = next(
        message
        for message in pending_messages
        if str(message.id) == body["outbox_message_id"]
    )
    assert retry_outbox_message.payload["workflow_run_id"] == run["id"]
    assert retry_outbox_message.payload["node_run_id"] == node_run["id"]
    assert conflict_response.status_code == 409
    assert "status is queued" in conflict_response.json()["detail"]


def test_outbox_message_routes_inspect_and_requeue_terminal_failures() -> None:
    store = InMemoryDataStore()
    failed_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )
    failed_message.mark_failed("nats unavailable")
    failed_message.mark_permanently_failed("nats unavailable")
    pending_message = OutboxMessage(
        subject="events.workflow_run.queued",
        message_type="WorkflowRunEvent",
        payload={"workflow_run_id": str(uuid4()), "event_type": "queued"},
    )
    asyncio.run(_add_outbox_messages(store, [failed_message, pending_message]))
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        failed_response = client.get("/v1/outbox-messages?status=failed")
        pending_response = client.get("/v1/outbox-messages")
        filtered_response = client.get(
            "/v1/outbox-messages",
            params={
                "status": "pending",
                "subject_prefix": "events.",
                "limit": 1,
                "offset": 0,
            },
        )
        get_response = client.get(f"/v1/outbox-messages/{failed_message.id}")
        requeue_response = client.post(
            f"/v1/outbox-messages/{failed_message.id}/requeue"
        )
        conflict_response = client.post(
            f"/v1/outbox-messages/{failed_message.id}/requeue"
        )
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert failed_response.status_code == 200
    assert [message["id"] for message in failed_response.json()] == [
        str(failed_message.id)
    ]
    assert pending_response.status_code == 200
    assert [message["id"] for message in pending_response.json()] == [
        str(pending_message.id)
    ]
    assert filtered_response.status_code == 200
    assert [message["id"] for message in filtered_response.json()] == [
        str(pending_message.id)
    ]
    assert get_response.status_code == 200
    assert get_response.json()["status"] == "failed"
    assert get_response.json()["attempts"] == 1
    assert get_response.json()["error"] == "nats unavailable"
    assert requeue_response.status_code == 200
    assert requeue_response.json()["status"] == "pending"
    assert requeue_response.json()["attempts"] == 0
    assert requeue_response.json()["error"] is None
    assert conflict_response.status_code == 409
    assert "status is pending" in conflict_response.json()["detail"]


def test_outbox_cleanup_previews_and_deletes_only_terminal_messages() -> None:
    store = InMemoryDataStore()
    older_than = "2026-02-01T00:00:00+00:00"
    old_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    new_timestamp = datetime(2026, 3, 1, tzinfo=UTC)
    published_message = OutboxMessage(
        subject="events.workflow_run.succeeded",
        message_type="WorkflowRunEvent",
        payload={"workflow_run_id": str(uuid4()), "event_type": "succeeded"},
        created_at=old_timestamp,
    )
    published_message.mark_published()
    failed_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
        created_at=old_timestamp,
    )
    failed_message.mark_permanently_failed("poison message")
    pending_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
        created_at=old_timestamp,
    )
    newer_failed_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
        created_at=new_timestamp,
    )
    newer_failed_message.mark_permanently_failed("recent failure")
    asyncio.run(
        _add_outbox_messages(
            store,
            [
                published_message,
                failed_message,
                pending_message,
                newer_failed_message,
            ],
        )
    )
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        dry_run_response = client.post(
            "/v1/outbox-messages/cleanup",
            json={
                "statuses": ["published", "failed"],
                "older_than": older_than,
                "subject_prefix": "events.",
                "dry_run": True,
            },
        )
        execute_response = client.post(
            "/v1/outbox-messages/cleanup",
            json={
                "statuses": ["failed"],
                "older_than": older_than,
                "message_type": "NodeRunExecuteRequested",
                "dry_run": False,
            },
        )
        unsafe_response = client.post(
            "/v1/outbox-messages/cleanup",
            json={
                "statuses": ["pending"],
                "older_than": older_than,
                "dry_run": False,
            },
        )
        failed_list_response = client.get("/v1/outbox-messages?status=failed")
        pending_list_response = client.get("/v1/outbox-messages?status=pending")
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert dry_run_response.status_code == 200
    dry_run_body = dry_run_response.json()
    assert dry_run_body["dry_run"] is True
    assert dry_run_body["matched_count"] == 1
    assert dry_run_body["deleted_count"] == 0
    assert [message["id"] for message in dry_run_body["messages"]] == [
        str(published_message.id)
    ]
    assert execute_response.status_code == 200
    execute_body = execute_response.json()
    assert execute_body["dry_run"] is False
    assert execute_body["matched_count"] == 1
    assert execute_body["deleted_count"] == 1
    assert [message["id"] for message in execute_body["messages"]] == [
        str(failed_message.id)
    ]
    assert unsafe_response.status_code == 422
    assert "only supports published and failed" in unsafe_response.json()["detail"]
    assert failed_list_response.status_code == 200
    assert [message["id"] for message in failed_list_response.json()] == [
        str(newer_failed_message.id)
    ]
    assert pending_list_response.status_code == 200
    assert [message["id"] for message in pending_list_response.json()] == [
        str(pending_message.id)
    ]


def test_outbox_dlq_summary_groups_by_consumer_error_and_original_subject() -> None:
    store = InMemoryDataStore()
    first_message = dlq_node_run_execute_outbox_message(
        original_subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        original_message_id=str(uuid4()),
        consumer_name="local-node-run-outbox-drainer",
        failure=ErrorContext(
            operation="drain_node_run_outbox_message",
            error_code="node_run_failed_permanent",
            error_message="bad node config",
            retryable=False,
        ),
        attempt_count=1,
        workflow_run_id=uuid4(),
        node_run_id=uuid4(),
    )
    second_message = dlq_node_run_execute_outbox_message(
        original_subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        original_message_id=str(uuid4()),
        consumer_name="local-node-run-outbox-drainer",
        failure=ErrorContext(
            operation="drain_node_run_outbox_message",
            error_code="node_run_failed_permanent",
            error_message="attempts exhausted",
            retryable=False,
        ),
        attempt_count=2,
        workflow_run_id=uuid4(),
        node_run_id=uuid4(),
    )
    third_message = dlq_node_run_execute_outbox_message(
        original_subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        original_message_id=str(uuid4()),
        consumer_name="local-node-run-outbox-drainer",
        failure=ErrorContext(
            operation="drain_node_run_outbox_message",
            error_code="invalid_node_run_execute_requested",
            error_message="missing workflow_run_id",
            retryable=False,
        ),
        attempt_count=1,
    )
    asyncio.run(_add_outbox_messages(store, [first_message, second_message, third_message]))
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)

        summary_response = client.get("/v1/outbox-messages/dlq-summary")
        filtered_response = client.get(
            "/v1/outbox-messages/dlq-summary",
            params={
                "consumer_name": "local-node-run-outbox-drainer",
                "error_code": "node_run_failed_permanent",
                "original_subject": NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
            },
        )
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert summary_response.status_code == 200
    summaries = summary_response.json()
    assert [
        (summary["consumer_name"], summary["error_code"], summary["count"])
        for summary in summaries
    ] == [
        (
            "local-node-run-outbox-drainer",
            "invalid_node_run_execute_requested",
            1,
        ),
        ("local-node-run-outbox-drainer", "node_run_failed_permanent", 2),
    ]
    assert filtered_response.status_code == 200
    assert filtered_response.json()[0]["error_code"] == "node_run_failed_permanent"
    assert filtered_response.json()[0]["original_subject"] == (
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    )
    assert filtered_response.json()[0]["count"] == 2


def test_create_experiment_launches_parameter_grid_variants(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Prompt variants",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "baseline"},
                }
            ],
        },
    )

    response = platform_client.post(
        "/v1/experiments",
        json={
            "name": "Debug prompt matrix",
            "workflow_version_id": version["id"],
            "parameters": [
                {
                    "name": "text",
                    "node_id": "emit",
                    "config_path": ["text"],
                    "values": ["variant A", "variant B"],
                }
            ],
            "metadata": {"metric": "manual_review"},
        },
    )

    assert response.status_code == 201
    experiment = response.json()
    listed_experiments = platform_client.get("/v1/experiments").json()
    version_experiments = platform_client.get(
        f"/v1/workflow-versions/{version['id']}/experiments"
    ).json()
    fetched_experiment = platform_client.get(
        f"/v1/experiments/{experiment['id']}"
    ).json()
    run_ids = experiment["workflow_run_ids"]
    first_node_runs = platform_client.get(
        f"/v1/workflow-runs/{run_ids[0]}/node-runs"
    ).json()
    second_node_runs = platform_client.get(
        f"/v1/workflow-runs/{run_ids[1]}/node-runs"
    ).json()
    runs = platform_client.get(
        f"/v1/workflow-versions/{version['id']}/runs"
    ).json()

    assert experiment["name"] == "Debug prompt matrix"
    assert experiment["status"] == "queued"
    assert experiment["metadata"] == {"metric": "manual_review"}
    assert experiment["variants"] == [
        {
            "id": experiment["variants"][0]["id"],
            "key": "variant-0001",
            "ordinal": 1,
            "parameter_values": {"text": "variant A"},
            "workflow_run_id": run_ids[0],
            "metadata": {},
        },
        {
            "id": experiment["variants"][1]["id"],
            "key": "variant-0002",
            "ordinal": 2,
            "parameter_values": {"text": "variant B"},
            "workflow_run_id": run_ids[1],
            "metadata": {},
        },
    ]
    assert [item["id"] for item in listed_experiments] == [experiment["id"]]
    assert [item["id"] for item in version_experiments] == [experiment["id"]]
    assert fetched_experiment == experiment
    assert first_node_runs[0]["metadata"]["workflow_node_config"] == {
        "text": "variant A"
    }
    assert second_node_runs[0]["metadata"]["workflow_node_config"] == {
        "text": "variant B"
    }
    assert [run["metadata"]["experiment_variant_key"] for run in runs] == [
        "variant-0001",
        "variant-0002",
    ]


def test_create_experiment_accepts_ocr_engine_parameter_preset(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry(VALID_NODE_SPEC_REGISTRY)
    version = _create_workflow_version(
        platform_client,
        {
            "name": "OCR engine variants",
            "nodes": [
                {
                    "id": "ocr",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                    "config": {"engine": "local.text"},
                }
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )
    page_refs = [_artifact_ref_payload("source.page_image", "page-one-hash")]

    response = platform_client.post(
        "/v1/experiments",
        json={
            "name": "OCR engine matrix",
            "workflow_version_id": version["id"],
            "parameter_presets": [
                {
                    "kind": "ocr_engine",
                    "node_id": "ocr",
                    "values": ["local.text", "mistral.ocr"],
                }
            ],
            "input_artifact_refs": page_refs,
        },
    )

    assert response.status_code == 201
    experiment = response.json()
    run_ids = experiment["workflow_run_ids"]
    first_node_runs = platform_client.get(
        f"/v1/workflow-runs/{run_ids[0]}/node-runs"
    ).json()
    second_node_runs = platform_client.get(
        f"/v1/workflow-runs/{run_ids[1]}/node-runs"
    ).json()

    assert experiment["parameters"] == [
        {
            "name": "engine",
            "node_id": "ocr",
            "config_path": ["engine"],
            "values": ["local.text", "mistral.ocr"],
            "description": None,
        }
    ]
    assert experiment["variants"][0]["parameter_values"] == {"engine": "local.text"}
    assert experiment["variants"][1]["parameter_values"] == {"engine": "mistral.ocr"}
    assert first_node_runs[0]["metadata"]["workflow_node_config"] == {
        "engine": "local.text"
    }
    assert second_node_runs[0]["metadata"]["workflow_node_config"] == {
        "engine": "mistral.ocr"
    }
    assert first_node_runs[0]["input_artifact_refs"] == {"pages": page_refs}


def test_create_experiment_accepts_named_parameter_presets_for_parallel_nodes(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry(VALID_NODE_SPEC_REGISTRY)
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Parallel OCR engine variants",
            "nodes": [
                {
                    "id": "ocr_a",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                    "config": {"engine": "local.text"},
                },
                {
                    "id": "ocr_b",
                    "operator_id": "test.ocr",
                    "operator_version": "1.0.0",
                    "config": {"engine": "local.text"},
                },
            ],
            "declared_inputs": [PAGE_INPUT_PAYLOAD],
        },
    )

    response = platform_client.post(
        "/v1/experiments",
        json={
            "name": "Parallel OCR engine matrix",
            "workflow_version_id": version["id"],
            "parameter_presets": [
                {
                    "kind": "ocr_engine",
                    "node_id": "ocr_a",
                    "name": "ocr_a_engine",
                    "values": ["local.text"],
                },
                {
                    "kind": "ocr_engine",
                    "node_id": "ocr_b",
                    "name": "ocr_b_engine",
                    "values": ["mistral.ocr"],
                },
            ],
            "input_artifact_refs": [
                _artifact_ref_payload("source.page_image", "page-one-hash")
            ],
        },
    )

    assert response.status_code == 201
    experiment = response.json()
    node_runs = platform_client.get(
        f"/v1/workflow-runs/{experiment['workflow_run_ids'][0]}/node-runs"
    ).json()
    configs_by_node_id = {
        node_run["workflow_node_id"]: node_run["metadata"]["workflow_node_config"]
        for node_run in node_runs
    }

    assert [parameter["name"] for parameter in experiment["parameters"]] == [
        "ocr_a_engine",
        "ocr_b_engine",
    ]
    assert experiment["variants"][0]["parameter_values"] == {
        "ocr_a_engine": "local.text",
        "ocr_b_engine": "mistral.ocr",
    }
    assert configs_by_node_id["ocr_a"] == {"engine": "local.text"}
    assert configs_by_node_id["ocr_b"] == {"engine": "mistral.ocr"}


def test_create_experiment_accepts_extraction_parameter_presets(
    platform_client: TestClient,
) -> None:
    schema_variant = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    model_parameters_variant = {"temperature": 0, "max_tokens": 250}
    policy_settings_variant = {"window_size": 3}
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Contextual extraction preset variants",
            "nodes": [
                {
                    "id": "prompt",
                    "operator_id": "prompt.template.define",
                    "operator_version": "1.0.0",
                    "config": {
                        "name": "Prompt",
                        "template": "Base prompt",
                    },
                },
                {
                    "id": "schema",
                    "operator_id": "extraction.schema.define",
                    "operator_version": "1.0.0",
                    "config": {
                        "name": "Schema",
                        "json_schema": {"type": "object"},
                    },
                },
                {
                    "id": "model",
                    "operator_id": "model.binding.define",
                    "operator_version": "1.0.0",
                    "config": {
                        "provider": "local",
                        "model": "echo",
                        "parameters": {},
                    },
                },
                {
                    "id": "policy",
                    "operator_id": "input.policy.define",
                    "operator_version": "1.0.0",
                    "config": {
                        "name": "Policy",
                        "policy_type": "stateless",
                        "settings": {},
                    },
                },
            ],
        },
    )

    response = platform_client.post(
        "/v1/experiments",
        json={
            "name": "Extraction preset matrix",
            "workflow_version_id": version["id"],
            "parameter_presets": [
                {
                    "kind": "model_provider",
                    "node_id": "model",
                    "values": ["openai-compatible"],
                },
                {
                    "kind": "model_name",
                    "node_id": "model",
                    "values": ["gpt-4.1-mini"],
                },
                {
                    "kind": "model_parameters",
                    "node_id": "model",
                    "values": [model_parameters_variant],
                },
                {
                    "kind": "prompt_template",
                    "node_id": "prompt",
                    "values": ["Extract {{ TEXT }}"],
                },
                {
                    "kind": "extraction_schema",
                    "node_id": "schema",
                    "values": [schema_variant],
                },
                {
                    "kind": "input_policy_type",
                    "node_id": "policy",
                    "values": ["sliding_window"],
                },
                {
                    "kind": "input_policy_settings",
                    "node_id": "policy",
                    "values": [policy_settings_variant],
                },
            ],
        },
    )

    assert response.status_code == 201
    experiment = response.json()
    node_runs = platform_client.get(
        f"/v1/workflow-runs/{experiment['workflow_run_ids'][0]}/node-runs"
    ).json()
    configs_by_node_id = {
        node_run["workflow_node_id"]: node_run["metadata"]["workflow_node_config"]
        for node_run in node_runs
    }

    assert [parameter["name"] for parameter in experiment["parameters"]] == [
        "provider",
        "model",
        "parameters",
        "template",
        "json_schema",
        "policy_type",
        "settings",
    ]
    assert experiment["variants"][0]["parameter_values"] == {
        "provider": "openai-compatible",
        "model": "gpt-4.1-mini",
        "parameters": model_parameters_variant,
        "template": "Extract {{ TEXT }}",
        "json_schema": schema_variant,
        "policy_type": "sliding_window",
        "settings": policy_settings_variant,
    }
    assert configs_by_node_id["model"] == {
        "provider": "openai-compatible",
        "model": "gpt-4.1-mini",
        "parameters": model_parameters_variant,
    }
    assert configs_by_node_id["prompt"]["template"] == "Extract {{ TEXT }}"
    assert configs_by_node_id["schema"]["json_schema"] == schema_variant
    assert configs_by_node_id["policy"] == {
        "name": "Policy",
        "policy_type": "sliding_window",
        "settings": policy_settings_variant,
    }


def test_create_experiment_accepts_workflow_control_parameter_presets(
    platform_client: TestClient,
) -> None:
    _override_node_spec_registry(VALID_NODE_SPEC_REGISTRY)
    engine_config_variant = {"deskew": True, "timeout_seconds": 30}
    language_hints_variant = ["pl", "la"]
    static_context_variant = {"corpus": "schematism", "period": "1900s"}
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Workflow control preset variants",
            "nodes": [
                {
                    "id": "ocr",
                    "operator_id": "test.config",
                    "operator_version": "1.0.0",
                    "config": {
                        "engine": "local.text",
                        "language_hints": [],
                        "engine_config": {},
                    },
                },
                {
                    "id": "compare",
                    "operator_id": "test.config",
                    "operator_version": "1.0.0",
                    "config": {
                        "candidate_a_label": "candidate_a",
                        "candidate_b_label": "candidate_b",
                    },
                },
                {
                    "id": "select",
                    "operator_id": "test.config",
                    "operator_version": "1.0.0",
                    "config": {
                        "selected_candidate": "candidate_a",
                        "decision_note": "baseline",
                    },
                },
                {
                    "id": "context",
                    "operator_id": "test.config",
                    "operator_version": "1.0.0",
                    "config": {"context": {}},
                },
                {
                    "id": "export",
                    "operator_id": "test.config",
                    "operator_version": "1.0.0",
                    "config": {"format": "json", "filename": "baseline.json"},
                },
            ],
        },
    )

    response = platform_client.post(
        "/v1/experiments",
        json={
            "name": "Workflow control preset matrix",
            "workflow_version_id": version["id"],
            "parameter_presets": [
                {
                    "kind": "ocr_engine_config",
                    "node_id": "ocr",
                    "values": [engine_config_variant],
                },
                {
                    "kind": "ocr_language_hints",
                    "node_id": "ocr",
                    "values": [language_hints_variant],
                },
                {
                    "kind": "ocr_candidate_a_label",
                    "node_id": "compare",
                    "values": ["baseline OCR"],
                },
                {
                    "kind": "ocr_candidate_b_label",
                    "node_id": "compare",
                    "values": ["provider OCR"],
                },
                {
                    "kind": "ocr_selected_candidate",
                    "node_id": "select",
                    "values": ["candidate_b"],
                },
                {
                    "kind": "ocr_selection_note",
                    "node_id": "select",
                    "values": ["prefer provider OCR"],
                },
                {
                    "kind": "static_context",
                    "node_id": "context",
                    "values": [static_context_variant],
                },
                {
                    "kind": "export_format",
                    "node_id": "export",
                    "values": ["csv"],
                },
                {
                    "kind": "export_filename",
                    "node_id": "export",
                    "values": ["records.csv"],
                },
            ],
        },
    )

    assert response.status_code == 201
    experiment = response.json()
    node_runs = platform_client.get(
        f"/v1/workflow-runs/{experiment['workflow_run_ids'][0]}/node-runs"
    ).json()
    configs_by_node_id = {
        node_run["workflow_node_id"]: node_run["metadata"]["workflow_node_config"]
        for node_run in node_runs
    }

    assert [parameter["name"] for parameter in experiment["parameters"]] == [
        "engine_config",
        "language_hints",
        "candidate_a_label",
        "candidate_b_label",
        "selected_candidate",
        "decision_note",
        "context",
        "format",
        "filename",
    ]
    assert experiment["variants"][0]["parameter_values"] == {
        "engine_config": engine_config_variant,
        "language_hints": language_hints_variant,
        "candidate_a_label": "baseline OCR",
        "candidate_b_label": "provider OCR",
        "selected_candidate": "candidate_b",
        "decision_note": "prefer provider OCR",
        "context": static_context_variant,
        "format": "csv",
        "filename": "records.csv",
    }
    assert configs_by_node_id["ocr"] == {
        "engine": "local.text",
        "language_hints": language_hints_variant,
        "engine_config": engine_config_variant,
    }
    assert configs_by_node_id["compare"] == {
        "candidate_a_label": "baseline OCR",
        "candidate_b_label": "provider OCR",
    }
    assert configs_by_node_id["select"] == {
        "selected_candidate": "candidate_b",
        "decision_note": "prefer provider OCR",
    }
    assert configs_by_node_id["context"] == {"context": static_context_variant}
    assert configs_by_node_id["export"] == {
        "format": "csv",
        "filename": "records.csv",
    }


def test_execute_experiment_processes_all_variant_runs(
    platform_client: TestClient,
) -> None:
    version = _create_workflow_version(
        platform_client,
        {
            "name": "Executable prompt variants",
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "baseline"},
                }
            ],
        },
    )
    experiment = platform_client.post(
        "/v1/experiments",
        json={
            "name": "Executable matrix",
            "workflow_version_id": version["id"],
            "parameters": [
                {
                    "name": "text",
                    "node_id": "emit",
                    "config_path": ["text"],
                    "values": ["variant A", "variant B"],
                }
            ],
        },
    ).json()

    response = platform_client.post(
        f"/v1/experiments/{experiment['id']}/execute",
        json={"max_node_runs_per_variant": 10},
    )

    assert response.status_code == 200
    execution = response.json()
    updated_experiment = platform_client.get(
        f"/v1/experiments/{experiment['id']}"
    ).json()
    comparison = platform_client.get(
        f"/v1/experiments/{experiment['id']}/comparison"
    ).json()
    outputs = platform_client.get(
        f"/v1/experiments/{experiment['id']}/outputs",
        params={"artifact_type": "debug.text", "include_traces": "true"},
    ).json()

    assert execution["experiment"]["status"] == "succeeded"
    assert updated_experiment["status"] == "succeeded"
    assert [variant["variant_key"] for variant in execution["variants"]] == [
        "variant-0001",
        "variant-0002",
    ]
    assert [variant["workflow_run_status"] for variant in comparison["variants"]] == [
        "succeeded",
        "succeeded",
    ]
    for variant in execution["variants"]:
        assert variant["errors"] == []
        assert len(variant["processed_node_run_ids"]) == 1
        assert variant["workflow_run"]["status"] == "succeeded"
    for variant in comparison["variants"]:
        assert variant["node_run_status_counts"] == {"succeeded": 1}
        assert variant["artifact_counts"] == {"debug.text": 1}
    assert outputs["experiment"]["id"] == experiment["id"]
    assert [variant["variant_key"] for variant in outputs["variants"]] == [
        "variant-0001",
        "variant-0002",
    ]
    assert [
        variant["output_bundle"]["artifacts"][0]["artifact"]["metadata"]["text"]
        for variant in outputs["variants"]
    ] == ["variant A", "variant B"]
    assert [
        variant["output_bundle"]["traces"][0]["node_run"]["workflow_run_id"]
        for variant in outputs["variants"]
    ] == [
        variant["workflow_run_id"]
        for variant in outputs["variants"]
    ]
    assert [
        variant["output_bundle"]["traces"][0]["invocation_traces"][0][
            "invocation_type"
        ]
        for variant in outputs["variants"]
    ] == ["debug.emit_text", "debug.emit_text"]
    assert [
        variant["output_bundle"]["artifact_sequences"]
        for variant in outputs["variants"]
    ] == [[], []]


def test_experiment_comparison_aggregates_variant_evidence() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Comparable variants",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "baseline"},
                    }
                ],
            },
        )
        experiment = client.post(
            "/v1/experiments",
            json={
                "name": "Comparison matrix",
                "workflow_version_id": version["id"],
                "parameters": [
                    {
                        "name": "text",
                        "node_id": "emit",
                        "config_path": ["text"],
                        "values": ["strong", "weak"],
                    }
                ],
            },
        ).json()
        asyncio.run(_seed_experiment_comparison_evidence(store, experiment["id"]))

        response = client.get(f"/v1/experiments/{experiment['id']}/comparison")

        assert response.status_code == 200
        comparison = response.json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert comparison["experiment_id"] == experiment["id"]
    assert comparison["workflow_version_id"] == version["id"]
    assert comparison["variant_count"] == 2
    assert comparison["metric_names"] == [
        "ocr_comparison.mean_similarity_ratio",
        "ocr_comparison.page_count",
        "summary.invocation_count",
        "summary.total_cost",
        "summary.total_duration_ms",
        "summary.validation_error_count",
    ]
    assert comparison["variants"][0]["variant_key"] == "variant-0001"
    assert comparison["variants"][0]["parameter_values"] == {"text": "strong"}
    assert comparison["variants"][0]["workflow_run_status"] == "succeeded"
    assert comparison["variants"][0]["node_run_status_counts"] == {"succeeded": 1}
    assert comparison["variants"][0]["artifact_counts"] == {
        "evaluation.metrics": 1,
        "extraction.document_result": 1,
    }
    assert comparison["variants"][0]["invocation_count"] == 2
    assert comparison["variants"][0]["validation_error_count"] == 0
    assert comparison["variants"][0]["total_duration_ms"] == 15.0
    assert comparison["variants"][0]["total_cost"] == pytest.approx(0.13)
    assert comparison["variants"][0]["evaluation_metrics"][0]["metadata"] == {
        "metric_family": "ocr_comparison",
        "page_count": 2,
        "mean_similarity_ratio": 0.95,
    }
    first_metric_values = {
        metric["name"]: metric
        for metric in comparison["variants"][0]["metric_values"]
    }
    assert first_metric_values["summary.invocation_count"]["value"] == 2
    assert first_metric_values["summary.validation_error_count"]["value"] == 0
    assert first_metric_values["summary.total_duration_ms"]["value"] == 15.0
    assert first_metric_values["summary.total_cost"]["value"] == pytest.approx(0.13)
    assert first_metric_values["ocr_comparison.page_count"]["value"] == 2
    assert first_metric_values["ocr_comparison.mean_similarity_ratio"]["value"] == 0.95
    assert (
        first_metric_values["ocr_comparison.mean_similarity_ratio"]["artifact_id"]
        == comparison["variants"][0]["evaluation_metrics"][0]["artifact_id"]
    )
    assert first_metric_values["ocr_comparison.mean_similarity_ratio"]["source"] == (
        "evaluation.metrics"
    )
    assert comparison["variants"][0]["errors"] == []

    assert comparison["variants"][1]["variant_key"] == "variant-0002"
    assert comparison["variants"][1]["parameter_values"] == {"text": "weak"}
    assert comparison["variants"][1]["workflow_run_status"] == "succeeded"
    assert comparison["variants"][1]["validation_error_count"] == 3
    assert comparison["variants"][1]["total_duration_ms"] == 22.0
    assert comparison["variants"][1]["total_cost"] == pytest.approx(0.21)
    assert comparison["variants"][1]["evaluation_metrics"][0]["metadata"] == {
        "metric_family": "ocr_comparison",
        "page_count": 2,
        "mean_similarity_ratio": 0.72,
    }
    second_metric_values = {
        metric["name"]: metric
        for metric in comparison["variants"][1]["metric_values"]
    }
    assert second_metric_values["summary.validation_error_count"]["value"] == 3
    assert second_metric_values["summary.total_duration_ms"]["value"] == 22.0
    assert second_metric_values["summary.total_cost"]["value"] == pytest.approx(0.21)
    assert second_metric_values["ocr_comparison.mean_similarity_ratio"]["value"] == 0.72


def test_cancel_experiment_variant_marks_open_attempt_cancelled() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Cancelable variant",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "queued"},
                    }
                ],
            },
        )
        experiment = client.post(
            "/v1/experiments",
            json={
                "name": "Cancelable experiment",
                "workflow_version_id": version["id"],
            },
        ).json()
        variant = experiment["variants"][0]

        response = client.post(
            f"/v1/experiments/{experiment['id']}/variants/"
            f"{variant['id']}/cancel"
        )

        assert response.status_code == 200
        updated_experiment = response.json()
        run = client.get(f"/v1/workflow-runs/{variant['workflow_run_id']}").json()
        node_runs = client.get(
            f"/v1/workflow-runs/{variant['workflow_run_id']}/node-runs"
        ).json()
        pending_messages = asyncio.run(_pending_outbox_messages(store))
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert updated_experiment["status"] == "cancelled"
    assert updated_experiment["variants"][0]["workflow_run_id"] == (
        variant["workflow_run_id"]
    )
    assert run["status"] == "cancelled"
    assert [node_run["status"] for node_run in node_runs] == ["cancelled"]
    assert [message.subject for message in pending_messages].count(
        WORKFLOW_RUN_CANCELLED_EVENT_SUBJECT
    ) == 1
    assert [message.subject for message in pending_messages].count(
        NODE_RUN_CANCELLED_EVENT_SUBJECT
    ) == 1


def test_cancel_experiment_marks_all_open_attempts_cancelled() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Cancelable experiment",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "baseline"},
                    }
                ],
            },
        )
        experiment = client.post(
            "/v1/experiments",
            json={
                "name": "Cancelable experiment",
                "workflow_version_id": version["id"],
                "parameters": [
                    {
                        "name": "text",
                        "node_id": "emit",
                        "config_path": ["text"],
                        "values": ["first", "second"],
                    }
                ],
            },
        ).json()

        response = client.post(f"/v1/experiments/{experiment['id']}/cancel")

        assert response.status_code == 200
        updated_experiment = response.json()
        runs = [
            client.get(
                f"/v1/workflow-runs/{variant['workflow_run_id']}"
            ).json()
            for variant in updated_experiment["variants"]
        ]
        node_run_lists = [
            client.get(
                f"/v1/workflow-runs/{variant['workflow_run_id']}/node-runs"
            ).json()
            for variant in updated_experiment["variants"]
        ]
        pending_messages = asyncio.run(_pending_outbox_messages(store))
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert updated_experiment["status"] == "cancelled"
    assert [run["status"] for run in runs] == ["cancelled", "cancelled"]
    assert [[node_run["status"] for node_run in item] for item in node_run_lists] == [
        ["cancelled"],
        ["cancelled"],
    ]
    assert [message.subject for message in pending_messages].count(
        WORKFLOW_RUN_CANCELLED_EVENT_SUBJECT
    ) == 2
    assert [message.subject for message in pending_messages].count(
        NODE_RUN_CANCELLED_EVENT_SUBJECT
    ) == 2


def test_rerun_experiment_variant_replaces_failed_current_attempt() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Rerunnable variant",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "baseline"},
                    }
                ],
            },
        )
        experiment = client.post(
            "/v1/experiments",
            json={
                "name": "Rerunnable experiment",
                "workflow_version_id": version["id"],
                "parameters": [
                    {
                        "name": "text",
                        "node_id": "emit",
                        "config_path": ["text"],
                        "values": ["rerun text"],
                    }
                ],
            },
        ).json()
        variant = experiment["variants"][0]
        old_run_id = variant["workflow_run_id"]

        conflict_response = client.post(
            f"/v1/experiments/{experiment['id']}/variants/"
            f"{variant['id']}/rerun"
        )
        asyncio.run(_mark_first_experiment_variant_failed(store, experiment["id"]))

        response = client.post(
            f"/v1/experiments/{experiment['id']}/variants/"
            f"{variant['id']}/rerun"
        )

        assert conflict_response.status_code == 409
        assert response.status_code == 200
        updated_experiment = response.json()
        updated_variant = updated_experiment["variants"][0]
        new_run_id = updated_variant["workflow_run_id"]
        old_run = client.get(f"/v1/workflow-runs/{old_run_id}").json()
        new_run = client.get(f"/v1/workflow-runs/{new_run_id}").json()
        new_node_runs = client.get(
            f"/v1/workflow-runs/{new_run_id}/node-runs"
        ).json()
        comparison = client.get(
            f"/v1/experiments/{experiment['id']}/comparison"
        ).json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert updated_experiment["status"] == "queued"
    assert new_run_id != old_run_id
    assert updated_variant["metadata"] == {
        "previous_workflow_run_ids": [old_run_id],
        "rerun_count": 1,
        "rerun_of_workflow_run_id": old_run_id,
    }
    assert old_run["status"] == "failed_permanent"
    assert new_run["status"] == "queued"
    assert new_run["metadata"]["experiment_variant_id"] == updated_variant["id"]
    assert new_run["metadata"]["experiment_variant_key"] == "variant-0001"
    assert new_run["metadata"]["experiment_parameter_values"] == {
        "text": "rerun text"
    }
    assert new_run["metadata"]["experiment_rerun_of_workflow_run_id"] == old_run_id
    assert [node_run["status"] for node_run in new_node_runs] == ["queued"]
    assert new_node_runs[0]["metadata"]["workflow_node_config"] == {
        "text": "rerun text"
    }
    assert comparison["variants"][0]["workflow_run_id"] == new_run_id
    assert comparison["variants"][0]["workflow_run_status"] == "queued"


def test_rerun_failed_experiment_variants_replaces_only_failed_attempts() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        client = TestClient(app)
        version = _create_workflow_version(
            client,
            {
                "name": "Bulk rerunnable variants",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "baseline"},
                    }
                ],
            },
        )
        experiment = client.post(
            "/v1/experiments",
            json={
                "name": "Bulk rerunnable experiment",
                "workflow_version_id": version["id"],
                "parameters": [
                    {
                        "name": "text",
                        "node_id": "emit",
                        "config_path": ["text"],
                        "values": ["failed text", "queued text"],
                    }
                ],
            },
        ).json()
        first_variant = experiment["variants"][0]
        second_variant = experiment["variants"][1]
        old_first_run_id = first_variant["workflow_run_id"]
        second_run_id = second_variant["workflow_run_id"]

        conflict_response = client.post(
            f"/v1/experiments/{experiment['id']}/rerun-failed"
        )
        asyncio.run(_mark_first_experiment_variant_failed(store, experiment["id"]))

        response = client.post(f"/v1/experiments/{experiment['id']}/rerun-failed")

        assert conflict_response.status_code == 409
        assert response.status_code == 200
        body = response.json()
        updated_experiment = body["experiment"]
        rerun_variants = body["variants"]
        updated_first_variant = updated_experiment["variants"][0]
        updated_second_variant = updated_experiment["variants"][1]
        new_first_run_id = updated_first_variant["workflow_run_id"]
        old_first_run = client.get(f"/v1/workflow-runs/{old_first_run_id}").json()
        new_first_run = client.get(f"/v1/workflow-runs/{new_first_run_id}").json()
        second_run = client.get(f"/v1/workflow-runs/{second_run_id}").json()
        new_first_node_runs = client.get(
            f"/v1/workflow-runs/{new_first_run_id}/node-runs"
        ).json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert updated_experiment["status"] == "queued"
    assert len(rerun_variants) == 1
    assert rerun_variants[0] == {
        "variant_id": first_variant["id"],
        "variant_key": "variant-0001",
        "previous_workflow_run_id": old_first_run_id,
        "workflow_run_id": new_first_run_id,
    }
    assert new_first_run_id != old_first_run_id
    assert updated_first_variant["metadata"] == {
        "previous_workflow_run_ids": [old_first_run_id],
        "rerun_count": 1,
        "rerun_of_workflow_run_id": old_first_run_id,
    }
    assert updated_second_variant["workflow_run_id"] == second_run_id
    assert updated_second_variant["metadata"] == {}
    assert old_first_run["status"] == "failed_permanent"
    assert new_first_run["status"] == "queued"
    assert second_run["status"] == "queued"
    assert new_first_node_runs[0]["metadata"]["workflow_node_config"] == {
        "text": "failed text"
    }


@pytest.mark.parametrize(
    ("registry", "workflow_payload", "expected_detail"),
    [
        (
            {},
            {
                "name": "Unknown operator",
                "nodes": [
                    {
                        "id": "ocr",
                        "operator_id": "test.ocr",
                        "operator_version": "1.0.0",
                    }
                ],
                "declared_inputs": [PAGE_INPUT_PAYLOAD],
            },
            "Unknown operator",
        ),
        (
            {
                (OCR_SPEC.id, OCR_SPEC.version): NodeSpec(
                    id="test.ocr",
                    version="1.0.0",
                    execution_mode=ExecutionMode.MAP,
                    inputs=(PAGE_INPUT,),
                    outputs=(
                        PortSpec(
                            name="ocr_pages",
                            artifact_type="text.markdown",
                            schema_version=1,
                            sequence=True,
                        ),
                    ),
                ),
                (EXPORT_SPEC.id, EXPORT_SPEC.version): EXPORT_SPEC,
            },
            {
                "name": "Incompatible graph",
                "nodes": [
                    {
                        "id": "ocr",
                        "operator_id": "test.ocr",
                        "operator_version": "1.0.0",
                    },
                    {
                        "id": "export",
                        "operator_id": "test.export",
                        "operator_version": "1.0.0",
                    },
                ],
                "edges": [
                    {
                        "from_node_id": "ocr",
                        "from_port": "ocr_pages",
                        "to_node_id": "export",
                        "to_port": "records",
                    }
                ],
                "declared_inputs": [PAGE_INPUT_PAYLOAD],
            },
            "incompatible artifact contracts",
        ),
    ],
)
def test_create_workflow_run_rejects_graph_compile_errors(
    platform_client: TestClient,
    registry: dict[tuple[str, str], NodeSpec],
    workflow_payload: dict[str, object],
    expected_detail: str,
) -> None:
    _override_node_spec_registry(registry)
    version = _create_workflow_version(platform_client, workflow_payload)

    response = platform_client.post(
        "/v1/workflow-runs",
        json={
            "workflow_version_id": version["id"],
            "input_artifact_refs": [
                _artifact_ref_payload("source.page_image", "page-one-hash")
            ],
        },
    )

    assert response.status_code == 422
    assert expected_detail in response.json()["detail"]

    listed_runs = platform_client.get(
        f"/v1/workflow-versions/{version['id']}/runs"
    ).json()
    assert listed_runs == []


def _create_workflow_version(
    client: TestClient,
    workflow_payload: dict[str, object],
) -> dict[str, object]:
    definition_response = client.post("/v1/workflows", json=workflow_payload)
    assert definition_response.status_code == 201
    definition = definition_response.json()

    version_response = client.post(
        f"/v1/workflows/{definition['id']}/versions",
        json={"change_note": "Compile through API"},
    )
    assert version_response.status_code == 201
    return version_response.json()


def _override_node_spec_registry(registry: dict[tuple[str, str], NodeSpec]) -> None:
    app.dependency_overrides[_node_spec_registry_dependency()] = lambda: registry


def _artifact_graph_edge_keys(
    graph: dict[str, object],
) -> set[tuple[str, str, str, str, str, str | None]]:
    edges = graph["edges"]
    if not isinstance(edges, list):
        raise AssertionError("Artifact graph edges is not a list")
    edge_keys: set[tuple[str, str, str, str, str, str | None]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise AssertionError("Artifact graph edge is not an object")
        port_name = edge["port_name"]
        if port_name is not None and not isinstance(port_name, str):
            raise AssertionError("Artifact graph edge port_name is invalid")
        edge_keys.add(
            (
                str(edge["edge_type"]),
                str(edge["from_kind"]),
                str(edge["from_id"]),
                str(edge["to_kind"]),
                str(edge["to_id"]),
                port_name,
            )
        )
    return edge_keys


async def _mark_node_run_failed(
    store: InMemoryDataStore,
    node_run_id: UUID,
    error: str,
    *,
    retryable: bool,
) -> None:
    async with InMemoryUnitOfWork(store) as uow:
        node_run = await uow.node_runs.get(node_run_id)
        assert node_run is not None
        workflow_run = await uow.workflow_runs.get(node_run.workflow_run_id)
        assert workflow_run is not None

        node_run.mark_running()
        node_run.mark_failed(error, retryable=retryable)
        workflow_run.mark_running()
        workflow_run.mark_failed(error, retryable=retryable)
        await uow.node_runs.update(node_run)
        await uow.workflow_runs.update(workflow_run)
        await uow.commit()


async def _pending_outbox_messages(
    store: InMemoryDataStore,
) -> list[OutboxMessage]:
    async with InMemoryUnitOfWork(store) as uow:
        return await uow.outbox_messages.list_pending()


async def _add_outbox_messages(
    store: InMemoryDataStore,
    messages: list[OutboxMessage],
) -> None:
    async with InMemoryUnitOfWork(store) as uow:
        for message in messages:
            await uow.outbox_messages.add(message)
        await uow.commit()


async def _seed_workflow_run_events(
    store: InMemoryDataStore,
    workflow_runs: list[WorkflowRun],
    messages: list[OutboxMessage],
) -> None:
    async with InMemoryUnitOfWork(store) as uow:
        for workflow_run in workflow_runs:
            await uow.workflow_runs.add(workflow_run)
        for message in messages:
            await uow.outbox_messages.add(message)
        await uow.commit()


def _set_outbox_event_time(
    message: OutboxMessage,
    payload_field: str,
    timestamp: datetime,
) -> None:
    message.created_at = timestamp
    message.payload[payload_field] = timestamp.isoformat()


async def _seed_sequence_outputs(
    store: InMemoryDataStore,
    workflow_version_id: UUID | None = None,
) -> UUID:
    workflow_run = WorkflowRun(workflow_version_id=workflow_version_id or uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.extract_pages",
        operator_version="1.0.0",
    )
    first_artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="artifact://ocr/page-1.json",
        content_hash="page-1",
        metadata={"page_number": 1},
    )
    second_artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="artifact://ocr/page-2.json",
        content_hash="page-2",
        metadata={"page_number": 2},
    )
    debug_artifact = Artifact(
        artifact_type="debug.text",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="artifact://debug/output.json",
    )
    sequence = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_artifact.ref(), second_artifact.ref()],
        index_key="page_number",
        metadata={"page_count": 2},
    )
    node_run.mark_running()
    node_run.mark_succeeded({"ocr_pages": sequence.ref()})
    workflow_run.mark_running()
    workflow_run.mark_succeeded(
        [
            first_artifact.ref(),
            second_artifact.ref(),
            debug_artifact.ref(),
        ]
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.artifacts.add(first_artifact)
        await uow.artifacts.add(second_artifact)
        await uow.artifacts.add(debug_artifact)
        await uow.artifact_sequences.add(sequence)
        await uow.commit()
    return workflow_run.id


async def _seed_experiment_sequence_outputs(store: InMemoryDataStore) -> UUID:
    workflow_version_id = uuid4()
    workflow_run_id = await _seed_sequence_outputs(store, workflow_version_id)
    experiment = Experiment(
        name="Sequence output experiment",
        workflow_version_id=workflow_version_id,
        parameters=[],
        variants=[
            ExperimentVariant(
                key="variant-0001",
                ordinal=1,
                parameter_values={},
                workflow_run_id=workflow_run_id,
            )
        ],
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.experiments.add(experiment)
        await uow.commit()
    return experiment.id


async def _seed_node_run_traces(store: InMemoryDataStore) -> UUID:
    first_created_at = datetime(2026, 1, 1, tzinfo=UTC)
    second_created_at = datetime(2026, 1, 2, tzinfo=UTC)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="select",
        operator_id="ocr.select_pages",
        operator_version="1.0.0",
    )
    input_artifact = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=node_run.workflow_run_id,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/page-1.png",
        content_hash="input-hash",
    )
    output_artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=node_run.workflow_run_id,
        producer_node_run_id=node_run.id,
        payload_ref="artifact://ocr-page-results/page-1.json",
        content_hash="output-hash",
    )
    first_input_trace = InputAssemblyTrace(
        node_run_id=node_run.id,
        selected_inputs={
            "pages": [input_artifact.ref()],
            "selected_page": output_artifact.ref(),
        },
        omitted_inputs={"history": "disabled"},
        policies={"history": "none"},
        metadata={"source": "unit-test", "order": "first"},
        created_at=first_created_at,
    )
    second_input_trace = InputAssemblyTrace(
        node_run_id=node_run.id,
        selected_inputs={"selected_page": output_artifact.ref()},
        omitted_inputs={"comparison": "not-provided"},
        policies={"history": "none"},
        metadata={"source": "unit-test", "order": "second"},
        created_at=second_created_at,
    )
    first_invocation_trace = InvocationTrace(
        node_run_id=node_run.id,
        invocation_type="ocr.select_pages",
        input_artifact_refs=[input_artifact.ref()],
        output_artifact_refs=[output_artifact.ref()],
        provider="local",
        model="configured_selection",
        request_ref="artifact://requests/request.json",
        response_ref="artifact://responses/response.json",
        runtime={"page_count": 1},
        metadata={"output_sequence_id": "sequence-1", "order": "first"},
        created_at=first_created_at,
    )
    second_invocation_trace = InvocationTrace(
        node_run_id=node_run.id,
        invocation_type="ocr.compare_pages",
        input_artifact_refs=[input_artifact.ref()],
        output_artifact_refs=[output_artifact.ref()],
        provider="local",
        model="difflib.SequenceMatcher",
        runtime={"page_count": 1, "mean_similarity_ratio": 1.0},
        metadata={"output_sequence_id": "sequence-2", "order": "second"},
        created_at=second_created_at,
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.node_runs.add(node_run)
        await uow.artifacts.add(input_artifact)
        await uow.artifacts.add(output_artifact)
        await uow.input_assembly_traces.add(second_input_trace)
        await uow.input_assembly_traces.add(first_input_trace)
        await uow.invocation_traces.add(second_invocation_trace)
        await uow.invocation_traces.add(first_invocation_trace)
        await uow.commit()

    return node_run.id


async def _seed_experiment_comparison_evidence(
    store: InMemoryDataStore,
    experiment_id: str,
) -> None:
    async with InMemoryUnitOfWork(store) as uow:
        experiment = await uow.experiments.get(UUID(experiment_id))
        assert isinstance(experiment, Experiment)
        for index, variant in enumerate(experiment.variants):
            run = await uow.workflow_runs.get(variant.workflow_run_id)
            assert run is not None
            node_runs = await uow.node_runs.list_for_workflow_run(run.id)
            assert len(node_runs) == 1
            node_run = node_runs[0]
            if index == 0:
                similarity_ratio = 0.95
                validation_error_count = 0
                trace_specs = [
                    {"duration_ms": 10, "cost_usd": 0.10},
                    {"duration_ms": 5, "cost_usd": 0.03},
                ]
            else:
                similarity_ratio = 0.72
                validation_error_count = 3
                trace_specs = [{"duration_ms": 22, "cost_usd": 0.21}]

            metrics_artifact = Artifact(
                artifact_type="evaluation.metrics",
                schema_version=1,
                workflow_run_id=run.id,
                producer_node_run_id=node_run.id,
                payload_ref=f"artifact://metrics/{variant.key}.json",
                metadata={
                    "metric_family": "ocr_comparison",
                    "page_count": 2,
                    "mean_similarity_ratio": similarity_ratio,
                },
            )
            document_artifact = Artifact(
                artifact_type="extraction.document_result",
                schema_version=1,
                workflow_run_id=run.id,
                producer_node_run_id=node_run.id,
                payload_ref=f"artifact://documents/{variant.key}.json",
                metadata={
                    "page_count": 2,
                    "record_count": 2,
                    "validation_error_count": validation_error_count,
                },
            )
            await uow.artifacts.add(metrics_artifact)
            await uow.artifacts.add(document_artifact)

            for trace_index, trace_spec in enumerate(trace_specs, start=1):
                await uow.invocation_traces.add(
                    InvocationTrace(
                        node_run_id=node_run.id,
                        invocation_type="experiment.synthetic",
                        provider="local",
                        model="comparison",
                        runtime={
                            "duration_ms": trace_spec["duration_ms"],
                            "validation_error_count": validation_error_count,
                        },
                        metadata={"cost_usd": trace_spec["cost_usd"]},
                        created_at=datetime(
                            2026,
                            1,
                            index + 1,
                            12,
                            trace_index,
                            tzinfo=UTC,
                        ),
                    )
                )

            node_run.mark_running()
            node_run.mark_succeeded(
                {
                    "metrics": metrics_artifact.ref(),
                    "document": document_artifact.ref(),
                }
            )
            run.mark_running()
            run.mark_succeeded([metrics_artifact.ref(), document_artifact.ref()])
            await uow.node_runs.update(node_run)
            await uow.workflow_runs.update(run)
        await uow.commit()


async def _mark_first_experiment_variant_failed(
    store: InMemoryDataStore,
    experiment_id: str,
) -> None:
    async with InMemoryUnitOfWork(store) as uow:
        experiment = await uow.experiments.get(UUID(experiment_id))
        assert isinstance(experiment, Experiment)
        variant = experiment.variants[0]
        run = await uow.workflow_runs.get(variant.workflow_run_id)
        assert run is not None
        node_runs = await uow.node_runs.list_for_workflow_run(run.id)
        assert node_runs
        for node_run in node_runs:
            node_run.mark_running()
            node_run.mark_failed("Synthetic variant failure", retryable=False)
            await uow.node_runs.update(node_run)
        run.mark_running()
        run.mark_failed("Synthetic variant failure", retryable=False)
        await uow.workflow_runs.update(run)
        await uow.commit()


def _node_spec_registry_dependency() -> object:
    return getattr(
        api_deps,
        "get_node_spec_registry",
        _missing_node_spec_registry_dependency,
    )


def _missing_node_spec_registry_dependency() -> dict[tuple[str, str], NodeSpec]:
    return {}

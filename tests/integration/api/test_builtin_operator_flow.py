import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_core.domain.models import (
    Artifact,
    ArtifactSequence,
    ArtifactSequenceRef,
    InvocationTrace,
    NodeRun,
    NodeRunStatus,
)
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)
from notarius_storage import LocalArtifactPayloadStorage, parse_artifact_payload_ref
from notarius_worker.node_execution import NodeRunExecutor
from notarius_worker.operators import OcrDocumentResultPayload, builtin_node_handlers


def test_builtin_emit_text_workflow_launches_and_executes_from_api() -> None:
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        with TestClient(app) as client:
            definition_response = client.post(
                "/v1/workflows",
                json={
                    "name": "Debug emit",
                    "nodes": [
                        {
                            "id": "emit",
                            "operator_id": "debug.emit_text",
                            "operator_version": "1.0.0",
                            "config": {"text": "hello from graph"},
                        },
                    ],
                },
            )
            assert definition_response.status_code == 201

            definition = definition_response.json()
            version_response = client.post(
                f"/v1/workflows/{definition['id']}/versions",
                json={"change_note": "Built-in operator smoke flow"},
            )
            assert version_response.status_code == 201
            version = version_response.json()

            run_response = client.post(
                "/v1/workflow-runs",
                json={"workflow_version_id": version["id"]},
            )
            assert run_response.status_code == 201
            run = run_response.json()

            node_runs_response = client.get(
                f"/v1/workflow-runs/{run['id']}/node-runs"
            )
            assert node_runs_response.status_code == 200
            node_runs = node_runs_response.json()

        assert [node_run["workflow_node_id"] for node_run in node_runs] == ["emit"]
        assert node_runs[0]["status"] == "queued"
        processed_id = asyncio.run(_execute_next_node_run(store))
        node_run, artifacts, invocation_traces = asyncio.run(
            _read_node_run_outputs(store, UUID(node_runs[0]["id"]))
        )
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert str(processed_id) == node_runs[0]["id"]
    assert node_run.status == NodeRunStatus.SUCCEEDED
    assert node_run.output_artifact_refs == {"text": artifacts[0].ref()}
    assert artifacts[0].artifact_type == "debug.text"
    assert artifacts[0].metadata == {"text": "hello from graph"}
    assert invocation_traces[0].output_artifact_refs == [artifacts[0].ref()]


def test_builtin_define_nodes_materialize_spec_artifacts_from_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "artifacts"
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(payload_root))
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        with TestClient(app) as client:
            definition_response = client.post(
                "/v1/workflows",
                json={
                    "name": "Define extraction inputs",
                    "nodes": [
                        {
                            "id": "prompt",
                            "operator_id": "prompt.template.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Page prompt",
                                "template": "Extract {{ CURRENT_PAGE_TEXT }}",
                                "variables": ["CURRENT_PAGE_TEXT"],
                            },
                        },
                        {
                            "id": "schema",
                            "operator_id": "extraction.schema.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Record schema",
                                "json_schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                },
                            },
                        },
                        {
                            "id": "model",
                            "operator_id": "model.binding.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "provider": "openai-compatible",
                                "model": "vision-model",
                                "capabilities": ["vision", "structured_output"],
                                "credential_ref": "secret://providers/openai",
                            },
                        },
                        {
                            "id": "policy",
                            "operator_id": "input.policy.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Sliding context",
                                "policy_type": "sliding_window",
                                "settings": {"window_size": 3},
                                "applies_to": ["pages", "text"],
                            },
                        },
                    ],
                },
            )
            assert definition_response.status_code == 201
            definition = definition_response.json()
            version = client.post(
                f"/v1/workflows/{definition['id']}/versions",
                json={"change_note": "Define structured extraction inputs"},
            ).json()
            run_response = client.post(
                "/v1/workflow-runs",
                json={"workflow_version_id": version["id"]},
            )
            assert run_response.status_code == 201
            run = run_response.json()
            node_runs = client.get(
                f"/v1/workflow-runs/{run['id']}/node-runs"
            ).json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    storage = LocalArtifactPayloadStorage(payload_root)
    processed_node_run_ids = [
        asyncio.run(_execute_next_node_run(store, storage)) for _ in range(4)
    ]
    assert all(processed_node_run_ids)

    node_runs_by_workflow_node_id = asyncio.run(
        _read_node_runs_by_workflow_node_id(store, UUID(run["id"]))
    )
    prompt_artifacts = asyncio.run(
        _read_node_artifacts(store, node_runs_by_workflow_node_id["prompt"].id)
    )
    schema_artifacts = asyncio.run(
        _read_node_artifacts(store, node_runs_by_workflow_node_id["schema"].id)
    )
    model_artifacts = asyncio.run(
        _read_node_artifacts(store, node_runs_by_workflow_node_id["model"].id)
    )
    policy_artifacts = asyncio.run(
        _read_node_artifacts(store, node_runs_by_workflow_node_id["policy"].id)
    )

    assert [node_run["status"] for node_run in node_runs] == [
        "queued",
        "queued",
        "queued",
        "queued",
    ]
    assert all(
        node_run.status == NodeRunStatus.SUCCEEDED
        for node_run in node_runs_by_workflow_node_id.values()
    )
    assert prompt_artifacts[0].artifact_type == "prompt.template"
    assert schema_artifacts[0].artifact_type == "extraction.schema"
    assert model_artifacts[0].artifact_type == "model.binding"
    assert policy_artifacts[0].artifact_type == "input.policy"
    assert node_runs_by_workflow_node_id["prompt"].output_artifact_refs == {
        "template": prompt_artifacts[0].ref()
    }
    assert node_runs_by_workflow_node_id["schema"].output_artifact_refs == {
        "schema": schema_artifacts[0].ref()
    }
    assert node_runs_by_workflow_node_id["model"].output_artifact_refs == {
        "binding": model_artifacts[0].ref()
    }
    assert node_runs_by_workflow_node_id["policy"].output_artifact_refs == {
        "policy": policy_artifacts[0].ref()
    }

    prompt_payload = _load_json_payload(storage, prompt_artifacts[0])
    model_payload = _load_json_payload(storage, model_artifacts[0])
    assert prompt_payload["template"] == "Extract {{ CURRENT_PAGE_TEXT }}"
    assert model_payload["credential_ref"] == "secret://providers/openai"
    assert "api_key" not in model_payload


def test_builtin_contextual_extraction_workflow_runs_from_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "artifacts"
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(payload_root))
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        with TestClient(app) as client:
            project = client.post(
                "/v1/projects",
                json={"name": "Contextual extraction"},
            ).json()
            uploaded = client.post(
                f"/v1/projects/{project['id']}/sources/images",
                data={"name": "Two pages"},
                files=[
                    ("files", ("page-1.png", b"Alpha page", "image/png")),
                    ("files", ("page-2.png", b"Beta page", "image/png")),
                ],
            ).json()
            sequence = uploaded["sequence"]

            definition_response = client.post(
                "/v1/workflows",
                json={
                    "name": "OCR then contextual extraction",
                    "nodes": [
                        {
                            "id": "prompt",
                            "operator_id": "prompt.template.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Page prompt",
                                "template": (
                                    "Extract {{ CURRENT_PAGE_TEXT }} "
                                    "after {{ PREVIOUS_RECORD.text if "
                                    "PREVIOUS_RECORD else 'none' }}"
                                ),
                                "variables": [
                                    "CURRENT_PAGE_TEXT",
                                    "PREVIOUS_RECORD",
                                ],
                            },
                        },
                        {
                            "id": "schema",
                            "operator_id": "extraction.schema.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Record schema",
                                "json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "page_number": {"type": "integer"},
                                    },
                                    "required": ["text", "page_number"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        {
                            "id": "model",
                            "operator_id": "model.binding.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "provider": "local",
                                "model": "echo",
                                "capabilities": ["structured_output"],
                            },
                        },
                        {
                            "id": "policy",
                            "operator_id": "input.policy.define",
                            "operator_version": "1.0.0",
                            "config": {
                                "name": "Accumulate context",
                                "policy_type": "accumulating",
                                "applies_to": ["text"],
                            },
                        },
                        {
                            "id": "ocr",
                            "operator_id": "ocr.extract_pages",
                            "operator_version": "1.0.0",
                            "config": {"engine": "local.text"},
                        },
                        {
                            "id": "extract",
                            "operator_id": "extraction.contextual_structured",
                            "operator_version": "1.0.0",
                            "config": {},
                        },
                    ],
                    "edges": [
                        {
                            "from_node_id": "ocr",
                            "from_port": "ocr_pages",
                            "to_node_id": "extract",
                            "to_port": "text",
                        },
                        {
                            "from_node_id": "prompt",
                            "from_port": "template",
                            "to_node_id": "extract",
                            "to_port": "template",
                        },
                        {
                            "from_node_id": "schema",
                            "from_port": "schema",
                            "to_node_id": "extract",
                            "to_port": "schema",
                        },
                        {
                            "from_node_id": "model",
                            "from_port": "binding",
                            "to_node_id": "extract",
                            "to_port": "binding",
                        },
                        {
                            "from_node_id": "policy",
                            "from_port": "policy",
                            "to_node_id": "extract",
                            "to_port": "policy",
                        },
                    ],
                    "declared_inputs": [
                        {
                            "name": "pages",
                            "artifact_type": "source.page_image",
                            "schema_version": 1,
                            "sequence": True,
                        }
                    ],
                },
            )
            assert definition_response.status_code == 201
            definition = definition_response.json()
            version = client.post(
                f"/v1/workflows/{definition['id']}/versions",
                json={"change_note": "Run contextual extraction"},
            ).json()
            run_response = client.post(
                "/v1/workflow-runs",
                json={
                    "workflow_version_id": version["id"],
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
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    storage = LocalArtifactPayloadStorage(payload_root)
    processed_node_run_ids = [
        asyncio.run(_execute_next_node_run(store, storage)) for _ in range(6)
    ]
    assert all(processed_node_run_ids)

    node_runs_by_workflow_node_id = asyncio.run(
        _read_node_runs_by_workflow_node_id(store, UUID(run["id"]))
    )
    extract_run = node_runs_by_workflow_node_id["extract"]
    document_ref = extract_run.output_artifact_refs["document_result"]
    page_results_ref = extract_run.output_artifact_refs["page_results"]
    model_inputs_ref = extract_run.output_artifact_refs["model_inputs"]
    document_artifact = asyncio.run(_read_artifact(store, document_ref.artifact_id))
    page_results_sequence = asyncio.run(_read_output_sequence(store, page_results_ref))
    model_inputs_sequence = asyncio.run(_read_output_sequence(store, model_inputs_ref))
    invocation_traces = asyncio.run(_read_invocation_traces(store, extract_run.id))
    document_payload = _load_json_payload(storage, document_artifact)

    assert all(
        node_run.status == NodeRunStatus.SUCCEEDED
        for node_run in node_runs_by_workflow_node_id.values()
    )
    assert page_results_sequence.artifact_type == "extraction.record_result"
    assert model_inputs_sequence.artifact_type == "model.input"
    assert len(page_results_sequence.item_refs) == 2
    assert len(model_inputs_sequence.item_refs) == 2
    assert document_payload["page_count"] == 2
    assert document_payload["records"] == [
        {"text": "Alpha page", "page_number": 1},
        {"text": "Beta page", "page_number": 2},
    ]
    assert len(invocation_traces) == 2
    assert invocation_traces[0].provider == "local"
    assert invocation_traces[0].model == "echo"


def test_builtin_ocr_workflow_runs_uploaded_page_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "artifacts"
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(payload_root))
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        with TestClient(app) as client:
            project = client.post("/v1/projects", json={"name": "OCR run"}).json()
            uploaded = client.post(
                f"/v1/projects/{project['id']}/sources/images",
                data={"name": "Two pages"},
                files=[
                    ("files", ("page-1.png", b"Alpha page", "image/png")),
                    ("files", ("page-2.png", b"Beta page", "image/png")),
                ],
            ).json()
            sequence = uploaded["sequence"]

            definition_response = client.post(
                "/v1/workflows",
                json={
                    "name": "OCR uploaded pages",
                    "nodes": [
                        {
                            "id": "ocr",
                            "operator_id": "ocr.extract_pages",
                            "operator_version": "1.0.0",
                            "config": {"engine": "local.text"},
                        }
                    ],
                    "declared_inputs": [
                        {
                            "name": "pages",
                            "artifact_type": "source.page_image",
                            "schema_version": 1,
                            "sequence": True,
                        }
                    ],
                },
            )
            assert definition_response.status_code == 201
            definition = definition_response.json()
            version = client.post(
                f"/v1/workflows/{definition['id']}/versions",
                json={"change_note": "Run OCR over uploaded page sequence"},
            ).json()
            run_response = client.post(
                "/v1/workflow-runs",
                json={
                    "workflow_version_id": version["id"],
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
            node_runs = client.get(
                f"/v1/workflow-runs/{run['id']}/node-runs"
            ).json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    storage = LocalArtifactPayloadStorage(payload_root)
    processed_id = asyncio.run(_execute_next_node_run(store, storage))
    node_run, artifacts, sequences, invocation_traces = asyncio.run(
        _read_node_run_ocr_outputs(store, UUID(node_runs[0]["id"]))
    )

    assert str(processed_id) == node_runs[0]["id"]
    assert node_run.status == NodeRunStatus.SUCCEEDED
    page_result_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "ocr.page_result"
    ]
    document_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "ocr.document_result"
    ]
    document_artifact = document_artifacts[0]
    assert node_run.output_artifact_refs == {
        "ocr_pages": sequences[0].ref(),
        "ocr_document": document_artifact.ref(),
    }
    assert [artifact.metadata["page_number"] for artifact in page_result_artifacts] == [
        1,
        2,
    ]
    assert [artifact.artifact_type for artifact in page_result_artifacts] == [
        "ocr.page_result",
        "ocr.page_result",
    ]
    assert sequences[0].artifact_type == "ocr.page_result"
    assert sequences[0].item_refs == [
        artifact.ref() for artifact in page_result_artifacts
    ]
    document_payload = _load_json_payload(storage, document_artifact)
    assert OcrDocumentResultPayload.model_validate(document_payload).text == (
        "Alpha page\n\nBeta page"
    )
    assert invocation_traces[0].model == "local.text"
    assert invocation_traces[0].request_ref is not None
    assert invocation_traces[0].response_ref is not None

    first_payload_location = parse_artifact_payload_ref(
        page_result_artifacts[0].payload_ref
    )
    first_payload = storage.load(
        first_payload_location.bucket,
        first_payload_location.key,
    )
    assert b'"text": "Alpha page"' in first_payload.payload


def test_builtin_ocr_compare_select_workflow_runs_uploaded_page_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "artifacts"
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(payload_root))
    store = InMemoryDataStore()
    app.dependency_overrides[api_deps.create_uow] = lambda: InMemoryUnitOfWork(store)
    try:
        with TestClient(app) as client:
            project = client.post(
                "/v1/projects",
                json={"name": "OCR compare select"},
            ).json()
            uploaded = client.post(
                f"/v1/projects/{project['id']}/sources/images",
                data={"name": "Two pages"},
                files=[
                    ("files", ("page-1.png", b"Alpha page", "image/png")),
                    ("files", ("page-2.png", b"Beta page", "image/png")),
                ],
            ).json()
            sequence = uploaded["sequence"]

            definition_response = client.post(
                "/v1/workflows",
                json={
                    "name": "OCR compare and select",
                    "nodes": [
                        {
                            "id": "ocr_a",
                            "operator_id": "ocr.extract_pages",
                            "operator_version": "1.0.0",
                            "config": {"engine": "local.text"},
                        },
                        {
                            "id": "ocr_b",
                            "operator_id": "ocr.extract_pages",
                            "operator_version": "1.0.0",
                            "config": {"engine": "local.text"},
                        },
                        {
                            "id": "compare",
                            "operator_id": "ocr.compare_pages",
                            "operator_version": "1.0.0",
                            "config": {
                                "candidate_a_label": "A",
                                "candidate_b_label": "B",
                            },
                        },
                        {
                            "id": "select",
                            "operator_id": "ocr.select_pages",
                            "operator_version": "1.0.0",
                            "config": {
                                "selected_candidate": "candidate_b",
                                "decision_note": "Use branch B for downstream extraction",
                            },
                        },
                    ],
                    "edges": [
                        {
                            "from_node_id": "ocr_a",
                            "from_port": "ocr_pages",
                            "to_node_id": "compare",
                            "to_port": "candidate_a_pages",
                        },
                        {
                            "from_node_id": "ocr_b",
                            "from_port": "ocr_pages",
                            "to_node_id": "compare",
                            "to_port": "candidate_b_pages",
                        },
                        {
                            "from_node_id": "ocr_a",
                            "from_port": "ocr_pages",
                            "to_node_id": "select",
                            "to_port": "candidate_a_pages",
                        },
                        {
                            "from_node_id": "ocr_b",
                            "from_port": "ocr_pages",
                            "to_node_id": "select",
                            "to_port": "candidate_b_pages",
                        },
                        {
                            "from_node_id": "compare",
                            "from_port": "comparison_pages",
                            "to_node_id": "select",
                            "to_port": "comparison_pages",
                        },
                    ],
                    "declared_inputs": [
                        {
                            "name": "pages",
                            "artifact_type": "source.page_image",
                            "schema_version": 1,
                            "sequence": True,
                        }
                    ],
                },
            )
            assert definition_response.status_code == 201
            definition = definition_response.json()
            version = client.post(
                f"/v1/workflows/{definition['id']}/versions",
                json={"change_note": "Compare and select OCR branches"},
            ).json()
            run_response = client.post(
                "/v1/workflow-runs",
                json={
                    "workflow_version_id": version["id"],
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
            node_runs = client.get(
                f"/v1/workflow-runs/{run['id']}/node-runs"
            ).json()
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    storage = LocalArtifactPayloadStorage(payload_root)
    processed_node_run_ids = [
        asyncio.run(_execute_next_node_run(store, storage)) for _ in range(4)
    ]
    assert all(processed_node_run_ids)

    node_runs_by_workflow_node_id = asyncio.run(
        _read_node_runs_by_workflow_node_id(store, UUID(run["id"]))
    )
    compare_run = node_runs_by_workflow_node_id["compare"]
    select_run = node_runs_by_workflow_node_id["select"]
    selected_sequence = asyncio.run(
        _read_output_sequence(store, select_run.output_artifact_refs["selected_pages"])
    )
    compare_artifacts = asyncio.run(_read_node_artifacts(store, compare_run.id))

    assert [node_run["status"] for node_run in node_runs] == [
        "queued",
        "queued",
        "blocked",
        "blocked",
    ]
    assert all(
        node_run.status == NodeRunStatus.SUCCEEDED
        for node_run in node_runs_by_workflow_node_id.values()
    )
    assert [artifact.artifact_type for artifact in compare_artifacts] == [
        "ocr.comparison_result",
        "ocr.comparison_result",
        "evaluation.metrics",
    ]
    assert selected_sequence.artifact_type == "ocr.page_result"
    assert selected_sequence.metadata["selected_candidate"] == "candidate_b"
    assert selected_sequence.metadata["page_count"] == 2


async def _execute_next_node_run(
    store: InMemoryDataStore,
    payload_storage: LocalArtifactPayloadStorage | None = None,
) -> UUID | None:
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        builtin_node_handlers(payload_storage),
    )
    return await executor.execute_next_node_run()


async def _read_node_run_outputs(
    store: InMemoryDataStore,
    node_run_id: UUID,
) -> tuple[NodeRun, list[Artifact], list[InvocationTrace]]:
    async with InMemoryUnitOfWork(store) as uow:
        node_run = await uow.node_runs.get(node_run_id)
        artifacts = await uow.artifacts.list_for_node_run(node_run_id)
        invocation_traces = await uow.invocation_traces.list_for_node_run(node_run_id)

    assert node_run is not None
    return node_run, artifacts, invocation_traces


async def _read_node_run_ocr_outputs(
    store: InMemoryDataStore,
    node_run_id: UUID,
) -> tuple[NodeRun, list[Artifact], list[ArtifactSequence], list[InvocationTrace]]:
    async with InMemoryUnitOfWork(store) as uow:
        node_run = await uow.node_runs.get(node_run_id)
        artifacts = await uow.artifacts.list_for_node_run(node_run_id)
        sequences = await uow.artifact_sequences.list_by_artifact_type(
            "ocr.page_result"
        )
        invocation_traces = await uow.invocation_traces.list_for_node_run(node_run_id)

    assert node_run is not None
    return node_run, artifacts, sequences, invocation_traces


async def _read_node_runs_by_workflow_node_id(
    store: InMemoryDataStore,
    workflow_run_id: UUID,
) -> dict[str, NodeRun]:
    async with InMemoryUnitOfWork(store) as uow:
        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)

    return {node_run.workflow_node_id: node_run for node_run in node_runs}


async def _read_output_sequence(
    store: InMemoryDataStore,
    sequence_ref: ArtifactSequenceRef,
) -> ArtifactSequence:
    async with InMemoryUnitOfWork(store) as uow:
        sequence = await uow.artifact_sequences.get(sequence_ref.sequence_id)

    assert sequence is not None
    return sequence


async def _read_node_artifacts(
    store: InMemoryDataStore,
    node_run_id: UUID,
) -> list[Artifact]:
    async with InMemoryUnitOfWork(store) as uow:
        return await uow.artifacts.list_for_node_run(node_run_id)


async def _read_artifact(
    store: InMemoryDataStore,
    artifact_id: UUID,
) -> Artifact:
    async with InMemoryUnitOfWork(store) as uow:
        artifact = await uow.artifacts.get(artifact_id)

    assert artifact is not None
    return artifact


async def _read_invocation_traces(
    store: InMemoryDataStore,
    node_run_id: UUID,
) -> list[InvocationTrace]:
    async with InMemoryUnitOfWork(store) as uow:
        return await uow.invocation_traces.list_for_node_run(node_run_id)


def _load_json_payload(
    storage: LocalArtifactPayloadStorage,
    artifact: Artifact,
) -> dict[str, object]:
    location = parse_artifact_payload_ref(artifact.payload_ref)
    stored = storage.load(location.bucket, location.key)
    return json.loads(stored.payload.decode("utf-8"))

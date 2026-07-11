from pathlib import Path
from types import TracebackType
from typing import Self

import fitz
import pytest
from fastapi.testclient import TestClient

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_core.domain.errors import ValidationError


class RejectingValidator:
    async def validate(self, data: object) -> None:
        raise ValidationError("Injected validator was used")


class FailingUnitOfWork:
    async def __aenter__(self) -> Self:
        raise RuntimeError("database unavailable")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def test_health_live_and_legacy_health_are_ok() -> None:
    client = TestClient(app)

    legacy_response = client.get("/health")
    live_response = client.get("/health/live")

    assert legacy_response.status_code == 200
    assert legacy_response.json() == {
        "status": "ok",
        "database": {"status": "ok", "detail": None},
        "nats": None,
    }
    assert live_response.status_code == 200
    assert live_response.json() == legacy_response.json()


def test_health_ready_checks_database_and_reports_optional_nats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NATS_URL", raising=False)
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": {"status": "ok", "detail": None},
        "nats": {
            "status": "not_configured",
            "detail": "NATS_URL is not set; local outbox draining is available.",
        },
    }


def test_health_ready_returns_unavailable_when_database_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NATS_URL", raising=False)
    app.dependency_overrides[api_deps.create_uow] = lambda: FailingUnitOfWork()
    try:
        client = TestClient(app)

        response = client.get("/health/ready")
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": {
            "status": "error",
            "detail": "RuntimeError: database unavailable",
        },
        "nats": {
            "status": "not_configured",
            "detail": "NATS_URL is not set; local outbox draining is available.",
        },
    }


def test_api_job_lifecycle_routes() -> None:
    client = TestClient(app)

    project = client.post("/v1/projects", json={"name": "Demo"}).json()
    source = client.post(
        f"/v1/projects/{project['id']}/sources",
        json={
            "name": "Source",
            "items": [
                {"order": 1, "text": "first"},
                {"order": 2, "text": "second"},
            ],
        },
    ).json()
    schema = client.post(
        f"/v1/projects/{project['id']}/schemas",
        json={"name": "Schema", "json_schema": {"type": "object"}},
    ).json()
    recipe = client.post(
        f"/v1/projects/{project['id']}/recipes",
        json={"name": "Recipe", "schema_id": schema["id"]},
    ).json()
    job = client.post(
        "/v1/jobs",
        json={
            "project_id": project["id"],
            "source_id": source["id"],
            "recipe_id": recipe["id"],
        },
    ).json()

    items = client.get(f"/v1/jobs/{job['id']}/items").json()
    jobs = client.get(f"/v1/jobs/projects/{project['id']}").json()

    assert job["status"] == "queued"
    assert len(items) == 2
    assert [listed_job["id"] for listed_job in jobs] == [job["id"]]


def test_create_project_uses_injected_validator() -> None:
    app.dependency_overrides[api_deps.get_name_required_validator] = (
        lambda: RejectingValidator()
    )
    try:
        client = TestClient(app)

        response = client.post("/v1/projects", json={"name": "Valid name"})

        assert response.status_code == 422
        assert response.json() == {"detail": "Injected validator was used"}
    finally:
        app.dependency_overrides.pop(api_deps.get_name_required_validator, None)


def test_upload_pdf_source_creates_storage_backed_source_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NOTARIUS_OBJECT_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(tmp_path / "artifacts"))
    client = TestClient(app)
    project = client.post("/v1/projects", json={"name": "PDF workspace"}).json()
    pdf_payload = _pdf_bytes(["Alpha page", "Beta page"])

    response = client.post(
        f"/v1/projects/{project['id']}/sources/pdf",
        data={"name": "Uploaded PDF"},
        files={
            "file": (
                "sample.pdf",
                pdf_payload,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    uploaded = response.json()
    source = uploaded["source"]
    document_artifact = uploaded["document_artifact"]
    artifacts = uploaded["artifacts"]
    sequence = uploaded["sequence"]
    items = client.get(f"/v1/sources/{source['id']}/items").json()

    assert len(items) == 2
    assert document_artifact["artifact_type"] == "source.document"
    assert document_artifact["payload_ref"].startswith("artifact://source-documents/")
    assert document_artifact["metadata"]["filename"] == "sample.pdf"
    assert document_artifact["metadata"]["content_type"] == "application/pdf"
    assert document_artifact["metadata"]["byte_size"] == len(pdf_payload)
    assert items[0]["text"] is None
    assert [item["image_path"] for item in items] == [
        artifact["payload_ref"] for artifact in artifacts
    ]
    assert items[0]["metadata"]["text_object_uri"].startswith(
        "s3://notarius-studio/"
    )
    assert "Alpha page" in items[0]["metadata"]["text_preview"]
    assert items[0]["metadata"]["document_artifact_id"] == document_artifact["id"]
    assert items[0]["metadata"]["artifact_id"] == artifacts[0]["id"]
    assert items[0]["metadata"]["artifact_sequence_id"] == sequence["id"]
    assert [artifact["artifact_type"] for artifact in artifacts] == [
        "source.page_image",
        "source.page_image",
    ]
    assert [artifact["input_artifact_ids"] for artifact in artifacts] == [
        [document_artifact["id"]],
        [document_artifact["id"]],
    ]
    assert [artifact["metadata"]["page_number"] for artifact in artifacts] == [1, 2]
    assert [artifact["metadata"]["content_type"] for artifact in artifacts] == [
        "image/png",
        "image/png",
    ]
    assert sequence["artifact_type"] == "source.page_image"
    assert sequence["index_key"] == "page_number"
    assert sequence["metadata"]["document_artifact_id"] == document_artifact["id"]
    assert sequence["item_refs"] == [
        {
            "artifact_id": artifacts[0]["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": artifacts[0]["content_hash"],
        },
        {
            "artifact_id": artifacts[1]["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": artifacts[1]["content_hash"],
        },
    ]

    source_artifacts_response = client.get(f"/v1/sources/{source['id']}/artifacts")
    source_sequences_response = client.get(
        f"/v1/sources/{source['id']}/artifact-sequences"
    )
    document_payload_response = client.get(
        f"/v1/artifacts/{document_artifact['id']}/payload"
    )
    page_payload_response = client.get(f"/v1/artifacts/{artifacts[0]['id']}/payload")

    assert source_artifacts_response.status_code == 200
    assert source_sequences_response.status_code == 200
    assert [artifact["id"] for artifact in source_artifacts_response.json()] == [
        document_artifact["id"],
        artifacts[0]["id"],
        artifacts[1]["id"],
    ]
    assert source_sequences_response.json() == [sequence]
    assert document_payload_response.status_code == 200
    assert document_payload_response.content == pdf_payload
    assert page_payload_response.status_code == 200
    assert page_payload_response.headers["content-type"] == "image/png"
    assert page_payload_response.content.startswith(b"\x89PNG")


def test_upload_image_source_creates_artifacts_sequence_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(tmp_path / "artifacts"))
    client = TestClient(app)
    project = client.post("/v1/projects", json={"name": "Image workspace"}).json()

    response = client.post(
        f"/v1/projects/{project['id']}/sources/images",
        data={"name": "Four scanned pages"},
        files=[
            ("files", ("page-1.png", b"first-page-bytes", "image/png")),
            ("files", ("page-2.png", b"second-page-bytes", "image/png")),
        ],
    )

    assert response.status_code == 201
    uploaded = response.json()
    source = uploaded["source"]
    items = client.get(f"/v1/sources/{source['id']}/items").json()
    artifacts = uploaded["artifacts"]
    sequence = uploaded["sequence"]

    assert [item["order"] for item in items] == [1, 2]
    assert [item["image_path"] for item in items] == [
        artifact["payload_ref"] for artifact in artifacts
    ]
    assert [artifact["workflow_run_id"] for artifact in artifacts] == [None, None]
    assert [artifact["artifact_type"] for artifact in artifacts] == [
        "source.page_image",
        "source.page_image",
    ]
    assert sequence["artifact_type"] == "source.page_image"
    assert sequence["index_key"] == "page_number"
    assert sequence["item_refs"] == [
        {
            "artifact_id": artifacts[0]["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": artifacts[0]["content_hash"],
        },
        {
            "artifact_id": artifacts[1]["id"],
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "content_hash": artifacts[1]["content_hash"],
        },
    ]

    source_artifacts_response = client.get(f"/v1/sources/{source['id']}/artifacts")
    source_sequences_response = client.get(
        f"/v1/sources/{source['id']}/artifact-sequences"
    )
    payload_response = client.get(f"/v1/artifacts/{artifacts[1]['id']}/payload")

    assert source_artifacts_response.status_code == 200
    assert source_sequences_response.status_code == 200
    assert [artifact["id"] for artifact in source_artifacts_response.json()] == [
        artifacts[0]["id"],
        artifacts[1]["id"],
    ]
    assert source_sequences_response.json() == [sequence]
    assert payload_response.status_code == 200
    assert payload_response.content == b"second-page-bytes"
    assert payload_response.headers["content-type"] == "image/png"


def _pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()

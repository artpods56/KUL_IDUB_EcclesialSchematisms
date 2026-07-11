from pathlib import Path

import anyio
import fitz
import pytest
from fastapi.testclient import TestClient

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_api.messaging import NoOpJobPublisher
from notarius_core.domain.models import JobStatus
from notarius_persistence.unit_of_work import create_sqlite_uow_factory
from notarius_worker.runner import WorkerRunner


def test_api_and_worker_share_sqlite_uow(tmp_path: Path) -> None:
    factory = create_sqlite_uow_factory(f"sqlite:///{tmp_path / 'studio.db'}")
    app.dependency_overrides[api_deps.create_uow] = factory
    app.dependency_overrides[api_deps.get_job_publisher] = lambda: NoOpJobPublisher()
    try:
        client = TestClient(app)
        project = client.post("/v1/projects", json={"name": "SQL"}).json()
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

        runner = WorkerRunner(factory)
        anyio.run(runner.run_job, job["id"])

        processed = client.get(f"/v1/jobs/{job['id']}").json()
        jobs = client.get(f"/v1/jobs/projects/{project['id']}").json()
        items = client.get(f"/v1/jobs/{job['id']}/items").json()

        assert processed["status"] == JobStatus.SUCCEEDED
        assert [listed_job["id"] for listed_job in jobs] == [job["id"]]
        assert [item["structured_output"]["text"] for item in items] == [
            "first",
            "second",
        ]
        assert items[1]["context_trace"]["previous_domain_context"] == {
            "last_source_item_id": items[0]["source_item_id"]
        }
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)
        app.dependency_overrides.pop(api_deps.get_job_publisher, None)


def test_pdf_upload_source_runs_from_object_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NOTARIUS_OBJECT_STORAGE_DIR", str(tmp_path / "objects"))
    factory = create_sqlite_uow_factory(f"sqlite:///{tmp_path / 'studio.db'}")
    app.dependency_overrides[api_deps.create_uow] = factory
    app.dependency_overrides[api_deps.get_job_publisher] = lambda: NoOpJobPublisher()
    try:
        client = TestClient(app)
        project = client.post("/v1/projects", json={"name": "PDF SQL"}).json()
        uploaded = client.post(
            f"/v1/projects/{project['id']}/sources/pdf",
            data={"name": "Uploaded register"},
            files={
                "file": (
                    "register.pdf",
                    _pdf_bytes(["Alpha source page", "Beta source page"]),
                    "application/pdf",
                )
            },
        ).json()
        source = uploaded["source"]
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

        runner = WorkerRunner(factory)
        anyio.run(runner.run_job, job["id"])

        processed = client.get(f"/v1/jobs/{job['id']}").json()
        items = client.get(f"/v1/jobs/{job['id']}/items").json()

        assert processed["status"] == JobStatus.SUCCEEDED
        assert [item["structured_output"]["text"] for item in items] == [
            "Alpha source page",
            "Beta source page",
        ]
        assert items[0]["structured_output"]["metadata"]["loaded_from_storage"] is True
        assert items[1]["context_trace"]["previous_domain_context"] == {
            "last_source_item_id": items[0]["source_item_id"]
        }
    finally:
        app.dependency_overrides.pop(api_deps.create_uow, None)
        app.dependency_overrides.pop(api_deps.get_job_publisher, None)


def _pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()

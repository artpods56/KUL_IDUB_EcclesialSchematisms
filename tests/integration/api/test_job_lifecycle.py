from fastapi.testclient import TestClient

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_core.domain.models import JobStatus
from notarius_persistence.adapters.in_memory import InMemoryUnitOfWork
from notarius_worker.runner import WorkerRunner


def test_api_create_job_worker_process_export_jsonl() -> None:
    client = TestClient(app)
    project = client.post("/v1/projects", json={"name": "Integration"}).json()
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

    runner = WorkerRunner(lambda: InMemoryUnitOfWork(api_deps.get_store()))
    import anyio

    anyio.run(runner.run_job, job["id"])

    processed = client.get(f"/v1/jobs/{job['id']}").json()
    exported = client.get(f"/v1/jobs/{job['id']}/exports/jsonl").text

    assert processed["status"] == JobStatus.SUCCEEDED
    assert "structured_output" in exported
    assert "first" in exported


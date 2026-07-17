from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from notarius_core.artifacts import ArtifactObject, ArtifactRef, InMemoryUnitOfWork
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork
from notarius_storage import LocalFileObjectStore

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.schemas.workbench import RunNodeRequest, RunRequest
from notarius_api.services.invocation_cache import (
    InvocationCacheAccessError,
    PersistentInvocationCache,
)
from notarius_api.services.workbench import WorkbenchService


def _raise_storage_outage(bucket: str, path: str) -> bool:
    raise RuntimeError(f"Storage unavailable for {bucket}/{path}")


def _run_output_artifact_id(
    client: TestClient,
    *,
    node_id: str,
    operator_id: str,
    config: dict[str, object],
) -> str:
    response = client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": node_id,
                    "operator_id": operator_id,
                    "operator_version": 1,
                    "config": config,
                }
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    return str(body["node_runs"][0]["outputs"][0]["value"]["artifact_id"])


def test_exact_builtin_reuses_artifact_for_the_same_invocation(
    builtin_client: TestClient,
) -> None:
    first = _run_output_artifact_id(
        builtin_client,
        node_id="text-source",
        operator_id="text.input",
        config={"text": "stable value"},
    )
    repeated = _run_output_artifact_id(
        builtin_client,
        node_id="text-source",
        operator_id="text.input",
        config={"text": "stable value"},
    )
    changed_config = _run_output_artifact_id(
        builtin_client,
        node_id="text-source",
        operator_id="text.input",
        config={"text": "changed value"},
    )
    changed_node = _run_output_artifact_id(
        builtin_client,
        node_id="other-text-source",
        operator_id="text.input",
        config={"text": "stable value"},
    )

    assert repeated == first
    assert changed_config != first
    assert changed_node != first


def test_external_node_default_policy_does_not_reuse_results(
    structural_projection_client: TestClient,
) -> None:
    first = _run_output_artifact_id(
        structural_projection_client,
        node_id="external",
        operator_id="test.api_response",
        config={},
    )
    repeated = _run_output_artifact_id(
        structural_projection_client,
        node_id="external",
        operator_id="test.api_response",
        config={},
    )

    assert repeated != first


@pytest.mark.asyncio
async def test_cache_evicts_an_entry_with_a_missing_artifact(tmp_path: Path) -> None:
    unit_of_work = InMemoryUnitOfWork()
    missing_artifact = ArtifactObject(
        id=UUID("00000000-0000-0000-0000-000000000301"),
        artifact_type="scalar.text",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": "missing"},
        sha256="a" * 64,
    )
    entry = InvocationCacheEntry(
        key_sha256="b" * 64,
        outputs={"text": missing_artifact.ref()},
    )
    async with unit_of_work as entered:
        assert await entered.invocation_cache.put_if_absent(entry)
        await entered.commit()

    cache = PersistentInvocationCache(
        unit_of_work=unit_of_work,
        storage=LocalFileObjectStore(tmp_path / "objects"),
    )
    assert await cache.get(entry.key_sha256) is None
    async with unit_of_work as entered:
        assert await entered.invocation_cache.get(entry.key_sha256) is None


@pytest.mark.asyncio
async def test_storage_outage_preserves_the_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    artifact = ArtifactObject(
        id=UUID("00000000-0000-0000-0000-000000000311"),
        artifact_type="image.raster",
        schema_version=1,
        content_type="image/png",
        storage_backend="local",
        bucket="artifacts",
        object_key="cached.png",
        byte_size=12,
        sha256="c" * 64,
    )
    entry = InvocationCacheEntry(
        key_sha256="d" * 64,
        outputs={"image": artifact.ref()},
    )
    async with unit_of_work as entered:
        await entered.artifacts.add(artifact)
        assert await entered.invocation_cache.put_if_absent(entry)
        await entered.commit()

    storage = LocalFileObjectStore(tmp_path / "objects")
    monkeypatch.setattr(storage, "exists", _raise_storage_outage)
    cache = PersistentInvocationCache(
        unit_of_work=unit_of_work,
        storage=storage,
    )
    with pytest.raises(InvocationCacheAccessError, match=str(artifact.id)):
        await cache.get(entry.key_sha256)

    async with unit_of_work as entered:
        preserved = await entered.invocation_cache.get(entry.key_sha256)
    assert preserved is not None
    assert preserved.generation == entry.generation


@pytest.mark.asyncio
async def test_sql_cache_survives_a_fresh_workbench_service(tmp_path: Path) -> None:
    database = create_database(
        f"sqlite+aiosqlite:///{tmp_path / 'persistent-cache.sqlite3'}"
    )
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    registry = build_plugin_registry(builtin_plugins(), external_plugins=())
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="persistent-text",
                operator_id="text.input",
                operator_version=1,
                config={"text": "survives restart"},
            )
        ]
    )

    try:
        first_service = WorkbenchService(
            plugin_registry=registry,
            workspace=tmp_path / "workbench",
            uow=SqlAlchemyUnitOfWork(database.sessions),
        )
        first_response = await first_service.run_graph(request)
        first_value = first_response.node_runs[0].outputs[0].value
        assert isinstance(first_value, ArtifactRef)

        fresh_service = WorkbenchService(
            plugin_registry=registry,
            workspace=tmp_path / "workbench",
            uow=SqlAlchemyUnitOfWork(database.sessions),
        )
        repeated_response = await fresh_service.run_graph(request)
        repeated_value = repeated_response.node_runs[0].outputs[0].value
        assert isinstance(repeated_value, ArtifactRef)

        assert repeated_value == first_value
    finally:
        await database.dispose()

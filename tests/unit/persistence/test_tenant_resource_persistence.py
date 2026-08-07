from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from notarius_core.artifacts import ArtifactObject
from notarius_core.domain.execution_history import GraphExecution
from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.node_secrets import EncryptedNodeSecret
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphDocument
from notarius_core.domain.staged_uploads import StagedUpload
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ONE = UUID("00000000-0000-0000-0000-000000000201")
WORKSPACE_TWO = UUID("00000000-0000-0000-0000-000000000202")


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = create_database(f"sqlite+aiosqlite:///{tmp_path / 'tenant.sqlite3'}")
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.execute(
            text(
                "INSERT INTO workspaces "
                "(id, slug, name, kind, created_at, updated_at) VALUES "
                "(:one, 'one', 'One', 'shared', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
                "(:two, 'two', 'Two', 'shared', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"one": WORKSPACE_ONE.hex, "two": WORKSPACE_TWO.hex},
        )
    try:
        yield database
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_sql_repositories_require_workspace_for_artifact_cache_and_upload(
    database: Database,
) -> None:
    artifact = ArtifactObject(
        workspace_id=WORKSPACE_ONE,
        id=UUID("00000000-0000-0000-0000-000000000203"),
        artifact_type="test.value",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": 1},
        metadata={},
    )
    artifact_two = ArtifactObject(
        workspace_id=WORKSPACE_TWO,
        id=UUID("00000000-0000-0000-0000-000000000207"),
        artifact_type="test.value",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": 2},
        metadata={},
    )
    cache_one = InvocationCacheEntry(
        workspace_id=WORKSPACE_ONE,
        key_sha256="c" * 64,
        generation=UUID("00000000-0000-0000-0000-000000000208"),
        outputs={"value": artifact.ref()},
    )
    cache_two = InvocationCacheEntry(
        workspace_id=WORKSPACE_TWO,
        key_sha256=cache_one.key_sha256,
        generation=UUID("00000000-0000-0000-0000-000000000209"),
        outputs={"value": artifact_two.ref()},
    )
    upload = StagedUpload(
        workspace_id=WORKSPACE_ONE,
        upload_key="legacy-upload",
        original_filename="source.csv",
        byte_size=4,
    )
    graph = SavedGraph(
        workspace_id=WORKSPACE_ONE,
        id=UUID("00000000-0000-0000-0000-000000000204"),
        name="Tenant graph",
        document=SavedGraphDocument(),
    )
    revision = graph.snapshot()
    secret = EncryptedNodeSecret(
        workspace_id=WORKSPACE_ONE,
        graph_id=graph.id,
        node_id="node",
        name="secret",
        operator_id="test.operator",
        operator_version=1,
        key_id="key",
        aad_version=2,
        dependency_sha256="d" * 64,
        nonce=b"0" * 12,
        ciphertext=b"ciphertext",
        created_at=graph.created_at,
        updated_at=graph.updated_at,
    )
    materialized = MaterializedNodeOutputs(
        workspace_id=WORKSPACE_ONE,
        graph_id=graph.id,
        graph_revision=graph.revision,
        node_id="node",
        workflow_run_id=UUID("00000000-0000-0000-0000-000000000205"),
        outputs={},
    )
    execution = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=UUID("00000000-0000-0000-0000-000000000206"),
        graph_id=graph.id,
        graph_revision=graph.revision,
        requested_node_ids=("node",),
        status="queued",
    )

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.graphs.add(graph)
        await unit_of_work.graphs.add_revision(revision)
        await unit_of_work.artifacts.add(artifact)
        await unit_of_work.artifacts.add(artifact_two)
        assert await unit_of_work.invocation_cache.put_if_absent(cache_one)
        assert await unit_of_work.invocation_cache.put_if_absent(cache_two)
        await unit_of_work.staged_uploads.add(upload)
        await unit_of_work.node_secrets.upsert(secret)
        await unit_of_work.materialized_outputs.upsert(materialized)
        await unit_of_work.execution_history.add(execution)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.artifacts.get(WORKSPACE_ONE, artifact.id) is not None
        assert await unit_of_work.artifacts.get(WORKSPACE_TWO, artifact.id) is None
        assert await unit_of_work.graphs.get(WORKSPACE_TWO, graph.id) is None
        assert (
            await unit_of_work.graphs.get_revision(
                WORKSPACE_TWO,
                graph.id,
                graph.revision,
            )
            is None
        )
        assert (
            await unit_of_work.node_secrets.get(
                WORKSPACE_TWO,
                graph.id,
                secret.node_id,
                secret.name,
            )
            is None
        )
        assert (
            await unit_of_work.materialized_outputs.get(
                WORKSPACE_TWO,
                graph.id,
                graph.revision,
                materialized.node_id,
            )
            is None
        )
        assert (
            await unit_of_work.execution_history.get(
                WORKSPACE_TWO,
                execution.execution_id,
            )
            is None
        )
        assert (
            await unit_of_work.execution_history.list_for_graph(
                WORKSPACE_TWO,
                graph.id,
                limit=10,
            )
        ).items == ()
        with pytest.raises(NotFoundError):
            await unit_of_work.execution_history.update(
                replace(execution, workspace_id=WORKSPACE_TWO)
            )
        stored_cache_one = await unit_of_work.invocation_cache.get(
            WORKSPACE_ONE,
            cache_one.key_sha256,
        )
        stored_cache_two = await unit_of_work.invocation_cache.get(
            WORKSPACE_TWO,
            cache_two.key_sha256,
        )
        assert stored_cache_one is not None
        assert stored_cache_two is not None
        assert stored_cache_one.generation == cache_one.generation
        assert stored_cache_one.outputs == cache_one.outputs
        assert stored_cache_two.generation == cache_two.generation
        assert stored_cache_two.outputs == cache_two.outputs
        assert stored_cache_one != stored_cache_two
        assert (
            await unit_of_work.staged_uploads.get(
                WORKSPACE_TWO,
                upload.upload_key,
            )
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_key",
    ["", ".", "..", "../escape", r"..\escape", "\x00key", "x" * 1025],
)
async def test_sql_staged_upload_key_constraints_reject_path_like_keys(
    database: Database,
    invalid_key: str,
) -> None:
    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO staged_uploads "
                    "(workspace_id, upload_key, original_filename, byte_size, created_at) "
                    "VALUES (:workspace_id, :upload_key, 'input.csv', 0, CURRENT_TIMESTAMP)"
                ),
                {
                    "workspace_id": WORKSPACE_ONE.hex,
                    "upload_key": invalid_key,
                },
            )

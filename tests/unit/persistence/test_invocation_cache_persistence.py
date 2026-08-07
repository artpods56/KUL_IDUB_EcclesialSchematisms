import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import event, text

from notarius_core.artifacts import ArtifactObject, ArtifactRefSequence
from notarius_core.domain.invocation_cache import InvocationCacheEntry

from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000901")


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database_path = tmp_path / "invocation-cache.sqlite3"
    created = create_database(f"sqlite+aiosqlite:///{database_path}")
    async with created.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.execute(
            text(
                "INSERT INTO workspaces "
                "(id, slug, name, kind, created_at, updated_at) VALUES "
                "(:id, 'cache-workspace', 'Cache workspace', 'shared', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": WORKSPACE_ID.hex},
        )
    try:
        yield created
    finally:
        await created.dispose()


def _artifact(artifact_id: str, content_hash: str) -> ArtifactObject:
    return ArtifactObject(
        workspace_id=WORKSPACE_ID,
        id=UUID(artifact_id),
        artifact_type="scalar.integer",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": 3},
        byte_size=11,
        sha256=content_hash,
    )


@pytest.mark.asyncio
async def test_cache_entry_and_batch_artifacts_round_trip_in_a_fresh_session(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    first = _artifact(
        "00000000-0000-0000-0000-000000000201",
        "1" * 64,
    )
    second = _artifact(
        "00000000-0000-0000-0000-000000000202",
        "2" * 64,
    )
    sequence = ArtifactRefSequence(
        sequence_id=UUID("00000000-0000-0000-0000-000000000203"),
        artifact_type="scalar.integer",
        schema_version=1,
        item_refs=[first.ref(), second.ref()],
        ordered=True,
        index_key="source_order",
        metadata={"source_sequence_id": "source"},
    )
    entry = InvocationCacheEntry(
        workspace_id=WORKSPACE_ID,
        key_sha256="d" * 64,
        outputs={"single": first.ref(), "sequence": sequence},
        generation=UUID("00000000-0000-0000-0000-000000000204"),
        created_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
    )

    async with unit_of_work as entered:
        await entered.artifacts.add(first)
        await entered.artifacts.add(second)
        assert await entered.invocation_cache.put_if_absent(entry)
        await entered.commit()

    async with unit_of_work as entered:
        loaded = await entered.invocation_cache.get(WORKSPACE_ID, entry.key_sha256)
        artifacts = await entered.artifacts.get_many(
            WORKSPACE_ID,
            {
                first.id,
                second.id,
                UUID("00000000-0000-0000-0000-000000000205"),
            }
        )

    assert loaded is not None
    assert loaded is not entry
    assert loaded.generation == entry.generation
    assert loaded.created_at == entry.created_at
    assert loaded.outputs["single"] == first.ref()
    assert loaded.outputs["sequence"] == sequence
    assert set(artifacts) == {first.id, second.id}


@pytest.mark.asyncio
async def test_concurrent_cache_publication_is_first_writer_wins(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    first = InvocationCacheEntry(
        workspace_id=WORKSPACE_ID,
        key_sha256="e" * 64,
        outputs={
            "first": _artifact(
                "00000000-0000-0000-0000-000000000211",
                "3" * 64,
            ).ref()
        },
    )
    second = InvocationCacheEntry(
        workspace_id=WORKSPACE_ID,
        key_sha256=first.key_sha256,
        outputs={
            "second": _artifact(
                "00000000-0000-0000-0000-000000000212",
                "4" * 64,
            ).ref()
        },
    )
    first_statement_finished = asyncio.Event()
    second_statement_started = asyncio.Event()
    release_first_commit = asyncio.Event()
    insert_count = 0

    def observe_cache_insert(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        nonlocal insert_count
        if statement.lstrip().startswith("INSERT INTO invocation_cache_entries"):
            insert_count += 1
            if insert_count == 2:
                second_statement_started.set()

    async def write_first() -> bool:
        async with unit_of_work as entered:
            published = await entered.invocation_cache.put_if_absent(first)
            first_statement_finished.set()
            await asyncio.wait_for(release_first_commit.wait(), timeout=5)
            await entered.commit()
            return published

    async def write_second() -> bool:
        await asyncio.wait_for(first_statement_finished.wait(), timeout=5)
        async with unit_of_work as entered:
            published = await entered.invocation_cache.put_if_absent(second)
            await entered.commit()
            return published

    event.listen(
        database.engine.sync_engine,
        "before_cursor_execute",
        observe_cache_insert,
    )
    try:
        first_task = asyncio.create_task(write_first())
        second_task = asyncio.create_task(write_second())
        await asyncio.wait_for(second_statement_started.wait(), timeout=5)
        assert not second_task.done()
        release_first_commit.set()
        assert await first_task
        assert not await second_task
    finally:
        event.remove(
            database.engine.sync_engine,
            "before_cursor_execute",
            observe_cache_insert,
        )

    async with unit_of_work as entered:
        loaded = await entered.invocation_cache.get(WORKSPACE_ID, first.key_sha256)

    assert insert_count == 2
    assert loaded is not None
    assert loaded.generation == first.generation
    assert loaded.outputs == first.outputs


@pytest.mark.asyncio
async def test_stale_generation_cannot_remove_a_new_cache_entry(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    stale = InvocationCacheEntry(
        workspace_id=WORKSPACE_ID, key_sha256="f" * 64, outputs={}
    )
    fresh = InvocationCacheEntry(
        workspace_id=WORKSPACE_ID, key_sha256=stale.key_sha256, outputs={}
    )

    async with unit_of_work as entered:
        assert await entered.invocation_cache.put_if_absent(stale)
        await entered.commit()
    async with unit_of_work as entered:
        assert await entered.invocation_cache.remove_if_current(
            WORKSPACE_ID,
            stale.key_sha256,
            stale.generation,
        )
        await entered.commit()
    async with unit_of_work as entered:
        assert await entered.invocation_cache.put_if_absent(fresh)
        await entered.commit()
    async with unit_of_work as entered:
        assert not await entered.invocation_cache.remove_if_current(
            WORKSPACE_ID,
            stale.key_sha256,
            stale.generation,
        )
        await entered.commit()
    async with unit_of_work as entered:
        loaded = await entered.invocation_cache.get(WORKSPACE_ID, stale.key_sha256)

    assert loaded is not None
    assert loaded.generation == fresh.generation

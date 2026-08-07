from dataclasses import replace
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select

from notarius_core.artifacts import ArtifactObject, ArtifactRefSequence
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionNodeResult,
    GraphExecutionStatus,
)
from notarius_core.domain.saved_graphs import (
    SavedGraph,
    SavedGraphDocument,
    SavedGraphRevision,
)

from notarius_persistence import schema
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ONE = UUID("00000000-0000-0000-0000-000000000001")
WORKSPACE_TWO = UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    created = create_database(
        f"sqlite+aiosqlite:///{tmp_path / 'execution-history.sqlite3'}"
    )
    async with created.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.execute(
            schema.workspaces.insert(),
            [
                {
                    "id": WORKSPACE_ONE,
                    "slug": "one",
                    "name": "One",
                    "kind": "shared",
                    "created_at": datetime(2026, 7, 1, tzinfo=UTC),
                    "updated_at": datetime(2026, 7, 1, tzinfo=UTC),
                },
                {
                    "id": WORKSPACE_TWO,
                    "slug": "two",
                    "name": "Two",
                    "kind": "shared",
                    "created_at": datetime(2026, 7, 1, tzinfo=UTC),
                    "updated_at": datetime(2026, 7, 1, tzinfo=UTC),
                },
            ],
        )
    try:
        yield created
    finally:
        await created.dispose()


async def _persist_graph_revisions(
    unit_of_work: SqlAlchemyUnitOfWork,
    graph_id: UUID,
    workspace_id: UUID,
) -> None:
    document = SavedGraphDocument()
    created_at = datetime(2026, 7, 18, 7, 0, tzinfo=UTC)
    async with unit_of_work as entered:
        await entered.graphs.add(
            SavedGraph(
                workspace_id=workspace_id,
                id=graph_id,
                name="Execution history graph",
                document=document,
                revision=2,
                created_at=created_at,
                updated_at=created_at,
            )
        )
        await entered.commit()
    async with unit_of_work as entered:
        for revision in (1, 2):
            await entered.graphs.add_revision(
                SavedGraphRevision(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    revision=revision,
                    name="Execution history graph",
                    document=document,
                    created_at=created_at,
                )
            )
        await entered.commit()


@pytest.mark.asyncio
async def test_execution_lifecycle_and_node_outputs_round_trip_without_replacing_head(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    graph_id = UUID("00000000-0000-0000-0000-000000000101")
    execution_id = UUID("00000000-0000-0000-0000-000000000102")
    await _persist_graph_revisions(unit_of_work, graph_id, WORKSPACE_ONE)
    created_at = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    started_at = created_at + timedelta(seconds=1)
    finished_at = started_at + timedelta(seconds=2)
    execution = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=execution_id,
        graph_id=graph_id,
        graph_revision=2,
        status="queued",
        scope="selected-with-dependencies",
        requested_node_ids=("upload", "extract"),
        created_at=created_at,
    )
    image = ArtifactObject(
        workspace_id=WORKSPACE_ONE,
        id=UUID("00000000-0000-0000-0000-000000000103"),
        artifact_type="image.raster",
        schema_version=1,
        content_type="image/png",
        storage_backend="local",
        object_key="images/page.png",
    )
    second_image = ArtifactObject(
        workspace_id=WORKSPACE_ONE,
        id=UUID("00000000-0000-0000-0000-000000000104"),
        artifact_type="image.raster",
        schema_version=1,
        content_type="image/png",
        storage_backend="local",
        object_key="images/page-2.png",
    )
    sequence = ArtifactRefSequence(
        sequence_id=UUID("00000000-0000-0000-0000-000000000105"),
        artifact_type="image.raster",
        schema_version=1,
        item_refs=[image.ref(), second_image.ref()],
    )

    async with unit_of_work as entered:
        await entered.execution_history.add(execution)
        await entered.commit()
    running = replace(execution, status="running", started_at=started_at)
    async with unit_of_work as entered:
        await entered.execution_history.update(running)
        await entered.commit()
    succeeded = replace(
        running,
        status="succeeded",
        workflow_run_id=UUID("00000000-0000-0000-0000-000000000106"),
        finished_at=finished_at,
    )
    async with unit_of_work as entered:
        await entered.artifacts.add(image)
        await entered.artifacts.add(second_image)
        await entered.execution_history.add_node_result(
            GraphExecutionNodeResult(
                workspace_id=WORKSPACE_ONE,
                execution_id=execution_id,
                node_id="upload",
                position=0,
                status="succeeded",
                outputs={"images": sequence},
                completed_at=started_at + timedelta(seconds=1),
            )
        )
        await entered.execution_history.add_node_result(
            GraphExecutionNodeResult(
                workspace_id=WORKSPACE_ONE,
                execution_id=execution_id,
                node_id="extract",
                position=1,
                status="failed",
                outputs={},
                error="provider timed out",
                completed_at=finished_at,
            )
        )
        await entered.execution_history.update(succeeded)
        await entered.commit()

    async with unit_of_work as entered:
        detail = await entered.execution_history.get(WORKSPACE_ONE, execution_id)
        page = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE, graph_id, limit=20
        )
        materialized = await entered.materialized_outputs.list_for_graph(
            WORKSPACE_ONE, graph_id, 2
        )

    assert detail is not None
    assert detail.execution == succeeded
    assert detail.execution.requested_node_ids == ("upload", "extract")
    assert [result.node_id for result in detail.node_results] == ["upload", "extract"]
    assert detail.node_results[0].outputs == {"images": sequence}
    assert detail.node_results[1].error == "provider timed out"
    assert page.items[0].node_count == 2
    assert page.items[0].artifact_count == 2
    assert materialized == []


@pytest.mark.asyncio
async def test_cursor_paging_is_stable_when_execution_timestamps_tie(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    graph_id = UUID("00000000-0000-0000-0000-000000000201")
    await _persist_graph_revisions(unit_of_work, graph_id, WORKSPACE_ONE)
    created_at = datetime(2026, 7, 18, 9, 0, tzinfo=UTC)
    execution_ids = [
        UUID("00000000-0000-0000-0000-000000000201"),
        UUID("00000000-0000-0000-0000-000000000202"),
        UUID("00000000-0000-0000-0000-000000000203"),
    ]
    async with unit_of_work as entered:
        for execution_id in execution_ids:
            await entered.execution_history.add(
                GraphExecution(
                    workspace_id=WORKSPACE_ONE,
                    execution_id=execution_id,
                    graph_id=graph_id,
                    graph_revision=1,
                    status="queued",
                    requested_node_ids=("extract",),
                    created_at=created_at,
                )
            )
        await entered.commit()

    async with unit_of_work as entered:
        first = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE, graph_id, limit=2
        )
        assert first.next_cursor is not None
        second = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE,
            graph_id,
            limit=2,
            cursor=first.next_cursor,
        )

    assert [item.execution.execution_id for item in first.items] == [
        execution_ids[2],
        execution_ids[1],
    ]
    assert [item.execution.execution_id for item in second.items] == [
        execution_ids[0]
    ]
    assert second.next_cursor is None


@pytest.mark.asyncio
async def test_filters_select_executions_without_filtering_detail_node_rows(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    graph_id = UUID("00000000-0000-0000-0000-000000000301")
    await _persist_graph_revisions(unit_of_work, graph_id, WORKSPACE_ONE)
    base_time = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
    completed_id = UUID("00000000-0000-0000-0000-000000000302")
    queued_id = UUID("00000000-0000-0000-0000-000000000303")
    unrelated_id = UUID("00000000-0000-0000-0000-000000000304")
    cancelled_id = UUID("00000000-0000-0000-0000-000000000305")
    completed = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=completed_id,
        graph_id=graph_id,
        graph_revision=1,
        status="succeeded",
        scope="selected",
        requested_node_ids=("target", "other"),
        created_at=base_time,
        started_at=base_time,
        finished_at=base_time + timedelta(seconds=1),
    )
    queued = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=queued_id,
        graph_id=graph_id,
        graph_revision=2,
        status="queued",
        scope="selected",
        requested_node_ids=("target",),
        created_at=base_time + timedelta(minutes=1),
    )
    unrelated = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=unrelated_id,
        graph_id=graph_id,
        graph_revision=1,
        status="queued",
        scope="selected",
        requested_node_ids=("other",),
        created_at=base_time + timedelta(minutes=2),
    )
    cancelled = GraphExecution(
        workspace_id=WORKSPACE_ONE,
        execution_id=cancelled_id,
        graph_id=graph_id,
        graph_revision=1,
        status="cancelled",
        scope="selected",
        requested_node_ids=("target",),
        created_at=base_time + timedelta(minutes=3),
        finished_at=base_time + timedelta(minutes=3, seconds=1),
    )
    async with unit_of_work as entered:
        for execution in (completed, queued, unrelated, cancelled):
            await entered.execution_history.add(execution)
        with pytest.raises(ValueError, match="did not request node 'unexpected'"):
            await entered.execution_history.add_node_result(
                GraphExecutionNodeResult(
                    workspace_id=WORKSPACE_ONE,
                    execution_id=completed_id,
                    node_id="unexpected",
                    position=2,
                    status="skipped",
                    outputs={},
                    completed_at=base_time + timedelta(seconds=1),
                )
            )
        for position, node_id in enumerate(("target", "other")):
            await entered.execution_history.add_node_result(
                GraphExecutionNodeResult(
                    workspace_id=WORKSPACE_ONE,
                    execution_id=completed_id,
                    node_id=node_id,
                    position=position,
                    status="succeeded",
                    outputs={},
                    completed_at=base_time + timedelta(seconds=1),
                )
            )
        await entered.commit()

    async with unit_of_work as entered:
        target_page = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE,
            graph_id,
            limit=10,
            node_id="target",
        )
        revision_page = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE,
            graph_id,
            limit=10,
            graph_revision=2,
        )
        succeeded_page = await entered.execution_history.list_for_graph(
            WORKSPACE_ONE,
            graph_id,
            limit=10,
            status="succeeded",
        )
        detail = await entered.execution_history.get(WORKSPACE_ONE, completed_id)

    assert [item.execution.execution_id for item in target_page.items] == [
        cancelled_id,
        queued_id,
        completed_id,
    ]
    assert [item.execution.execution_id for item in revision_page.items] == [queued_id]
    assert [item.execution.execution_id for item in succeeded_page.items] == [
        completed_id
    ]
    assert detail is not None
    assert [result.node_id for result in detail.node_results] == ["target", "other"]


@pytest.mark.asyncio
async def test_restart_recovery_fails_only_active_executions(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    graph_one_id = UUID("00000000-0000-0000-0000-000000000401")
    graph_two_id = UUID("00000000-0000-0000-0000-000000000402")
    await _persist_graph_revisions(unit_of_work, graph_one_id, WORKSPACE_ONE)
    await _persist_graph_revisions(unit_of_work, graph_two_id, WORKSPACE_TWO)
    created_at = datetime(2026, 7, 18, 11, 0, tzinfo=UTC)
    statuses: tuple[GraphExecutionStatus, ...] = (
        "queued",
        "running",
        "cancelling",
        "succeeded",
    )
    execution_ids = [
        UUID(f"00000000-0000-0000-0000-{index:012d}")
        for index in range(410, 414)
    ]
    second_workspace_execution_ids = [
        UUID(f"00000000-0000-0000-0000-{index:012d}")
        for index in range(420, 424)
    ]
    async with unit_of_work as entered:
        for execution_id, status in zip(execution_ids, statuses, strict=True):
            terminal = status == "succeeded"
            await entered.execution_history.add(
                GraphExecution(
                    workspace_id=WORKSPACE_ONE,
                    execution_id=execution_id,
                    graph_id=graph_one_id,
                    graph_revision=1,
                    status=status,
                    created_at=created_at,
                    started_at=(created_at if status != "queued" else None),
                    finished_at=(created_at if terminal else None),
                )
            )
        for execution_id, status in zip(
            second_workspace_execution_ids, statuses, strict=True
        ):
            terminal = status == "succeeded"
            await entered.execution_history.add(
                GraphExecution(
                    workspace_id=WORKSPACE_TWO,
                    execution_id=execution_id,
                    graph_id=graph_two_id,
                    graph_revision=1,
                    status=status,
                    created_at=created_at,
                    started_at=(created_at if status != "queued" else None),
                    finished_at=(created_at if terminal else None),
                )
            )
        await entered.commit()

    recovered_at = created_at + timedelta(minutes=5)
    async with unit_of_work as entered:
        interrupted = await entered.execution_history.interrupt_all_active(
            finished_at=recovered_at,
            error="API restarted before execution completed",
        )
        await entered.commit()

    async with unit_of_work as entered:
        details = [
            await entered.execution_history.get(WORKSPACE_ONE, execution_id)
            for execution_id in execution_ids
        ]
        second_workspace_details = [
            await entered.execution_history.get(WORKSPACE_TWO, execution_id)
            for execution_id in second_workspace_execution_ids
        ]

    assert interrupted == 6
    assert all(detail is not None for detail in details)
    assert [detail.execution.status for detail in details if detail is not None] == [
        "failed",
        "failed",
        "failed",
        "succeeded",
    ]
    assert details[0] is not None
    assert details[0].execution.finished_at == recovered_at
    assert details[0].execution.error == "API restarted before execution completed"
    assert [
        detail.execution.status
        for detail in second_workspace_details
        if detail is not None
    ] == ["failed", "failed", "failed", "succeeded"]


@pytest.mark.asyncio
async def test_deleting_graph_cascades_history_and_preserves_artifacts(
    database: Database,
) -> None:
    unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    graph_id = UUID("00000000-0000-0000-0000-000000000501")
    execution_id = UUID("00000000-0000-0000-0000-000000000502")
    artifact = ArtifactObject(
        workspace_id=WORKSPACE_ONE,
        id=UUID("00000000-0000-0000-0000-000000000503"),
        artifact_type="scalar.integer",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": 5},
    )
    await _persist_graph_revisions(unit_of_work, graph_id, WORKSPACE_ONE)
    async with unit_of_work as entered:
        await entered.artifacts.add(artifact)
        await entered.execution_history.add(
            GraphExecution(
                workspace_id=WORKSPACE_ONE,
                execution_id=execution_id,
                graph_id=graph_id,
                graph_revision=1,
                status="queued",
            )
        )
        await entered.commit()
    async with unit_of_work as entered:
        graph = await entered.graphs.get(WORKSPACE_ONE, graph_id)
        assert graph is not None
        await entered.graphs.remove(WORKSPACE_ONE, graph)
        await entered.commit()

    async with unit_of_work as entered:
        assert await entered.execution_history.get(WORKSPACE_ONE, execution_id) is None
        assert await entered.artifacts.get(WORKSPACE_ONE, artifact.id) is not None
    async with database.sessions() as session:
        assert (
            await session.scalar(
                select(schema.graph_execution_requested_nodes.c.execution_id)
            )
            is None
        )
        assert (
            await session.scalar(select(schema.graph_executions.c.execution_id))
            is None
        )

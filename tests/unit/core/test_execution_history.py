from datetime import UTC, datetime
from uuid import UUID

import pytest

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    InMemoryUnitOfWork,
)
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionNodeResult,
)


def test_execution_history_models_validate_lifecycle_and_count_artifacts() -> None:
    with pytest.raises(ValueError, match="requires finished_at"):
        GraphExecution(
            execution_id=UUID("00000000-0000-0000-0000-000000000001"),
            graph_id=UUID("00000000-0000-0000-0000-000000000002"),
            graph_revision=1,
            status="failed",
        )

    first = ArtifactRef(
        artifact_id=UUID("00000000-0000-0000-0000-000000000010"),
        artifact_type="scalar.integer",
        schema_version=1,
    )
    sequence = ArtifactRefSequence(
        sequence_id=UUID("00000000-0000-0000-0000-000000000011"),
        artifact_type="scalar.integer",
        schema_version=1,
        item_refs=[
            first,
            first.model_copy(
                update={
                    "artifact_id": UUID(
                        "00000000-0000-0000-0000-000000000012"
                    )
                }
            ),
        ],
    )
    result = GraphExecutionNodeResult(
        execution_id=UUID("00000000-0000-0000-0000-000000000001"),
        node_id="  extract  ",
        position=0,
        status="succeeded",
        outputs={"single": first, "many": sequence},
    )

    assert result.node_id == "extract"
    assert result.artifact_count == 3
    assert GraphExecutionNodeResult.outputs_from_storage(
        result.storage_envelopes()
    ) == result.outputs


@pytest.mark.asyncio
async def test_in_memory_execution_history_obeys_commit_and_node_filter_semantics() -> (
    None
):
    unit_of_work = InMemoryUnitOfWork()
    execution = GraphExecution(
        execution_id=UUID("00000000-0000-0000-0000-000000000021"),
        graph_id=UUID("00000000-0000-0000-0000-000000000022"),
        graph_revision=3,
        status="queued",
        scope="selected-with-dependencies",
        requested_node_ids=("source", "target"),
        created_at=datetime(2026, 7, 18, 8, 0, tzinfo=UTC),
    )

    async with unit_of_work as entered:
        await entered.execution_history.add(execution)
        await entered.commit()
    async with unit_of_work as entered:
        page = await entered.execution_history.list_for_graph(
            execution.graph_id,
            limit=10,
            node_id="target",
        )
        missing = await entered.execution_history.list_for_graph(
            execution.graph_id,
            limit=10,
            node_id="not-requested",
        )

    assert [item.execution.execution_id for item in page.items] == [
        execution.execution_id
    ]
    assert page.items[0].node_count == 0
    assert missing.items == ()


@pytest.mark.asyncio
async def test_in_memory_execution_history_rejects_unrequested_node_results() -> None:
    unit_of_work = InMemoryUnitOfWork()
    execution = GraphExecution(
        execution_id=UUID("00000000-0000-0000-0000-000000000031"),
        graph_id=UUID("00000000-0000-0000-0000-000000000032"),
        graph_revision=1,
        status="queued",
        requested_node_ids=("requested",),
    )
    async with unit_of_work as entered:
        await entered.execution_history.add(execution)
        with pytest.raises(ValueError, match="did not request node 'unexpected'"):
            await entered.execution_history.add_node_result(
                GraphExecutionNodeResult(
                    execution_id=execution.execution_id,
                    node_id="unexpected",
                    position=0,
                    status="skipped",
                    outputs={},
                )
            )

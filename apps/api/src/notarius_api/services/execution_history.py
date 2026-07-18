from datetime import UTC, datetime
from uuid import UUID

from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionDetail,
    GraphExecutionNodeResult,
    GraphExecutionPage,
    GraphExecutionScope,
    GraphExecutionStatus,
)
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.ports.execution_history import ExecutionHistoryUnitOfWorkPort

from notarius_api.services.execution.models import GraphExecutionResult


class ExecutionHistoryService:
    """Own the durable lifecycle and browsing of saved-graph executions."""

    def __init__(
        self,
        unit_of_work: ExecutionHistoryUnitOfWorkPort,
        saved_graphs: SavedGraphService | None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._saved_graphs = saved_graphs

    async def create_queued(
        self,
        *,
        execution_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        scope: GraphExecutionScope,
        requested_node_ids: tuple[str, ...],
    ) -> GraphExecution:
        if self._saved_graphs is None:
            raise RuntimeError(
                "Saved graph context is not configured for execution history"
            )
        await self._saved_graphs.get_revision(graph_id, graph_revision)
        execution = GraphExecution(
            execution_id=execution_id,
            graph_id=graph_id,
            graph_revision=graph_revision,
            scope=scope,
            status="queued",
            requested_node_ids=requested_node_ids,
        )
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.add(execution)
            await unit_of_work.commit()
        return execution

    async def mark_running(self, execution: GraphExecution) -> None:
        execution.status = "running"
        execution.started_at = datetime.now(UTC)
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            await unit_of_work.commit()

    async def mark_cancelling(self, execution: GraphExecution) -> None:
        execution.status = "cancelling"
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            await unit_of_work.commit()

    async def complete(
        self,
        execution: GraphExecution,
        *,
        status: GraphExecutionStatus,
        result: GraphExecutionResult | None,
        error: str | None,
    ) -> None:
        if status not in {"cancelled", "succeeded", "failed"}:
            raise ValueError(f"Execution completion status {status!r} is not terminal")
        completed_at = datetime.now(UTC)
        execution.status = status
        execution.finished_at = completed_at
        execution.error = error
        if result is not None:
            execution.workflow_run_id = result.workflow_run_id

        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            if result is not None:
                for position, node_result in enumerate(result.node_results):
                    await unit_of_work.execution_history.add_node_result(
                        GraphExecutionNodeResult(
                            execution_id=execution.execution_id,
                            node_id=node_result.node_id,
                            position=position,
                            status=node_result.status,
                            outputs=dict(node_result.outputs),
                            error=node_result.error,
                            completed_at=completed_at,
                        )
                    )
            await unit_of_work.commit()

    async def get_for_graph(
        self,
        graph_id: UUID,
        execution_id: UUID,
    ) -> GraphExecutionDetail | None:
        async with self._unit_of_work as unit_of_work:
            detail = await unit_of_work.execution_history.get(execution_id)
        if detail is None or detail.execution.graph_id != graph_id:
            return None
        return detail

    async def list_for_graph(
        self,
        graph_id: UUID,
        *,
        limit: int,
        cursor: GraphExecutionCursor | None = None,
        graph_revision: int | None = None,
        status: GraphExecutionStatus | None = None,
        node_id: str | None = None,
    ) -> GraphExecutionPage:
        async with self._unit_of_work as unit_of_work:
            return await unit_of_work.execution_history.list_for_graph(
                graph_id,
                limit=limit,
                cursor=cursor,
                graph_revision=graph_revision,
                status=status,
                node_id=node_id,
            )

    async def interrupt_active(self) -> int:
        async with self._unit_of_work as unit_of_work:
            interrupted = await unit_of_work.execution_history.interrupt_active(
                finished_at=datetime.now(UTC),
                error=(
                    "Execution was interrupted because the API process stopped "
                    "before reporting a terminal result"
                ),
            )
            await unit_of_work.commit()
        return interrupted


__all__ = ["ExecutionHistoryService"]

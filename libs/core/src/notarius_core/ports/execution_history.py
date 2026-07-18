from datetime import datetime
from typing import Protocol
from uuid import UUID

from notarius_core.artifacts import UnitOfWorkPort
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionDetail,
    GraphExecutionNodeResult,
    GraphExecutionPage,
    GraphExecutionStatus,
)


class GraphExecutionHistoryRepositoryPort(Protocol):
    async def add(self, execution: GraphExecution) -> None: ...

    async def update(self, execution: GraphExecution) -> None: ...

    async def add_node_result(self, result: GraphExecutionNodeResult) -> None: ...

    async def get(self, execution_id: UUID) -> GraphExecutionDetail | None: ...

    async def list_for_graph(
        self,
        graph_id: UUID,
        *,
        limit: int,
        cursor: GraphExecutionCursor | None = None,
        graph_revision: int | None = None,
        status: GraphExecutionStatus | None = None,
        node_id: str | None = None,
    ) -> GraphExecutionPage: ...

    async def interrupt_active(
        self,
        *,
        finished_at: datetime,
        error: str,
    ) -> int: ...


class ExecutionHistoryUnitOfWorkPort(UnitOfWorkPort, Protocol):
    @property
    def execution_history(self) -> GraphExecutionHistoryRepositoryPort: ...


__all__ = [
    "ExecutionHistoryUnitOfWorkPort",
    "GraphExecutionHistoryRepositoryPort",
]

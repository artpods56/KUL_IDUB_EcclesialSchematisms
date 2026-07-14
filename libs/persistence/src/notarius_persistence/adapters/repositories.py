from typing import override
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from notarius_core.domain.saved_graphs import SavedGraph
from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort

from notarius_persistence import schema


class SqlSavedGraphRepository(SavedGraphRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, graph: SavedGraph) -> None:
        self._session.add(graph)

    @override
    async def get(self, graph_id: UUID) -> SavedGraph | None:
        return await self._session.get(SavedGraph, graph_id)

    @override
    async def list(self) -> list[SavedGraph]:
        result = await self._session.scalars(
            select(SavedGraph).order_by(
                schema.saved_graphs.c.updated_at.desc(),
                schema.saved_graphs.c.id.asc(),
            )
        )
        return list(result)

    @override
    async def remove(self, graph: SavedGraph) -> None:
        await self._session.delete(graph)

from collections.abc import Callable
from uuid import UUID

from notarius_core.domain.errors import (
    ConcurrentWriteError,
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphDocument
from notarius_core.ports.saved_graphs import SavedGraphUnitOfWorkPort


class SavedGraphService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], SavedGraphUnitOfWorkPort],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def create(
        self,
        *,
        name: str,
        document: SavedGraphDocument,
    ) -> SavedGraph:
        graph = SavedGraph(name=name, document=document)
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.graphs.add(graph)
            await unit_of_work.commit()
        return graph

    async def list(self) -> list[SavedGraph]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.graphs.list()

    async def get(self, graph_id: UUID) -> SavedGraph:
        async with self._unit_of_work_factory() as unit_of_work:
            graph = await unit_of_work.graphs.get(graph_id)
        if graph is None:
            raise NotFoundError("Saved graph", str(graph_id))
        return graph

    async def replace(
        self,
        graph_id: UUID,
        *,
        name: str,
        document: SavedGraphDocument,
        expected_revision: int,
    ) -> SavedGraph:
        async with self._unit_of_work_factory() as unit_of_work:
            graph = await unit_of_work.graphs.get(graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            graph.replace(
                name=name,
                document=document,
                expected_revision=expected_revision,
            )
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                ) from exc
        return graph

    async def delete(self, graph_id: UUID, *, expected_revision: int) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            graph = await unit_of_work.graphs.get(graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            graph.ensure_revision(expected_revision)
            await unit_of_work.graphs.remove(graph)
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=graph.revision,
                    actual_revision=None,
                ) from exc

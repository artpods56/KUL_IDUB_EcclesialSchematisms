from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from notarius_core.domain.saved_graphs import SavedGraph


class SavedGraphRepositoryPort(Protocol):
    async def add(self, graph: SavedGraph) -> None: ...

    async def get(self, graph_id: UUID) -> SavedGraph | None: ...

    async def list(self) -> list[SavedGraph]: ...

    async def remove(self, graph: SavedGraph) -> None: ...


class SavedGraphUnitOfWorkPort(Protocol):
    @property
    def graphs(self) -> SavedGraphRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

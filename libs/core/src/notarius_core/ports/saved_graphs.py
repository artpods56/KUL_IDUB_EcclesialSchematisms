from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self
from uuid import UUID

from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphRevision

if TYPE_CHECKING:
    from notarius_core.ports.node_secrets import NodeSecretRepositoryPort


class SavedGraphRepositoryPort(Protocol):
    async def add(self, graph: SavedGraph) -> None: ...

    async def add_revision(self, revision: SavedGraphRevision) -> None: ...

    async def lock_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        expected_revision: int,
    ) -> None: ...

    async def get(self, workspace_id: UUID, graph_id: UUID) -> SavedGraph | None: ...

    async def get_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        revision: int,
    ) -> SavedGraphRevision | None: ...

    async def list_revisions(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> list[SavedGraphRevision]: ...

    async def list(self, workspace_id: UUID) -> list[SavedGraph]: ...

    async def remove(self, workspace_id: UUID, graph: SavedGraph) -> None: ...


class SavedGraphUnitOfWorkPort(Protocol):
    @property
    def graphs(self) -> SavedGraphRepositoryPort: ...

    @property
    def node_secrets(self) -> "NodeSecretRepositoryPort": ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

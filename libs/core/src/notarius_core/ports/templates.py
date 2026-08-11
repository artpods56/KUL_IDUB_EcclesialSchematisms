from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from notarius_core.domain.templates import Template
from notarius_core.ports.collaboration import CollaborationRepositoryPort
from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort


class TemplateRepositoryPort(Protocol):
    async def add(self, template: Template) -> None: ...

    async def get(
        self,
        workspace_id: UUID,
        template_id: UUID,
    ) -> Template | None: ...

    async def list(
        self,
        workspace_id: UUID,
        *,
        query: str | None,
        include_archived: bool,
    ) -> list[Template]: ...


class TemplateUnitOfWorkPort(Protocol):
    @property
    def graphs(self) -> SavedGraphRepositoryPort: ...

    @property
    def collaboration(self) -> CollaborationRepositoryPort: ...

    @property
    def templates(self) -> TemplateRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


__all__ = ["TemplateRepositoryPort", "TemplateUnitOfWorkPort"]

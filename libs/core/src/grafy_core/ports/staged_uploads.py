from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from grafy_core.domain.staged_uploads import StagedUpload


class StagedUploadRepositoryPort(Protocol):
    async def add(self, upload: StagedUpload) -> None: ...

    async def get(
        self,
        workspace_id: UUID,
        upload_key: str,
    ) -> StagedUpload | None: ...

    async def list_for_workspace(self, workspace_id: UUID) -> list[StagedUpload]: ...

    async def remove(self, workspace_id: UUID, upload_key: str) -> None: ...


class StagedUploadUnitOfWorkPort(Protocol):
    @property
    def staged_uploads(self) -> StagedUploadRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

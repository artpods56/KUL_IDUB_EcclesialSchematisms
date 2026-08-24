from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from grafy_core.domain.plugin_releases import PluginRelease, PluginRuntimeArtifact
from grafy_core.ports.identity import IdentityRepositoryPort


class PluginReleaseRepositoryPort(Protocol):
    async def add(self, release: PluginRelease) -> None: ...

    async def get_by_source_digest(
        self,
        workspace_id: UUID,
        slug: str,
        source_digest: str,
    ) -> PluginRelease | None: ...

    async def get_by_descriptor_digest(
        self,
        workspace_id: UUID,
        slug: str,
        descriptor_digest: str,
    ) -> PluginRelease | None: ...

    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginRelease | None: ...

    async def next_revision(self, workspace_id: UUID, slug: str) -> int: ...

    async def list_current(self, workspace_id: UUID) -> list[PluginRelease]: ...

    async def list_runtime_artifacts(self) -> list[PluginRuntimeArtifact]: ...


class PluginReleaseUnitOfWorkPort(Protocol):
    @property
    def plugin_releases(self) -> PluginReleaseRepositoryPort: ...

    @property
    def identity(self) -> IdentityRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


__all__ = ["PluginReleaseRepositoryPort", "PluginReleaseUnitOfWorkPort"]

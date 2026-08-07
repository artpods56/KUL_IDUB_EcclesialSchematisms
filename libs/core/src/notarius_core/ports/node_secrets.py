from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self
from uuid import UUID

from pydantic import SecretStr

from notarius_core.domain.node_secrets import (
    EncryptedNodeSecret,
    JsonValue,
)

if TYPE_CHECKING:
    from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort


class NodeSecretUnavailableError(RuntimeError):
    pass


class NodeSecretResolverPort(Protocol):
    async def resolve_secret(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID | None,
        graph_revision: int | None,
        node_id: str | None,
        name: str,
        dependencies: Mapping[str, JsonValue],
    ) -> SecretStr: ...

    async def cache_revision(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID | None,
        graph_revision: int | None,
        node_id: str | None,
        name: str,
        dependencies: Mapping[str, JsonValue],
    ) -> str: ...


class UnavailableNodeSecretResolver(NodeSecretResolverPort):
    async def resolve_secret(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID | None,
        graph_revision: int | None,
        node_id: str | None,
        name: str,
        dependencies: Mapping[str, JsonValue],
    ) -> SecretStr:
        del workspace_id, graph_id, graph_revision, node_id, name, dependencies
        raise NodeSecretUnavailableError("Node secrets are unavailable in this runtime")

    async def cache_revision(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID | None,
        graph_revision: int | None,
        node_id: str | None,
        name: str,
        dependencies: Mapping[str, JsonValue],
    ) -> str:
        del workspace_id, graph_id, graph_revision, node_id, name, dependencies
        raise NodeSecretUnavailableError("Node secrets are unavailable in this runtime")


class NodeSecretRepositoryPort(Protocol):
    async def upsert(self, secret: EncryptedNodeSecret) -> None: ...

    async def get(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        node_id: str,
        name: str,
    ) -> EncryptedNodeSecret | None: ...

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> list[EncryptedNodeSecret]: ...

    async def remove(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        node_id: str,
        name: str,
    ) -> None: ...


class NodeSecretUnitOfWorkPort(Protocol):
    @property
    def graphs(self) -> "SavedGraphRepositoryPort": ...

    @property
    def node_secrets(self) -> NodeSecretRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

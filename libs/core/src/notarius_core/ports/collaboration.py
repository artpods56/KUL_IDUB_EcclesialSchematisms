from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self
from uuid import UUID

from notarius_core.domain.collaboration import (
    CollaborativeGraphHead,
    GraphCheckpointMapping,
    GraphCommandJournalEntry,
    GraphCommandReceipt,
    GraphExecutionIdempotencyRecord,
    GraphActiveExecutionSlot,
)

if TYPE_CHECKING:
    from notarius_core.ports.identity import (
        IdentityRepositoryPort,
        SecurityAuditRepositoryPort,
    )
    from notarius_core.ports.node_secrets import NodeSecretRepositoryPort
    from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort


class CollaborationRepositoryPort(Protocol):
    async def add_head(self, head: CollaborativeGraphHead) -> None: ...

    async def get_head(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> CollaborativeGraphHead | None: ...

    async def lock_head(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> CollaborativeGraphHead | None: ...

    async def save_head(self, head: CollaborativeGraphHead) -> None: ...

    async def remove_head(self, workspace_id: UUID, graph_id: UUID) -> None: ...

    async def list_graphs_missing_heads(self) -> list[tuple[UUID, UUID]]:
        """Return workspace/graph ids for saved graphs without a collaborative head."""
        ...

    async def add_journal_entry(self, entry: GraphCommandJournalEntry) -> None: ...

    async def clear_journal(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> None:
        """Drop journal rows so a room-epoch reset can reuse sequence numbers.

        Receipt tombstones remain for obsolete-epoch command resolve.
        """
        ...

    async def get_receipt(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        command_id: UUID,
    ) -> GraphCommandReceipt | None: ...

    async def add_receipt(self, receipt: GraphCommandReceipt) -> None: ...

    async def get_checkpoint_mapping(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        room_epoch: UUID,
        collaboration_sequence: int,
    ) -> GraphCheckpointMapping | None: ...

    async def add_checkpoint_mapping(
        self,
        mapping: GraphCheckpointMapping,
    ) -> None: ...

    async def get_execution_idempotency(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        client_request_id: UUID,
    ) -> GraphExecutionIdempotencyRecord | None: ...

    async def add_execution_idempotency(
        self,
        record: GraphExecutionIdempotencyRecord,
    ) -> None: ...

    async def get_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> GraphActiveExecutionSlot | None: ...

    async def acquire_active_execution_slot(
        self,
        slot: GraphActiveExecutionSlot,
    ) -> bool:
        """Insert the active slot. Returns False when the graph already has one."""
        ...

    async def clear_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        execution_id: UUID | None = None,
    ) -> None: ...

    async def clear_all_active_execution_slots(self) -> int: ...


class CollaborationUnitOfWorkPort(Protocol):
    @property
    def collaboration(self) -> CollaborationRepositoryPort: ...

    @property
    def graphs(self) -> "SavedGraphRepositoryPort": ...

    @property
    def node_secrets(self) -> "NodeSecretRepositoryPort": ...

    @property
    def identity(self) -> "IdentityRepositoryPort": ...

    @property
    def security_audit(self) -> "SecurityAuditRepositoryPort": ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

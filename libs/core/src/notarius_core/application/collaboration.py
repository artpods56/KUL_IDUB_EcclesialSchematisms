from collections.abc import Callable
from typing import cast
from uuid import UUID, uuid4

import hmac

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.collaboration import (
    CollaborationActorKind,
    CollaborativeGraphHead,
    CommandReceiptOutcome,
    GraphCheckpointMapping,
    GraphCommand,
    GraphCommandJournalEntry,
    GraphCommandKind,
    GraphCommandReceipt,
    apply_graph_command,
    command_hmac_digest,
    empty_collaborative_document,
)
from notarius_core.domain.errors import (
    CollaborationActiveExecutionError,
    CollaborationHeadConflictError,
    CollaborationIdempotencyMismatchError,
    CollaborationUncheckpointedError,
    ConcurrentWriteError,
    MissingCollaborativeHeadError,
    NotFoundError,
    SavedGraphRevisionConflictError,
    UserDisabledError,
)
from notarius_core.domain.identity import (
    ActorContext,
    WorkspaceAccess,
    WorkspaceCapability,
)
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphDocument
from notarius_core.domain.security_audit import (
    SecurityAuditActorKind,
    SecurityAuditEvent,
    SecurityAuditOutcome,
)
from notarius_core.plugins import PluginRegistry
from notarius_core.ports.collaboration import CollaborationUnitOfWorkPort
from notarius_core.ports.saved_graphs import SavedGraphUnitOfWorkPort


class CollaborationService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], CollaborationUnitOfWorkPort],
        plugin_registry: PluginRegistry,
        *,
        command_hmac_key: bytes,
        command_hmac_key_version: int,
        saved_graphs: SavedGraphService | None = None,
    ) -> None:
        if not command_hmac_key:
            raise ValueError("Command HMAC key must not be empty")
        if command_hmac_key_version < 1:
            raise ValueError("Command HMAC key version must be at least 1")
        self._unit_of_work_factory = unit_of_work_factory
        self._plugin_registry = plugin_registry
        self._command_hmac_key = command_hmac_key
        self._command_hmac_key_version = command_hmac_key_version
        self._saved_graphs = saved_graphs or SavedGraphService(
            cast(
                Callable[[], SavedGraphUnitOfWorkPort],
                unit_of_work_factory,
            ),
            plugin_registry,
        )

    async def initialize_head_for_existing_graph(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> CollaborativeGraphHead:
        async with self._unit_of_work_factory() as unit_of_work:
            existing = await unit_of_work.collaboration.get_head(
                workspace_id,
                graph_id,
            )
            if existing is not None:
                return existing
            graph = await unit_of_work.graphs.get(workspace_id, graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            head = CollaborativeGraphHead.for_existing_saved_graph(
                workspace_id=workspace_id,
                graph_id=graph_id,
                name=graph.name,
                document=graph.document,
                checkpoint_revision=graph.revision,
                updated_at=graph.updated_at,
            )
            await unit_of_work.collaboration.add_head(head)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.SYSTEM,
                    operation="collaboration.head.bootstrap",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(graph_id),
                )
            )
            await unit_of_work.commit()
        return head

    async def bootstrap_graph(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        command_id: UUID,
        command: GraphCommand,
        graph_id: UUID | None = None,
        graph_room_session_id: UUID | None = None,
    ) -> tuple[SavedGraph, CollaborativeGraphHead, GraphCommandReceipt]:
        async with self._unit_of_work_factory() as unit_of_work:
            access = await self._require_capability(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
                capability=WorkspaceCapability.CREATE_GRAPH,
            )
            access.require(WorkspaceCapability.EDIT_GRAPH)
            access.require(WorkspaceCapability.CHECKPOINT_GRAPH)

            resolved_graph_id = uuid4() if graph_id is None else graph_id
            room_epoch = uuid4()
            next_name, next_document = apply_graph_command(
                name="Untitled graph",
                document=empty_collaborative_document(),
                command=command,
            )
            digest = command_hmac_digest(
                self._command_hmac_key,
                key_version=self._command_hmac_key_version,
                workspace_id=workspace_id,
                graph_id=resolved_graph_id,
                actor_user_id=actor.user_id,
                room_epoch=room_epoch,
                observed_sequence=0,
                command=command,
            )
            existing_receipt = await unit_of_work.collaboration.get_receipt(
                workspace_id,
                resolved_graph_id,
                command_id,
            )
            if existing_receipt is not None:
                if not hmac.compare_digest(existing_receipt.command_hmac, digest):
                    raise CollaborationIdempotencyMismatchError(
                        workspace_id=workspace_id,
                        graph_id=resolved_graph_id,
                        command_id=command_id,
                    )
                graph = await unit_of_work.graphs.get(workspace_id, resolved_graph_id)
                head = await unit_of_work.collaboration.get_head(
                    workspace_id,
                    resolved_graph_id,
                )
                if graph is None or head is None:
                    raise MissingCollaborativeHeadError(
                        workspace_id=workspace_id,
                        graph_id=resolved_graph_id,
                    )
                return graph, head, existing_receipt

            graph = SavedGraph(
                workspace_id=workspace_id,
                created_by_user_id=actor.user_id,
                id=resolved_graph_id,
                name=next_name,
                document=next_document,
                revision=1,
            )
            head = CollaborativeGraphHead(
                workspace_id=workspace_id,
                graph_id=resolved_graph_id,
                room_epoch=room_epoch,
                collaboration_sequence=1,
                checkpoint_sequence=1,
                checkpoint_revision=1,
                name=next_name,
                document=next_document,
            )
            receipt = GraphCommandReceipt(
                workspace_id=workspace_id,
                graph_id=resolved_graph_id,
                command_id=command_id,
                command_hmac=digest,
                hmac_key_version=self._command_hmac_key_version,
                actor_kind=CollaborationActorKind.USER,
                actor_user_id=actor.user_id,
                room_epoch=room_epoch,
                accepted_sequence=1,
                outcome=CommandReceiptOutcome.ACCEPTED,
            )
            journal_entry = GraphCommandJournalEntry(
                workspace_id=workspace_id,
                graph_id=resolved_graph_id,
                room_epoch=room_epoch,
                command_id=command_id,
                command_hmac=digest,
                hmac_key_version=self._command_hmac_key_version,
                accepted_sequence=1,
                actor_kind=CollaborationActorKind.USER,
                actor_user_id=actor.user_id,
                graph_room_session_id=graph_room_session_id,
                authorization_version=access.membership.authorization_version,
                command_kind=GraphCommandKind(command.kind),
                command_payload=command.model_dump(mode="json"),
            )
            mapping = GraphCheckpointMapping(
                workspace_id=workspace_id,
                graph_id=resolved_graph_id,
                room_epoch=room_epoch,
                collaboration_sequence=1,
                saved_revision=1,
            )
            await unit_of_work.graphs.add(graph)
            await unit_of_work.graphs.add_revision(graph.snapshot())
            await unit_of_work.collaboration.add_head(head)
            await unit_of_work.collaboration.add_checkpoint_mapping(mapping)
            await unit_of_work.collaboration.add_journal_entry(journal_entry)
            await unit_of_work.collaboration.add_receipt(receipt)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="collaboration.graph.bootstrap",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(resolved_graph_id),
                )
            )
            await unit_of_work.commit()
        return graph, head, receipt

    async def accept_command(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
        command_id: UUID,
        observed_sequence: int,
        observed_room_epoch: UUID,
        command: GraphCommand,
        graph_room_session_id: UUID | None = None,
    ) -> tuple[CollaborativeGraphHead, GraphCommandReceipt]:
        async with self._unit_of_work_factory() as unit_of_work:
            access = await self._require_capability(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
                capability=WorkspaceCapability.EDIT_GRAPH,
            )
            digest = command_hmac_digest(
                self._command_hmac_key,
                key_version=self._command_hmac_key_version,
                workspace_id=workspace_id,
                graph_id=graph_id,
                actor_user_id=actor.user_id,
                room_epoch=observed_room_epoch,
                observed_sequence=observed_sequence,
                command=command,
            )
            existing_receipt = await unit_of_work.collaboration.get_receipt(
                workspace_id,
                graph_id,
                command_id,
            )
            if existing_receipt is not None:
                if not hmac.compare_digest(existing_receipt.command_hmac, digest):
                    raise CollaborationIdempotencyMismatchError(
                        workspace_id=workspace_id,
                        graph_id=graph_id,
                        command_id=command_id,
                    )
                head = await unit_of_work.collaboration.get_head(
                    workspace_id,
                    graph_id,
                )
                if head is None:
                    raise MissingCollaborativeHeadError(
                        workspace_id=workspace_id,
                        graph_id=graph_id,
                    )
                replay = existing_receipt.model_copy(
                    update={"outcome": CommandReceiptOutcome.IDEMPOTENT_REPLAY}
                )
                return head, replay

            head = await unit_of_work.collaboration.lock_head(workspace_id, graph_id)
            if head is None:
                raise MissingCollaborativeHeadError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                )
            if (
                head.room_epoch != observed_room_epoch
                or head.collaboration_sequence != observed_sequence
            ):
                raise CollaborationHeadConflictError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    expected_sequence=observed_sequence,
                    actual_sequence=head.collaboration_sequence,
                    room_epoch=head.room_epoch,
                )
            next_name, next_document = apply_graph_command(
                name=head.name,
                document=head.document,
                command=command,
            )
            SavedGraphDocument.model_validate(next_document.model_dump(mode="json"))
            head.apply_accepted_command(name=next_name, document=next_document)
            receipt = GraphCommandReceipt(
                workspace_id=workspace_id,
                graph_id=graph_id,
                command_id=command_id,
                command_hmac=digest,
                hmac_key_version=self._command_hmac_key_version,
                actor_kind=CollaborationActorKind.USER,
                actor_user_id=actor.user_id,
                room_epoch=head.room_epoch,
                accepted_sequence=head.collaboration_sequence,
                outcome=CommandReceiptOutcome.ACCEPTED,
            )
            journal_entry = GraphCommandJournalEntry(
                workspace_id=workspace_id,
                graph_id=graph_id,
                room_epoch=head.room_epoch,
                command_id=command_id,
                command_hmac=digest,
                hmac_key_version=self._command_hmac_key_version,
                accepted_sequence=head.collaboration_sequence,
                actor_kind=CollaborationActorKind.USER,
                actor_user_id=actor.user_id,
                graph_room_session_id=graph_room_session_id,
                authorization_version=access.membership.authorization_version,
                command_kind=GraphCommandKind(command.kind),
                command_payload=command.model_dump(mode="json"),
            )
            await unit_of_work.collaboration.save_head(head)
            await unit_of_work.collaboration.add_journal_entry(journal_entry)
            await unit_of_work.collaboration.add_receipt(receipt)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="collaboration.command.accept",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(graph_id),
                )
            )
            await unit_of_work.commit()
        return head, receipt

    async def checkpoint(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
        expected_sequence: int,
        expected_room_epoch: UUID,
    ) -> tuple[CollaborativeGraphHead, int]:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_capability(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
                capability=WorkspaceCapability.CHECKPOINT_GRAPH,
            )
            # Fixed lock order: saved-graph row, then collaborative head.
            graph = await unit_of_work.graphs.get(workspace_id, graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            await unit_of_work.graphs.lock_revision(
                workspace_id,
                graph_id,
                graph.revision,
            )
            graph = await unit_of_work.graphs.get(workspace_id, graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            head = await unit_of_work.collaboration.lock_head(workspace_id, graph_id)
            if head is None:
                raise MissingCollaborativeHeadError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                )
            if (
                head.room_epoch != expected_room_epoch
                or head.collaboration_sequence != expected_sequence
            ):
                raise CollaborationHeadConflictError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    expected_sequence=expected_sequence,
                    actual_sequence=head.collaboration_sequence,
                    room_epoch=head.room_epoch,
                )
            existing_mapping = await unit_of_work.collaboration.get_checkpoint_mapping(
                workspace_id,
                graph_id,
                room_epoch=head.room_epoch,
                collaboration_sequence=head.collaboration_sequence,
            )
            if (
                existing_mapping is not None
                and head.checkpoint_sequence == head.collaboration_sequence
            ):
                return head, existing_mapping.saved_revision

            if graph.revision != head.checkpoint_revision:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=head.checkpoint_revision,
                    actual_revision=graph.revision,
                )
            checkpoint_expected_revision = head.checkpoint_revision
            await self._saved_graphs.apply_replacement_in_unit_of_work(
                unit_of_work,
                graph,
                name=head.name,
                document=head.document,
                expected_revision=checkpoint_expected_revision,
                physically_remove_orphaned_secrets=False,
            )
            mapping = GraphCheckpointMapping(
                workspace_id=workspace_id,
                graph_id=graph_id,
                room_epoch=head.room_epoch,
                collaboration_sequence=head.collaboration_sequence,
                saved_revision=graph.revision,
            )
            head.record_checkpoint(
                sequence=head.collaboration_sequence,
                revision=graph.revision,
            )
            await unit_of_work.collaboration.add_checkpoint_mapping(mapping)
            await unit_of_work.collaboration.save_head(head)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="collaboration.checkpoint",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(graph_id),
                )
            )
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=checkpoint_expected_revision,
                    actual_revision=None,
                ) from exc
        return head, graph.revision

    async def replace_complete_document(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
        name: str,
        document: SavedGraphDocument,
        expected_revision: int,
    ) -> tuple[SavedGraph, CollaborativeGraphHead]:
        async with self._unit_of_work_factory() as unit_of_work:
            access = await self._require_capability(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
                capability=WorkspaceCapability.EDIT_GRAPH,
            )
            # Fixed lock order: saved-graph row, then collaborative head.
            await unit_of_work.graphs.lock_revision(
                workspace_id,
                graph_id,
                expected_revision,
            )
            graph = await unit_of_work.graphs.get(workspace_id, graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            head = await unit_of_work.collaboration.lock_head(workspace_id, graph_id)
            if head is None:
                raise MissingCollaborativeHeadError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                )
            if not head.is_fully_checkpointed:
                raise CollaborationUncheckpointedError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    head_sequence=head.collaboration_sequence,
                    checkpoint_sequence=head.checkpoint_sequence,
                )
            if (
                graph.revision != expected_revision
                or head.checkpoint_revision != expected_revision
            ):
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=expected_revision,
                    actual_revision=graph.revision,
                )
            await self._saved_graphs.apply_replacement_in_unit_of_work(
                unit_of_work,
                graph,
                name=name,
                document=document,
                expected_revision=expected_revision,
                physically_remove_orphaned_secrets=access.membership.grants(
                    WorkspaceCapability.MANAGE_SECRETS
                ),
            )
            room_epoch = uuid4()
            head.room_epoch = room_epoch
            head.collaboration_sequence = 0
            head.checkpoint_sequence = 0
            head.checkpoint_revision = graph.revision
            head.name = graph.name
            head.document = graph.document
            head.updated_at = graph.updated_at
            mapping = GraphCheckpointMapping(
                workspace_id=workspace_id,
                graph_id=graph_id,
                room_epoch=room_epoch,
                collaboration_sequence=0,
                saved_revision=graph.revision,
            )
            await unit_of_work.collaboration.save_head(head)
            await unit_of_work.collaboration.add_checkpoint_mapping(mapping)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="collaboration.graph.replace",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(graph_id),
                )
            )
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                ) from exc
        return graph, head

    async def delete_graph(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
        expected_revision: int,
        expected_room_epoch: UUID | None = None,
        expected_sequence: int | None = None,
    ) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_capability(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
                capability=WorkspaceCapability.DELETE_GRAPH,
            )
            # Fixed lock order: saved-graph row, then collaborative head.
            await unit_of_work.graphs.lock_revision(
                workspace_id,
                graph_id,
                expected_revision,
            )
            graph = await unit_of_work.graphs.get(workspace_id, graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            graph.ensure_revision(expected_revision)
            head = await unit_of_work.collaboration.lock_head(workspace_id, graph_id)
            if head is None:
                raise MissingCollaborativeHeadError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                )
            active_slot = await unit_of_work.collaboration.get_active_execution_slot(
                workspace_id,
                graph_id,
            )
            if active_slot is not None:
                raise CollaborationActiveExecutionError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    execution_id=active_slot.execution_id,
                )
            if expected_room_epoch is not None and expected_sequence is not None:
                if (
                    head.room_epoch != expected_room_epoch
                    or head.collaboration_sequence != expected_sequence
                ):
                    raise CollaborationHeadConflictError(
                        workspace_id=workspace_id,
                        graph_id=graph_id,
                        expected_sequence=expected_sequence,
                        actual_sequence=head.collaboration_sequence,
                        room_epoch=head.room_epoch,
                    )
            elif not head.is_fully_checkpointed:
                raise CollaborationUncheckpointedError(
                    workspace_id=workspace_id,
                    graph_id=graph_id,
                    head_sequence=head.collaboration_sequence,
                    checkpoint_sequence=head.checkpoint_sequence,
                )
            await unit_of_work.collaboration.remove_head(workspace_id, graph_id)
            await unit_of_work.graphs.remove(workspace_id, graph)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="collaboration.graph.delete",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="saved_graph",
                    resource_id=str(graph_id),
                )
            )
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                ) from exc

    async def _require_capability(
        self,
        unit_of_work: CollaborationUnitOfWorkPort,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        capability: WorkspaceCapability,
    ) -> WorkspaceAccess:
        user = await unit_of_work.identity.get_user(actor.user_id)
        if user is None or not user.active:
            raise UserDisabledError()
        membership = await unit_of_work.identity.get_membership(
            workspace_id=workspace_id,
            user_id=actor.user_id,
        )
        if membership is None or not membership.is_active:
            raise NotFoundError("Workspace", str(workspace_id))
        access = WorkspaceAccess(
            actor=actor,
            workspace_id=workspace_id,
            membership=membership,
        )
        access.require(capability)
        return access

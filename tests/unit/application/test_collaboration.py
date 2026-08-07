from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

import pytest

from notarius_core.application.collaboration import CollaborationService
from notarius_core.domain.collaboration import (
    CollaborativeGraphHead,
    CommandReceiptOutcome,
    GraphActiveExecutionSlot,
    GraphCheckpointMapping,
    GraphCommandJournalEntry,
    GraphCommandReceipt,
    MoveNodePosition,
    MoveNodesCommand,
    RenameGraphCommand,
    ReplaceDocumentCommand,
)
from notarius_core.domain.errors import (
    CollaborationActiveExecutionError,
    CollaborationCommandRejectedError,
    CollaborationHeadConflictError,
    CollaborationIdempotencyMismatchError,
    CollaborationUncheckpointedError,
    MissingCollaborativeHeadError,
)
from notarius_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphDocument,
    SavedGraphNode,
    SavedGraphRevision,
)
from notarius_core.domain.security_audit import SecurityAuditEvent
from notarius_core.plugins import PluginRegistry


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000201")
USER_ID = UUID("00000000-0000-0000-0000-000000000202")
HMAC_KEY = b"test-command-hmac-key-phase3"


class FakeCollaborationRepository:
    def __init__(self) -> None:
        self.heads: dict[tuple[UUID, UUID], CollaborativeGraphHead] = {}
        self.receipts: dict[tuple[UUID, UUID, UUID], GraphCommandReceipt] = {}
        self.journal: list[GraphCommandJournalEntry] = []
        self.mappings: dict[
            tuple[UUID, UUID, UUID, int], GraphCheckpointMapping
        ] = {}
        self.active_slots: dict[tuple[UUID, UUID], GraphActiveExecutionSlot] = {}
        self.locked: list[tuple[UUID, UUID]] = []

    async def add_head(self, head: CollaborativeGraphHead) -> None:
        key = (head.workspace_id, head.graph_id)
        if key in self.heads:
            raise ValueError("head exists")
        self.heads[key] = head

    async def get_head(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> CollaborativeGraphHead | None:
        return self.heads.get((workspace_id, graph_id))

    async def lock_head(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> CollaborativeGraphHead | None:
        self.locked.append((workspace_id, graph_id))
        return self.heads.get((workspace_id, graph_id))

    async def save_head(self, head: CollaborativeGraphHead) -> None:
        self.heads[(head.workspace_id, head.graph_id)] = head

    async def remove_head(self, workspace_id: UUID, graph_id: UUID) -> None:
        self.heads.pop((workspace_id, graph_id), None)

    async def add_journal_entry(self, entry: GraphCommandJournalEntry) -> None:
        self.journal.append(entry)

    async def get_receipt(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        command_id: UUID,
    ) -> GraphCommandReceipt | None:
        return self.receipts.get((workspace_id, graph_id, command_id))

    async def add_receipt(self, receipt: GraphCommandReceipt) -> None:
        self.receipts[(receipt.workspace_id, receipt.graph_id, receipt.command_id)] = (
            receipt
        )

    async def get_checkpoint_mapping(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        room_epoch: UUID,
        collaboration_sequence: int,
    ) -> GraphCheckpointMapping | None:
        return self.mappings.get(
            (workspace_id, graph_id, room_epoch, collaboration_sequence)
        )

    async def add_checkpoint_mapping(self, mapping: GraphCheckpointMapping) -> None:
        self.mappings[
            (
                mapping.workspace_id,
                mapping.graph_id,
                mapping.room_epoch,
                mapping.collaboration_sequence,
            )
        ] = mapping

    async def get_execution_idempotency(self, *args, **kwargs):
        del args, kwargs
        return None

    async def add_execution_idempotency(self, record) -> None:
        del record

    async def get_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> GraphActiveExecutionSlot | None:
        return self.active_slots.get((workspace_id, graph_id))

    async def upsert_active_execution_slot(self, slot: GraphActiveExecutionSlot) -> None:
        self.active_slots[(slot.workspace_id, slot.graph_id)] = slot

    async def clear_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> None:
        self.active_slots.pop((workspace_id, graph_id), None)


class FakeSavedGraphRepository:
    def __init__(self) -> None:
        self.graphs: dict[UUID, SavedGraph] = {}
        self.revisions: dict[tuple[UUID, int], SavedGraphRevision] = {}
        self.locked_revisions: list[tuple[UUID, UUID, int]] = []

    async def add(self, graph: SavedGraph) -> None:
        self.graphs[graph.id] = graph

    async def add_revision(self, revision: SavedGraphRevision) -> None:
        self.revisions[(revision.graph_id, revision.revision)] = revision

    async def lock_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        expected_revision: int,
    ) -> None:
        self.locked_revisions.append((workspace_id, graph_id, expected_revision))

    async def get(self, workspace_id: UUID, graph_id: UUID) -> SavedGraph | None:
        graph = self.graphs.get(graph_id)
        if graph is None or graph.workspace_id != workspace_id:
            return None
        return graph

    async def get_revision(self, *args, **kwargs):
        del args, kwargs
        return None

    async def list_revisions(self, *args, **kwargs):
        del args, kwargs
        return []

    async def list(self, workspace_id: UUID) -> list[SavedGraph]:
        return [
            graph for graph in self.graphs.values() if graph.workspace_id == workspace_id
        ]

    async def remove(self, workspace_id: UUID, graph: SavedGraph) -> None:
        if graph.workspace_id == workspace_id:
            self.graphs.pop(graph.id, None)


class FakeNodeSecretRepository:
    async def list_for_graph(self, *args, **kwargs):
        del args, kwargs
        return []

    async def remove(self, *args, **kwargs) -> None:
        del args, kwargs


class FakeIdentityRepository:
    def __init__(
        self,
        user: User,
        memberships: list[WorkspaceMembership],
    ) -> None:
        self.user = user
        self.memberships = {
            (membership.workspace_id, membership.user_id): membership
            for membership in memberships
        }

    async def get_user(self, user_id: UUID) -> User | None:
        if user_id != self.user.id:
            return None
        return self.user

    async def get_membership(self, *, workspace_id: UUID, user_id: UUID):
        return self.memberships.get((workspace_id, user_id))


class FakeSecurityAuditRepository:
    def __init__(self) -> None:
        self.events: list[SecurityAuditEvent] = []

    async def add(self, event: SecurityAuditEvent) -> None:
        self.events.append(event)


class FakeCollaborationUnitOfWork:
    def __init__(
        self,
        *,
        collaboration: FakeCollaborationRepository,
        graphs: FakeSavedGraphRepository,
        identity: FakeIdentityRepository,
        security_audit: FakeSecurityAuditRepository,
    ) -> None:
        self.collaboration = collaboration
        self.graphs = graphs
        self.node_secrets = FakeNodeSecretRepository()
        self.identity = identity
        self.security_audit = security_audit
        self.commit_count = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        return None


TARGET_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000301")


class FakeFactory:
    def __init__(self, role: WorkspaceRole = WorkspaceRole.OWNER) -> None:
        self.user = User(id=USER_ID, email="editor@example.com", display_name="Editor")
        self.workspace = Workspace.shared(slug="phase3", name="Phase 3")
        self.workspace.id = WORKSPACE_ID
        self.membership = WorkspaceMembership(
            workspace_id=WORKSPACE_ID,
            user_id=USER_ID,
            role=role,
        )
        self.target_membership = WorkspaceMembership(
            workspace_id=TARGET_WORKSPACE_ID,
            user_id=USER_ID,
            role=role,
        )
        self.collaboration = FakeCollaborationRepository()
        self.graphs = FakeSavedGraphRepository()
        self.security_audit = FakeSecurityAuditRepository()
        self.identity = FakeIdentityRepository(
            self.user,
            [self.membership, self.target_membership],
        )
        self.created: list[FakeCollaborationUnitOfWork] = []

    def __call__(self) -> FakeCollaborationUnitOfWork:
        unit = FakeCollaborationUnitOfWork(
            collaboration=self.collaboration,
            graphs=self.graphs,
            identity=self.identity,
            security_audit=self.security_audit,
        )
        self.created.append(unit)
        return unit


def _service(factory: FakeFactory) -> CollaborationService:
    return CollaborationService(
        factory,
        PluginRegistry(),
        command_hmac_key=HMAC_KEY,
        command_hmac_key_version=1,
    )


@pytest.mark.asyncio
async def test_initialize_head_for_existing_graph_is_idempotent() -> None:
    factory = FakeFactory()
    graph = SavedGraph(
        workspace_id=WORKSPACE_ID,
        created_by_user_id=USER_ID,
        name="Legacy",
        document=SavedGraphDocument(),
        revision=3,
    )
    factory.graphs.graphs[graph.id] = graph
    service = _service(factory)

    first = await service.initialize_head_for_existing_graph(
        workspace_id=WORKSPACE_ID,
        graph_id=graph.id,
    )
    second = await service.initialize_head_for_existing_graph(
        workspace_id=WORKSPACE_ID,
        graph_id=graph.id,
    )

    assert first.collaboration_sequence == 0
    assert first.checkpoint_revision == 3
    assert second.room_epoch == first.room_epoch
    assert factory.created[0].commit_count == 1
    assert factory.created[1].commit_count == 0


@pytest.mark.asyncio
async def test_bootstrap_graph_commits_head_checkpoint_and_receipt() -> None:
    factory = FakeFactory()
    service = _service(factory)
    command_id = uuid4()
    graph_id = uuid4()
    command = ReplaceDocumentCommand(
        name="Bootstrapped",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="n1",
                    operator_id="example.operator",
                    operator_version=1,
                    position=GraphPoint(x=1, y=2),
                ),
            )
        ),
    )

    graph, head, receipt = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=command_id,
        command=command,
        graph_id=graph_id,
    )

    assert graph.id == graph_id
    assert graph.revision == 1
    assert head.collaboration_sequence == 1
    assert head.checkpoint_sequence == 1
    assert head.checkpoint_revision == 1
    assert receipt.outcome is CommandReceiptOutcome.ACCEPTED
    assert factory.collaboration.journal[0].accepted_sequence == 1
    assert (WORKSPACE_ID, graph_id, head.room_epoch, 1) in factory.collaboration.mappings
    assert factory.created[-1].commit_count == 1


@pytest.mark.asyncio
async def test_accept_command_advances_sequence_without_checkpoint() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )

    updated_head, receipt = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=head.collaboration_sequence,
        observed_room_epoch=head.room_epoch,
        command=RenameGraphCommand(name="Renamed", expected_name="Draft"),
    )

    assert updated_head.collaboration_sequence == 2
    assert updated_head.checkpoint_sequence == 1
    assert updated_head.name == "Renamed"
    assert receipt.accepted_sequence == 2
    assert factory.graphs.graphs[graph_id].revision == 1


@pytest.mark.asyncio
async def test_accept_command_idempotent_replay_and_mismatch() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    command_id = uuid4()
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    command = RenameGraphCommand(name="Once", expected_name="Draft")
    observed_sequence = head.collaboration_sequence
    observed_room_epoch = head.room_epoch
    first_head, first_receipt = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=command_id,
        observed_sequence=observed_sequence,
        observed_room_epoch=observed_room_epoch,
        command=command,
    )
    replay_head, replay_receipt = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=command_id,
        observed_sequence=observed_sequence,
        observed_room_epoch=observed_room_epoch,
        command=command,
    )

    assert first_head.collaboration_sequence == replay_head.collaboration_sequence == 2
    assert replay_receipt.outcome is CommandReceiptOutcome.IDEMPOTENT_REPLAY
    assert first_receipt.outcome is CommandReceiptOutcome.ACCEPTED

    with pytest.raises(CollaborationIdempotencyMismatchError):
        await service.accept_command(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            command_id=command_id,
            observed_sequence=observed_sequence,
            observed_room_epoch=observed_room_epoch,
            command=RenameGraphCommand(name="Different", expected_name="Draft"),
        )


@pytest.mark.asyncio
async def test_checkpoint_advances_saved_revision_and_preserves_secrets() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    head, _ = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=head.collaboration_sequence,
        observed_room_epoch=head.room_epoch,
        command=RenameGraphCommand(name="Checkpoint me", expected_name="Draft"),
    )

    checkpointed_head, revision = await service.checkpoint(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        expected_sequence=head.collaboration_sequence,
        expected_room_epoch=head.room_epoch,
    )

    assert revision == 2
    assert checkpointed_head.checkpoint_sequence == 2
    assert checkpointed_head.checkpoint_revision == 2
    assert factory.graphs.graphs[graph_id].name == "Checkpoint me"
    assert factory.graphs.revisions[(graph_id, 2)].name == "Checkpoint me"


@pytest.mark.asyncio
async def test_accept_command_requires_existing_head() -> None:
    factory = FakeFactory()
    service = _service(factory)
    with pytest.raises(MissingCollaborativeHeadError):
        await service.accept_command(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=uuid4(),
            command_id=uuid4(),
            observed_sequence=0,
            observed_room_epoch=uuid4(),
            command=RenameGraphCommand(name="Nope", expected_name="Nope"),
        )


@pytest.mark.asyncio
async def test_replace_complete_document_resets_epoch_when_checkpointed() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Original", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    prior_epoch = head.room_epoch

    replaced, new_head = await service.replace_complete_document(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        name="Replaced",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="n1",
                    operator_id="example.operator",
                    operator_version=1,
                    position=GraphPoint(x=3, y=4),
                ),
            )
        ),
        expected_revision=graph.revision,
    )

    assert replaced.revision == 2
    assert replaced.name == "Replaced"
    assert new_head.room_epoch != prior_epoch
    assert new_head.collaboration_sequence == 0
    assert new_head.checkpoint_sequence == 0
    assert new_head.checkpoint_revision == 2
    assert (WORKSPACE_ID, graph_id, new_head.room_epoch, 0) in (
        factory.collaboration.mappings
    )


@pytest.mark.asyncio
async def test_replace_complete_document_rejects_uncheckpointed_head() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=head.collaboration_sequence,
        observed_room_epoch=head.room_epoch,
        command=RenameGraphCommand(name="Uncheckpointed", expected_name="Draft"),
    )

    with pytest.raises(CollaborationUncheckpointedError):
        await service.replace_complete_document(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            name="Nope",
            document=SavedGraphDocument(),
            expected_revision=graph.revision,
        )


@pytest.mark.asyncio
async def test_delete_graph_removes_head_and_graph() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, _, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Delete me", document=SavedGraphDocument()),
        graph_id=graph_id,
    )

    await service.delete_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        expected_revision=graph.revision,
    )

    assert graph_id not in factory.graphs.graphs
    assert (WORKSPACE_ID, graph_id) not in factory.collaboration.heads


@pytest.mark.asyncio
async def test_delete_graph_rejects_uncheckpointed_legacy_delete() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=head.collaboration_sequence,
        observed_room_epoch=head.room_epoch,
        command=RenameGraphCommand(name="Pending", expected_name="Draft"),
    )

    with pytest.raises(CollaborationUncheckpointedError):
        await service.delete_graph(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            expected_revision=graph.revision,
        )


@pytest.mark.asyncio
async def test_delete_graph_rejects_active_execution() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, _, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Busy", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    execution_id = uuid4()
    factory.collaboration.active_slots[(WORKSPACE_ID, graph_id)] = (
        GraphActiveExecutionSlot(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            execution_id=execution_id,
        )
    )

    with pytest.raises(CollaborationActiveExecutionError):
        await service.delete_graph(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            expected_revision=graph.revision,
        )


@pytest.mark.asyncio
async def test_get_head_returns_live_document() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Live", document=SavedGraphDocument()),
        graph_id=graph_id,
    )

    loaded = await service.get_head(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
    )

    assert loaded.room_epoch == head.room_epoch
    assert loaded.collaboration_sequence == 1
    assert loaded.name == "Live"


@pytest.mark.asyncio
async def test_accept_command_rebases_move_against_newer_head() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    node = SavedGraphNode(
        id="n1",
        operator_id="example.operator",
        operator_version=1,
        position=GraphPoint(x=0, y=0),
    )
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(
            name="Draft",
            document=SavedGraphDocument(nodes=(node,)),
        ),
        graph_id=graph_id,
    )
    observed_sequence = head.collaboration_sequence
    observed_epoch = head.room_epoch
    await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=observed_sequence,
        observed_room_epoch=observed_epoch,
        command=RenameGraphCommand(name="Moved ahead", expected_name="Draft"),
    )

    updated, receipt = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=observed_sequence,
        observed_room_epoch=observed_epoch,
        command=MoveNodesCommand(
            positions=(MoveNodePosition(node_id="n1", x=9, y=8),)
        ),
    )

    assert updated.collaboration_sequence == 3
    assert updated.document.nodes[0].position == GraphPoint(x=9, y=8)
    assert receipt.accepted_sequence == 3


@pytest.mark.asyncio
async def test_accept_command_rejects_stale_rename_with_field_conflict() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    _, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    observed_sequence = head.collaboration_sequence
    observed_epoch = head.room_epoch
    await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=observed_sequence,
        observed_room_epoch=observed_epoch,
        command=RenameGraphCommand(name="Other", expected_name="Draft"),
    )

    with pytest.raises(CollaborationCommandRejectedError) as exc:
        await service.accept_command(
            actor=ActorContext(user_id=USER_ID),
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            command_id=uuid4(),
            observed_sequence=observed_sequence,
            observed_room_epoch=observed_epoch,
            command=RenameGraphCommand(name="Mine", expected_name="Draft"),
        )
    assert exc.value.error_code == "field_conflict"


@pytest.mark.asyncio
async def test_copy_exact_head_bootstraps_target_without_source_secrets() -> None:
    factory = FakeFactory()
    service = _service(factory)
    source_graph_id = uuid4()
    _, source_head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(
            name="Source",
            document=SavedGraphDocument(
                nodes=(
                    SavedGraphNode(
                        id="n1",
                        operator_id="example.operator",
                        operator_version=1,
                        position=GraphPoint(x=1, y=2),
                        config={
                            "uploads": [
                                {
                                    "upload_key": "x.png",
                                    "filename": "x.png",
                                    "byte_size": 1,
                                }
                            ],
                            "label": "keep",
                        },
                    ),
                )
            ),
        ),
        graph_id=source_graph_id,
    )

    graph, head, receipt = await service.copy_exact_head(
        actor=ActorContext(user_id=USER_ID),
        source_workspace_id=WORKSPACE_ID,
        source_graph_id=source_graph_id,
        target_workspace_id=TARGET_WORKSPACE_ID,
        expected_room_epoch=source_head.room_epoch,
        expected_sequence=source_head.collaboration_sequence,
        command_id=uuid4(),
    )

    assert graph.workspace_id == TARGET_WORKSPACE_ID
    assert graph.revision == 1
    assert head.collaboration_sequence == 1
    assert head.checkpoint_sequence == 1
    assert head.checkpoint_revision == 1
    assert receipt.accepted_sequence == 1
    assert graph.document.nodes[0].config_dict() == {"label": "keep"}
    assert source_graph_id in factory.graphs.graphs
    assert factory.graphs.graphs[source_graph_id].workspace_id == WORKSPACE_ID


@pytest.mark.asyncio
async def test_copy_exact_head_rejects_moved_source() -> None:
    factory = FakeFactory()
    service = _service(factory)
    source_graph_id = uuid4()
    _, source_head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Source", document=SavedGraphDocument()),
        graph_id=source_graph_id,
    )
    expected_room_epoch = source_head.room_epoch
    expected_sequence = source_head.collaboration_sequence
    await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=source_graph_id,
        command_id=uuid4(),
        observed_sequence=expected_sequence,
        observed_room_epoch=expected_room_epoch,
        command=RenameGraphCommand(name="Moved", expected_name="Source"),
    )

    with pytest.raises(CollaborationHeadConflictError):
        await service.copy_exact_head(
            actor=ActorContext(user_id=USER_ID),
            source_workspace_id=WORKSPACE_ID,
            source_graph_id=source_graph_id,
            target_workspace_id=TARGET_WORKSPACE_ID,
            expected_room_epoch=expected_room_epoch,
            expected_sequence=expected_sequence,
            command_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_delete_graph_collaboration_aware_discards_uncheckpointed() -> None:
    factory = FakeFactory()
    service = _service(factory)
    graph_id = uuid4()
    graph, head, _ = await service.bootstrap_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(name="Draft", document=SavedGraphDocument()),
        graph_id=graph_id,
    )
    head, _ = await service.accept_command(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        command_id=uuid4(),
        observed_sequence=head.collaboration_sequence,
        observed_room_epoch=head.room_epoch,
        command=RenameGraphCommand(name="Discard", expected_name="Draft"),
    )

    await service.delete_graph(
        actor=ActorContext(user_id=USER_ID),
        workspace_id=WORKSPACE_ID,
        graph_id=graph_id,
        expected_revision=graph.revision,
        expected_room_epoch=head.room_epoch,
        expected_sequence=head.collaboration_sequence,
    )

    assert graph_id not in factory.graphs.graphs
    assert (WORKSPACE_ID, graph_id) not in factory.collaboration.heads

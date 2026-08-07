from collections.abc import Collection
from datetime import datetime
from typing import cast, override
from uuid import UUID

from sqlalchemy import and_, delete, func, insert, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRepositoryPort,
    ArtifactTypeKey,
)
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.identity import (
    AuthSession,
    OidcBootstrapOwnerMapping,
    OidcIdentity,
    OidcLoginTransaction,
    PersonalAccessToken,
    User,
    Workspace,
    WorkspaceKind,
    WorkspaceMembership,
)
from notarius_core.domain.errors import NotFoundError, ObjectAlreadyExistsError
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionDetail,
    GraphExecutionListItem,
    GraphExecutionNodeResult,
    GraphExecutionPage,
    GraphExecutionStatus,
)
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.node_secrets import EncryptedNodeSecret
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphRevision
from notarius_core.domain.security_audit import SecurityAuditEvent
from notarius_core.domain.staged_uploads import StagedUpload
from notarius_core.ports.identity import (
    IdentityRepositoryPort,
    SecurityAuditRepositoryPort,
)
from notarius_core.ports.invocation_cache import InvocationCacheRepositoryPort
from notarius_core.ports.execution_history import (
    GraphExecutionHistoryRepositoryPort,
)
from notarius_core.ports.materialized_outputs import (
    MaterializedNodeOutputsRepositoryPort,
)
from notarius_core.ports.node_secrets import NodeSecretRepositoryPort
from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort
from notarius_core.ports.staged_uploads import StagedUploadRepositoryPort

from notarius_persistence import schema
from notarius_persistence.orm import GraphExecutionRecord, SavedGraphRevisionRecord


class SqlIdentityRepository(IdentityRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add_user(self, user: User) -> None:
        self._session.add(user)

    @override
    async def get_user(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)

    @override
    async def get_oidc_identity(
        self,
        *,
        issuer: str,
        subject: str,
    ) -> OidcIdentity | None:
        return await self._session.scalar(
            select(OidcIdentity).where(
                schema.oidc_identities.c.issuer == issuer,
                schema.oidc_identities.c.subject == subject,
            )
        )

    @override
    async def add_oidc_identity(self, identity: OidcIdentity) -> None:
        await self._session.flush()
        self._session.add(identity)

    @override
    async def add_workspace(self, workspace: Workspace) -> None:
        await self._session.flush()
        self._session.add(workspace)

    @override
    async def get_workspace(self, workspace_id: UUID) -> Workspace | None:
        return await self._session.get(Workspace, workspace_id)

    @override
    async def get_workspace_by_slug(self, slug: str) -> Workspace | None:
        return await self._session.scalar(
            select(Workspace).where(schema.workspaces.c.slug == slug)
        )

    @override
    async def lock_workspace_for_membership_mutation(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        statement = select(Workspace).where(
            schema.workspaces.c.id == workspace_id,
        )
        if self._session.get_bind().dialect.name == "sqlite":
            await self._session.execute(text("BEGIN IMMEDIATE"))
        else:
            statement = statement.with_for_update()
        return await self._session.scalar(statement)

    @override
    async def lock_workspace_by_slug_for_membership_mutation(
        self,
        slug: str,
    ) -> Workspace | None:
        statement = select(Workspace).where(schema.workspaces.c.slug == slug)
        if self._session.get_bind().dialect.name == "sqlite":
            await self._session.execute(text("BEGIN IMMEDIATE"))
        else:
            statement = statement.with_for_update()
        return await self._session.scalar(statement)

    @override
    async def get_personal_workspace(self, user_id: UUID) -> Workspace | None:
        return await self._session.scalar(
            select(Workspace).where(
                schema.workspaces.c.kind == WorkspaceKind.PERSONAL.value,
                schema.workspaces.c.personal_owner_user_id == user_id,
            )
        )

    @override
    async def list_workspaces_for_user(self, user_id: UUID) -> list[Workspace]:
        result = await self._session.scalars(
            select(Workspace)
            .join(
                schema.workspace_memberships,
                schema.workspace_memberships.c.workspace_id == schema.workspaces.c.id,
            )
            .where(
                schema.workspace_memberships.c.user_id == user_id,
                schema.workspace_memberships.c.revoked_at.is_(None),
            )
            .order_by(schema.workspaces.c.slug.asc())
        )
        return list(result)

    @override
    async def list_memberships_for_user(
        self,
        user_id: UUID,
    ) -> list[WorkspaceMembership]:
        result = await self._session.scalars(
            select(WorkspaceMembership)
            .where(schema.workspace_memberships.c.user_id == user_id)
            .order_by(schema.workspace_memberships.c.workspace_id.asc())
        )
        return list(result)

    @override
    async def add_membership(self, membership: WorkspaceMembership) -> None:
        await self._session.flush()
        self._session.add(membership)

    @override
    async def get_membership(
        self,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> WorkspaceMembership | None:
        return await self._session.get(WorkspaceMembership, (workspace_id, user_id))

    @override
    async def list_memberships(self, workspace_id: UUID) -> list[WorkspaceMembership]:
        result = await self._session.scalars(
            select(WorkspaceMembership)
            .where(schema.workspace_memberships.c.workspace_id == workspace_id)
            .order_by(schema.workspace_memberships.c.user_id.asc())
        )
        return list(result)

    @override
    async def count_active_owners(self, workspace_id: UUID) -> int:
        count = await self._session.scalar(
            select(func.count())
            .select_from(schema.workspace_memberships)
            .where(
                schema.workspace_memberships.c.workspace_id == workspace_id,
                schema.workspace_memberships.c.role == "owner",
                schema.workspace_memberships.c.revoked_at.is_(None),
            )
        )
        return int(count or 0)

    @override
    async def get_unconsumed_bootstrap_mapping(
        self,
        workspace_id: UUID,
    ) -> OidcBootstrapOwnerMapping | None:
        return await self._session.scalar(
            select(OidcBootstrapOwnerMapping).where(
                schema.oidc_bootstrap_owner_mappings.c.workspace_id == workspace_id,
                schema.oidc_bootstrap_owner_mappings.c.consumed_at.is_(None),
            )
        )

    @override
    async def add_bootstrap_mapping(
        self,
        mapping: OidcBootstrapOwnerMapping,
    ) -> None:
        self._session.add(mapping)

    @override
    async def add_login_transaction(self, transaction: OidcLoginTransaction) -> None:
        self._session.add(transaction)

    @override
    async def get_login_transaction(
        self,
        transaction_id: UUID,
    ) -> OidcLoginTransaction | None:
        return await self._session.get(OidcLoginTransaction, transaction_id)

    @override
    async def lock_login_transaction(
        self,
        transaction_id: UUID,
    ) -> OidcLoginTransaction | None:
        statement = select(OidcLoginTransaction).where(
            schema.oidc_login_transactions.c.id == transaction_id,
        )
        if self._session.get_bind().dialect.name == "sqlite":
            await self._session.execute(text("BEGIN IMMEDIATE"))
        else:
            statement = statement.with_for_update()
        return await self._session.scalar(statement)

    @override
    async def add_auth_session(self, session: AuthSession) -> None:
        self._session.add(session)

    @override
    async def get_auth_session(self, session_id: UUID) -> AuthSession | None:
        return await self._session.get(AuthSession, session_id)

    @override
    async def list_auth_sessions_for_user(self, user_id: UUID) -> list[AuthSession]:
        result = await self._session.scalars(
            select(AuthSession).where(schema.auth_sessions.c.user_id == user_id)
        )
        return list(result)

    @override
    async def get_auth_session_for_user(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
    ) -> AuthSession | None:
        return await self._session.scalar(
            select(AuthSession).where(
                schema.auth_sessions.c.id == session_id,
                schema.auth_sessions.c.user_id == user_id,
            )
        )

    @override
    async def delete_expired_login_transactions(self, expired_before: datetime) -> int:
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(schema.oidc_login_transactions).where(
                    or_(
                        schema.oidc_login_transactions.c.expires_at < expired_before,
                        schema.oidc_login_transactions.c.consumed_at.is_not(None),
                    )
                )
            ),
        )
        return result.rowcount

    @override
    async def add_personal_access_token(self, token: PersonalAccessToken) -> None:
        self._session.add(token)

    @override
    async def get_personal_access_token_by_digest(
        self,
        secret_digest: bytes,
    ) -> PersonalAccessToken | None:
        return await self._session.scalar(
            select(PersonalAccessToken).where(
                schema.personal_access_tokens.c.secret_digest == secret_digest
            )
        )

    @override
    async def list_personal_access_tokens_for_user(
        self,
        user_id: UUID,
    ) -> list[PersonalAccessToken]:
        result = await self._session.scalars(
            select(PersonalAccessToken).where(
                schema.personal_access_tokens.c.user_id == user_id
            )
        )
        return list(result)

    @override
    async def list_personal_access_tokens_for_user_workspace(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
    ) -> list[PersonalAccessToken]:
        result = await self._session.scalars(
            select(PersonalAccessToken)
            .where(
                schema.personal_access_tokens.c.user_id == user_id,
                schema.personal_access_tokens.c.workspace_id == workspace_id,
            )
            .order_by(schema.personal_access_tokens.c.created_at.desc())
        )
        return list(result)

    @override
    async def get_personal_access_token_for_user_workspace(
        self,
        *,
        token_id: UUID,
        user_id: UUID,
        workspace_id: UUID,
    ) -> PersonalAccessToken | None:
        return await self._session.scalar(
            select(PersonalAccessToken).where(
                schema.personal_access_tokens.c.id == token_id,
                schema.personal_access_tokens.c.user_id == user_id,
                schema.personal_access_tokens.c.workspace_id == workspace_id,
            )
        )

    @override
    async def delete_expired_sessions(self, expired_before: datetime) -> int:
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(schema.auth_sessions).where(
                    schema.auth_sessions.c.expires_at < expired_before
                )
            ),
        )
        return result.rowcount

    @override
    async def delete_expired_personal_access_tokens(
        self,
        expired_before: datetime,
    ) -> int:
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(schema.personal_access_tokens).where(
                    schema.personal_access_tokens.c.expires_at < expired_before
                )
            ),
        )
        return result.rowcount


class SqlSecurityAuditRepository(SecurityAuditRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, event: SecurityAuditEvent) -> None:
        self._session.add(event)

    @override
    async def list_for_workspace(
        self,
        workspace_id: UUID,
        *,
        limit: int,
    ) -> list[SecurityAuditEvent]:
        if limit < 1:
            raise ValueError("Security audit event limit must be positive")
        result = await self._session.scalars(
            select(SecurityAuditEvent)
            .where(schema.security_audit_events.c.workspace_id == workspace_id)
            .order_by(schema.security_audit_events.c.occurred_at.desc())
            .limit(limit)
        )
        return list(result)

    @override
    async def delete_before(self, occurred_before: datetime) -> int:
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(schema.security_audit_events).where(
                    schema.security_audit_events.c.occurred_at < occurred_before
                )
            ),
        )
        return result.rowcount


class SqlSavedGraphRepository(SavedGraphRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, graph: SavedGraph) -> None:
        self._session.add(graph)

    @override
    async def add_revision(self, revision: SavedGraphRevision) -> None:
        self._session.add(
            SavedGraphRevisionRecord(
                workspace_id=revision.workspace_id,
                graph_id=revision.graph_id,
                revision=revision.revision,
                name=revision.name,
                document=revision.document,
                created_at=revision.created_at,
            ),
        )

    @override
    async def lock_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        expected_revision: int,
    ) -> None:
        table = schema.saved_graphs
        await self._session.execute(
            update(table)
            .where(
                table.c.id == graph_id,
                table.c.workspace_id == workspace_id,
                table.c.revision == expected_revision,
            )
            .values(revision=table.c.revision)
        )

    @override
    async def get(self, workspace_id: UUID, graph_id: UUID) -> SavedGraph | None:
        return await self._session.scalar(
            select(SavedGraph).where(
                schema.saved_graphs.c.workspace_id == workspace_id,
                schema.saved_graphs.c.id == graph_id,
            )
        )

    @override
    async def get_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        revision: int,
    ) -> SavedGraphRevision | None:
        record = await self._session.get(
            SavedGraphRevisionRecord,
            (workspace_id, graph_id, revision),
        )
        if record is None:
            return None
        return SavedGraphRevision(
            graph_id=record.graph_id,
            workspace_id=record.workspace_id,
            revision=record.revision,
            name=record.name,
            document=record.document,
            created_at=record.created_at,
        )

    @override
    async def list_revisions(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> list[SavedGraphRevision]:
        result = await self._session.scalars(
            select(SavedGraphRevisionRecord)
            .where(schema.saved_graph_revisions.c.graph_id == graph_id)
            .where(schema.saved_graph_revisions.c.workspace_id == workspace_id)
            .order_by(schema.saved_graph_revisions.c.revision.desc())
        )
        return [
            SavedGraphRevision(
                graph_id=record.graph_id,
                workspace_id=record.workspace_id,
                revision=record.revision,
                name=record.name,
                document=record.document,
                created_at=record.created_at,
            )
            for record in result
        ]

    @override
    async def list(self, workspace_id: UUID) -> list[SavedGraph]:
        result = await self._session.scalars(
            select(SavedGraph).order_by(
                schema.saved_graphs.c.updated_at.desc(),
                schema.saved_graphs.c.id.asc(),
            )
            .where(schema.saved_graphs.c.workspace_id == workspace_id)
        )
        return list(result)

    @override
    async def remove(self, workspace_id: UUID, graph: SavedGraph) -> None:
        await self._session.execute(
            delete(schema.saved_graphs).where(
                schema.saved_graphs.c.workspace_id == workspace_id,
                schema.saved_graphs.c.id == graph.id,
            )
        )


class SqlArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, artifact: ArtifactObject) -> None:
        self._session.add(artifact)

    @override
    async def get(
        self,
        workspace_id: UUID,
        artifact_id: UUID,
    ) -> ArtifactObject | None:
        return await self._session.scalar(
            select(ArtifactObject).where(
                schema.artifact_objects.c.workspace_id == workspace_id,
                schema.artifact_objects.c.id == artifact_id,
            )
        )

    @override
    async def get_many(
        self,
        workspace_id: UUID,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]:
        if not artifact_ids:
            return {}
        result = await self._session.scalars(
            select(ArtifactObject).where(
                schema.artifact_objects.c.id.in_(set(artifact_ids)),
                schema.artifact_objects.c.workspace_id == workspace_id,
            )
        )
        return {artifact.id: artifact for artifact in result}

    @override
    async def remove(self, workspace_id: UUID, artifact: ArtifactObject) -> None:
        await self._session.execute(
            delete(schema.artifact_objects).where(
                schema.artifact_objects.c.workspace_id == workspace_id,
                schema.artifact_objects.c.id == artifact.id,
            )
        )

    @override
    async def list_by_type(
        self,
        workspace_id: UUID,
        key: ArtifactTypeKey,
    ) -> list[ArtifactObject]:
        result = await self._session.scalars(
            select(ArtifactObject)
            .where(
                schema.artifact_objects.c.artifact_type == key.id,
                schema.artifact_objects.c.schema_version == key.schema_version,
                schema.artifact_objects.c.workspace_id == workspace_id,
            )
            .order_by(schema.artifact_objects.c.id.asc())
        )
        return list(result)


class SqlInvocationCacheRepository(InvocationCacheRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def get(
        self,
        workspace_id: UUID,
        key_sha256: str,
    ) -> InvocationCacheEntry | None:
        return await self._session.get(
            InvocationCacheEntry,
            (workspace_id, key_sha256),
        )

    @override
    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool:
        table = schema.invocation_cache_entries
        dialect_name = self._session.get_bind().dialect.name
        if dialect_name == "sqlite":
            insert_statement = sqlite_insert(table)
        elif dialect_name == "postgresql":
            insert_statement = postgresql_insert(table)
        else:
            raise NotImplementedError(
                "Invocation cache publication requires SQLite or PostgreSQL; "
                f"received dialect {dialect_name!r}"
            )

        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                insert_statement.values(
                    key_sha256=entry.key_sha256,
                    workspace_id=entry.workspace_id,
                    generation=entry.generation,
                    outputs=entry.outputs,
                    created_at=entry.created_at,
                ).on_conflict_do_nothing(
                    index_elements=(table.c.workspace_id, table.c.key_sha256),
                )
            ),
        )
        return result.rowcount == 1

    @override
    async def remove_if_current(
        self,
        workspace_id: UUID,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        table = schema.invocation_cache_entries
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(table).where(
                    table.c.workspace_id == workspace_id,
                    table.c.key_sha256 == key_sha256,
                    table.c.generation == generation,
                )
            ),
        )
        return result.rowcount == 1


class SqlMaterializedNodeOutputsRepository(
    MaterializedNodeOutputsRepositoryPort,
):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def upsert(self, value: MaterializedNodeOutputs) -> None:
        table = schema.materialized_node_outputs
        dialect_name = self._session.get_bind().dialect.name
        if dialect_name == "sqlite":
            insert_statement = sqlite_insert(table)
        elif dialect_name == "postgresql":
            insert_statement = postgresql_insert(table)
        else:
            raise NotImplementedError(
                "Materialized output upsert requires SQLite or PostgreSQL; "
                f"received dialect {dialect_name!r}"
            )

        insert_statement = insert_statement.values(
            workspace_id=value.workspace_id,
            graph_id=value.graph_id,
            graph_revision=value.graph_revision,
            node_id=value.node_id,
            workflow_run_id=value.workflow_run_id,
            outputs=value.outputs,
            materialized_at=value.materialized_at,
        )
        await self._session.execute(
            insert_statement.on_conflict_do_update(
                index_elements=(
                    table.c.workspace_id,
                    table.c.graph_id,
                    table.c.graph_revision,
                    table.c.node_id,
                ),
                set_={
                    "workflow_run_id": insert_statement.excluded.workflow_run_id,
                    "outputs": insert_statement.excluded.outputs,
                    "materialized_at": insert_statement.excluded.materialized_at,
                },
            )
        )

    @override
    async def get(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> MaterializedNodeOutputs | None:
        return await self._session.get(
            MaterializedNodeOutputs,
            (workspace_id, graph_id, graph_revision, node_id),
        )

    @override
    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
    ) -> list[MaterializedNodeOutputs]:
        result = await self._session.scalars(
            select(MaterializedNodeOutputs)
            .where(
                schema.materialized_node_outputs.c.graph_id == graph_id,
                schema.materialized_node_outputs.c.graph_revision == graph_revision,
                schema.materialized_node_outputs.c.workspace_id == workspace_id,
            )
            .order_by(schema.materialized_node_outputs.c.node_id.asc())
        )
        return list(result)


class SqlGraphExecutionHistoryRepository(
    GraphExecutionHistoryRepositoryPort,
):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, execution: GraphExecution) -> None:
        table = schema.graph_executions
        revision_exists = await self._session.scalar(
            select(schema.saved_graph_revisions.c.graph_id).where(
                schema.saved_graph_revisions.c.workspace_id == execution.workspace_id,
                schema.saved_graph_revisions.c.graph_id == execution.graph_id,
                schema.saved_graph_revisions.c.revision == execution.graph_revision,
            )
        )
        if revision_exists is None:
            raise NotFoundError(
                "Saved graph revision",
                f"{execution.graph_id}/r{execution.graph_revision}",
            )
        try:
            await self._session.execute(
                insert(table).values(
                    execution_id=execution.execution_id,
                    workspace_id=execution.workspace_id,
                    graph_id=execution.graph_id,
                    graph_revision=execution.graph_revision,
                    status=execution.status,
                    scope=execution.scope,
                    workflow_run_id=execution.workflow_run_id,
                    error=execution.error,
                    created_at=execution.created_at,
                    started_at=execution.started_at,
                    finished_at=execution.finished_at,
                )
            )
        except IntegrityError as exc:
            raise ObjectAlreadyExistsError(
                f"Graph execution already exists: {execution.execution_id}"
            ) from exc
        if execution.requested_node_ids:
            await self._session.execute(
                insert(schema.graph_execution_requested_nodes),
                [
                    {
                        "workspace_id": execution.workspace_id,
                        "execution_id": execution.execution_id,
                        "node_id": node_id,
                        "position": position,
                    }
                    for position, node_id in enumerate(execution.requested_node_ids)
                ],
            )

    @override
    async def update(self, execution: GraphExecution) -> None:
        current_record = await self._session.scalar(
            select(GraphExecutionRecord).where(
                schema.graph_executions.c.workspace_id == execution.workspace_id,
                schema.graph_executions.c.execution_id == execution.execution_id,
            )
        )
        if current_record is None:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        requested_node_ids = await self._requested_node_ids(
            execution.workspace_id,
            execution.execution_id,
        )
        current = current_record.to_domain(requested_node_ids)
        if (
            current.graph_id != execution.graph_id
            or current.graph_revision != execution.graph_revision
            or current.scope != execution.scope
            or current.requested_node_ids != execution.requested_node_ids
            or current.created_at != execution.created_at
        ):
            raise ValueError(
                f"Graph execution {execution.execution_id} identity and request "
                "fields are immutable"
            )

        await self._session.execute(
            update(schema.graph_executions)
            .where(
                schema.graph_executions.c.workspace_id == execution.workspace_id,
                schema.graph_executions.c.execution_id == execution.execution_id,
            )
            .values(
                status=execution.status,
                workflow_run_id=execution.workflow_run_id,
                error=execution.error,
                started_at=execution.started_at,
                finished_at=execution.finished_at,
            )
        )

    @override
    async def add_node_result(self, result: GraphExecutionNodeResult) -> None:
        execution_exists = await self._session.scalar(
            select(schema.graph_executions.c.execution_id).where(
                schema.graph_executions.c.workspace_id == result.workspace_id,
                schema.graph_executions.c.execution_id == result.execution_id
            )
        )
        if execution_exists is None:
            raise NotFoundError("Graph execution", str(result.execution_id))
        requested_node_exists = await self._session.scalar(
            select(schema.graph_execution_requested_nodes.c.execution_id).where(
                schema.graph_execution_requested_nodes.c.workspace_id
                == result.workspace_id,
                schema.graph_execution_requested_nodes.c.execution_id
                == result.execution_id,
                schema.graph_execution_requested_nodes.c.node_id == result.node_id,
            )
        )
        if requested_node_exists is None:
            raise ValueError(
                f"Graph execution {result.execution_id} did not request node "
                f"{result.node_id!r}"
            )

        table = schema.graph_execution_node_results
        try:
            await self._session.execute(
                insert(table).values(
                    workspace_id=result.workspace_id,
                    execution_id=result.execution_id,
                    node_id=result.node_id,
                    position=result.position,
                    status=result.status,
                    outputs=result.outputs,
                    artifact_count=result.artifact_count,
                    error=result.error,
                    completed_at=result.completed_at,
                )
            )
        except IntegrityError as exc:
            raise ObjectAlreadyExistsError(
                "Graph execution node result already exists: "
                f"{result.execution_id}/{result.node_id}"
            ) from exc

    @override
    async def get(
        self,
        workspace_id: UUID,
        execution_id: UUID,
    ) -> GraphExecutionDetail | None:
        record = await self._session.scalar(
            select(GraphExecutionRecord).where(
                schema.graph_executions.c.workspace_id == workspace_id,
                schema.graph_executions.c.execution_id == execution_id,
            )
        )
        if record is None:
            return None
        execution = record.to_domain(
            await self._requested_node_ids(workspace_id, execution_id)
        )
        results = await self._session.scalars(
            select(GraphExecutionNodeResult)
            .where(schema.graph_execution_node_results.c.execution_id == execution_id)
            .where(schema.graph_execution_node_results.c.workspace_id == workspace_id)
            .order_by(
                schema.graph_execution_node_results.c.position.asc(),
                schema.graph_execution_node_results.c.node_id.asc(),
            )
        )
        return GraphExecutionDetail(
            execution=execution,
            node_results=tuple(results),
        )

    @override
    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        limit: int,
        cursor: GraphExecutionCursor | None = None,
        graph_revision: int | None = None,
        status: GraphExecutionStatus | None = None,
        node_id: str | None = None,
    ) -> GraphExecutionPage:
        if limit < 1:
            raise ValueError("Graph execution page limit must be at least 1")
        if graph_revision is not None and graph_revision < 1:
            raise ValueError("Graph execution revision filter must be at least 1")
        normalized_node_id = None
        if node_id is not None:
            normalized_node_id = node_id.strip()
            if normalized_node_id == "":
                raise ValueError("Graph execution node filter must not be blank")

        executions = schema.graph_executions
        requested_nodes = schema.graph_execution_requested_nodes
        node_results = schema.graph_execution_node_results
        counts = (
            select(
                node_results.c.workspace_id,
                node_results.c.execution_id,
                func.count(node_results.c.node_id).label("node_count"),
                func.coalesce(func.sum(node_results.c.artifact_count), 0).label(
                    "artifact_count"
                ),
            )
            .group_by(
                node_results.c.workspace_id,
                node_results.c.execution_id,
            )
            .subquery()
        )
        statement = (
            select(
                GraphExecutionRecord,
                func.coalesce(counts.c.node_count, 0),
                func.coalesce(counts.c.artifact_count, 0),
            )
            .outerjoin(
                counts,
                and_(
                    counts.c.workspace_id == executions.c.workspace_id,
                    counts.c.execution_id == executions.c.execution_id,
                ),
            )
            .where(
                executions.c.workspace_id == workspace_id,
                executions.c.graph_id == graph_id,
            )
        )
        if graph_revision is not None:
            statement = statement.where(executions.c.graph_revision == graph_revision)
        if status is not None:
            statement = statement.where(executions.c.status == status)
        if normalized_node_id is not None:
            statement = statement.where(
                select(1)
                .where(
                    requested_nodes.c.execution_id == executions.c.execution_id,
                    requested_nodes.c.workspace_id == workspace_id,
                    requested_nodes.c.node_id == normalized_node_id,
                )
                .exists()
            )
        if cursor is not None:
            statement = statement.where(
                or_(
                    executions.c.created_at < cursor.created_at,
                    (
                        (executions.c.created_at == cursor.created_at)
                        & (executions.c.execution_id < cursor.execution_id)
                    ),
                )
            )
        statement = statement.order_by(
            executions.c.created_at.desc(),
            executions.c.execution_id.desc(),
        ).limit(limit + 1)
        rows = list((await self._session.execute(statement)).all())
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        requested_by_execution: dict[UUID, list[tuple[int, str]]] = {}
        execution_ids = [row[0].execution_id for row in page_rows]
        if execution_ids:
            requested_rows = (
                await self._session.execute(
                    select(
                        requested_nodes.c.execution_id,
                        requested_nodes.c.position,
                        requested_nodes.c.node_id,
                    )
                    .where(requested_nodes.c.execution_id.in_(execution_ids))
                    .where(requested_nodes.c.workspace_id == workspace_id)
                    .order_by(
                        requested_nodes.c.execution_id.asc(),
                        requested_nodes.c.position.asc(),
                    )
                )
            ).all()
            for requested_execution_id, position, requested_node_id in requested_rows:
                requested_by_execution.setdefault(requested_execution_id, []).append(
                    (position, requested_node_id)
                )
        items = tuple(
            GraphExecutionListItem(
                execution=row[0].to_domain(
                    tuple(
                        node_id
                        for _, node_id in requested_by_execution.get(
                            row[0].execution_id,
                            [],
                        )
                    )
                ),
                node_count=int(row[1]),
                artifact_count=int(row[2]),
            )
            for row in page_rows
        )
        next_cursor = None
        if has_more and items:
            last = items[-1].execution
            next_cursor = GraphExecutionCursor(
                created_at=last.created_at,
                execution_id=last.execution_id,
            )
        return GraphExecutionPage(items=items, next_cursor=next_cursor)

    @override
    async def interrupt_active(
        self,
        *,
        workspace_id: UUID,
        finished_at: datetime,
        error: str,
    ) -> int:
        if finished_at.tzinfo is None:
            raise ValueError(
                "Graph execution interruption timestamp must be timezone-aware"
            )
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                update(schema.graph_executions)
                .where(
                    schema.graph_executions.c.workspace_id == workspace_id,
                    schema.graph_executions.c.status.in_(
                        ("queued", "running", "cancelling")
                    )
                )
                .values(
                    status="failed",
                    finished_at=finished_at,
                    error=error,
                )
            ),
        )
        return result.rowcount

    @override
    async def interrupt_all_active(
        self,
        *,
        finished_at: datetime,
        error: str,
    ) -> int:
        if finished_at.tzinfo is None:
            raise ValueError(
                "Graph execution interruption timestamp must be timezone-aware"
            )
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                update(schema.graph_executions)
                .where(
                    schema.graph_executions.c.status.in_(
                        ("queued", "running", "cancelling")
                    )
                )
                .values(
                    status="failed",
                    finished_at=finished_at,
                    error=error,
                )
            ),
        )
        return result.rowcount

    async def _requested_node_ids(
        self,
        workspace_id: UUID,
        execution_id: UUID,
    ) -> tuple[str, ...]:
        requested_nodes = schema.graph_execution_requested_nodes
        result = await self._session.scalars(
            select(requested_nodes.c.node_id)
            .where(
                requested_nodes.c.workspace_id == workspace_id,
                requested_nodes.c.execution_id == execution_id,
            )
            .order_by(requested_nodes.c.position.asc())
        )
        return tuple(result)


class SqlNodeSecretRepository(NodeSecretRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def upsert(self, secret: EncryptedNodeSecret) -> None:
        await self._session.flush()
        table = schema.node_secrets
        dialect_name = self._session.get_bind().dialect.name
        if dialect_name == "sqlite":
            insert_statement = sqlite_insert(table)
        elif dialect_name == "postgresql":
            insert_statement = postgresql_insert(table)
        else:
            raise NotImplementedError(
                "Node secret upsert requires SQLite or PostgreSQL; "
                f"received dialect {dialect_name!r}"
            )
        insert_statement = insert_statement.values(
            workspace_id=secret.workspace_id,
            graph_id=secret.graph_id,
            node_id=secret.node_id,
            name=secret.name,
            operator_id=secret.operator_id,
            operator_version=secret.operator_version,
            key_id=secret.key_id,
            aad_version=secret.aad_version,
            dependency_sha256=secret.dependency_sha256,
            nonce=secret.nonce,
            ciphertext=secret.ciphertext,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )
        await self._session.execute(
            insert_statement.on_conflict_do_update(
                index_elements=(
                    table.c.workspace_id,
                    table.c.graph_id,
                    table.c.node_id,
                    table.c.name,
                ),
                set_={
                    "operator_id": insert_statement.excluded.operator_id,
                    "operator_version": insert_statement.excluded.operator_version,
                    "key_id": insert_statement.excluded.key_id,
                    "aad_version": insert_statement.excluded.aad_version,
                    "dependency_sha256": (insert_statement.excluded.dependency_sha256),
                    "nonce": insert_statement.excluded.nonce,
                    "ciphertext": insert_statement.excluded.ciphertext,
                    "updated_at": insert_statement.excluded.updated_at,
                },
            )
        )

    @override
    async def get(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        node_id: str,
        name: str,
    ) -> EncryptedNodeSecret | None:
        return await self._session.get(
            EncryptedNodeSecret,
            (workspace_id, graph_id, node_id, name),
        )

    @override
    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> list[EncryptedNodeSecret]:
        result = await self._session.scalars(
            select(EncryptedNodeSecret)
            .where(schema.node_secrets.c.graph_id == graph_id)
            .where(schema.node_secrets.c.workspace_id == workspace_id)
            .order_by(
                schema.node_secrets.c.node_id.asc(),
                schema.node_secrets.c.name.asc(),
            )
        )
        return list(result)

    @override
    async def remove(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        node_id: str,
        name: str,
    ) -> None:
        await self._session.execute(
            delete(schema.node_secrets).where(
                schema.node_secrets.c.workspace_id == workspace_id,
                schema.node_secrets.c.graph_id == graph_id,
                schema.node_secrets.c.node_id == node_id,
                schema.node_secrets.c.name == name,
            )
        )


class SqlStagedUploadRepository(StagedUploadRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, upload: StagedUpload) -> None:
        self._session.add(upload)

    @override
    async def get(
        self,
        workspace_id: UUID,
        upload_key: str,
    ) -> StagedUpload | None:
        return await self._session.get(StagedUpload, (workspace_id, upload_key))

    @override
    async def list_for_workspace(self, workspace_id: UUID) -> list[StagedUpload]:
        result = await self._session.scalars(
            select(StagedUpload)
            .where(schema.staged_uploads.c.workspace_id == workspace_id)
            .order_by(
                schema.staged_uploads.c.created_at.asc(),
                schema.staged_uploads.c.upload_key.asc(),
            )
        )
        return list(result)

    @override
    async def remove(self, workspace_id: UUID, upload_key: str) -> None:
        await self._session.execute(
            delete(schema.staged_uploads).where(
                schema.staged_uploads.c.workspace_id == workspace_id,
                schema.staged_uploads.c.upload_key == upload_key,
            )
        )

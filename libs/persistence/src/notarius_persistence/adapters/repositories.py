from collections.abc import Collection
from typing import cast, override
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRepositoryPort,
    ArtifactTypeKey,
)
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.node_secrets import EncryptedNodeSecret
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphRevision
from notarius_core.ports.invocation_cache import InvocationCacheRepositoryPort
from notarius_core.ports.materialized_outputs import (
    MaterializedNodeOutputsRepositoryPort,
)
from notarius_core.ports.node_secrets import NodeSecretRepositoryPort
from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort

from notarius_persistence import schema
from notarius_persistence.orm import SavedGraphRevisionRecord


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
                graph_id=revision.graph_id,
                revision=revision.revision,
                name=revision.name,
                document=revision.document,
                created_at=revision.created_at,
            ),
        )

    @override
    async def lock_revision(self, graph_id: UUID, expected_revision: int) -> None:
        table = schema.saved_graphs
        await self._session.execute(
            update(table)
            .where(
                table.c.id == graph_id,
                table.c.revision == expected_revision,
            )
            .values(revision=table.c.revision)
        )

    @override
    async def get(self, graph_id: UUID) -> SavedGraph | None:
        return await self._session.get(SavedGraph, graph_id)

    @override
    async def get_revision(
        self,
        graph_id: UUID,
        revision: int,
    ) -> SavedGraphRevision | None:
        record = await self._session.get(
            SavedGraphRevisionRecord,
            (graph_id, revision),
        )
        if record is None:
            return None
        return SavedGraphRevision(
            graph_id=record.graph_id,
            revision=record.revision,
            name=record.name,
            document=record.document,
            created_at=record.created_at,
        )

    @override
    async def list_revisions(self, graph_id: UUID) -> list[SavedGraphRevision]:
        result = await self._session.scalars(
            select(SavedGraphRevisionRecord)
            .where(schema.saved_graph_revisions.c.graph_id == graph_id)
            .order_by(schema.saved_graph_revisions.c.revision.desc())
        )
        return [
            SavedGraphRevision(
                graph_id=record.graph_id,
                revision=record.revision,
                name=record.name,
                document=record.document,
                created_at=record.created_at,
            )
            for record in result
        ]

    @override
    async def list(self) -> list[SavedGraph]:
        result = await self._session.scalars(
            select(SavedGraph).order_by(
                schema.saved_graphs.c.updated_at.desc(),
                schema.saved_graphs.c.id.asc(),
            )
        )
        return list(result)

    @override
    async def remove(self, graph: SavedGraph) -> None:
        await self._session.delete(graph)


class SqlArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, artifact: ArtifactObject) -> None:
        self._session.add(artifact)

    @override
    async def get(self, artifact_id: UUID) -> ArtifactObject | None:
        return await self._session.get(ArtifactObject, artifact_id)

    @override
    async def get_many(
        self,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]:
        if not artifact_ids:
            return {}
        result = await self._session.scalars(
            select(ArtifactObject).where(
                schema.artifact_objects.c.id.in_(set(artifact_ids))
            )
        )
        return {artifact.id: artifact for artifact in result}

    @override
    async def remove(self, artifact: ArtifactObject) -> None:
        await self._session.delete(artifact)

    @override
    async def list_by_type(self, key: ArtifactTypeKey) -> list[ArtifactObject]:
        result = await self._session.scalars(
            select(ArtifactObject)
            .where(
                schema.artifact_objects.c.artifact_type == key.id,
                schema.artifact_objects.c.schema_version == key.schema_version,
            )
            .order_by(schema.artifact_objects.c.id.asc())
        )
        return list(result)


class SqlInvocationCacheRepository(InvocationCacheRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def get(self, key_sha256: str) -> InvocationCacheEntry | None:
        return await self._session.get(InvocationCacheEntry, key_sha256)

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
                    generation=entry.generation,
                    outputs=entry.outputs,
                    created_at=entry.created_at,
                ).on_conflict_do_nothing(
                    index_elements=(table.c.key_sha256,),
                )
            ),
        )
        return result.rowcount == 1

    @override
    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        table = schema.invocation_cache_entries
        result = cast(
            CursorResult[tuple[object, ...]],
            await self._session.execute(
                delete(table).where(
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
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> MaterializedNodeOutputs | None:
        return await self._session.get(
            MaterializedNodeOutputs,
            (graph_id, graph_revision, node_id),
        )

    @override
    async def list_for_graph(
        self,
        graph_id: UUID,
        graph_revision: int,
    ) -> list[MaterializedNodeOutputs]:
        result = await self._session.scalars(
            select(MaterializedNodeOutputs)
            .where(
                schema.materialized_node_outputs.c.graph_id == graph_id,
                schema.materialized_node_outputs.c.graph_revision == graph_revision,
            )
            .order_by(schema.materialized_node_outputs.c.node_id.asc())
        )
        return list(result)


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
            graph_id=secret.graph_id,
            node_id=secret.node_id,
            name=secret.name,
            operator_id=secret.operator_id,
            operator_version=secret.operator_version,
            key_id=secret.key_id,
            dependency_sha256=secret.dependency_sha256,
            nonce=secret.nonce,
            ciphertext=secret.ciphertext,
            created_at=secret.created_at,
            updated_at=secret.updated_at,
        )
        await self._session.execute(
            insert_statement.on_conflict_do_update(
                index_elements=(table.c.graph_id, table.c.node_id, table.c.name),
                set_={
                    "operator_id": insert_statement.excluded.operator_id,
                    "operator_version": insert_statement.excluded.operator_version,
                    "key_id": insert_statement.excluded.key_id,
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
        graph_id: UUID,
        node_id: str,
        name: str,
    ) -> EncryptedNodeSecret | None:
        return await self._session.get(
            EncryptedNodeSecret,
            (graph_id, node_id, name),
        )

    @override
    async def list_for_graph(self, graph_id: UUID) -> list[EncryptedNodeSecret]:
        result = await self._session.scalars(
            select(EncryptedNodeSecret)
            .where(schema.node_secrets.c.graph_id == graph_id)
            .order_by(
                schema.node_secrets.c.node_id.asc(),
                schema.node_secrets.c.name.asc(),
            )
        )
        return list(result)

    @override
    async def remove(self, graph_id: UUID, node_id: str, name: str) -> None:
        await self._session.execute(
            delete(schema.node_secrets).where(
                schema.node_secrets.c.graph_id == graph_id,
                schema.node_secrets.c.node_id == node_id,
                schema.node_secrets.c.name == name,
            )
        )

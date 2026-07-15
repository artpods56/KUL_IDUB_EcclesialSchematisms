from typing import override
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRepositoryPort,
    ArtifactTypeKey,
)
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.saved_graphs import SavedGraph
from notarius_core.ports.materialized_outputs import (
    MaterializedNodeOutputsRepositoryPort,
)
from notarius_core.ports.saved_graphs import SavedGraphRepositoryPort

from notarius_persistence import schema


class SqlSavedGraphRepository(SavedGraphRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @override
    async def add(self, graph: SavedGraph) -> None:
        self._session.add(graph)

    @override
    async def get(self, graph_id: UUID) -> SavedGraph | None:
        return await self._session.get(SavedGraph, graph_id)

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

from datetime import UTC, datetime
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
)
from sqlalchemy import Uuid as SaUuid
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from notarius_core.domain.materialized_outputs import (
    MaterializedNodeOutputs,
    MaterializedOutputValue,
)
from notarius_core.domain.saved_graphs import SavedGraphDocument


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class SavedGraphDocumentType(TypeDecorator[SavedGraphDocument]):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: SavedGraphDocument | None,
        dialect: Dialect,
    ) -> dict[str, object] | None:
        del dialect
        if value is None:
            return None
        return value.model_dump(mode="json")

    def process_result_value(
        self,
        value: object | None,
        dialect: Dialect,
    ) -> SavedGraphDocument | None:
        del dialect
        if value is None:
            return None
        return SavedGraphDocument.model_validate(value)


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UTCDateTime requires a timezone-aware datetime")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class MaterializedOutputsType(
    TypeDecorator[dict[str, MaterializedOutputValue]],
):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: dict[str, MaterializedOutputValue] | None,
        dialect: Dialect,
    ) -> list[dict[str, object]] | None:
        del dialect
        if value is None:
            return None
        return MaterializedNodeOutputs.outputs_to_storage(value)

    def process_result_value(
        self,
        value: object | None,
        dialect: Dialect,
    ) -> dict[str, MaterializedOutputValue] | None:
        del dialect
        if value is None:
            return None
        return MaterializedNodeOutputs.outputs_from_storage(value)


saved_graphs = Table(
    "saved_graphs",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("name", String(160), nullable=False),
    Column("document", SavedGraphDocumentType(), nullable=False),
    Column("revision", Integer, nullable=False, default=1),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    Index("ix_saved_graphs_updated_at", "updated_at"),
)


artifact_objects = Table(
    "artifact_objects",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("artifact_type", String(255), nullable=False),
    Column("schema_version", Integer, nullable=False),
    Column("content_type", String(255), nullable=False),
    Column("storage_backend", String(40), nullable=False),
    Column("bucket", String(255), nullable=True),
    Column("object_key", String(2048), nullable=True),
    Column("inline_payload", JSON, nullable=True),
    Column("byte_size", BigInteger, nullable=True),
    Column("sha256", String(64), nullable=True),
    Column("metadata", JSON, nullable=False),
    Index("ix_artifact_objects_type", "artifact_type", "schema_version"),
    Index("ix_artifact_objects_sha256", "sha256"),
)


materialized_node_outputs = Table(
    "materialized_node_outputs",
    metadata,
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        ForeignKey("saved_graphs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("graph_revision", Integer, primary_key=True),
    Column("node_id", String(255), primary_key=True),
    Column("workflow_run_id", SaUuid(as_uuid=True), nullable=False),
    Column("outputs", MaterializedOutputsType(), nullable=False),
    Column("materialized_at", UTCDateTime(), nullable=False),
    Index(
        "ix_materialized_node_outputs_graph_revision",
        "graph_id",
        "graph_revision",
        "materialized_at",
    ),
)

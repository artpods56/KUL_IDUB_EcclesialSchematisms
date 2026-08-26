from datetime import UTC, datetime
from typing import cast
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Uuid as SaUuid
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from grafy_core.domain.artifact_outputs import (
    ArtifactOutputValue,
    artifact_outputs_from_storage,
    artifact_outputs_to_storage,
)
from grafy_core.domain.saved_graphs import SavedGraphDocument
from grafy_core.domain.identity import (
    WorkspaceCapability,
    WorkspaceInvitationStatus,
    WorkspaceKind,
    WorkspaceRole,
)
from grafy_core.domain.module_library import ModulePublicationState
from grafy_core.domain.plugin_releases import (
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginDistribution,
    PluginExecutionPolicy,
    PluginReleaseScope,
    PluginRuntimeArtifact,
)
from grafy_core.domain.plugin_selection import PluginFamilyLifecycle
from grafy_core.domain.plugin_revocations import PluginReleaseRevocationReason
from grafy_core.domain.templates import TemplateState
from grafy_core.domain.security_audit import (
    SecurityAuditActorKind,
    SecurityAuditOutcome,
)


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


class PluginCatalogManifestType(TypeDecorator[PluginCatalogManifest]):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginCatalogManifest | None,
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
    ) -> PluginCatalogManifest | None:
        del dialect
        if value is None:
            return None
        return PluginCatalogManifest.model_validate(value)


class PluginCapabilityManifestType(TypeDecorator[PluginCapabilityManifest]):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginCapabilityManifest | None,
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
    ) -> PluginCapabilityManifest | None:
        del dialect
        if value is None:
            return None
        return PluginCapabilityManifest.model_validate(value)


class PluginRuntimeArtifactType(TypeDecorator[PluginRuntimeArtifact]):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginRuntimeArtifact | None,
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
    ) -> PluginRuntimeArtifact | None:
        del dialect
        if value is None:
            return None
        return PluginRuntimeArtifact.model_validate(value)


class PluginReleaseScopeType(TypeDecorator[PluginReleaseScope]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginReleaseScope | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else PluginReleaseScope(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> PluginReleaseScope | None:
        del dialect
        return None if value is None else PluginReleaseScope(value)


class PluginReleaseRevocationReasonType(
    TypeDecorator[PluginReleaseRevocationReason]
):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginReleaseRevocationReason | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        if value is None:
            return None
        return PluginReleaseRevocationReason(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> PluginReleaseRevocationReason | None:
        del dialect
        if value is None:
            return None
        return PluginReleaseRevocationReason(value)


class PluginExecutionPolicyType(TypeDecorator[PluginExecutionPolicy]):
    impl = String(24)
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginExecutionPolicy | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else PluginExecutionPolicy(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> PluginExecutionPolicy | None:
        del dialect
        return None if value is None else PluginExecutionPolicy(value)


class PluginDistributionType(TypeDecorator[PluginDistribution]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginDistribution | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else PluginDistribution(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> PluginDistribution | None:
        del dialect
        return None if value is None else PluginDistribution(value)


class PluginFamilyLifecycleType(TypeDecorator[PluginFamilyLifecycle]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: PluginFamilyLifecycle | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else PluginFamilyLifecycle(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> PluginFamilyLifecycle | None:
        del dialect
        return None if value is None else PluginFamilyLifecycle(value)


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


class ArtifactOutputsType(
    TypeDecorator[dict[str, ArtifactOutputValue]],
):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: dict[str, ArtifactOutputValue] | None,
        dialect: Dialect,
    ) -> list[dict[str, object]] | None:
        del dialect
        if value is None:
            return None
        return artifact_outputs_to_storage(value)

    def process_result_value(
        self,
        value: object | None,
        dialect: Dialect,
    ) -> dict[str, ArtifactOutputValue] | None:
        del dialect
        if value is None:
            return None
        return artifact_outputs_from_storage(value)


class WorkspaceCapabilityTupleType(TypeDecorator[tuple[WorkspaceCapability, ...]]):
    impl = JSON
    cache_ok = True

    def process_bind_param(
        self,
        value: tuple[WorkspaceCapability, ...] | None,
        dialect: Dialect,
    ) -> list[str] | None:
        del dialect
        return None if value is None else [capability.value for capability in value]

    def process_result_value(
        self,
        value: object | None,
        dialect: Dialect,
    ) -> tuple[WorkspaceCapability, ...] | None:
        del dialect
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("Stored string tuple is not a JSON string list")
        items: list[str] = []
        for item in cast(list[object], value):
            if not isinstance(item, str):
                raise ValueError("Stored string tuple is not a JSON string list")
            items.append(item)
        try:
            return tuple(WorkspaceCapability(item) for item in items)
        except ValueError as exc:
            raise ValueError("Stored workspace capability is unknown") from exc


class WorkspaceKindType(TypeDecorator[WorkspaceKind]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: WorkspaceKind | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else WorkspaceKind(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> WorkspaceKind | None:
        del dialect
        return None if value is None else WorkspaceKind(value)


class WorkspaceRoleType(TypeDecorator[WorkspaceRole]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: WorkspaceRole | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else WorkspaceRole(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> WorkspaceRole | None:
        del dialect
        return None if value is None else WorkspaceRole(value)


class WorkspaceInvitationStatusType(TypeDecorator[WorkspaceInvitationStatus]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: WorkspaceInvitationStatus | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else WorkspaceInvitationStatus(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> WorkspaceInvitationStatus | None:
        del dialect
        return None if value is None else WorkspaceInvitationStatus(value)


class ModulePublicationStateType(TypeDecorator[ModulePublicationState]):
    impl = String(32)
    cache_ok = True

    def process_bind_param(
        self,
        value: ModulePublicationState | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else ModulePublicationState(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> ModulePublicationState | None:
        del dialect
        return None if value is None else ModulePublicationState(value)


class TemplateStateType(TypeDecorator[TemplateState]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: TemplateState | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else TemplateState(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> TemplateState | None:
        del dialect
        return None if value is None else TemplateState(value)


class SecurityAuditActorKindType(TypeDecorator[SecurityAuditActorKind]):
    impl = String(24)
    cache_ok = True

    def process_bind_param(
        self,
        value: SecurityAuditActorKind | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else SecurityAuditActorKind(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> SecurityAuditActorKind | None:
        del dialect
        return None if value is None else SecurityAuditActorKind(value)


class SecurityAuditOutcomeType(TypeDecorator[SecurityAuditOutcome]):
    impl = String(16)
    cache_ok = True

    def process_bind_param(
        self,
        value: SecurityAuditOutcome | None,
        dialect: Dialect,
    ) -> str | None:
        del dialect
        return None if value is None else SecurityAuditOutcome(value).value

    def process_result_value(
        self,
        value: str | None,
        dialect: Dialect,
    ) -> SecurityAuditOutcome | None:
        del dialect
        return None if value is None else SecurityAuditOutcome(value)


graph_folders = Table(
    "graph_folders",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String(160), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("workspace_id", "id", name="uq_graph_folders_workspace_id_id"),
    UniqueConstraint(
        "workspace_id",
        "name",
        name="uq_graph_folders_workspace_id_name",
    ),
    Index("ix_graph_folders_workspace_name", "workspace_id", "name"),
)


saved_graphs = Table(
    "saved_graphs",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "created_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("name", String(160), nullable=False),
    Column("document", SavedGraphDocumentType(), nullable=False),
    Column("revision", Integer, nullable=False, default=1),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("workspace_id", "id", name="uq_saved_graphs_workspace_id_id"),
    Index("ix_saved_graphs_workspace_updated_at", "workspace_id", "updated_at"),
    Index("ix_saved_graphs_workspace_id", "workspace_id", "id"),
)


saved_graph_revisions = Table(
    "saved_graph_revisions",
    metadata,
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column("revision", Integer, primary_key=True),
    Column("name", String(160), nullable=False),
    Column("document", SavedGraphDocumentType(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        ondelete="CASCADE",
    ),
    Index(
        "ix_saved_graph_revisions_workspace_graph_revision",
        "workspace_id",
        "graph_id",
        "revision",
    ),
)


graph_organizations = Table(
    "graph_organizations",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), primary_key=True),
    Column("folder_id", SaUuid(as_uuid=True), nullable=True),
    Column("archived_at", UTCDateTime(), nullable=True),
    Column("updated_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ("workspace_id", "folder_id"),
        ("graph_folders.workspace_id", "graph_folders.id"),
        ondelete="RESTRICT",
    ),
    Index(
        "ix_graph_organizations_workspace_folder_archived",
        "workspace_id",
        "folder_id",
        "archived_at",
    ),
)


user_graph_states = Table(
    "user_graph_states",
    metadata,
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column("graph_id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("starred", Boolean, nullable=False, default=False),
    Column("last_opened_at", UTCDateTime(), nullable=True),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        ondelete="CASCADE",
    ),
    Index(
        "ix_user_graph_states_user_starred",
        "user_id",
        "starred",
    ),
    Index(
        "ix_user_graph_states_user_last_opened",
        "user_id",
        "last_opened_at",
    ),
)


artifact_objects = Table(
    "artifact_objects",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    ),
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
    UniqueConstraint("workspace_id", "id", name="uq_artifact_objects_workspace_id_id"),
    Index(
        "ix_artifact_objects_workspace_type",
        "workspace_id",
        "artifact_type",
        "schema_version",
    ),
    Index("ix_artifact_objects_workspace_sha256", "workspace_id", "sha256"),
)


invocation_cache_entries = Table(
    "invocation_cache_entries",
    metadata,
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("key_sha256", String(64), primary_key=True),
    Column("generation", SaUuid(as_uuid=True), nullable=False),
    Column("outputs", ArtifactOutputsType(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
)


materialized_node_outputs = Table(
    "materialized_node_outputs",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column("graph_revision", Integer, primary_key=True),
    Column("node_id", String(255), primary_key=True),
    Column("workflow_run_id", SaUuid(as_uuid=True), nullable=False),
    Column("outputs", ArtifactOutputsType(), nullable=False),
    Column("materialized_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id", "graph_revision"),
        (
            "saved_graph_revisions.workspace_id",
            "saved_graph_revisions.graph_id",
            "saved_graph_revisions.revision",
        ),
        ondelete="CASCADE",
    ),
    Index(
        "ix_materialized_node_outputs_graph_revision",
        "workspace_id",
        "graph_id",
        "graph_revision",
        "materialized_at",
    ),
)


graph_executions = Table(
    "graph_executions",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), nullable=False),
    Column("execution_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), nullable=False),
    Column("graph_revision", Integer, nullable=False),
    Column("status", String(24), nullable=False),
    Column("scope", String(32), nullable=False),
    Column("submitted_request", JSON, nullable=True),
    Column("idempotency_key", String(255), nullable=True),
    Column("submitted_by_actor_id", SaUuid(as_uuid=True), nullable=True),
    Column("workflow_run_id", SaUuid(as_uuid=True), nullable=True),
    Column("error", Text, nullable=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("started_at", UTCDateTime(), nullable=True),
    Column("finished_at", UTCDateTime(), nullable=True),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id", "graph_revision"),
        (
            "saved_graph_revisions.workspace_id",
            "saved_graph_revisions.graph_id",
            "saved_graph_revisions.revision",
        ),
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "workspace_id",
        "execution_id",
        name="uq_graph_executions_workspace_id_execution_id",
    ),
    UniqueConstraint(
        "workspace_id",
        "idempotency_key",
        name="uq_graph_executions_workspace_idempotency_key",
    ),
    Index(
        "ix_graph_executions_graph_created",
        "workspace_id",
        "graph_id",
        "created_at",
        "execution_id",
    ),
    Index(
        "ix_graph_executions_graph_revision_created",
        "workspace_id",
        "graph_id",
        "graph_revision",
        "created_at",
        "execution_id",
    ),
    Index("ix_graph_executions_workspace_status", "workspace_id", "status"),
    Index(
        "ix_graph_executions_queue_order",
        "status",
        "created_at",
        "execution_id",
    ),
    Index(
        "uq_graph_executions_one_active_per_graph",
        "workspace_id",
        "graph_id",
        unique=True,
        sqlite_where=text("status IN ('queued', 'running', 'cancelling')"),
        postgresql_where=text("status IN ('queued', 'running', 'cancelling')"),
    ),
)


graph_execution_nodes = Table(
    "graph_execution_nodes",
    metadata,
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column(
        "execution_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column("node_id", String(255), primary_key=True),
    Column("position", Integer, nullable=False),
    Column("result_status", String(16), nullable=True),
    Column("result_position", Integer, nullable=True),
    Column("outputs", ArtifactOutputsType(), nullable=True),
    Column("artifact_count", Integer, nullable=True),
    Column("error", Text, nullable=True),
    Column("diagnostics", JSON, nullable=True),
    Column("completed_at", UTCDateTime(), nullable=True),
    ForeignKeyConstraint(
        ("workspace_id", "execution_id"),
        ("graph_executions.workspace_id", "graph_executions.execution_id"),
        name="fk_exec_nodes_workspace_execution",
        ondelete="CASCADE",
    ),
    CheckConstraint(
        "(result_status IS NULL AND result_position IS NULL AND outputs IS NULL "
        "AND artifact_count IS NULL AND diagnostics IS NULL "
        "AND completed_at IS NULL) OR "
        "(result_status IN ('succeeded', 'failed', 'skipped') "
        "AND result_position IS NOT NULL AND outputs IS NOT NULL "
        "AND artifact_count IS NOT NULL AND artifact_count >= 0 "
        "AND completed_at IS NOT NULL)",
        name="ck_graph_execution_nodes_result_shape",
    ),
    UniqueConstraint(
        "workspace_id",
        "execution_id",
        "position",
        name="uq_graph_execution_nodes_execution_position",
    ),
    Index(
        "uq_graph_execution_nodes_execution_result_position",
        "workspace_id",
        "execution_id",
        "result_position",
        unique=True,
        sqlite_where=text("result_position IS NOT NULL"),
        postgresql_where=text("result_position IS NOT NULL"),
    ),
    Index(
        "ix_graph_execution_nodes_node_execution",
        "workspace_id",
        "node_id",
        "execution_id",
    ),
)


node_secrets = Table(
    "node_secrets",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        primary_key=True,
    ),
    Column("node_id", String(255), primary_key=True),
    Column("name", String(255), primary_key=True),
    Column("operator_id", String(255), nullable=False),
    Column("operator_version", Integer, nullable=False),
    Column("key_id", String(64), nullable=False),
    Column("aad_version", Integer, nullable=False),
    Column("dependency_sha256", String(64), nullable=False),
    Column("nonce", LargeBinary(12), nullable=False),
    Column("ciphertext", LargeBinary(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        ondelete="CASCADE",
    ),
    CheckConstraint("aad_version IN (1, 2)", name="ck_node_secrets_aad_version"),
    Index("ix_node_secrets_workspace_graph", "workspace_id", "graph_id"),
)


staged_uploads = Table(
    "staged_uploads",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("upload_key", String(1024), primary_key=True),
    Column(
        "created_by_user_id",
        SaUuid(as_uuid=True),
        nullable=True,
    ),
    Column("original_filename", String(255), nullable=False),
    Column("byte_size", BigInteger, nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id",),
        ("workspaces.id",),
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ("created_by_user_id",),
        ("users.id",),
        ondelete="SET NULL",
    ),
    CheckConstraint("byte_size >= 0", name="ck_staged_uploads_byte_size_nonnegative"),
    CheckConstraint(
        "length(original_filename) BETWEEN 1 AND 255",
        name="ck_staged_uploads_original_filename_bounded",
    ),
    CheckConstraint(
        "length(upload_key) BETWEEN 1 AND 1024",
        name="ck_staged_uploads_upload_key_bounded",
    ),
    CheckConstraint(
        "upload_key NOT IN ('.', '..')",
        name="ck_staged_uploads_upload_key_not_dot_path",
    ),
    CheckConstraint(
        "upload_key NOT LIKE '%/%'",
        name="ck_staged_uploads_upload_key_no_slash",
    ),
    CheckConstraint(
        "instr(upload_key, char(92)) = 0",
        name="ck_staged_uploads_upload_key_no_backslash",
    ).ddl_if(dialect="sqlite"),
    CheckConstraint(
        "position(chr(92) in upload_key) = 0",
        name="ck_staged_uploads_upload_key_no_backslash",
    ).ddl_if(dialect="postgresql"),
    CheckConstraint(
        "instr(upload_key, char(0)) = 0",
        name="ck_staged_uploads_upload_key_no_nul",
    ).ddl_if(dialect="sqlite"),
    Index("ix_staged_uploads_workspace_created_at", "workspace_id", "created_at"),
)


users = Table(
    "users",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("email", String(320), nullable=True),
    Column("normalized_email", String(320), nullable=True),
    Column("email_verified", Boolean, nullable=False, default=False),
    Column("display_name", String(160), nullable=True),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    Index("ix_users_active_updated_at", "active", "updated_at"),
    Index(
        "ix_users_invitation_email",
        "normalized_email",
        "email_verified",
        "active",
    ),
)


oidc_identities = Table(
    "oidc_identities",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("issuer", String(2048), nullable=False),
    Column("subject", String(512), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("issuer", "subject", name="uq_oidc_identities_issuer_subject"),
    Index("ix_oidc_identities_user_id", "user_id"),
)


oidc_login_transactions = Table(
    "oidc_login_transactions",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("state_digest", LargeBinary(64), nullable=False),
    Column("nonce_digest", LargeBinary(64), nullable=False),
    Column("encrypted_pkce_verifier", LargeBinary(), nullable=False),
    Column("pkce_key_version", Integer, nullable=False),
    Column("return_path", String(2048), nullable=False),
    Column("expires_at", UTCDateTime(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("consumed_at", UTCDateTime(), nullable=True),
    CheckConstraint(
        "pkce_key_version >= 1",
        name="ck_oidc_login_transactions_pkce_key_version_positive",
    ),
    Index(
        "ix_oidc_login_transactions_expiry_consumed",
        "expires_at",
        "consumed_at",
    ),
)


workspaces = Table(
    "workspaces",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("slug", String(80), nullable=False),
    Column("name", String(160), nullable=False),
    Column("kind", WorkspaceKindType(), nullable=False),
    Column(
        "personal_owner_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("slug", name="uq_workspaces_slug"),
    UniqueConstraint(
        "personal_owner_user_id",
        name="uq_workspaces_personal_owner_user_id",
    ),
    CheckConstraint(
        "length(slug) BETWEEN 1 AND 80 AND "
        "slug = lower(trim(slug)) AND "
        "slug NOT LIKE '-%' AND slug NOT LIKE '%-'",
        name="ck_workspaces_slug_normalized",
    ),
    CheckConstraint(
        "kind IN ('personal', 'shared')",
        name="ck_workspaces_kind_choice",
    ),
    CheckConstraint(
        "(kind = 'personal' AND personal_owner_user_id IS NOT NULL) OR "
        "(kind = 'shared' AND personal_owner_user_id IS NULL)",
        name="ck_workspaces_personal_owner_shape",
    ),
    Index("ix_workspaces_kind", "kind"),
)


workspace_memberships = Table(
    "workspace_memberships",
    metadata,
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("role", WorkspaceRoleType(), nullable=False),
    Column("authorization_version", BigInteger, nullable=False, default=1),
    Column("revoked_at", UTCDateTime(), nullable=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    CheckConstraint(
        "role IN ('viewer', 'editor', 'owner')",
        name="ck_workspace_memberships_role_choice",
    ),
    CheckConstraint(
        "authorization_version >= 1",
        name="ck_workspace_memberships_authorization_version_positive",
    ),
    Index(
        "ix_workspace_memberships_user_active",
        "user_id",
        "revoked_at",
    ),
    Index(
        "ix_workspace_memberships_workspace_role_active",
        "workspace_id",
        "role",
        "revoked_at",
    ),
)


workspace_invitations = Table(
    "workspace_invitations",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "invitee_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "invited_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("role", WorkspaceRoleType(), nullable=False),
    Column("status", WorkspaceInvitationStatusType(), nullable=False),
    Column("expires_at", UTCDateTime(), nullable=False),
    Column("resolved_at", UTCDateTime(), nullable=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    CheckConstraint(
        "role IN ('viewer', 'editor', 'owner')",
        name="ck_workspace_invitations_role_choice",
    ),
    CheckConstraint(
        "status IN ('pending', 'accepted', 'declined', 'cancelled', 'expired')",
        name="ck_workspace_invitations_status_choice",
    ),
    CheckConstraint(
        "(status = 'pending' AND resolved_at IS NULL) OR "
        "(status != 'pending' AND resolved_at IS NOT NULL)",
        name="ck_workspace_invitations_resolution_shape",
    ),
    CheckConstraint(
        "expires_at > created_at",
        name="ck_workspace_invitations_expiry_after_creation",
    ),
    Index(
        "ix_workspace_invitations_invitee_status_expiry",
        "invitee_user_id",
        "status",
        "expires_at",
    ),
    Index(
        "ix_workspace_invitations_workspace_status_expiry",
        "workspace_id",
        "status",
        "expires_at",
    ),
    Index(
        "uq_workspace_invitations_pending_recipient",
        "workspace_id",
        "invitee_user_id",
        unique=True,
        sqlite_where=text("status = 'pending'"),
        postgresql_where=text("status = 'pending'"),
    ),
)


oidc_bootstrap_owner_mappings = Table(
    "oidc_bootstrap_owner_mappings",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("issuer", String(2048), nullable=False),
    Column("subject", String(512), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("consumed_at", UTCDateTime(), nullable=True),
    UniqueConstraint(
        "workspace_id",
        name="uq_oidc_bootstrap_owner_mappings_workspace_id",
    ),
    Index(
        "ix_oidc_bootstrap_owner_mappings_unconsumed",
        "workspace_id",
        "consumed_at",
    ),
)


auth_sessions = Table(
    "auth_sessions",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("secret_digest", LargeBinary(64), nullable=False),
    Column("csrf_digest", LargeBinary(64), nullable=False),
    Column("expires_at", UTCDateTime(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("last_used_at", UTCDateTime(), nullable=True),
    Column("revoked_at", UTCDateTime(), nullable=True),
    UniqueConstraint("secret_digest", name="uq_auth_sessions_secret_digest"),
    Index("ix_auth_sessions_user_revoked", "user_id", "revoked_at"),
    Index("ix_auth_sessions_expiry_revoked", "expires_at", "revoked_at"),
)


personal_access_tokens = Table(
    "personal_access_tokens",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("public_prefix", String(32), nullable=False),
    Column("secret_digest", LargeBinary(64), nullable=False),
    Column("label", String(160), nullable=False),
    Column("scopes", WorkspaceCapabilityTupleType(), nullable=False),
    Column("expires_at", UTCDateTime(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("last_used_at", UTCDateTime(), nullable=True),
    Column("revoked_at", UTCDateTime(), nullable=True),
    UniqueConstraint("public_prefix", name="uq_personal_access_tokens_public_prefix"),
    UniqueConstraint("secret_digest", name="uq_personal_access_tokens_secret_digest"),
    Index(
        "ix_personal_access_tokens_workspace_revoked",
        "workspace_id",
        "revoked_at",
    ),
    Index(
        "ix_personal_access_tokens_expiry_revoked",
        "expires_at",
        "revoked_at",
    ),
)


security_audit_events = Table(
    "security_audit_events",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("occurred_at", UTCDateTime(), nullable=False),
    Column("actor_kind", SecurityAuditActorKindType(), nullable=False),
    Column("user_id", SaUuid(as_uuid=True), nullable=True),
    Column("credential_reference", String(120), nullable=True),
    Column("workspace_id", SaUuid(as_uuid=True), nullable=True),
    Column("resource_type", String(80), nullable=True),
    Column("resource_id", String(255), nullable=True),
    Column("operation", String(120), nullable=False),
    Column("outcome", SecurityAuditOutcomeType(), nullable=False),
    Column("error_code", String(80), nullable=True),
    CheckConstraint(
        "actor_kind IN ('authenticated', 'unauthenticated', 'system')",
        name="ck_security_audit_events_actor_kind_choice",
    ),
    CheckConstraint(
        "outcome IN ('success', 'failure')",
        name="ck_security_audit_events_outcome_choice",
    ),
    Index(
        "ix_security_audit_events_workspace_occurred_at",
        "workspace_id",
        "occurred_at",
    ),
    Index(
        "ix_security_audit_events_actor_occurred_at",
        "actor_kind",
        "user_id",
        "occurred_at",
    ),
    Index(
        "ix_security_audit_events_operation_occurred_at",
        "operation",
        "occurred_at",
    ),
    Index("ix_security_audit_events_retention", "occurred_at"),
)


collaborative_graph_heads = Table(
    "collaborative_graph_heads",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), primary_key=True),
    Column("room_epoch", SaUuid(as_uuid=True), nullable=False),
    Column("collaboration_sequence", Integer, nullable=False),
    Column("checkpoint_sequence", Integer, nullable=False),
    Column("checkpoint_revision", Integer, nullable=False),
    Column("name", String(160), nullable=False),
    Column("document", SavedGraphDocumentType(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    CheckConstraint(
        "collaboration_sequence >= 0",
        name="ck_collaborative_graph_heads_collaboration_sequence_nonneg",
    ),
    CheckConstraint(
        "checkpoint_sequence >= 0",
        name="ck_collaborative_graph_heads_checkpoint_sequence_nonneg",
    ),
    CheckConstraint(
        "checkpoint_sequence <= collaboration_sequence",
        name="ck_collaborative_graph_heads_checkpoint_lte_head",
    ),
    CheckConstraint(
        "checkpoint_revision >= 1",
        name="ck_collaborative_graph_heads_checkpoint_revision_positive",
    ),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        ondelete="CASCADE",
    ),
    Index(
        "ix_collaborative_graph_heads_workspace_updated_at",
        "workspace_id",
        "updated_at",
    ),
)


graph_command_receipts = Table(
    "graph_command_receipts",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), primary_key=True),
    Column("command_id", SaUuid(as_uuid=True), primary_key=True),
    Column("command_hmac", LargeBinary(64), nullable=False),
    Column("hmac_key_version", Integer, nullable=False),
    Column("actor_kind", String(32), nullable=False),
    Column("actor_user_id", SaUuid(as_uuid=True), nullable=True),
    Column("room_epoch", SaUuid(as_uuid=True), nullable=False),
    Column("accepted_sequence", Integer, nullable=False),
    Column("outcome", String(40), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        (
            "collaborative_graph_heads.workspace_id",
            "collaborative_graph_heads.graph_id",
        ),
        ondelete="CASCADE",
    ),
)


graph_checkpoint_mappings = Table(
    "graph_checkpoint_mappings",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), primary_key=True),
    Column("room_epoch", SaUuid(as_uuid=True), primary_key=True),
    Column("collaboration_sequence", Integer, primary_key=True),
    Column("saved_revision", Integer, nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id"),
        (
            "collaborative_graph_heads.workspace_id",
            "collaborative_graph_heads.graph_id",
        ),
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ("workspace_id", "graph_id", "saved_revision"),
        (
            "saved_graph_revisions.workspace_id",
            "saved_graph_revisions.graph_id",
            "saved_graph_revisions.revision",
        ),
        ondelete="RESTRICT",
    ),
)


modules = Table(
    "modules",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("source_graph_id", SaUuid(as_uuid=True), nullable=False),
    Column("name", String(160), nullable=False),
    Column("description", String(1000), nullable=True),
    Column("publication_state", ModulePublicationStateType(), nullable=False),
    Column("current_library_release", Integer, nullable=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("workspace_id", "id", name="uq_modules_workspace_id_id"),
    UniqueConstraint(
        "workspace_id",
        "source_graph_id",
        name="uq_modules_workspace_source_graph",
    ),
    ForeignKeyConstraint(
        ("workspace_id", "source_graph_id"),
        ("saved_graphs.workspace_id", "saved_graphs.id"),
        name="fk_modules_source_graph_id_saved_graphs",
    ),
    CheckConstraint(
        "publication_state IN ('published', 'deprecated', 'withdrawn')",
        name="module_publication_state",
    ),
    CheckConstraint(
        "current_library_release IS NULL OR current_library_release >= 1",
        name="module_current_library_release",
    ),
    Index("ix_modules_workspace_updated_at", "workspace_id", "updated_at"),
)


module_releases = Table(
    "module_releases",
    metadata,
    Column("workspace_id", SaUuid(as_uuid=True), primary_key=True),
    Column("module_id", SaUuid(as_uuid=True), primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("source_graph_id", SaUuid(as_uuid=True), nullable=False),
    Column("published_at", UTCDateTime(), nullable=False),
    Column(
        "published_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    ),
    ForeignKeyConstraint(
        ("workspace_id", "module_id"),
        ("modules.workspace_id", "modules.id"),
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ("workspace_id", "source_graph_id", "revision"),
        (
            "saved_graph_revisions.workspace_id",
            "saved_graph_revisions.graph_id",
            "saved_graph_revisions.revision",
        ),
        ondelete="RESTRICT",
        name="fk_module_releases_saved_graph_revision",
    ),
    CheckConstraint("revision >= 1", name="module_release_revision"),
    Index(
        "ix_module_releases_workspace_module_revision",
        "workspace_id",
        "module_id",
        "revision",
    ),
)


plugin_releases = Table(
    "plugin_releases",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("scope", PluginReleaseScopeType(), nullable=False),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("slug", String(100), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("catalog", PluginCatalogManifestType(), nullable=False),
    Column("contract_digest", String(64), nullable=True),
    Column("capabilities", PluginCapabilityManifestType(), nullable=False),
    Column("capability_digest", String(64), nullable=False),
    Column("protocol_digest", String(64), nullable=True),
    Column("profile_digest", String(64), nullable=True),
    Column("source_object_key", String(2048), nullable=False),
    Column("source_digest", String(64), nullable=False),
    Column("lock_digest", String(64), nullable=False),
    Column("runtime_profile", String(100), nullable=False),
    Column("runtime_image_digest", String(64), nullable=True),
    Column("runtime_artifact", PluginRuntimeArtifactType(), nullable=True),
    Column("descriptor_digest", String(64), nullable=True),
    Column("execution_policy", PluginExecutionPolicyType(), nullable=False),
    Column("distribution", PluginDistributionType(), nullable=True),
    Column(
        "published_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("published_by_platform_actor", String(255), nullable=True),
    Column("published_at", UTCDateTime(), nullable=False),
    CheckConstraint("revision >= 1", name="plugin_release_revision"),
    CheckConstraint(
        "scope IN ('system', 'workspace')",
        name="plugin_release_scope",
    ),
    CheckConstraint(
        "(scope = 'system' AND workspace_id IS NULL) OR "
        "(scope = 'workspace' AND workspace_id IS NOT NULL)",
        name="plugin_release_scope_workspace",
    ),
    CheckConstraint(
        "execution_policy IN ('host-eligible', 'isolated-only')",
        name="plugin_release_execution_policy",
    ),
    CheckConstraint(
        "(scope = 'system' AND distribution IN "
        "('bundled', 'optional', 'published')) OR "
        "(scope = 'workspace' AND distribution IS NULL "
        "AND execution_policy = 'isolated-only')",
        name="plugin_release_scope_policy",
    ),
    CheckConstraint(
        "(scope = 'system' AND published_by_user_id IS NULL "
        "AND published_by_platform_actor IS NOT NULL "
        "AND length(trim(published_by_platform_actor)) BETWEEN 1 AND 255) OR "
        "(scope = 'workspace' AND published_by_platform_actor IS NULL)",
        name="plugin_release_scope_publisher",
    ),
    CheckConstraint(
        "length(capability_digest) = 64",
        name="plugin_release_capability_digest",
    ),
    CheckConstraint(
        "length(source_digest) = 64",
        name="plugin_release_source_digest",
    ),
    CheckConstraint(
        "length(lock_digest) = 64",
        name="plugin_release_lock_digest",
    ),
    CheckConstraint(
        "runtime_image_digest IS NULL OR length(runtime_image_digest) = 64",
        name="plugin_release_runtime_image_digest",
    ),
    CheckConstraint(
        "descriptor_digest IS NULL OR length(descriptor_digest) = 64",
        name="plugin_release_descriptor_digest",
    ),
    CheckConstraint(
        "contract_digest IS NULL OR length(contract_digest) = 64",
        name="plugin_release_contract_digest",
    ),
    CheckConstraint(
        "protocol_digest IS NULL OR length(protocol_digest) = 64",
        name="plugin_release_protocol_digest",
    ),
    CheckConstraint(
        "profile_digest IS NULL OR length(profile_digest) = 64",
        name="plugin_release_profile_digest",
    ),
    Index(
        "uq_plugin_releases_system_slug_revision",
        "slug",
        "revision",
        unique=True,
        sqlite_where=text("scope = 'system'"),
        postgresql_where=text("scope = 'system'"),
    ),
    Index(
        "uq_plugin_releases_workspace_slug_revision",
        "workspace_id",
        "slug",
        "revision",
        unique=True,
        sqlite_where=text("scope = 'workspace'"),
        postgresql_where=text("scope = 'workspace'"),
    ),
    Index(
        "uq_plugin_releases_system_slug_descriptor",
        "slug",
        "descriptor_digest",
        unique=True,
        sqlite_where=text("scope = 'system' AND descriptor_digest IS NOT NULL"),
        postgresql_where=text("scope = 'system' AND descriptor_digest IS NOT NULL"),
    ),
    Index(
        "uq_plugin_releases_workspace_slug_descriptor",
        "workspace_id",
        "slug",
        "descriptor_digest",
        unique=True,
        sqlite_where=text("scope = 'workspace' AND descriptor_digest IS NOT NULL"),
        postgresql_where=text("scope = 'workspace' AND descriptor_digest IS NOT NULL"),
    ),
)


plugin_release_selections = Table(
    "plugin_release_selections",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("scope", PluginReleaseScopeType(), nullable=False),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("slug", String(100), nullable=False),
    Column(
        "selected_release_id",
        SaUuid(as_uuid=True),
        ForeignKey("plugin_releases.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("selected_revision", Integer, nullable=False),
    Column("lifecycle", PluginFamilyLifecycleType(), nullable=False),
    Column("generation", Integer, nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    Column("updated_by_actor", String(255), nullable=True),
    CheckConstraint(
        "scope IN ('system', 'workspace')",
        name="plugin_release_selection_scope",
    ),
    CheckConstraint(
        "(scope = 'system' AND workspace_id IS NULL) OR "
        "(scope = 'workspace' AND workspace_id IS NOT NULL)",
        name="plugin_release_selection_scope_workspace",
    ),
    CheckConstraint(
        "selected_revision >= 1",
        name="plugin_release_selection_revision",
    ),
    CheckConstraint(
        "generation >= 1",
        name="plugin_release_selection_generation",
    ),
    CheckConstraint(
        "lifecycle IN ('published', 'deprecated', 'withdrawn')",
        name="plugin_release_selection_lifecycle",
    ),
    CheckConstraint(
        "updated_by_actor IS NULL OR length(trim(updated_by_actor)) BETWEEN 1 AND 255",
        name="plugin_release_selection_actor",
    ),
    Index(
        "uq_plugin_release_selections_system_slug",
        "slug",
        unique=True,
        sqlite_where=text("scope = 'system'"),
        postgresql_where=text("scope = 'system'"),
    ),
    Index(
        "uq_plugin_release_selections_workspace_slug",
        "workspace_id",
        "slug",
        unique=True,
        sqlite_where=text("scope = 'workspace'"),
        postgresql_where=text("scope = 'workspace'"),
    ),
)


plugin_release_revocations = Table(
    "plugin_release_revocations",
    metadata,
    Column(
        "release_id",
        SaUuid(as_uuid=True),
        ForeignKey("plugin_releases.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("scope", PluginReleaseScopeType(), nullable=False),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("slug", String(100), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("reason", PluginReleaseRevocationReasonType(), nullable=False),
    Column(
        "revoked_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
    ),
    Column("revoked_by_platform_actor", String(255), nullable=True),
    Column("revoked_at", UTCDateTime(), nullable=False),
    CheckConstraint(
        "scope IN ('system', 'workspace')",
        name="revocation_scope",
    ),
    CheckConstraint(
        "(scope = 'system' AND workspace_id IS NULL) OR "
        "(scope = 'workspace' AND workspace_id IS NOT NULL)",
        name="revocation_scope_workspace",
    ),
    CheckConstraint(
        "revision >= 1",
        name="revocation_revision",
    ),
    CheckConstraint(
        "reason IN ('security', 'integrity', 'policy', 'operational')",
        name="revocation_reason",
    ),
    CheckConstraint(
        "(scope = 'system' AND revoked_by_user_id IS NULL "
        "AND revoked_by_platform_actor IS NOT NULL "
        "AND length(trim(revoked_by_platform_actor)) BETWEEN 1 AND 255) OR "
        "(scope = 'workspace' AND revoked_by_user_id IS NOT NULL "
        "AND revoked_by_platform_actor IS NULL)",
        name="revocation_actor",
    ),
    Index(
        "ix_plugin_release_revocations_scoped_identity",
        "scope",
        "workspace_id",
        "slug",
        "revision",
    ),
)


templates = Table(
    "templates",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column(
        "workspace_id",
        SaUuid(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("source_graph_id", SaUuid(as_uuid=True), nullable=False),
    Column("source_revision", Integer, nullable=False),
    Column("source_graph_name", String(160), nullable=False),
    Column("snapshot_document", SavedGraphDocumentType(), nullable=False),
    Column("name", String(160), nullable=False),
    Column("description", String(1000), nullable=True),
    Column("state", TemplateStateType(), nullable=False),
    Column(
        "created_by_user_id",
        SaUuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    UniqueConstraint("workspace_id", "id", name="uq_templates_workspace_id_id"),
    CheckConstraint("source_revision >= 1", name="template_source_revision"),
    CheckConstraint(
        "state IN ('active', 'archived')",
        name="template_state",
    ),
    Index("ix_templates_workspace_name", "workspace_id", "name"),
    Index("ix_templates_workspace_updated_at", "workspace_id", "updated_at"),
)

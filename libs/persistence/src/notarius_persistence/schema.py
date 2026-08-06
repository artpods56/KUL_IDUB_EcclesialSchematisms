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
)
from sqlalchemy import Uuid as SaUuid
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from notarius_core.domain.artifact_outputs import (
    ArtifactOutputValue,
    artifact_outputs_from_storage,
    artifact_outputs_to_storage,
)
from notarius_core.domain.saved_graphs import SavedGraphDocument
from notarius_core.domain.identity import (
    WorkspaceCapability,
    WorkspaceKind,
    WorkspaceRole,
)
from notarius_core.domain.security_audit import (
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


saved_graph_revisions = Table(
    "saved_graph_revisions",
    metadata,
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        ForeignKey("saved_graphs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("revision", Integer, primary_key=True),
    Column("name", String(160), nullable=False),
    Column("document", SavedGraphDocumentType(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
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


invocation_cache_entries = Table(
    "invocation_cache_entries",
    metadata,
    Column("key_sha256", String(64), primary_key=True),
    Column("generation", SaUuid(as_uuid=True), nullable=False),
    Column("outputs", ArtifactOutputsType(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
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
    Column("outputs", ArtifactOutputsType(), nullable=False),
    Column("materialized_at", UTCDateTime(), nullable=False),
    Index(
        "ix_materialized_node_outputs_graph_revision",
        "graph_id",
        "graph_revision",
        "materialized_at",
    ),
)


graph_executions = Table(
    "graph_executions",
    metadata,
    Column("execution_id", SaUuid(as_uuid=True), primary_key=True),
    Column("graph_id", SaUuid(as_uuid=True), nullable=False),
    Column("graph_revision", Integer, nullable=False),
    Column("status", String(24), nullable=False),
    Column("scope", String(32), nullable=False),
    Column("workflow_run_id", SaUuid(as_uuid=True), nullable=True),
    Column("error", Text, nullable=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("started_at", UTCDateTime(), nullable=True),
    Column("finished_at", UTCDateTime(), nullable=True),
    ForeignKeyConstraint(
        ("graph_id", "graph_revision"),
        ("saved_graph_revisions.graph_id", "saved_graph_revisions.revision"),
        ondelete="CASCADE",
    ),
    Index(
        "ix_graph_executions_graph_created",
        "graph_id",
        "created_at",
        "execution_id",
    ),
    Index(
        "ix_graph_executions_graph_revision_created",
        "graph_id",
        "graph_revision",
        "created_at",
        "execution_id",
    ),
    Index("ix_graph_executions_status", "status"),
)


graph_execution_requested_nodes = Table(
    "graph_execution_requested_nodes",
    metadata,
    Column(
        "execution_id",
        SaUuid(as_uuid=True),
        ForeignKey("graph_executions.execution_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("node_id", String(255), primary_key=True),
    Column("position", Integer, nullable=False),
    UniqueConstraint(
        "execution_id",
        "position",
        name="uq_graph_execution_requested_nodes_execution_position",
    ),
    Index(
        "ix_graph_execution_requested_nodes_node_execution",
        "node_id",
        "execution_id",
    ),
)


graph_execution_node_results = Table(
    "graph_execution_node_results",
    metadata,
    Column(
        "execution_id",
        SaUuid(as_uuid=True),
        ForeignKey("graph_executions.execution_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("node_id", String(255), primary_key=True),
    Column("position", Integer, nullable=False),
    Column("status", String(16), nullable=False),
    Column("outputs", ArtifactOutputsType(), nullable=False),
    Column("artifact_count", Integer, nullable=False),
    Column("error", Text, nullable=True),
    Column("completed_at", UTCDateTime(), nullable=False),
    UniqueConstraint(
        "execution_id",
        "position",
        name="uq_graph_execution_node_results_execution_position",
    ),
    Index(
        "ix_graph_execution_node_results_node_execution",
        "node_id",
        "execution_id",
    ),
)


node_secrets = Table(
    "node_secrets",
    metadata,
    Column(
        "graph_id",
        SaUuid(as_uuid=True),
        ForeignKey("saved_graphs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("node_id", String(255), primary_key=True),
    Column("name", String(255), primary_key=True),
    Column("operator_id", String(255), nullable=False),
    Column("operator_version", Integer, nullable=False),
    Column("key_id", String(64), nullable=False),
    Column("dependency_sha256", String(64), nullable=False),
    Column("nonce", LargeBinary(12), nullable=False),
    Column("ciphertext", LargeBinary(), nullable=False),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    Index("ix_node_secrets_graph_id", "graph_id"),
)


users = Table(
    "users",
    metadata,
    Column("id", SaUuid(as_uuid=True), primary_key=True),
    Column("email", String(320), nullable=True),
    Column("display_name", String(160), nullable=True),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", UTCDateTime(), nullable=False),
    Column("updated_at", UTCDateTime(), nullable=False),
    Index("ix_users_active_updated_at", "active", "updated_at"),
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

"""Scope existing resource persistence to an owning workspace.

Revision ID: 0008_tenant_existing_resources
Revises: 0007_identity_workspace_foundation
Create Date: 2026-08-07
"""

from collections.abc import Sequence
from uuid import UUID

from alembic import op
import sqlalchemy as sa


revision: str = "0008_tenant_existing_resources"
down_revision: str | Sequence[str] | None = "0007_identity_workspace_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LOCAL_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")
_LOCAL_ID = LOCAL_WORKSPACE_ID.hex

_RESOURCE_TABLES = (
    "saved_graphs",
    "saved_graph_revisions",
    "artifact_objects",
    "invocation_cache_entries",
    "materialized_node_outputs",
    "node_secrets",
    "graph_executions",
    "graph_execution_requested_nodes",
    "graph_execution_node_results",
    "staged_uploads",
)


def _table_exists(connection: sa.Connection, table_name: str) -> bool:
    return table_name in sa.inspect(connection).get_table_names()


def _assert_no_orphans(connection: sa.Connection) -> None:
    checks = (
        (
            "saved_graph_revisions",
            "NOT EXISTS (SELECT 1 FROM saved_graphs WHERE "
            "saved_graphs.id = saved_graph_revisions.graph_id)",
        ),
        (
            "materialized_node_outputs",
            "NOT EXISTS (SELECT 1 FROM saved_graph_revisions WHERE "
            "saved_graph_revisions.graph_id = materialized_node_outputs.graph_id "
            "AND saved_graph_revisions.revision = "
            "materialized_node_outputs.graph_revision)",
        ),
        (
            "node_secrets",
            "NOT EXISTS (SELECT 1 FROM saved_graphs WHERE "
            "saved_graphs.id = node_secrets.graph_id)",
        ),
        (
            "graph_executions",
            "NOT EXISTS (SELECT 1 FROM saved_graph_revisions WHERE "
            "saved_graph_revisions.graph_id = graph_executions.graph_id "
            "AND saved_graph_revisions.revision = graph_executions.graph_revision)",
        ),
        (
            "graph_execution_requested_nodes",
            "NOT EXISTS (SELECT 1 FROM graph_executions WHERE "
            "graph_executions.execution_id = "
            "graph_execution_requested_nodes.execution_id)",
        ),
        (
            "graph_execution_node_results",
            "NOT EXISTS (SELECT 1 FROM graph_executions WHERE "
            "graph_executions.execution_id = "
            "graph_execution_node_results.execution_id)",
        ),
    )
    for child, orphan_predicate in checks:
        orphan_count = connection.scalar(
            sa.text(f"SELECT COUNT(*) FROM {child} WHERE {orphan_predicate}")
        )
        if orphan_count:
            raise RuntimeError(
                f"Cannot tenant migration: {child} contains {orphan_count} orphan row(s)"
            )


def _create_tenant_tables() -> None:
    op.create_table(
        "_0008_saved_graphs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_saved_graphs"),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_saved_graphs_workspace_id_id",
        ),
    )
    op.create_table(
        "_0008_saved_graph_revisions",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id"],
            ["_0008_saved_graphs.workspace_id", "_0008_saved_graphs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "graph_id",
            "revision",
            name="pk_saved_graph_revisions",
        ),
    )
    op.create_table(
        "_0008_artifact_objects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type", sa.String(length=255), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("storage_backend", sa.String(length=40), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=True),
        sa.Column("object_key", sa.String(length=2048), nullable=True),
        sa.Column("inline_payload", sa.JSON(), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_artifact_objects"),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_artifact_objects_workspace_id_id",
        ),
    )
    op.create_table(
        "_0008_invocation_cache_entries",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("key_sha256", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_invocation_cache_entries_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "key_sha256",
            name="pk_invocation_cache_entries",
        ),
    )
    op.create_table(
        "_0008_materialized_node_outputs",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("materialized_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id", "graph_revision"],
            [
                "_0008_saved_graph_revisions.workspace_id",
                "_0008_saved_graph_revisions.graph_id",
                "_0008_saved_graph_revisions.revision",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "graph_id",
            "graph_revision",
            "node_id",
            name="pk_materialized_node_outputs",
        ),
    )
    op.create_table(
        "_0008_node_secrets",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("operator_id", sa.String(length=255), nullable=False),
        sa.Column("operator_version", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("dependency_sha256", sa.String(length=64), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id"],
            ["_0008_saved_graphs.workspace_id", "_0008_saved_graphs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "graph_id",
            "node_id",
            "name",
            name="pk_node_secrets",
        ),
    )
    op.create_table(
        "_0008_graph_executions",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id", "graph_revision"],
            [
                "_0008_saved_graph_revisions.workspace_id",
                "_0008_saved_graph_revisions.graph_id",
                "_0008_saved_graph_revisions.revision",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("execution_id", name="pk_graph_executions"),
        sa.UniqueConstraint(
            "workspace_id",
            "execution_id",
            name="uq_graph_executions_workspace_id_execution_id",
        ),
    )
    for table_name, kind in (
        ("_0008_graph_execution_requested_nodes", "requested"),
        ("_0008_graph_execution_node_results", "results"),
    ):
        constraint_name = (
            "uq_graph_execution_requested_nodes_execution_position"
            if kind == "requested"
            else "uq_graph_execution_node_results_execution_position"
        )
        primary_key_name = (
            "pk_graph_execution_requested_nodes"
            if kind == "requested"
            else "pk_graph_execution_node_results"
        )
        columns = [
            sa.Column("workspace_id", sa.Uuid(), nullable=False),
            sa.Column("execution_id", sa.Uuid(), nullable=False),
            sa.Column("node_id", sa.String(length=255), nullable=False),
            sa.Column("position", sa.Integer(), nullable=False),
        ]
        if kind == "results":
            columns.extend(
                [
                    sa.Column("status", sa.String(length=16), nullable=False),
                    sa.Column("outputs", sa.JSON(), nullable=False),
                    sa.Column("artifact_count", sa.Integer(), nullable=False),
                    sa.Column("error", sa.Text(), nullable=True),
                    sa.Column("completed_at", sa.DateTime(), nullable=False),
                ]
            )
        op.create_table(
            table_name,
            *columns,
            sa.ForeignKeyConstraint(
                ["workspace_id", "execution_id"],
                [
                    "_0008_graph_executions.workspace_id",
                    "_0008_graph_executions.execution_id",
                ],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint(
                "workspace_id",
                "execution_id",
                "node_id",
                name=primary_key_name,
            ),
            sa.UniqueConstraint(
                "workspace_id",
                "execution_id",
                "position",
                name=constraint_name,
            ),
        )


def _create_indexes() -> None:
    op.create_index(
        "ix_saved_graphs_workspace_updated_at",
        "saved_graphs",
        ["workspace_id", "updated_at"],
    )
    op.create_index(
        "ix_saved_graphs_workspace_id",
        "saved_graphs",
        ["workspace_id", "id"],
    )
    op.create_index(
        "ix_saved_graph_revisions_workspace_graph_revision",
        "saved_graph_revisions",
        ["workspace_id", "graph_id", "revision"],
    )
    op.create_index(
        "ix_artifact_objects_workspace_type",
        "artifact_objects",
        ["workspace_id", "artifact_type", "schema_version"],
    )
    op.create_index(
        "ix_artifact_objects_workspace_sha256",
        "artifact_objects",
        ["workspace_id", "sha256"],
    )
    op.create_index(
        "ix_materialized_node_outputs_graph_revision",
        "materialized_node_outputs",
        ["workspace_id", "graph_id", "graph_revision", "materialized_at"],
    )
    op.create_index(
        "ix_graph_executions_graph_created",
        "graph_executions",
        ["workspace_id", "graph_id", "created_at", "execution_id"],
    )
    op.create_index(
        "ix_graph_executions_graph_revision_created",
        "graph_executions",
        [
            "workspace_id",
            "graph_id",
            "graph_revision",
            "created_at",
            "execution_id",
        ],
    )
    op.create_index("ix_graph_executions_status", "graph_executions", ["status"])
    op.create_index(
        "ix_graph_execution_requested_nodes_node_execution",
        "graph_execution_requested_nodes",
        ["workspace_id", "node_id", "execution_id"],
    )
    op.create_index(
        "ix_graph_execution_node_results_node_execution",
        "graph_execution_node_results",
        ["workspace_id", "node_id", "execution_id"],
    )
    op.create_index(
        "ix_node_secrets_workspace_graph",
        "node_secrets",
        ["workspace_id", "graph_id"],
    )
    op.create_index(
        "ix_staged_uploads_workspace_created_at",
        "staged_uploads",
        ["workspace_id", "created_at"],
    )


def _copy_and_replace_tables(connection: sa.Connection) -> None:
    counts = {
        table_name: int(
            connection.scalar(sa.text(f"SELECT COUNT(*) FROM {table_name}")) or 0
        )
        for table_name in _RESOURCE_TABLES
        if table_name != "staged_uploads"
    }
    _create_tenant_tables()
    copy_statements = (
        (
            "_0008_saved_graphs",
            "INSERT INTO _0008_saved_graphs "
            "(id, workspace_id, created_by_user_id, name, document, revision, "
            "created_at, updated_at) "
            "SELECT id, :local_id, NULL, name, document, revision, created_at, "
            "updated_at FROM saved_graphs",
        ),
        (
            "_0008_saved_graph_revisions",
            "INSERT INTO _0008_saved_graph_revisions "
            "(workspace_id, graph_id, revision, name, document, created_at) "
            "SELECT :local_id, revisions.graph_id, revisions.revision, "
            "revisions.name, revisions.document, revisions.created_at "
            "FROM saved_graph_revisions AS revisions "
            "JOIN saved_graphs AS graphs ON graphs.id = revisions.graph_id",
        ),
        (
            "_0008_artifact_objects",
            "INSERT INTO _0008_artifact_objects "
            "(id, workspace_id, artifact_type, schema_version, content_type, "
            "storage_backend, bucket, object_key, inline_payload, byte_size, "
            "sha256, metadata) "
            "SELECT id, :local_id, artifact_type, schema_version, content_type, "
            "storage_backend, bucket, object_key, inline_payload, byte_size, "
            "sha256, metadata FROM artifact_objects",
        ),
        (
            "_0008_invocation_cache_entries",
            "INSERT INTO _0008_invocation_cache_entries "
            "(workspace_id, key_sha256, generation, outputs, created_at) "
            "SELECT :local_id, key_sha256, generation, outputs, created_at "
            "FROM invocation_cache_entries",
        ),
        (
            "_0008_materialized_node_outputs",
            "INSERT INTO _0008_materialized_node_outputs "
            "(workspace_id, graph_id, graph_revision, node_id, workflow_run_id, "
            "outputs, materialized_at) "
            "SELECT :local_id, outputs.graph_id, outputs.graph_revision, "
            "outputs.node_id, outputs.workflow_run_id, outputs.outputs, "
            "outputs.materialized_at FROM materialized_node_outputs AS outputs "
            "JOIN saved_graph_revisions AS revisions ON revisions.graph_id = "
            "outputs.graph_id AND revisions.revision = outputs.graph_revision",
        ),
        (
            "_0008_node_secrets",
            "INSERT INTO _0008_node_secrets "
            "(workspace_id, graph_id, node_id, name, operator_id, operator_version, "
            "key_id, dependency_sha256, nonce, ciphertext, created_at, updated_at) "
            "SELECT :local_id, secrets.graph_id, secrets.node_id, secrets.name, "
            "secrets.operator_id, secrets.operator_version, secrets.key_id, "
            "secrets.dependency_sha256, secrets.nonce, secrets.ciphertext, "
            "secrets.created_at, secrets.updated_at FROM node_secrets AS secrets "
            "JOIN saved_graphs AS graphs ON graphs.id = secrets.graph_id",
        ),
        (
            "_0008_graph_executions",
            "INSERT INTO _0008_graph_executions "
            "(workspace_id, execution_id, graph_id, graph_revision, status, scope, "
            "workflow_run_id, error, created_at, started_at, finished_at) "
            "SELECT :local_id, executions.execution_id, executions.graph_id, "
            "executions.graph_revision, executions.status, executions.scope, "
            "executions.workflow_run_id, executions.error, executions.created_at, "
            "executions.started_at, executions.finished_at "
            "FROM graph_executions AS executions JOIN saved_graph_revisions AS "
            "revisions ON revisions.graph_id = executions.graph_id AND "
            "revisions.revision = executions.graph_revision",
        ),
        (
            "_0008_graph_execution_requested_nodes",
            "INSERT INTO _0008_graph_execution_requested_nodes "
            "(workspace_id, execution_id, node_id, position) "
            "SELECT :local_id, nodes.execution_id, nodes.node_id, nodes.position "
            "FROM graph_execution_requested_nodes AS nodes JOIN graph_executions "
            "AS executions ON executions.execution_id = nodes.execution_id",
        ),
        (
            "_0008_graph_execution_node_results",
            "INSERT INTO _0008_graph_execution_node_results "
            "(workspace_id, execution_id, node_id, position, status, outputs, "
            "artifact_count, error, completed_at) "
            "SELECT :local_id, results.execution_id, results.node_id, results.position, "
            "results.status, results.outputs, results.artifact_count, results.error, "
            "results.completed_at FROM graph_execution_node_results AS results "
            "JOIN graph_executions AS executions ON executions.execution_id = "
            "results.execution_id",
        ),
    )
    for temporary_table, statement in copy_statements:
        connection.execute(sa.text(statement), {"local_id": _LOCAL_ID})

    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    for table_name in (
        "graph_execution_node_results",
        "graph_execution_requested_nodes",
        "graph_executions",
        "materialized_node_outputs",
        "node_secrets",
        "saved_graph_revisions",
        "artifact_objects",
        "invocation_cache_entries",
        "saved_graphs",
    ):
        connection.exec_driver_sql(f"DROP TABLE {table_name}")
    for table_name in (
        "saved_graphs",
        "saved_graph_revisions",
        "artifact_objects",
        "invocation_cache_entries",
        "materialized_node_outputs",
        "node_secrets",
        "graph_executions",
        "graph_execution_requested_nodes",
        "graph_execution_node_results",
    ):
        connection.exec_driver_sql(
            f"ALTER TABLE _0008_{table_name} RENAME TO {table_name}"
        )
    op.create_table(
        "staged_uploads",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("upload_key", sa.String(length=1024), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("workspace_id", "upload_key", name="pk_staged_uploads"),
        sa.CheckConstraint("byte_size >= 0", name="ck_staged_uploads_byte_size_nonnegative"),
        sa.CheckConstraint(
            "length(original_filename) BETWEEN 1 AND 255",
            name="ck_staged_uploads_original_filename_bounded",
        ),
    )
    _create_indexes()
    connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    foreign_key_errors = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise RuntimeError(f"Tenant migration produced foreign-key errors: {foreign_key_errors}")
    for table_name, expected_count in counts.items():
        actual_count = int(connection.scalar(sa.text(f"SELECT COUNT(*) FROM {table_name}")) or 0)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Tenant migration changed {table_name} row count from "
                f"{expected_count} to {actual_count}"
            )


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        raise NotImplementedError("Tenant resource migration currently requires SQLite")
    _assert_no_orphans(connection)
    _copy_and_replace_tables(connection)


def _assert_downgrade_is_safe(connection: sa.Connection) -> None:
    for table_name in _RESOURCE_TABLES:
        if not _table_exists(connection, table_name) or table_name == "staged_uploads":
            continue
        outside_local = connection.scalar(
            sa.text(f"SELECT COUNT(*) FROM {table_name} WHERE workspace_id != :local_id"),
            {"local_id": _LOCAL_ID},
        )
        if outside_local:
            raise RuntimeError(
                f"Cannot downgrade tenant migration: {table_name} contains "
                f"{outside_local} row(s) outside the deterministic local workspace"
            )
    staged_count = connection.scalar(sa.text("SELECT COUNT(*) FROM staged_uploads"))
    if staged_count:
        raise RuntimeError(
            "Cannot downgrade tenant migration while staged uploads exist; "
            "run the reverse file migration first"
        )
    for table_name in ("saved_graphs", "staged_uploads"):
        attributed = connection.scalar(
            sa.text(
                f"SELECT COUNT(*) FROM {table_name} "
                "WHERE created_by_user_id IS NOT NULL"
            )
        )
        if attributed:
            raise RuntimeError(
                f"Cannot downgrade tenant migration: {table_name} contains "
                f"{attributed} creator-attributed row(s)"
            )
    collisions = connection.scalar(
        sa.text(
            "SELECT COUNT(*) FROM (SELECT key_sha256 FROM "
            "invocation_cache_entries GROUP BY key_sha256 HAVING COUNT(*) > 1)"
        )
    )
    if collisions:
        raise RuntimeError(
            "Cannot downgrade tenant migration: invocation cache keys would collide"
        )
    users = int(connection.scalar(sa.text("SELECT COUNT(*) FROM users")) or 0)
    workspaces = int(connection.scalar(sa.text("SELECT COUNT(*) FROM workspaces")) or 0)
    if users or workspaces != 1:
        raise RuntimeError(
            "Cannot downgrade tenant migration: identity/workspace data would "
            "be discarded or merged"
        )


def _rebuild_legacy_tables(connection: sa.Connection) -> None:
    op.create_table(
        "_0008d_saved_graphs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_saved_graphs"),
    )
    op.create_table(
        "_0008d_saved_graph_revisions",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id"], ["_0008d_saved_graphs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("graph_id", "revision", name="pk_saved_graph_revisions"),
    )
    op.create_table(
        "_0008d_artifact_objects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type", sa.String(length=255), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("storage_backend", sa.String(length=40), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=True),
        sa.Column("object_key", sa.String(length=2048), nullable=True),
        sa.Column("inline_payload", sa.JSON(), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_artifact_objects"),
    )
    op.create_table(
        "_0008d_invocation_cache_entries",
        sa.Column("key_sha256", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key_sha256", name="pk_invocation_cache_entries"),
    )
    op.create_table(
        "_0008d_materialized_node_outputs",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("materialized_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id", "graph_revision"],
            [
                "_0008d_saved_graph_revisions.graph_id",
                "_0008d_saved_graph_revisions.revision",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "graph_id",
            "graph_revision",
            "node_id",
            name="pk_materialized_node_outputs",
        ),
    )
    op.create_table(
        "_0008d_node_secrets",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("operator_id", sa.String(length=255), nullable=False),
        sa.Column("operator_version", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("dependency_sha256", sa.String(length=64), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id"], ["_0008d_saved_graphs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("graph_id", "node_id", "name", name="pk_node_secrets"),
    )
    op.create_table(
        "_0008d_graph_executions",
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["graph_id", "graph_revision"],
            [
                "_0008d_saved_graph_revisions.graph_id",
                "_0008d_saved_graph_revisions.revision",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("execution_id", name="pk_graph_executions"),
    )
    for table_name, kind in (
        ("_0008d_graph_execution_requested_nodes", "requested"),
        ("_0008d_graph_execution_node_results", "results"),
    ):
        constraint_name = (
            "uq_graph_execution_requested_nodes_execution_position"
            if kind == "requested"
            else "uq_graph_execution_node_results_execution_position"
        )
        primary_key_name = (
            "pk_graph_execution_requested_nodes"
            if kind == "requested"
            else "pk_graph_execution_node_results"
        )
        columns = [
            sa.Column("execution_id", sa.Uuid(), nullable=False),
            sa.Column("node_id", sa.String(length=255), nullable=False),
            sa.Column("position", sa.Integer(), nullable=False),
        ]
        if kind == "results":
            columns.extend(
                [
                    sa.Column("status", sa.String(length=16), nullable=False),
                    sa.Column("outputs", sa.JSON(), nullable=False),
                    sa.Column("artifact_count", sa.Integer(), nullable=False),
                    sa.Column("error", sa.Text(), nullable=True),
                    sa.Column("completed_at", sa.DateTime(), nullable=False),
                ]
            )
        op.create_table(
            table_name,
            *columns,
            sa.ForeignKeyConstraint(
                ["execution_id"],
                ["_0008d_graph_executions.execution_id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint(
                "execution_id",
                "node_id",
                name=primary_key_name,
            ),
            sa.UniqueConstraint(
                "execution_id",
                "position",
                name=constraint_name,
            ),
        )

    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    copy_statements = (
        (
            "_0008d_saved_graphs",
            "INSERT INTO _0008d_saved_graphs "
            "SELECT id, name, document, revision, created_at, updated_at "
            "FROM saved_graphs",
        ),
        (
            "_0008d_saved_graph_revisions",
            "INSERT INTO _0008d_saved_graph_revisions "
            "SELECT graph_id, revision, name, document, created_at "
            "FROM saved_graph_revisions",
        ),
        (
            "_0008d_artifact_objects",
            "INSERT INTO _0008d_artifact_objects "
            "SELECT id, artifact_type, schema_version, content_type, storage_backend, "
            "bucket, object_key, inline_payload, byte_size, sha256, metadata "
            "FROM artifact_objects",
        ),
        (
            "_0008d_invocation_cache_entries",
            "INSERT INTO _0008d_invocation_cache_entries "
            "SELECT key_sha256, generation, outputs, created_at "
            "FROM invocation_cache_entries",
        ),
        (
            "_0008d_materialized_node_outputs",
            "INSERT INTO _0008d_materialized_node_outputs "
            "SELECT graph_id, graph_revision, node_id, workflow_run_id, outputs, "
            "materialized_at FROM materialized_node_outputs",
        ),
        (
            "_0008d_node_secrets",
            "INSERT INTO _0008d_node_secrets "
            "SELECT graph_id, node_id, name, operator_id, operator_version, key_id, "
            "dependency_sha256, nonce, ciphertext, created_at, updated_at "
            "FROM node_secrets",
        ),
        (
            "_0008d_graph_executions",
            "INSERT INTO _0008d_graph_executions "
            "SELECT execution_id, graph_id, graph_revision, status, scope, "
            "workflow_run_id, error, created_at, started_at, finished_at "
            "FROM graph_executions",
        ),
        (
            "_0008d_graph_execution_requested_nodes",
            "INSERT INTO _0008d_graph_execution_requested_nodes "
            "SELECT execution_id, node_id, position "
            "FROM graph_execution_requested_nodes",
        ),
        (
            "_0008d_graph_execution_node_results",
            "INSERT INTO _0008d_graph_execution_node_results "
            "SELECT execution_id, node_id, position, status, outputs, artifact_count, "
            "error, completed_at FROM graph_execution_node_results",
        ),
    )
    for _, statement in copy_statements:
        connection.exec_driver_sql(statement)
    for table_name in (
        "staged_uploads",
        "graph_execution_node_results",
        "graph_execution_requested_nodes",
        "graph_executions",
        "materialized_node_outputs",
        "node_secrets",
        "saved_graph_revisions",
        "artifact_objects",
        "invocation_cache_entries",
        "saved_graphs",
    ):
        connection.exec_driver_sql(f"DROP TABLE {table_name}")
    for table_name in (
        "saved_graphs",
        "saved_graph_revisions",
        "artifact_objects",
        "invocation_cache_entries",
        "materialized_node_outputs",
        "node_secrets",
        "graph_executions",
        "graph_execution_requested_nodes",
        "graph_execution_node_results",
    ):
        connection.exec_driver_sql(
            f"ALTER TABLE _0008d_{table_name} RENAME TO {table_name}"
        )
    op.create_index(
        "ix_saved_graphs_updated_at", "saved_graphs", ["updated_at"]
    )
    op.create_index(
        "ix_artifact_objects_type",
        "artifact_objects",
        ["artifact_type", "schema_version"],
    )
    op.create_index("ix_artifact_objects_sha256", "artifact_objects", ["sha256"])
    op.create_index(
        "ix_materialized_node_outputs_graph_revision",
        "materialized_node_outputs",
        ["graph_id", "graph_revision", "materialized_at"],
    )
    op.create_index(
        "ix_graph_executions_graph_created",
        "graph_executions",
        ["graph_id", "created_at", "execution_id"],
    )
    op.create_index(
        "ix_graph_executions_graph_revision_created",
        "graph_executions",
        ["graph_id", "graph_revision", "created_at", "execution_id"],
    )
    op.create_index("ix_graph_executions_status", "graph_executions", ["status"])
    op.create_index(
        "ix_graph_execution_requested_nodes_node_execution",
        "graph_execution_requested_nodes",
        ["node_id", "execution_id"],
    )
    op.create_index(
        "ix_graph_execution_node_results_node_execution",
        "graph_execution_node_results",
        ["node_id", "execution_id"],
    )
    op.create_index("ix_node_secrets_graph_id", "node_secrets", ["graph_id"])
    connection.exec_driver_sql("PRAGMA foreign_keys=ON")


def downgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        raise NotImplementedError("Tenant resource migration currently requires SQLite")
    _assert_downgrade_is_safe(connection)
    _rebuild_legacy_tables(connection)

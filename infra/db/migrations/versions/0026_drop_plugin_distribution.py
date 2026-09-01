"""Wipe graph and Plugin data, then drop installation distribution metadata.

Revision ID: 0026_drop_plugin_distribution
Revises: 0025_platform_access_tokens
Create Date: 2026-09-01
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0026_drop_plugin_distribution"
down_revision: str | Sequence[str] | None = "0025_platform_access_tokens"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCOPE_POLICY_CONSTRAINT = "ck_plugin_installations_plugin_installation_scope_policy"
_WIPED_TABLES = (
    "graph_execution_nodes",
    "graph_executions",
    "materialized_node_outputs",
    "invocation_cache_entries",
    "graph_checkpoint_mappings",
    "graph_command_receipts",
    "collaborative_graph_heads",
    "node_secrets",
    "user_graph_states",
    "graph_organizations",
    "module_releases",
    "modules",
    "templates",
    "saved_graph_revisions",
    "saved_graphs",
    "graph_folders",
    "plugin_release_revocations",
    "plugin_release_selections",
    "plugin_installations",
    "plugin_releases",
    "staged_uploads",
)


def _wipe_cutover_rows(connection: sa.Connection) -> None:
    for table_name in _WIPED_TABLES:
        connection.execute(sa.text(f"DELETE FROM {table_name}"))


def upgrade() -> None:
    connection = op.get_bind()
    _wipe_cutover_rows(connection)
    with op.batch_alter_table("plugin_installations") as batch:
        batch.drop_constraint(_SCOPE_POLICY_CONSTRAINT, type_="check")
        batch.drop_column("distribution")
        batch.create_check_constraint(
            _SCOPE_POLICY_CONSTRAINT,
            "(scope = 'workspace' AND execution_policy = 'isolated-only') OR "
            "(scope = 'system')",
        )


def downgrade() -> None:
    with op.batch_alter_table("plugin_installations") as batch:
        batch.drop_constraint(_SCOPE_POLICY_CONSTRAINT, type_="check")
        batch.add_column(sa.Column("distribution", sa.String(length=16), nullable=True))
        batch.create_check_constraint(
            _SCOPE_POLICY_CONSTRAINT,
            "(scope = 'system' AND distribution IN "
            "('bundled', 'optional', 'published')) OR "
            "(scope = 'workspace' AND distribution IS NULL "
            "AND execution_policy = 'isolated-only')",
        )

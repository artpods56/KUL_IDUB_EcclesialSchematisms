"""Add workspace graph folders, organization, and per-user state.

Revision ID: 0011_graph_organization
Revises: 0010_module_library
Create Date: 2026-08-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0011_graph_organization"
down_revision: str | Sequence[str] | None = "0010_module_library"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "graph_folders",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_graph_folders_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graph_folders")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_graph_folders_workspace_id_id",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "name",
            name="uq_graph_folders_workspace_id_name",
        ),
    )
    op.create_index(
        "ix_graph_folders_workspace_name",
        "graph_folders",
        ["workspace_id", "name"],
    )

    op.create_table(
        "graph_organizations",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("folder_id", sa.Uuid(), nullable=True),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id", "folder_id"],
            ["graph_folders.workspace_id", "graph_folders.id"],
            name=op.f("fk_graph_organizations_workspace_id_graph_folders"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id"],
            ["saved_graphs.workspace_id", "saved_graphs.id"],
            name=op.f("fk_graph_organizations_workspace_id_saved_graphs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "graph_id",
            name=op.f("pk_graph_organizations"),
        ),
    )
    op.create_index(
        "ix_graph_organizations_workspace_folder_archived",
        "graph_organizations",
        ["workspace_id", "folder_id", "archived_at"],
    )

    op.create_table(
        "user_graph_states",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("starred", sa.Boolean(), nullable=False),
        sa.Column("last_opened_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_graph_states_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "graph_id"],
            ["saved_graphs.workspace_id", "saved_graphs.id"],
            name=op.f("fk_user_graph_states_workspace_id_saved_graphs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "graph_id",
            "user_id",
            name=op.f("pk_user_graph_states"),
        ),
    )
    op.create_index(
        "ix_user_graph_states_user_starred",
        "user_graph_states",
        ["user_id", "starred"],
    )
    op.create_index(
        "ix_user_graph_states_user_last_opened",
        "user_graph_states",
        ["user_id", "last_opened_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_graph_states_user_last_opened",
        table_name="user_graph_states",
    )
    op.drop_index(
        "ix_user_graph_states_user_starred",
        table_name="user_graph_states",
    )
    op.drop_table("user_graph_states")
    op.drop_index(
        "ix_graph_organizations_workspace_folder_archived",
        table_name="graph_organizations",
    )
    op.drop_table("graph_organizations")
    op.drop_index(
        "ix_graph_folders_workspace_name",
        table_name="graph_folders",
    )
    op.drop_table("graph_folders")

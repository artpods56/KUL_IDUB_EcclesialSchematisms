import json
from pathlib import Path
from uuid import UUID

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from notarius_api.settings import get_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_alembic_migration_upgrades_downgrades_and_has_no_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "fresh" / "nested" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")

    command.upgrade(config, "head")
    assert database_path.exists()
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "artifact_objects",
            "graph_execution_node_results",
            "graph_execution_requested_nodes",
            "graph_executions",
            "invocation_cache_entries",
            "materialized_node_outputs",
            "node_secrets",
            "users",
            "oidc_identities",
            "oidc_login_transactions",
            "oidc_bootstrap_owner_mappings",
            "workspaces",
            "workspace_memberships",
            "auth_sessions",
            "personal_access_tokens",
            "security_audit_events",
            "saved_graphs",
            "saved_graph_revisions",
        }
    command.check(config)

    command.downgrade(config, "base")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert inspect(connection).get_table_names() == ["alembic_version"]

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "artifact_objects",
            "graph_execution_node_results",
            "graph_execution_requested_nodes",
            "graph_executions",
            "invocation_cache_entries",
            "materialized_node_outputs",
            "node_secrets",
            "users",
            "oidc_identities",
            "oidc_login_transactions",
            "oidc_bootstrap_owner_mappings",
            "workspaces",
            "workspace_memberships",
            "auth_sessions",
            "personal_access_tokens",
            "security_audit_events",
            "saved_graphs",
            "saved_graph_revisions",
        }

    get_settings.cache_clear()


def test_identity_migration_creates_sealed_local_workspace_and_audit_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "identity" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")

    command.upgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        local = connection.execute(
            text(
                "SELECT slug, kind, personal_owner_user_id "
                "FROM workspaces WHERE slug = 'local'"
            )
        ).mappings().one()
        assert local == {
            "slug": "local",
            "kind": "shared",
            "personal_owner_user_id": None,
        }
        assert connection.execute(
            text("SELECT COUNT(*) FROM users")
        ).scalar_one() == 0
        workspaces_ddl = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'workspaces'"
            )
        ).scalar_one()
        assert "GLOB" not in workspaces_ddl.upper()
        assert "lower(trim(slug))" in workspaces_ddl
        indexes = {
            row[1]
            for row in connection.execute(
                text("PRAGMA index_list('security_audit_events')")
            )
        }
        assert indexes >= {
            "ix_security_audit_events_workspace_occurred_at",
            "ix_security_audit_events_actor_occurred_at",
            "ix_security_audit_events_operation_occurred_at",
            "ix_security_audit_events_retention",
        }
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO workspaces "
                    "(id, slug, name, kind, personal_owner_user_id, "
                    "created_at, updated_at) VALUES "
                    "(:id, :slug, :name, :kind, NULL, :created_at, :updated_at)"
                ),
                {
                    "id": "00000000000000000000000000000008",
                    "slug": "Not-Normalized",
                    "name": "Invalid",
                    "kind": "shared",
                    "created_at": "2026-08-07 00:00:00",
                    "updated_at": "2026-08-07 00:00:00",
                },
            )

    get_settings.cache_clear()


def test_saved_graph_revision_migration_backfills_the_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "backfill" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0003_node_secrets")

    graph_id = UUID("00000000-0000-0000-0000-000000000401")
    document: dict[str, object] = {
        "schema_version": 3,
        "nodes": [],
        "edges": [],
    }
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO saved_graphs "
                "(id, name, document, revision, created_at, updated_at) "
                "VALUES (:id, :name, :document, :revision, :created_at, :updated_at)"
            ),
            {
                "id": graph_id.hex,
                "name": "Existing graph",
                "document": json.dumps(document),
                "revision": 7,
                "created_at": "2026-07-14 08:00:00",
                "updated_at": "2026-07-16 09:30:00",
            },
        )

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT graph_id, revision, name, document, created_at "
                    "FROM saved_graph_revisions"
                )
            )
            .mappings()
            .one()
        )
        assert row["graph_id"] == graph_id.hex
        assert row["revision"] == 7
        assert row["name"] == "Existing graph"
        assert json.loads(row["document"]) == document
        assert str(row["created_at"]) == "2026-07-16 09:30:00"

    command.downgrade(config, "0003_node_secrets")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert "saved_graph_revisions" not in inspect(connection).get_table_names()

    get_settings.cache_clear()

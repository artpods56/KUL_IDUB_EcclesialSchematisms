import json
from pathlib import Path
from uuid import UUID

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect, text

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
            "saved_graphs",
            "saved_graph_revisions",
        }

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

import asyncio
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

import grafy_api.main as main_module
from grafy_persistence.database import Database, create_database
from grafy_persistence.orm import metadata

from grafy_api.settings import Settings, get_settings
from tests.testkit import client_with_overrides


class _SwitchableReadinessEngine:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self.available = True
        self.disposed = False

    def connect(self) -> AsyncConnection:
        if not self.available:
            raise OSError("database unavailable")
        return self._engine.connect()

    async def dispose(self) -> None:
        self.disposed = True
        await self._engine.dispose()


async def _create_schema(database: Database) -> None:
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


def _readiness_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Settings, _SwitchableReadinessEngine]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'readiness.sqlite3'}"
    database = create_database(database_url)
    asyncio.run(_create_schema(database))
    readiness_engine = _SwitchableReadinessEngine(database.engine)
    readiness_database = Database(
        engine=cast(AsyncEngine, readiness_engine),
        sessions=database.sessions,
    )

    def use_readiness_database(_database_url: str) -> Database:
        return readiness_database

    monkeypatch.setattr(
        main_module,
        "create_database",
        use_readiness_database,
    )
    return (
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            command_hmac_key=SecretStr("readiness-command-key"),
            require_single_api_owner=False,
        ),
        readiness_engine,
    )


def test_readiness_reports_database_failure_without_breaking_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, readiness_engine = _readiness_settings(tmp_path, monkeypatch)

    with client_with_overrides(settings=settings) as client:
        healthy = client.get("/ready")
        assert healthy.status_code == 200
        assert healthy.json() == {"status": "ok"}

        readiness_engine.available = False
        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"detail": "Service unavailable"}
        liveness = client.get("/health")
        assert liveness.status_code == 200
        assert liveness.json() == {"status": "ok"}


def test_readiness_uses_the_settings_attached_to_its_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFY_MAP_MAX_CONCURRENCY", "7")
    get_settings.cache_clear()
    settings, _readiness_engine = _readiness_settings(tmp_path, monkeypatch)

    try:
        assert get_settings().map_max_concurrency == 7
        with client_with_overrides(settings=settings) as client:
            response = client.get("/ready")

            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
    finally:
        get_settings.cache_clear()


def test_readiness_translates_uninitialized_resources_to_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _readiness_engine = _readiness_settings(tmp_path, monkeypatch)

    with client_with_overrides(settings=settings) as client:
        application = cast(FastAPI, client.app)
        resources = application.state.resources
        del application.state.resources
        try:
            response = client.get("/ready")
        finally:
            application.state.resources = resources

        assert response.status_code == 503
        assert response.json() == {"detail": "Service unavailable"}

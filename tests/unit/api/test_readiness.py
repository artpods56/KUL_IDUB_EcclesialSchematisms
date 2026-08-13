import asyncio
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata

import notarius_api.main as main_module
from notarius_api.main import create_app
from notarius_api.settings import Settings


class _SwitchableReadinessEngine:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self.available = True

    def connect(self) -> AsyncConnection:
        if not self.available:
            raise OSError("database unavailable")
        return self._engine.connect()

    async def dispose(self) -> None:
        await self._engine.dispose()


class _PrefectHealthClient:
    status_code = 200
    requested_urls: list[str] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    async def get(self, url: str) -> httpx.Response:
        self.requested_urls.append(url)
        return httpx.Response(
            self.status_code,
            request=httpx.Request("GET", url),
        )


async def _create_schema(database: Database) -> None:
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


def _readiness_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    execution_backend: Literal["prefect", "inline"],
) -> tuple[FastAPI, _SwitchableReadinessEngine]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'readiness.sqlite3'}"
    database = create_database(database_url)
    asyncio.run(_create_schema(database))
    readiness_engine = _SwitchableReadinessEngine(database.engine)
    readiness_database = Database(
        engine=cast(AsyncEngine, readiness_engine),
        sessions=database.sessions,
    )
    monkeypatch.setattr(
        main_module,
        "create_database",
        lambda _database_url: readiness_database,
    )
    application = create_app(
        Settings(
            _env_file=None,
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend=execution_backend,
            command_hmac_key=SecretStr("readiness-command-key"),
            require_single_api_owner=False,
        )
    )
    return application, readiness_engine


def test_readiness_reports_database_failure_without_breaking_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    application, readiness_engine = _readiness_application(
        tmp_path,
        monkeypatch,
        execution_backend="inline",
    )

    with TestClient(application) as client:
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


def test_prefect_readiness_checks_configured_health_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFECT_API_URL", "http://prefect.test:4200/api/")
    application, _readiness_engine = _readiness_application(
        tmp_path,
        monkeypatch,
        execution_backend="prefect",
    )
    _PrefectHealthClient.requested_urls = []
    _PrefectHealthClient.status_code = 200

    with TestClient(application) as client:
        monkeypatch.setattr(
            main_module.httpx,
            "AsyncClient",
            _PrefectHealthClient,
        )
        healthy = client.get("/ready")
        assert healthy.status_code == 200
        assert healthy.json() == {"status": "ok"}
        assert _PrefectHealthClient.requested_urls == [
            "http://prefect.test:4200/api/health"
        ]

        _PrefectHealthClient.status_code = 503
        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"detail": "Service unavailable"}
        liveness = client.get("/health")
        assert liveness.status_code == 200
        assert liveness.json() == {"status": "ok"}

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.workbench import WorkbenchService
from notarius_api.settings import Settings
from notarius_api.v1.routes.workbench import workbench_service


async def _create_schema(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
    finally:
        await database.dispose()


@pytest.fixture
def builtin_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    service = WorkbenchService(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
    )
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        )
    )
    application.dependency_overrides[workbench_service] = lambda: service
    with TestClient(application) as client:
        yield client

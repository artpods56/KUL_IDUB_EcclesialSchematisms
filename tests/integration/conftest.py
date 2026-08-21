"""Fixtures for API integration tests (full-app workbench clients)."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from grafy_core.artifacts import InMemoryUnitOfWork
from grafy_core.application.saved_graphs import SavedGraphService
from grafy_core.operators.tables import TableArtifactWriter
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemySavedGraphUnitOfWork

from grafy_api.builtins import builtin_plugins
from grafy_api.plugin_discovery import build_plugin_registry
from grafy_api.services.composition import (
    WorkbenchComponents,
    build_workbench_components,
)
from grafy_api.settings import Settings
from grafy_storage import LocalFileObjectStore

from tests.support.identity import create_schema
from tests.support.scenarios.conversion_path import CONVERSION_PATH_PLUGIN
from tests.support.scenarios.structural_projection import (
    STRUCTURAL_PROJECTION_PLUGIN,
)
from tests.support.workbench import workbench_dependency_overrides
from tests.testkit import client_with_overrides


@pytest.fixture
def builtin_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    saved_graph_database = create_database(database_url)
    saved_graphs = SavedGraphService(
        lambda: SqlAlchemySavedGraphUnitOfWork(saved_graph_database.sessions),
        registry,
    )
    components = build_workbench_components(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
        saved_graphs=saved_graphs,
    )
    try:
        with client_with_overrides(
            settings=Settings(
                workspace=tmp_path / "workbench",
                database_url=SecretStr(database_url),
            ),
            overrides=workbench_dependency_overrides(components),
        ) as client:
            yield client
    finally:
        asyncio.run(saved_graph_database.dispose())


@pytest.fixture
def table_artifact_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, TableArtifactWriter, WorkbenchComponents]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(create_schema(database_url))
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "workbench" / "objects")
    components = build_workbench_components(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
        unit_of_work=unit_of_work,
        storage=storage,
    )
    writer = TableArtifactWriter(
        storage=storage,
        uow=unit_of_work,
        bucket="workbench-artifacts",
        storage_backend="local",
    )
    with client_with_overrides(
        settings=Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        ),
        overrides=workbench_dependency_overrides(components),
    ) as client:
        yield client, writer, components


@pytest.fixture
def conversion_path_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, InMemoryUnitOfWork]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(create_schema(database_url))
    registry = build_plugin_registry(
        (*builtin_plugins(), CONVERSION_PATH_PLUGIN),
        external_plugins=(),
    )
    uow = InMemoryUnitOfWork()
    components = build_workbench_components(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
        unit_of_work=uow,
    )
    with client_with_overrides(
        settings=Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        ),
        overrides=workbench_dependency_overrides(components),
    ) as client:
        yield client, uow


@pytest.fixture
def structural_projection_client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite3'}"
    asyncio.run(create_schema(database_url))
    registry = build_plugin_registry(
        (*builtin_plugins(), STRUCTURAL_PROJECTION_PLUGIN),
        external_plugins=(),
    )
    components = build_workbench_components(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
    )
    with client_with_overrides(
        settings=Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        ),
        overrides=workbench_dependency_overrides(components),
    ) as client:
        yield client

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.workbench import WorkbenchService
from notarius_api.v1.routes.workbench import workbench_service


@pytest.fixture
def builtin_client(tmp_path: Path) -> Iterator[TestClient]:
    registry = build_plugin_registry(
        builtin_plugins(),
        external_plugins=(),
    )
    service = WorkbenchService(
        plugin_registry=registry,
        workspace=tmp_path / "workbench",
    )
    application = create_app()
    application.dependency_overrides[workbench_service] = lambda: service
    client = TestClient(application)
    yield client
    client.close()

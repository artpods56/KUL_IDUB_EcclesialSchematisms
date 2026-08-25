"""Phase 7 one-API-owner startup fence."""

from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from grafy_api.main import create_app
from grafy_api.settings import Settings
from grafy_api.single_owner import ApiOwnerLease, assert_single_http_worker
from grafy_persistence.database import create_database
from grafy_persistence.orm import metadata


pytestmark = pytest.mark.single_api_owner


def test_require_single_api_owner_defaults_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAFY_REQUIRE_SINGLE_API_OWNER", raising=False)
    assert Settings().require_single_api_owner is True


def test_assert_single_http_worker_rejects_multi_worker_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(RuntimeError, match="exactly one API HTTP worker"):
        assert_single_http_worker()


def test_api_owner_lease_rejects_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / ".grafy-api-owner.lock"
    first = ApiOwnerLease(lock_path)
    first.acquire()
    try:
        second = ApiOwnerLease(lock_path)
        with pytest.raises(RuntimeError, match="Another Grafy API owner"):
            second.acquire()
    finally:
        first.release()


@pytest.mark.asyncio
async def test_create_app_startup_acquires_owner_lease(tmp_path: Path) -> None:
    workspace = tmp_path / "workbench"
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'owner.sqlite3'}"
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    await database.dispose()

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        workspace=workspace,
        database_url=SecretStr(database_url),
        command_hmac_key=SecretStr("test-single-owner-hmac-key"),
        require_single_api_owner=True,
    )
    app = create_app(settings)
    async with LifespanManager(app):
        lock_path = workspace / ".grafy-api-owner.lock"
        assert lock_path.is_file()
        contested = ApiOwnerLease(lock_path)
        with pytest.raises(RuntimeError, match="Another Grafy API owner"):
            contested.acquire()

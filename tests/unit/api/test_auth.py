import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import SecretStr

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.services import AuthService, IssuedSession
from notarius_core.application.identity import IdentityService
from notarius_core.domain.identity import (
    User,
    Workspace,
    WorkspaceCapability,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


def _settings(database_url: str, *, idle_seconds: int = 1800) -> Settings:
    return Settings(
        public_origin="http://testserver",
        auth_cookie_secure=False,
        auth_session_idle_seconds=idle_seconds,
        database_url=SecretStr(database_url),
        execution_backend="inline",
    )


async def _seed(database_url: str) -> tuple[User, Workspace, IssuedSession]:
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    user = User(
        id=UUID(int=1),
        email="owner@example.test",
        display_name="Owner",
    )
    workspace = Workspace.shared(slug="local", name="Local workspace")
    membership = WorkspaceMembership(
        workspace_id=workspace.id,
        user_id=user.id,
        role=WorkspaceRole.OWNER,
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(user)
        await unit_of_work.identity.add_workspace(workspace)
        await unit_of_work.identity.add_membership(membership)
        await unit_of_work.commit()
    auth = AuthService(
        settings=_settings(database_url),
        unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
        identity_service=IdentityService(
            lambda: SqlAlchemyUnitOfWork(database.sessions)
        ),
    )
    issued = await auth.issue_session(user.id)
    await database.dispose()
    return user, workspace, issued


async def _set_session_expiry(database_url: str, session_id: UUID, *, expires_at: datetime) -> None:
    database = create_database(database_url)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        stored = await unit_of_work.identity.get_auth_session(session_id)
        assert stored is not None
        stored.expires_at = expires_at
        await unit_of_work.commit()
    await database.dispose()


def _seed_sync(database_url: str) -> tuple[User, Workspace, IssuedSession]:
    return asyncio.run(_seed(database_url))


def test_v1_routes_fail_closed_but_health_is_public(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'auth.sqlite3'}"
    _seed_sync(database_url)
    with TestClient(create_app(_settings(database_url))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/nodes").status_code == 401
        assert client.get("/v1/workspaces").status_code == 401
        assert client.get("/v1/auth/session").status_code == 401


def test_session_verification_failures_are_rate_limited(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'auth-rate.sqlite3'}"
    _seed_sync(database_url)
    settings = _settings(database_url)
    settings = settings.model_copy(update={"auth_session_failure_rate_limit": 1})
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/auth/session").status_code == 401
        assert client.get("/v1/auth/session").status_code == 429


def test_cookie_requests_require_exact_origin_and_csrf(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'csrf.sqlite3'}"
    _, _, issued = _seed_sync(database_url)
    with TestClient(create_app(_settings(database_url))) as client:
        client.cookies.set("notarius_session", issued.cookie_value)
        client.cookies.set("notarius_csrf", issued.csrf_value)
        assert client.post("/v1/workspaces", json={"slug": "team", "name": "Team"}).status_code == 403
        assert client.post(
            "/v1/workspaces",
            json={"slug": "team", "name": "Team"},
            headers={"Origin": "http://evil.example", "X-CSRF-Token": issued.csrf_value},
        ).status_code == 403


def test_session_idle_expiry_is_enforced_at_boundary(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'idle.sqlite3'}"
    user, _, issued = _seed_sync(database_url)
    old = datetime.now(UTC) - timedelta(seconds=1800)

    async def age_session() -> None:
        database = create_database(database_url)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            stored = await unit_of_work.identity.get_auth_session(issued.session.id)
            assert stored is not None
            stored.last_used_at = old
            await unit_of_work.commit()
        await database.dispose()

    asyncio.run(age_session())
    del user
    with TestClient(create_app(_settings(database_url, idle_seconds=1800))) as client:
        client.cookies.set("notarius_session", issued.cookie_value)
        client.cookies.set("notarius_csrf", issued.csrf_value)
        assert client.get("/v1/auth/session").status_code == 401


def test_session_absolute_expiry_is_enforced(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'absolute.sqlite3'}"
    _, _, issued = _seed_sync(database_url)
    asyncio.run(
        _set_session_expiry(
            database_url,
            issued.session.id,
            expires_at=datetime.now(UTC),
        )
    )
    with TestClient(create_app(_settings(database_url))) as client:
        client.cookies.set("notarius_session", issued.cookie_value)
        client.cookies.set("notarius_csrf", issued.csrf_value)
        assert client.get("/v1/auth/session").status_code == 401


def test_pat_create_shows_secret_once_and_revoke_is_audited(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'pat.sqlite3'}"
    _, workspace, issued = _seed_sync(database_url)
    with TestClient(create_app(_settings(database_url))) as client:
        client.cookies.set("notarius_session", issued.cookie_value)
        client.cookies.set("notarius_csrf", issued.csrf_value)
        headers = {"Origin": "http://testserver", "X-CSRF-Token": issued.csrf_value}
        response = client.post(
            f"/v1/workspaces/{workspace.id}/personal-access-tokens",
            headers=headers,
            json={
                "label": "read-only",
                "scopes": [WorkspaceCapability.VIEW_GRAPH.value],
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        )
        assert response.status_code == 201
        raw_token = response.json()["token"]
        listed = client.get(
            f"/v1/workspaces/{workspace.id}/personal-access-tokens"
        )
        assert listed.status_code == 200
        assert raw_token not in listed.text
        token_id = response.json()["id"]
        disallowed = client.post(
            f"/v1/workspaces/{workspace.id}/personal-access-tokens",
            headers=headers,
            json={
                "label": "admin",
                "scopes": [WorkspaceCapability.MANAGE_MEMBERS.value],
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        )
        assert disallowed.status_code == 422
        revoked = client.delete(
            f"/v1/workspaces/{workspace.id}/personal-access-tokens/{token_id}",
            headers=headers,
        )
        assert revoked.status_code == 204

    async def read_audits() -> list[str]:
        database = create_database(database_url)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            events = await unit_of_work.security_audit.list_for_workspace(
                workspace.id,
                limit=100,
            )
        await database.dispose()
        return [event.operation for event in events]

    assert "credential.pat.create" in asyncio.run(read_audits())
    assert "credential.pat.revoke" in asyncio.run(read_audits())

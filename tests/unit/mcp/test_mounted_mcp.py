"""Mounted Streamable HTTP MCP: PAT auth, live-head, and collaboration commands."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from asgi_lifespan import LifespanManager
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
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


def _settings(database_url: str) -> Settings:
    return Settings(
        public_origin="http://testserver",
        auth_cookie_secure=False,
        database_url=SecretStr(database_url),
        execution_backend="inline",
        command_hmac_key=SecretStr("test-mcp-command-hmac-key"),
    )


async def _seed_owner_workspace(
    database_url: str,
) -> tuple[User, Workspace, IssuedSession]:
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


async def _seed_two_workspaces(
    database_url: str,
) -> tuple[User, Workspace, Workspace, IssuedSession, IssuedSession]:
    """Owner of workspace A plus a distinct owner/session for workspace B."""

    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    owner_a = User(
        id=UUID(int=1),
        email="owner-a@example.test",
        display_name="Owner A",
    )
    owner_b = User(
        id=UUID(int=2),
        email="owner-b@example.test",
        display_name="Owner B",
    )
    workspace_a = Workspace.shared(slug="workspace-a", name="Workspace A")
    workspace_b = Workspace.shared(slug="workspace-b", name="Workspace B")
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(owner_a)
        await unit_of_work.identity.add_user(owner_b)
        await unit_of_work.identity.add_workspace(workspace_a)
        await unit_of_work.identity.add_workspace(workspace_b)
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=workspace_a.id,
                user_id=owner_a.id,
                role=WorkspaceRole.OWNER,
            )
        )
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=workspace_b.id,
                user_id=owner_b.id,
                role=WorkspaceRole.OWNER,
            )
        )
        await unit_of_work.commit()
    auth = AuthService(
        settings=_settings(database_url),
        unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
        identity_service=IdentityService(
            lambda: SqlAlchemyUnitOfWork(database.sessions)
        ),
    )
    issued_a = await auth.issue_session(owner_a.id)
    issued_b = await auth.issue_session(owner_b.id)
    await database.dispose()
    return owner_a, workspace_a, workspace_b, issued_a, issued_b


async def _seed_shared_with_member(
    database_url: str,
) -> tuple[User, User, Workspace, IssuedSession, IssuedSession]:
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    owner = User(
        id=UUID(int=1),
        email="owner@example.test",
        display_name="Owner",
    )
    member = User(
        id=UUID(int=2),
        email="member@example.test",
        display_name="Member",
    )
    workspace = Workspace.shared(slug="shared", name="Shared workspace")
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(owner)
        await unit_of_work.identity.add_user(member)
        await unit_of_work.identity.add_workspace(workspace)
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=workspace.id,
                user_id=owner.id,
                role=WorkspaceRole.OWNER,
            )
        )
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=workspace.id,
                user_id=member.id,
                role=WorkspaceRole.EDITOR,
            )
        )
        await unit_of_work.commit()
    auth = AuthService(
        settings=_settings(database_url),
        unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
        identity_service=IdentityService(
            lambda: SqlAlchemyUnitOfWork(database.sessions)
        ),
    )
    owner_session = await auth.issue_session(owner.id)
    member_session = await auth.issue_session(member.id)
    await database.dispose()
    return owner, member, workspace, owner_session, member_session


def _issue_pat(
    *,
    database_url: str,
    workspace_id: UUID,
    issued: IssuedSession,
    scopes: list[str],
) -> tuple[str, str]:
    with TestClient(create_app(_settings(database_url))) as client:
        client.cookies.set("notarius_session", issued.cookie_value)
        client.cookies.set("notarius_csrf", issued.csrf_value)
        response = client.post(
            f"/v1/workspaces/{workspace_id}/personal-access-tokens",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": issued.csrf_value,
            },
            json={
                "label": "mcp",
                "scopes": scopes,
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return body["token"], body["id"]


def _mcp_client_factory(app, raw_token: str):
    transport = ASGITransport(app=app)

    def factory(**kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {raw_token}"
        return AsyncClient(
            transport=transport,
            base_url="http://test",
            headers=headers,
            **kwargs,
        )

    return StreamableHttpTransport(
        url="http://test/mcp/",
        httpx_client_factory=factory,
    )


def _assert_permission_denied(exc_info: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(exc_info.value).lower()
    assert "permission" in message or "lacks permission" in message


def _assert_not_found(exc_info: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(exc_info.value).lower()
    assert "not found" in message


def _assert_conflict(exc_info: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(exc_info.value).lower()
    assert "changed" in message or "conflict" in message or "reconcile" in message


_WRITE_SCOPES = [
    WorkspaceCapability.VIEW_GRAPH.value,
    WorkspaceCapability.CREATE_GRAPH.value,
    WorkspaceCapability.EDIT_GRAPH.value,
]


@pytest.mark.asyncio
async def test_mcp_rejects_missing_and_revoked_pats(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-auth.sqlite3'}"
    _, workspace, issued = await _seed_owner_workspace(database_url)
    raw_token, token_id = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=[WorkspaceCapability.VIEW_GRAPH.value],
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            missing = await http.post("/mcp/", follow_redirects=False)
            assert missing.status_code == 401

        async with Client(_mcp_client_factory(app, raw_token)) as client:
            tools = await client.list_tools()
            assert "get_live_head" in {tool.name for tool in tools}
            listed = await client.call_tool("list_graphs")
            assert listed.structured_content["graphs"] == []

        with TestClient(create_app(_settings(database_url))) as browser:
            browser.cookies.set("notarius_session", issued.cookie_value)
            browser.cookies.set("notarius_csrf", issued.csrf_value)
            revoked = browser.delete(
                f"/v1/workspaces/{workspace.id}/personal-access-tokens/{token_id}",
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": issued.csrf_value,
                },
            )
            assert revoked.status_code == 204

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            after_revoke = await http.post(
                "/mcp/",
                headers={"Authorization": f"Bearer {raw_token}"},
                follow_redirects=False,
            )
            assert after_revoke.status_code == 401


@pytest.mark.asyncio
async def test_mcp_read_pat_cannot_mutate(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-read.sqlite3'}"
    _, workspace, issued = await _seed_owner_workspace(database_url)
    write_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=_WRITE_SCOPES,
    )
    read_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=[WorkspaceCapability.VIEW_GRAPH.value],
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, write_token)) as writer:
            created = await writer.call_tool(
                "create_graph",
                {"graph": {"name": "seed", "nodes": [], "edges": []}},
            )
            created_body = created.structured_content
            assert created_body is not None
            graph_id = created_body["id"]
            head = await writer.call_tool(
                "get_live_head",
                {"graph_id": graph_id},
            )
            head_body = head.structured_content
            assert head_body is not None

        async with Client(_mcp_client_factory(app, read_token)) as reader:
            listed = await reader.call_tool("list_graphs")
            assert len(listed.structured_content["graphs"]) == 1
            live = await reader.call_tool(
                "get_live_head",
                {"graph_id": graph_id},
            )
            assert live.structured_content["name"] == "seed"

            with pytest.raises(Exception) as create_exc:
                await reader.call_tool(
                    "create_graph",
                    {"graph": {"name": "blocked", "nodes": [], "edges": []}},
                )
            _assert_permission_denied(create_exc)

            with pytest.raises(Exception) as replace_exc:
                await reader.call_tool(
                    "replace_graph",
                    {
                        "graph_id": graph_id,
                        "expected_revision": created_body["revision"],
                        "graph": {"name": "blocked-replace", "nodes": [], "edges": []},
                    },
                )
            _assert_permission_denied(replace_exc)

            with pytest.raises(Exception) as command_exc:
                await reader.call_tool(
                    "submit_graph_command",
                    {
                        "graph_id": graph_id,
                        "command_id": str(uuid4()),
                        "room_epoch": head_body["room_epoch"],
                        "observed_sequence": head_body["collaboration_sequence"],
                        "command": {
                            "kind": "rename_graph",
                            "name": "blocked-rename",
                            "expected_name": "seed",
                        },
                    },
                )
            _assert_permission_denied(command_exc)


@pytest.mark.asyncio
async def test_mcp_write_pat_create_replace_and_command(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-write.sqlite3'}"
    _, workspace, issued = await _seed_owner_workspace(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=_WRITE_SCOPES,
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            created = await client.call_tool(
                "create_graph",
                {"graph": {"name": "Writable", "nodes": [], "edges": []}},
            )
            created_body = created.structured_content
            assert created_body is not None
            graph_id = created_body["id"]
            assert created_body["revision"] == 1

            replaced = await client.call_tool(
                "replace_graph",
                {
                    "graph_id": graph_id,
                    "expected_revision": 1,
                    "graph": {"name": "Replaced", "nodes": [], "edges": []},
                },
            )
            replaced_body = replaced.structured_content
            assert replaced_body is not None
            assert replaced_body["name"] == "Replaced"
            assert replaced_body["revision"] == 2

            head = await client.call_tool(
                "get_live_head",
                {"graph_id": graph_id},
            )
            head_body = head.structured_content
            assert head_body is not None
            assert head_body["name"] == "Replaced"
            assert head_body["checkpoint_revision"] == 2

            command_id = uuid4()
            submitted = await client.call_tool(
                "submit_graph_command",
                {
                    "graph_id": graph_id,
                    "command_id": str(command_id),
                    "room_epoch": head_body["room_epoch"],
                    "observed_sequence": head_body["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Commanded",
                        "expected_name": "Replaced",
                    },
                },
            )
            submitted_body = submitted.structured_content
            assert submitted_body is not None
            assert submitted_body["receipt"]["outcome"] == "accepted"
            assert submitted_body["head"]["name"] == "Commanded"


@pytest.mark.asyncio
async def test_mcp_foreign_workspace_pat_cannot_access_graph(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-foreign.sqlite3'}"
    _, workspace_a, workspace_b, issued_a, issued_b = await _seed_two_workspaces(
        database_url
    )
    token_a, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace_a.id,
        issued=issued_a,
        scopes=_WRITE_SCOPES,
    )
    token_b, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace_b.id,
        issued=issued_b,
        scopes=_WRITE_SCOPES,
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, token_a)) as client_a:
            created = await client_a.call_tool(
                "create_graph",
                {"graph": {"name": "A only", "nodes": [], "edges": []}},
            )
            graph_id = created.structured_content["id"]

        async with Client(_mcp_client_factory(app, token_b)) as client_b:
            listed = await client_b.call_tool("list_graphs")
            assert listed.structured_content["graphs"] == []
            with pytest.raises(Exception) as get_exc:
                await client_b.call_tool(
                    "get_live_head",
                    {"graph_id": graph_id},
                )
            _assert_not_found(get_exc)
            with pytest.raises(Exception) as replace_exc:
                await client_b.call_tool(
                    "replace_graph",
                    {
                        "graph_id": graph_id,
                        "expected_revision": 1,
                        "graph": {"name": "hijack", "nodes": [], "edges": []},
                    },
                )
            _assert_not_found(replace_exc)


@pytest.mark.asyncio
async def test_mcp_fails_closed_after_membership_removal(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-member.sqlite3'}"
    owner, member, workspace, owner_session, member_session = (
        await _seed_shared_with_member(database_url)
    )
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=member_session,
        scopes=_WRITE_SCOPES,
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            listed = await client.call_tool("list_graphs")
            assert listed.structured_content["graphs"] == []

        with TestClient(create_app(_settings(database_url))) as browser:
            browser.cookies.set("notarius_session", owner_session.cookie_value)
            browser.cookies.set("notarius_csrf", owner_session.csrf_value)
            removed = browser.delete(
                f"/v1/workspaces/{workspace.id}/members/{member.id}",
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": owner_session.csrf_value,
                },
            )
            assert removed.status_code == 204, removed.text

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            after_removal = await http.post(
                "/mcp/",
                headers={"Authorization": f"Bearer {raw_token}"},
                follow_redirects=False,
            )
            assert after_removal.status_code == 401


@pytest.mark.asyncio
async def test_mcp_fails_closed_after_user_disable(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-disable.sqlite3'}"
    user, workspace, issued = await _seed_owner_workspace(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=[WorkspaceCapability.VIEW_GRAPH.value],
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            listed = await client.call_tool("list_graphs")
            assert listed.structured_content["graphs"] == []

        database = create_database(database_url)
        identity = IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))
        await identity.disable_user(user_id=user.id)
        await database.dispose()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            after_disable = await http.post(
                "/mcp/",
                headers={"Authorization": f"Bearer {raw_token}"},
                follow_redirects=False,
            )
            assert after_disable.status_code == 401


@pytest.mark.asyncio
async def test_mcp_replace_conflict_and_command_idempotency(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-conflict.sqlite3'}"
    _, workspace, issued = await _seed_owner_workspace(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=_WRITE_SCOPES,
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            created = await client.call_tool(
                "create_graph",
                {"graph": {"name": "Conflict", "nodes": [], "edges": []}},
            )
            graph_id = created.structured_content["id"]

            with pytest.raises(Exception) as stale_exc:
                await client.call_tool(
                    "replace_graph",
                    {
                        "graph_id": graph_id,
                        "expected_revision": 99,
                        "graph": {"name": "Stale", "nodes": [], "edges": []},
                    },
                )
            _assert_conflict(stale_exc)

            head = await client.call_tool(
                "get_live_head",
                {"graph_id": graph_id},
            )
            head_body = head.structured_content
            assert head_body is not None
            command_id = uuid4()
            first = await client.call_tool(
                "submit_graph_command",
                {
                    "graph_id": graph_id,
                    "command_id": str(command_id),
                    "room_epoch": head_body["room_epoch"],
                    "observed_sequence": head_body["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Once",
                        "expected_name": "Conflict",
                    },
                },
            )
            assert first.structured_content["receipt"]["outcome"] == "accepted"
            replay = await client.call_tool(
                "submit_graph_command",
                {
                    "graph_id": graph_id,
                    "command_id": str(command_id),
                    "room_epoch": head_body["room_epoch"],
                    "observed_sequence": head_body["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Once",
                        "expected_name": "Conflict",
                    },
                },
            )
            assert replay.structured_content["receipt"]["outcome"] == "idempotent_replay"
            assert replay.structured_content["head"]["name"] == "Once"


@pytest.mark.asyncio
async def test_mcp_live_head_and_command_through_collaboration(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-collab.sqlite3'}"
    _, workspace, issued = await _seed_owner_workspace(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=_WRITE_SCOPES,
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            created = await client.call_tool(
                "create_graph",
                {
                    "graph": {
                        "name": "MCP graph",
                        "nodes": [],
                        "edges": [],
                    }
                },
            )
            created_body = created.structured_content
            assert created_body is not None
            graph_id = UUID(created_body["id"])
            head = await client.call_tool(
                "get_live_head",
                {"graph_id": str(graph_id)},
            )
            head_body = head.structured_content
            assert head_body is not None
            assert head_body["name"] == "MCP graph"
            assert head_body["collaboration_sequence"] == 1
            assert head_body["checkpoint_sequence"] == 1
            assert head_body["room_epoch"]
            command_id = uuid4()
            submitted = await client.call_tool(
                "submit_graph_command",
                {
                    "graph_id": str(graph_id),
                    "command_id": str(command_id),
                    "room_epoch": head_body["room_epoch"],
                    "observed_sequence": head_body["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Renamed by MCP",
                        "expected_name": "MCP graph",
                    },
                },
            )
            submitted_body = submitted.structured_content
            assert submitted_body is not None
            assert submitted_body["receipt"]["outcome"] == "accepted"
            assert submitted_body["head"]["name"] == "Renamed by MCP"
            assert submitted_body["head"]["collaboration_sequence"] == 2

            # Workspace must not be accepted as a tool argument.
            tools = await client.list_tools()
            schemas = {tool.name: tool.inputSchema for tool in tools}
            for tool_name in (
                "list_graphs",
                "get_live_head",
                "create_graph",
                "submit_graph_command",
                "replace_graph",
            ):
                properties = schemas[tool_name].get("properties", {})
                assert "workspace_id" not in properties
                assert "workspace" not in properties

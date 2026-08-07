"""Mounted Streamable HTTP MCP: PAT auth, live-head, and collaboration commands."""

import asyncio
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


@pytest.mark.asyncio
async def test_mcp_rejects_missing_and_revoked_pats(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-auth.sqlite3'}"
    _, workspace, issued = await _seed(database_url)
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
    _, workspace, issued = await _seed(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=[WorkspaceCapability.VIEW_GRAPH.value],
    )
    app = create_app(_settings(database_url))
    async with LifespanManager(app):
        async with Client(_mcp_client_factory(app, raw_token)) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "create_graph",
                    {
                        "graph": {
                            "name": "blocked",
                            "nodes": [],
                            "edges": [],
                        }
                    },
                )
            assert "lacks permission" in str(exc_info.value).lower() or "permission" in str(
                exc_info.value
            ).lower()


@pytest.mark.asyncio
async def test_mcp_live_head_and_command_through_collaboration(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-collab.sqlite3'}"
    _, workspace, issued = await _seed(database_url)
    raw_token, _ = _issue_pat(
        database_url=database_url,
        workspace_id=workspace.id,
        issued=issued,
        scopes=[
            WorkspaceCapability.VIEW_GRAPH.value,
            WorkspaceCapability.CREATE_GRAPH.value,
            WorkspaceCapability.EDIT_GRAPH.value,
        ],
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
            ):
                properties = schemas[tool_name].get("properties", {})
                assert "workspace_id" not in properties
                assert "workspace" not in properties

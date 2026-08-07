"""Pinned FastMCP Streamable HTTP mount compatibility gate."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_mcp.request_context import bind_mcp_request, current_mcp_binding, reset_mcp_request
from notarius_mcp.operations import McpCallerContext
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata


@pytest.mark.asyncio
async def test_fastmcp_streamable_http_mounts_under_fastapi_stateless() -> None:
    mcp = FastMCP("compat-probe")
    seen_tokens: list[str] = []

    @mcp.tool
    async def echo_authorization() -> str:
        from fastmcp.server.dependencies import get_http_request

        authorization = get_http_request().headers.get("authorization", "missing")
        seen_tokens.append(authorization)
        return authorization

    mcp_app = mcp.http_app(
        path="/",
        transport="streamable-http",
        stateless_http=True,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_app.lifespan(app):
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/mcp", mcp_app)

    async with LifespanManager(app):
        transport = ASGITransport(app=app)

        def factory(**kwargs):
            headers = dict(kwargs.pop("headers", {}) or {})
            headers["Authorization"] = "Bearer compat-token"
            return AsyncClient(
                transport=transport,
                base_url="http://test",
                headers=headers,
                **kwargs,
            )

        http_transport = StreamableHttpTransport(
            url="http://test/mcp/",
            httpx_client_factory=factory,
        )
        async with Client(http_transport) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools] == ["echo_authorization"]
            first = await client.call_tool("echo_authorization")
            second, third = await asyncio.gather(
                client.call_tool("echo_authorization"),
                client.call_tool("echo_authorization"),
            )
            assert first.data == "Bearer compat-token"
            assert second.data == "Bearer compat-token"
            assert third.data == "Bearer compat-token"
            assert seen_tokens == [
                "Bearer compat-token",
                "Bearer compat-token",
                "Bearer compat-token",
            ]


@pytest.mark.asyncio
async def test_request_binding_isolates_concurrent_callers() -> None:
    results: list[str] = []

    async def run_with_caller(label: str) -> None:
        caller = McpCallerContext(
            user_id=__import__("uuid").UUID(int=1),
            workspace_id=__import__("uuid").UUID(int=2),
            credential_reference=label,
            scopes=frozenset({"view_graph"}),
        )

        class _Ops:
            async def get_registry(self, bound_caller: McpCallerContext):
                del bound_caller
                raise NotImplementedError

        token = bind_mcp_request(caller, _Ops())  # type: ignore[arg-type]
        try:
            await asyncio.sleep(0)
            results.append(current_mcp_binding().caller.credential_reference)
        finally:
            reset_mcp_request(token)

    await asyncio.gather(run_with_caller("pat:a"), run_with_caller("pat:b"))
    assert sorted(results) == ["pat:a", "pat:b"]


@pytest.mark.asyncio
async def test_create_app_composes_mcp_lifespan(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'mcp-lifespan.sqlite3'}"
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    await database.dispose()

    settings = Settings(
        public_origin="http://testserver",
        auth_cookie_secure=False,
        database_url=SecretStr(database_url),
        execution_backend="inline",
        command_hmac_key=SecretStr("test-mcp-command-hmac-key"),
    )
    app = create_app(settings)
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            unauthenticated = await client.post(
                "/mcp/",
                follow_redirects=False,
            )
            assert unauthenticated.status_code == 401

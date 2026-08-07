"""Mount FastMCP Streamable HTTP under the FastAPI authority."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from notarius_mcp.operations import McpCallerContext
from notarius_mcp.request_context import bind_mcp_request, reset_mcp_request
from notarius_mcp.server import create_streamable_http_app

from notarius_api.app_state import get_identity
from notarius_api.mcp.operations import ApiGraphWorkspaceOperations
from notarius_api.v1.routes.auth.services import AuthService


class _McpPatMiddleware(BaseHTTPMiddleware):
    """Authenticate every `/mcp` request with a workspace-bound PAT."""

    def __init__(self, app: ASGIApp, *, api_app: FastAPI) -> None:
        super().__init__(app)
        self._api_app = api_app

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        auth: AuthService = get_identity(self._api_app).auth_service
        try:
            access = await auth.require_mcp_access(request)
        except HTTPException as error:
            return JSONResponse(
                status_code=error.status_code,
                content={"detail": error.detail},
            )
        caller = McpCallerContext(
            user_id=access.actor.user_id,
            workspace_id=access.workspace_id,
            credential_reference=(
                access.actor.credential_reference or f"pat:{access.token_id}"
            ),
            scopes=frozenset(scope.value for scope in access.scopes),
        )
        operations = ApiGraphWorkspaceOperations(self._api_app)
        token = bind_mcp_request(caller, operations)
        try:
            return await call_next(request)
        finally:
            reset_mcp_request(token)


def create_mounted_mcp_app(api_app: FastAPI) -> Starlette:
    """Create the stateless Streamable HTTP app bound to the parent API app."""

    mcp_app = create_streamable_http_app(path="/")
    mcp_app.add_middleware(_McpPatMiddleware, api_app=api_app)
    return mcp_app


__all__ = ["create_mounted_mcp_app"]

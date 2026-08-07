from typing import TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from notarius_mcp.models import (
    CreateSavedGraphRequest,
    NodeRegistryResponse,
    SavedGraphListResponse,
    SavedGraphResponse,
    UpdateSavedGraphRequest,
)


ResponseT = TypeVar("ResponseT", bound=BaseModel)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class NotariusApiError(RuntimeError):
    def __init__(
        self,
        *,
        method: str,
        path: str,
        status_code: int | None = None,
        detail: JsonValue | None = None,
        raw_body: str | None = None,
    ) -> None:
        self.method = method.upper()
        self.path = path
        self.status_code = status_code
        self.detail = detail
        self.raw_body = raw_body

        message = f"Notarius API {self.method} {path} failed"
        if status_code is not None:
            message = f"{message} with status {status_code}"
        if detail is not None:
            message = f"{message}: {detail}"
        elif raw_body is not None:
            message = f"{message}: {raw_body}"
        super().__init__(message)


class NotariusApiClient:
    """Workspace-scoped HTTP client for the standalone stdio MCP process.

    Paths target `/v1/workspaces/{workspace_id}/...`. Live calls still lack a
    credential accepted by those routes (browser AuthSession cookie only until
    Phase 6 mounts Streamable HTTP MCP at `/mcp` with a workspace-bound PAT).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        workspace_id: UUID,
    ) -> None:
        self._http_client = http_client
        self._workspace_id = workspace_id
        self._workspace_root = f"/v1/workspaces/{workspace_id}"

    async def get_registry(self) -> NodeRegistryResponse:
        return await self._request(
            "GET",
            f"{self._workspace_root}/nodes",
            expected_status=200,
            response_model=NodeRegistryResponse,
        )

    async def list_graphs(self) -> SavedGraphListResponse:
        return await self._request(
            "GET",
            f"{self._workspace_root}/graphs",
            expected_status=200,
            response_model=SavedGraphListResponse,
        )

    async def get_graph(self, graph_id: UUID) -> SavedGraphResponse:
        path = f"{self._workspace_root}/graphs/{graph_id}"
        return await self._request(
            "GET",
            path,
            expected_status=200,
            response_model=SavedGraphResponse,
        )

    async def create_graph(
        self,
        request: CreateSavedGraphRequest,
    ) -> SavedGraphResponse:
        return await self._request(
            "POST",
            f"{self._workspace_root}/graphs",
            expected_status=201,
            response_model=SavedGraphResponse,
            request=request,
        )

    async def replace_graph(
        self,
        graph_id: UUID,
        request: UpdateSavedGraphRequest,
    ) -> SavedGraphResponse:
        path = f"{self._workspace_root}/graphs/{graph_id}"
        return await self._request(
            "PUT",
            path,
            expected_status=200,
            response_model=SavedGraphResponse,
            request=request,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: int,
        response_model: type[ResponseT],
        request: BaseModel | None = None,
    ) -> ResponseT:
        content: str | None = None
        headers: dict[str, str] | None = None
        if request is not None:
            content = request.model_dump_json()
            headers = {"content-type": "application/json"}

        try:
            response = await self._http_client.request(
                method,
                path,
                content=content,
                headers=headers,
            )
        except httpx.TransportError as exc:
            raise NotariusApiError(
                method=method,
                path=path,
                detail=str(exc),
            ) from exc

        if response.status_code != expected_status:
            detail: JsonValue | None = None
            raw_body: str | None = None
            try:
                error_body = _JSON_VALUE_ADAPTER.validate_json(response.content)
            except ValidationError:
                raw_body = response.text
            else:
                if isinstance(error_body, dict) and "detail" in error_body:
                    detail = error_body["detail"]
                else:
                    detail = error_body
            raise NotariusApiError(
                method=method,
                path=path,
                status_code=response.status_code,
                detail=detail,
                raw_body=raw_body,
            )

        try:
            return response_model.model_validate_json(response.content)
        except ValidationError as exc:
            raise NotariusApiError(
                method=method,
                path=path,
                status_code=response.status_code,
                detail=(
                    f"Invalid {response_model.__name__} response from Notarius API: "
                    f"{exc}"
                ),
            ) from exc


__all__ = ["NotariusApiClient", "NotariusApiError"]

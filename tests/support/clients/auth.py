from __future__ import annotations

from typing import Mapping
from uuid import UUID

from httpx import Response
from starlette.testclient import TestClient

from grafy_api.v1.routes.auth.models import SessionResponse
from tests.support.clients._http import _expect, _parse, _parse_list


class AuthApi:
    """``/v1/auth`` session endpoints."""

    __slots__ = ("_client",)

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def get_session(self, *, headers: Mapping[str, str] | None = None) -> Response:
        return self._client.get("/v1/auth/session", headers=headers)

    def get_session_ok(
        self, *, headers: Mapping[str, str] | None = None
    ) -> SessionResponse:
        return _parse(SessionResponse, _expect(self.get_session(headers=headers), 200))

    def logout(self, *, headers: Mapping[str, str] | None = None) -> Response:
        return self._client.delete("/v1/auth/session", headers=headers)

    def list_sessions(self, *, headers: Mapping[str, str] | None = None) -> Response:
        return self._client.get("/v1/auth/sessions", headers=headers)

    def list_sessions_ok(
        self, *, headers: Mapping[str, str] | None = None
    ) -> list[SessionResponse]:
        response = _expect(self.list_sessions(headers=headers), 200)
        return _parse_list(SessionResponse, response)

    def revoke_session(
        self, session_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.delete(f"/v1/auth/sessions/{session_id}", headers=headers)

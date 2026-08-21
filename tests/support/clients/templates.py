from __future__ import annotations

from typing import Mapping
from uuid import UUID

from httpx import Response
from starlette.testclient import TestClient

from grafy_api.v1.routes.templates.models import (
    CreateTemplateRequest,
    InstantiateTemplateRequest,
    TemplateInstantiationResponse,
    TemplateListResponse,
    TemplateResponse,
    UpdateTemplateMetadataRequest,
)
from tests.support.clients._http import _expect, _parse, _request


class TemplatesApi:
    """The ``/v1/workspaces/{workspace_id}/templates`` library endpoints."""

    __slots__ = ("_client", "_workspace_id")

    def __init__(self, client: TestClient, workspace_id: UUID) -> None:
        self._client = client
        self._workspace_id = workspace_id

    def list(
        self,
        *,
        q: str | None = None,
        include_archived: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        params: dict[str, str] = {}
        if q is not None:
            params["q"] = q
        if include_archived:
            params["include_archived"] = "true"
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/templates",
            params=params,
            headers=headers,
        )

    def list_ok(
        self,
        *,
        q: str | None = None,
        include_archived: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> TemplateListResponse:
        return _parse(
            TemplateListResponse,
            _expect(
                self.list(q=q, include_archived=include_archived, headers=headers),
                200,
            ),
        )

    def create(
        self,
        payload: CreateTemplateRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "POST",
            f"/v1/workspaces/{self._workspace_id}/templates",
            payload=payload,
            headers=headers,
        )

    def create_ok(
        self,
        payload: CreateTemplateRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> TemplateResponse:
        return _parse(
            TemplateResponse, _expect(self.create(payload, headers=headers), 201)
        )

    def get(
        self, template_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/templates/{template_id}",
            headers=headers,
        )

    def get_ok(
        self, template_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> TemplateResponse:
        return _parse(
            TemplateResponse, _expect(self.get(template_id, headers=headers), 200)
        )

    def update_metadata(
        self,
        template_id: UUID,
        payload: UpdateTemplateMetadataRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "PUT",
            f"/v1/workspaces/{self._workspace_id}/templates/{template_id}",
            payload=payload,
            headers=headers,
        )

    def update_metadata_ok(
        self,
        template_id: UUID,
        payload: UpdateTemplateMetadataRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> TemplateResponse:
        return _parse(
            TemplateResponse,
            _expect(self.update_metadata(template_id, payload, headers=headers), 200),
        )

    def archive(
        self, template_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.post(
            f"/v1/workspaces/{self._workspace_id}/templates/{template_id}/archive",
            headers=headers,
        )

    def archive_ok(
        self, template_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> TemplateResponse:
        return _parse(
            TemplateResponse, _expect(self.archive(template_id, headers=headers), 200)
        )

    def instantiate(
        self,
        template_id: UUID,
        payload: InstantiateTemplateRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "POST",
            f"/v1/workspaces/{self._workspace_id}/templates/{template_id}/instantiate",
            payload=payload,
            headers=headers,
        )

    def instantiate_ok(
        self,
        template_id: UUID,
        payload: InstantiateTemplateRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> TemplateInstantiationResponse:
        return _parse(
            TemplateInstantiationResponse,
            _expect(self.instantiate(template_id, payload, headers=headers), 201),
        )

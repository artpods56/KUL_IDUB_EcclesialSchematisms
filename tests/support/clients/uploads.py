from __future__ import annotations

from typing import Mapping
from uuid import UUID

from httpx import Response
from starlette.testclient import TestClient

from grafy_api.v1.routes.uploads.models import ImageUploadItemResponse, SampleRequest
from tests.support.clients._http import _expect, _parse, _request


class UploadsApi:
    """Workbench uploads under ``/v1/workspaces/{workspace_id}``."""

    __slots__ = ("_client", "_workspace_id")

    def __init__(self, client: TestClient, workspace_id: UUID) -> None:
        self._client = client
        self._workspace_id = workspace_id

    def upload(
        self,
        filename: str,
        data: bytes,
        *,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """POST the single multipart ``file`` form field (not JSON)."""

        file_part = (
            (filename, data) if content_type is None else (filename, data, content_type)
        )
        return self._client.post(
            f"/v1/workspaces/{self._workspace_id}/uploads",
            files={"file": file_part},
            headers=headers,
        )

    def upload_ok(
        self,
        filename: str,
        data: bytes,
        *,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ImageUploadItemResponse:
        return _parse(
            ImageUploadItemResponse,
            _expect(
                self.upload(filename, data, content_type=content_type, headers=headers),
                200,
            ),
        )

    def create_samples(
        self,
        payload: SampleRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "POST",
            f"/v1/workspaces/{self._workspace_id}/samples",
            payload=payload,
            headers=headers,
        )

    def create_samples_ok(
        self,
        payload: SampleRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> list[ImageUploadItemResponse]:
        response = _expect(self.create_samples(payload, headers=headers), 200)
        return [_parse(ImageUploadItemResponse, item) for item in response.json()]

"""Shared request plumbing for the typed test clients (private module)."""

from __future__ import annotations

from typing import Mapping, TypeVar

from httpx import Response
from pydantic import BaseModel
from starlette.testclient import TestClient

ModelT = TypeVar("ModelT", bound=BaseModel)


def _request(
    client: TestClient,
    method: str,
    url: str,
    *,
    payload: BaseModel | None = None,
    headers: Mapping[str, str] | None = None,
) -> Response:
    """Send one request, serializing a typed payload when given."""

    body = payload.model_dump(mode="json") if payload is not None else None
    return client.request(method, url, json=body, headers=headers)


def _expect(response: Response, status_code: int) -> Response:
    """Assert the happy-path status, echoing the body on mismatch."""

    if response.status_code != status_code:
        raise AssertionError(
            f"expected HTTP {status_code}, got {response.status_code}: {response.text}"
        )
    return response


def _parse(model_type: type[ModelT], response: Response) -> ModelT:
    return model_type.model_validate(response.json())

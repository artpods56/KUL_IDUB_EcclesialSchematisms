from __future__ import annotations

from typing import Mapping
from uuid import UUID

from httpx import Response
from starlette.testclient import TestClient

from grafy_api.v1.routes.node_secrets.models import (
    ConfigureNodeSecretRequest,
    GraphNodeSecretsResponse,
    NodeSecretStatusResponse,
)
from tests.support.clients._http import _expect, _parse


class NodeSecretsApi:
    """Node secrets under ``/v1/workspaces/{workspace_id}/graphs``."""

    __slots__ = ("_client", "_workspace_id")

    def __init__(self, client: TestClient, workspace_id: UUID) -> None:
        self._client = client
        self._workspace_id = workspace_id

    # -- graph-scoped secret status ------------------------------------------

    def list_secrets(
        self, graph_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/graphs/{graph_id}/node-secrets",
            headers=headers,
        )

    def list_secrets_ok(
        self, graph_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> GraphNodeSecretsResponse:
        return _parse(
            GraphNodeSecretsResponse,
            _expect(self.list_secrets(graph_id, headers=headers), 200),
        )

    # -- per-node secret configuration ----------------------------------------

    def configure_secret(
        self,
        graph_id: UUID,
        node_id: str,
        name: str,
        payload: ConfigureNodeSecretRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Configure one node secret.

        ``ConfigureNodeSecretRequest.value`` is a redacting ``SecretStr``;
        the generic ``model_dump(mode="json")`` path would send the
        redaction sentinel, so the body is assembled from the model dump
        with only the secret un-redacted.
        """

        body = payload.model_dump(mode="json")
        body["value"] = payload.value.get_secret_value()
        return self._client.put(
            f"/v1/workspaces/{self._workspace_id}/graphs/{graph_id}"
            f"/nodes/{node_id}/secrets/{name}",
            json=body,
            headers=headers,
        )

    def configure_secret_ok(
        self,
        graph_id: UUID,
        node_id: str,
        name: str,
        payload: ConfigureNodeSecretRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> NodeSecretStatusResponse:
        return _parse(
            NodeSecretStatusResponse,
            _expect(
                self.configure_secret(
                    graph_id, node_id, name, payload, headers=headers
                ),
                200,
            ),
        )

    def remove_secret(
        self,
        graph_id: UUID,
        node_id: str,
        name: str,
        expected_graph_revision: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self._client.delete(
            f"/v1/workspaces/{self._workspace_id}/graphs/{graph_id}"
            f"/nodes/{node_id}/secrets/{name}",
            params={"expected_graph_revision": expected_graph_revision},
            headers=headers,
        )

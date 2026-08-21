from __future__ import annotations

from typing import Mapping
from uuid import UUID

from httpx import Response
from starlette.testclient import TestClient

from grafy_api.v1.routes.artifacts.models import (
    GeoExactFeatureResponse,
    GeoFeatureQueryRequest,
    GeoFeatureQueryResponse,
    GeoRasterTileJsonResponse,
    GeoRenderResponse,
    TableCellResponse,
    TablePageResponse,
    TableQueryRequest,
    TableSchemaResponse,
)
from tests.support.clients._http import _expect, _parse, _request


class ArtifactsApi:
    """Workbench artifacts scoped to one workspace.

    Covers the geo render/query/vector/raster, table page/schema/query/cell,
    and raw content/download endpoints under
    ``/v1/workspaces/{workspace_id}/artifacts``.
    """

    __slots__ = ("_client", "_workspace_id")

    def __init__(self, client: TestClient, workspace_id: UUID) -> None:
        self._client = client
        self._workspace_id = workspace_id

    # -- geo -----------------------------------------------------------------

    def geo_render(
        self, artifact_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/geo/render",
            headers=headers,
        )

    def geo_render_ok(
        self, artifact_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> GeoRenderResponse:
        return _parse(
            GeoRenderResponse,
            _expect(self.geo_render(artifact_id, headers=headers), 200),
        )

    def query_geo_features(
        self,
        artifact_id: UUID,
        payload: GeoFeatureQueryRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "POST",
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/geo/query",
            payload=payload,
            headers=headers,
        )

    def query_geo_features_ok(
        self,
        artifact_id: UUID,
        payload: GeoFeatureQueryRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> GeoFeatureQueryResponse:
        return _parse(
            GeoFeatureQueryResponse,
            _expect(
                self.query_geo_features(artifact_id, payload, headers=headers), 200
            ),
        )

    def geo_vector_pmtiles(
        self, source_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        """Stream the binary pmtiles archive (``Range`` via ``headers``)."""

        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}"
            f"/artifacts/{source_id}/geo/vector.pmtiles",
            headers=headers,
        )

    def geo_exact_feature(
        self,
        source_id: UUID,
        feature_index: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}"
            f"/artifacts/{source_id}/geo/features/{feature_index}",
            headers=headers,
        )

    def geo_exact_feature_ok(
        self,
        source_id: UUID,
        feature_index: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> GeoExactFeatureResponse:
        return _parse(
            GeoExactFeatureResponse,
            _expect(
                self.geo_exact_feature(source_id, feature_index, headers=headers), 200
            ),
        )

    def raster_tilejson(
        self, source_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}"
            f"/artifacts/{source_id}/geo/raster/tilejson.json",
            headers=headers,
        )

    def raster_tilejson_ok(
        self, source_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> GeoRasterTileJsonResponse:
        return _parse(
            GeoRasterTileJsonResponse,
            _expect(self.raster_tilejson(source_id, headers=headers), 200),
        )

    def raster_tile(
        self,
        source_id: UUID,
        z: int,
        x: int,
        y: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Fetch one PNG tile; the response has no JSON model."""

        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}"
            f"/artifacts/{source_id}/geo/raster/{z}/{x}/{y}.png",
            headers=headers,
        )

    # -- table ---------------------------------------------------------------

    def table_page(
        self,
        artifact_id: UUID,
        *,
        offset: int | None = None,
        limit: int | None = None,
        column_offset: int | None = None,
        column_limit: int | None = None,
        column_ids: list[str] | None = None,
        max_cell_characters: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        params = {
            key: value
            for key, value in {
                "offset": offset,
                "limit": limit,
                "column_offset": column_offset,
                "column_limit": column_limit,
                "column_ids": column_ids,
                "max_cell_characters": max_cell_characters,
            }.items()
            if value is not None
        }
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/table/page",
            params=params,
            headers=headers,
        )

    def table_page_ok(
        self,
        artifact_id: UUID,
        *,
        offset: int | None = None,
        limit: int | None = None,
        column_offset: int | None = None,
        column_limit: int | None = None,
        column_ids: list[str] | None = None,
        max_cell_characters: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> TablePageResponse:
        return _parse(
            TablePageResponse,
            _expect(
                self.table_page(
                    artifact_id,
                    offset=offset,
                    limit=limit,
                    column_offset=column_offset,
                    column_limit=column_limit,
                    column_ids=column_ids,
                    max_cell_characters=max_cell_characters,
                    headers=headers,
                ),
                200,
            ),
        )

    def table_schema(
        self, artifact_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/table/schema",
            headers=headers,
        )

    def table_schema_ok(
        self, artifact_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> TableSchemaResponse:
        return _parse(
            TableSchemaResponse,
            _expect(self.table_schema(artifact_id, headers=headers), 200),
        )

    def table_query(
        self,
        artifact_id: UUID,
        payload: TableQueryRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return _request(
            self._client,
            "POST",
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/table/query",
            payload=payload,
            headers=headers,
        )

    def table_query_ok(
        self,
        artifact_id: UUID,
        payload: TableQueryRequest,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> TablePageResponse:
        return _parse(
            TablePageResponse,
            _expect(self.table_query(artifact_id, payload, headers=headers), 200),
        )

    def table_cell(
        self,
        artifact_id: UUID,
        *,
        row_index: int,
        column_id: str,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/table/cell",
            params={"row_index": row_index, "column_id": column_id},
            headers=headers,
        )

    def table_cell_ok(
        self,
        artifact_id: UUID,
        *,
        row_index: int,
        column_id: str,
        headers: Mapping[str, str] | None = None,
    ) -> TableCellResponse:
        return _parse(
            TableCellResponse,
            _expect(
                self.table_cell(
                    artifact_id,
                    row_index=row_index,
                    column_id=column_id,
                    headers=headers,
                ),
                200,
            ),
        )

    # -- content -------------------------------------------------------------

    def content(
        self, artifact_id: UUID, *, headers: Mapping[str, str] | None = None
    ) -> Response:
        """Stream the stored artifact content; the body may be non-JSON."""

        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/content",
            headers=headers,
        )

    def download(
        self,
        artifact_id: UUID,
        *,
        format: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Stream an export; ``Content-Disposition`` carries the filename."""

        params = {"format": format} if format is not None else None
        return self._client.get(
            f"/v1/workspaces/{self._workspace_id}/artifacts/{artifact_id}/download",
            params=params,
            headers=headers,
        )

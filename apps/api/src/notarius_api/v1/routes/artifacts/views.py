import json
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from notarius_api.schemas.workbench import (
    TableCellResponse,
    TableColumnResponse,
    TablePageResponse,
    WorkbenchErrorResponse,
)
from notarius_api.services.errors import (
    ArtifactContentUnavailableError,
    WorkbenchOperationError,
)

from .dependencies import ArtifactDependency
from .models import table_cell_preview


router = APIRouter(tags=["workbench"])


@router.get(
    "/artifacts/{artifact_id}/table/page",
    response_model=TablePageResponse,
    responses={
        400: {"model": WorkbenchErrorResponse, "description": "Invalid table request"},
        404: {"model": WorkbenchErrorResponse, "description": "Artifact not found"},
        500: {
            "model": WorkbenchErrorResponse,
            "description": "Artifact content is unavailable",
        },
    },
)
async def get_table_artifact_page(
    artifact_id: UUID,
    service: ArtifactDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    column_offset: Annotated[int, Query(ge=0)] = 0,
    column_limit: Annotated[int, Query(ge=1, le=100)] = 25,
    max_cell_characters: Annotated[int, Query(ge=32, le=2_000)] = 256,
) -> TablePageResponse:
    if limit * column_limit * max_cell_characters > 2_000_000:
        raise HTTPException(
            status_code=400,
            detail=(
                "Requested table preview exceeds the 2,000,000-character "
                "response budget; reduce limit, column_limit, or "
                "max_cell_characters"
            ),
        )
    artifact = await service.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        page = await service.load_table_page(
            artifact,
            offset=offset,
            limit=limit,
        )
    except ArtifactContentUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    effective_column_offset = min(column_offset, len(page.columns))
    visible_columns = page.columns[
        effective_column_offset : effective_column_offset + column_limit
    ]
    return TablePageResponse(
        columns=[
            TableColumnResponse(
                id=column.id,
                title=column.title,
                value_type=column.value_type,
            )
            for column in visible_columns
        ],
        rows=[
            {
                column.id: table_cell_preview(
                    row[column.id],
                    max_cell_characters,
                )
                for column in visible_columns
            }
            for row in page.rows
        ],
        offset=page.offset,
        limit=limit,
        total_rows=page.total_rows,
        column_offset=effective_column_offset,
        column_limit=column_limit,
        total_columns=len(page.columns),
    )


@router.get(
    "/artifacts/{artifact_id}/table/cell",
    response_model=TableCellResponse,
    responses={
        400: {"model": WorkbenchErrorResponse, "description": "Invalid cell request"},
        404: {"model": WorkbenchErrorResponse, "description": "Artifact not found"},
        500: {
            "model": WorkbenchErrorResponse,
            "description": "Artifact content is unavailable",
        },
    },
)
async def get_table_artifact_cell(
    artifact_id: UUID,
    service: ArtifactDependency,
    row_index: Annotated[int, Query(ge=0)],
    column_id: Annotated[str, Query(min_length=1)],
) -> TableCellResponse:
    artifact = await service.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        value = await service.load_table_cell(
            artifact,
            row_index=row_index,
            column_id=column_id,
        )
    except ArtifactContentUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(value, int) and not isinstance(value, bool):
        response_value: str | float | bool | None = str(value)
        encoding: Literal["native", "integer", "json"] = "integer"
    elif value is None or isinstance(value, str | float | bool):
        response_value = value
        encoding = "native"
    else:
        response_value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        encoding = "json"
    return TableCellResponse(
        row_index=row_index,
        column_id=column_id,
        value=response_value,
        encoding=encoding,
    )


@router.get(
    "/artifacts/{artifact_id}/content",
    responses={
        404: {"model": WorkbenchErrorResponse, "description": "Artifact not found"},
        500: {
            "model": WorkbenchErrorResponse,
            "description": "Artifact content is unavailable",
        },
    },
)
async def get_artifact_content(
    artifact_id: UUID,
    service: ArtifactDependency,
) -> Response:
    artifact = await service.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        content = await service.load_content(artifact)
    except ArtifactContentUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    headers: dict[str, str] = {}
    download_name = artifact.metadata.get("download_name")
    if isinstance(download_name, str) and download_name != "":
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return Response(
        content=content,
        media_type=artifact.content_type,
        headers=headers,
    )


__all__ = [
    "get_artifact_content",
    "get_table_artifact_cell",
    "get_table_artifact_page",
    "router",
]

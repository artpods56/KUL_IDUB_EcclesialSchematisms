from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from grafy_core.domain.errors import NotFoundError
from grafy_core.domain.identity import WorkspaceCapability
from grafy_core.domain.templates import TemplateLibraryError

from grafy_api.v1.routes.auth.dependencies import (
    require_workspace_capability,
)
from grafy_api.v1.routes.templates.dependencies import TemplateDependency
from grafy_api.v1.routes.templates.models import (
    CreateTemplateRequest,
    InstantiateTemplateRequest,
    TemplateInstantiationResponse,
    TemplateListResponse,
    TemplateResponse,
    UpdateTemplateMetadataRequest,
)


router = APIRouter(prefix="/workspaces/{workspace_id}/templates", tags=["templates"])


@router.get("", response_model=TemplateListResponse)
async def list_templates(
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
    query: str | None = Query(default=None, alias="q", max_length=160),
    include_archived: bool = False,
) -> TemplateListResponse:
    templates = await service.list(
        access.workspace_id,
        query=query,
        include_archived=include_archived,
    )
    return TemplateListResponse.from_templates(templates)


@router.post("", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    body: CreateTemplateRequest,
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.CREATE_TEMPLATE),
) -> TemplateResponse:
    try:
        template = await service.create_from_graph_revision(
            actor=access.actor,
            workspace_id=access.workspace_id,
            source_graph_id=body.source_graph_id,
            source_revision=body.source_revision,
            name=body.name,
            description=body.description,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TemplateLibraryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TemplateResponse.from_template(template)


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: UUID,
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> TemplateResponse:
    try:
        template = await service.get(access.workspace_id, template_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TemplateResponse.from_template(template)


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template_metadata(
    template_id: UUID,
    body: UpdateTemplateMetadataRequest,
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.CREATE_TEMPLATE),
) -> TemplateResponse:
    try:
        template = await service.update_metadata(
            actor=access.actor,
            workspace_id=access.workspace_id,
            template_id=template_id,
            name=body.name,
            description=body.description,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TemplateLibraryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TemplateResponse.from_template(template)


@router.post("/{template_id}/archive", response_model=TemplateResponse)
async def archive_template(
    template_id: UUID,
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.MANAGE_TEMPLATE_LIBRARY),
) -> TemplateResponse:
    try:
        template = await service.archive(
            actor=access.actor,
            workspace_id=access.workspace_id,
            template_id=template_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TemplateResponse.from_template(template)


@router.post(
    "/{template_id}/instantiate",
    response_model=TemplateInstantiationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def instantiate_template(
    template_id: UUID,
    body: InstantiateTemplateRequest,
    service: TemplateDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> TemplateInstantiationResponse:
    try:
        result = await service.instantiate(
            actor=access.actor,
            source_workspace_id=access.workspace_id,
            template_id=template_id,
            destination_workspace_id=body.destination_workspace_id,
            name=body.name,
            folder_id=body.folder_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TemplateLibraryError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TemplateInstantiationResponse.from_instantiation(result)


__all__ = ["router"]

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from notarius_core.domain.models import Project
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.studio import ProjectCreate, ProjectResponse

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    name_validator: deps.NameRequiredValidatorDependency,
) -> ProjectResponse:
    async with uow:
        await name_validator.validate(body)
        project = Project(name=body.name, description=body.description)
        await uow.projects.add(project)
        await uow.commit()
        return ProjectResponse.from_project(project)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ProjectResponse]:
    async with uow:
        return [
            ProjectResponse.from_project(project)
            for project in await uow.projects.list()
        ]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ProjectResponse:
    async with uow:
        project = await deps.get_project_or_404(uow, project_id)
        return ProjectResponse.from_project(project)

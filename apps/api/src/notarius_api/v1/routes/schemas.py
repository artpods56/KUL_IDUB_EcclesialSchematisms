from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.studio import OutputSchemaCreate, OutputSchemaResponse
from notarius_core.domain.models import OutputSchema
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

router = APIRouter(tags=["schemas"])


@router.post(
    "/projects/{project_id}/schemas",
    response_model=OutputSchemaResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_schema(
    project_id: UUID,
    body: OutputSchemaCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    name_validator: deps.NameRequiredValidatorDependency,
) -> OutputSchemaResponse:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        await name_validator.validate(body)
        schema = OutputSchema(
            project_id=project_id,
            name=body.name,
            description=body.description,
            json_schema=body.json_schema,
        )
        await uow.output_schemas.add(schema)
        await uow.commit()
        return OutputSchemaResponse.from_output_schema(schema)


@router.get("/projects/{project_id}/schemas", response_model=list[OutputSchemaResponse])
async def list_project_schemas(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[OutputSchemaResponse]:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        return [
            OutputSchemaResponse.from_output_schema(schema)
            for schema in await uow.output_schemas.list_for_project(project_id)
        ]


@router.get("/schemas/{schema_id}", response_model=OutputSchemaResponse)
async def get_schema(
    schema_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> OutputSchemaResponse:
    async with uow:
        schema = await deps.get_schema_or_404(uow, schema_id)
        return OutputSchemaResponse.from_output_schema(schema)

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.studio import RecipeCreate, RecipeResponse
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import Recipe
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

router = APIRouter(tags=["recipes"])


@router.post(
    "/projects/{project_id}/recipes",
    response_model=RecipeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_recipe(
    project_id: UUID,
    body: RecipeCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    name_validator: deps.NameRequiredValidatorDependency,
) -> RecipeResponse:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        await name_validator.validate(body)
        schema = await deps.get_schema_or_404(uow, body.schema_id)
        if schema.project_id != project_id:
            raise ValidationError("Schema does not belong to project")
        recipe = Recipe(
            project_id=project_id,
            schema_id=body.schema_id,
            name=body.name,
            description=body.description,
            config=body.config,
        )
        await uow.recipes.add(recipe)
        await uow.commit()
        return RecipeResponse.from_recipe(recipe)


@router.get("/projects/{project_id}/recipes", response_model=list[RecipeResponse])
async def list_project_recipes(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[RecipeResponse]:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        return [
            RecipeResponse.from_recipe(recipe)
            for recipe in await uow.recipes.list_for_project(project_id)
        ]


@router.get("/recipes/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(
    recipe_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> RecipeResponse:
    async with uow:
        recipe = await deps.get_recipe_or_404(uow, recipe_id)
        return RecipeResponse.from_recipe(recipe)

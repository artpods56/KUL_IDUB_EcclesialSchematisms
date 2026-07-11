from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.messaging import JobPublisher
from notarius_api.schemas.studio import JobCreate, JobItemResponse, JobResponse
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import Job, JobItem, JobStatus
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    body: JobCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    job_publisher: Annotated[JobPublisher, Depends(deps.get_job_publisher)],
) -> JobResponse:
    async with uow:
        project = await deps.get_project_or_404(uow, body.project_id)
        source = await deps.get_source_or_404(uow, body.source_id)
        recipe = await deps.get_recipe_or_404(uow, body.recipe_id)
        if source.project_id != project.id or recipe.project_id != project.id:
            raise ValidationError("Source and recipe must belong to the project")

        source_items = await uow.source_items.list_for_source(source.id)
        job = Job(project_id=project.id, source_id=source.id, recipe_id=recipe.id)
        await uow.jobs.add(job)
        await uow.job_items.add_batch(
            [
                JobItem(
                    job_id=job.id,
                    source_item_id=item.id,
                    order=item.order,
                    status=JobStatus.QUEUED,
                )
                for item in source_items
            ]
        )
        await uow.commit()
        response = JobResponse.from_job(job)

    await job_publisher.publish_job_run_requested(job.id)
    return response


@router.get("/projects/{project_id}", response_model=list[JobResponse])
async def list_project_jobs(
    project_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[JobResponse]:
    async with uow:
        await deps.get_project_or_404(uow, project_id)
        return [
            JobResponse.from_job(job)
            for job in await uow.jobs.list_for_project(project_id)
        ]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> JobResponse:
    async with uow:
        job = await deps.get_job_or_404(uow, job_id)
        return JobResponse.from_job(job)


@router.get("/{job_id}/items", response_model=list[JobItemResponse])
async def list_job_items(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[JobItemResponse]:
    async with uow:
        await deps.get_job_or_404(uow, job_id)
        return [
            JobItemResponse.from_job_item(item)
            for item in await uow.job_items.list_for_job(job_id)
        ]


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> JobResponse:
    async with uow:
        job = await deps.get_job_or_404(uow, job_id)
        job.cancel()
        await uow.jobs.update(job)
        await uow.commit()
        return JobResponse.from_job(job)


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> JobResponse:
    async with uow:
        job = await deps.get_job_or_404(uow, job_id)
        job.retry()
        for item in await uow.job_items.list_for_job(job_id):
            item.status = JobStatus.QUEUED
            item.error = None
            item.structured_output = None
            item.context_trace = None
            await uow.job_items.update(item)
        await uow.jobs.update(job)
        await uow.commit()
        return JobResponse.from_job(job)

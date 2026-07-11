from pathlib import Path

import pytest

from notarius_core.domain.models import (
    Job,
    JobItem,
    JobStatus,
    OutputSchema,
    Project,
    Recipe,
    Source,
    SourceItem,
)
from notarius_persistence.adapters.in_memory import InMemoryDataStore, InMemoryUnitOfWork
from notarius_shared.object_store import LocalS3ObjectStore
from notarius_worker.runner import WorkerRunner


@pytest.mark.asyncio
async def test_worker_runner_processes_queued_job() -> None:
    store = InMemoryDataStore()
    async with InMemoryUnitOfWork(store) as uow:
        project = Project(name="Demo")
        source = Source(project_id=project.id, name="Source")
        item = SourceItem(source_id=source.id, order=1, text="abc")
        schema = OutputSchema(project_id=project.id, name="Schema", json_schema={})
        recipe = Recipe(project_id=project.id, schema_id=schema.id, name="Recipe")
        job = Job(project_id=project.id, source_id=source.id, recipe_id=recipe.id)
        job_item = JobItem(job_id=job.id, source_item_id=item.id, order=1)
        await uow.projects.add(project)
        await uow.sources.add(source)
        await uow.source_items.add(item)
        await uow.output_schemas.add(schema)
        await uow.recipes.add(recipe)
        await uow.jobs.add(job)
        await uow.job_items.add(job_item)
        await uow.commit()

    runner = WorkerRunner(lambda: InMemoryUnitOfWork(store))
    await runner.run_next_job()

    async with InMemoryUnitOfWork(store) as uow:
        processed_job = await uow.jobs.get(job.id)
        processed_items = await uow.job_items.list_for_job(job.id)

    assert processed_job.status == JobStatus.SUCCEEDED
    assert processed_items[0].structured_output["length"] == 3
    assert processed_items[0].context_trace.structured_output["text"] == "abc"


@pytest.mark.asyncio
async def test_worker_runner_loads_source_text_from_object_storage(tmp_path: Path) -> None:
    object_store = LocalS3ObjectStore(root=tmp_path / "objects", bucket="notarius-studio")
    text_uri = object_store.put_bytes("sources/one/pages/0001.txt", b"stored page text")
    store = InMemoryDataStore()
    async with InMemoryUnitOfWork(store) as uow:
        project = Project(name="PDF")
        source = Source(project_id=project.id, name="PDF source")
        item = SourceItem(
            source_id=source.id,
            order=1,
            text=None,
            metadata={"text_object_uri": text_uri},
        )
        schema = OutputSchema(project_id=project.id, name="Schema", json_schema={})
        recipe = Recipe(project_id=project.id, schema_id=schema.id, name="Recipe")
        job = Job(project_id=project.id, source_id=source.id, recipe_id=recipe.id)
        job_item = JobItem(job_id=job.id, source_item_id=item.id, order=1)
        await uow.projects.add(project)
        await uow.sources.add(source)
        await uow.source_items.add(item)
        await uow.output_schemas.add(schema)
        await uow.recipes.add(recipe)
        await uow.jobs.add(job)
        await uow.job_items.add(job_item)
        await uow.commit()

    runner = WorkerRunner(lambda: InMemoryUnitOfWork(store), object_store=object_store)
    await runner.run_next_job()

    async with InMemoryUnitOfWork(store) as uow:
        processed_items = await uow.job_items.list_for_job(job.id)

    assert processed_items[0].structured_output["text"] == "stored page text"
    assert processed_items[0].structured_output["metadata"]["loaded_from_storage"] is True

from collections.abc import Callable
from uuid import UUID

from notarius_core.domain.models import ContextTrace, JobStatus, SourceItem
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_shared.object_store import LocalS3ObjectStore, create_local_s3_object_store
from notarius_worker.pipeline.recipe_compiler import RecipeCompiler


class WorkerRunner:
    def __init__(
        self,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
        recipe_compiler: RecipeCompiler | None = None,
        object_store: LocalS3ObjectStore | None = None,
    ):
        self.uow_factory = uow_factory
        self.recipe_compiler = recipe_compiler or RecipeCompiler()
        self.object_store = object_store or create_local_s3_object_store()

    async def run_next_job(self) -> UUID | None:
        async with self.uow_factory() as uow:
            job = await uow.jobs.next_queued()
            if job is None:
                return None
            job_id = job.id
        await self.run_job(job_id)
        return job_id

    async def run_job(self, job_id: UUID | str) -> None:
        job_uuid = job_id if isinstance(job_id, UUID) else UUID(job_id)

        async with self.uow_factory() as uow:
            job = await uow.jobs.get(job_uuid)
            if job is None:
                return
            if job.status == JobStatus.CANCELED:
                return
            job.mark_running()
            await uow.jobs.update(job)
            await uow.commit()

        try:
            async with self.uow_factory() as uow:
                job = await uow.jobs.get(job_uuid)
                if job is None:
                    return
                recipe = await uow.recipes.get(job.recipe_id)
                if recipe is None:
                    raise RuntimeError(f"Recipe not found: {job.recipe_id}")
                compiled = self.recipe_compiler.compile(recipe.config)
                job_items = await uow.job_items.list_for_job(job.id)

                previous_context: dict | None = None
                for job_item in job_items:
                    source_item = await uow.source_items.get(job_item.source_item_id)
                    if source_item is None:
                        raise RuntimeError(f"Source item not found: {job_item.source_item_id}")
                    source_text, source_metadata = self._load_source_text(source_item)
                    output = compiled.extract(source_text, source_metadata)
                    output_context = {"last_source_item_id": str(source_item.id)}
                    job_item.status = JobStatus.SUCCEEDED
                    job_item.structured_output = output
                    job_item.context_trace = ContextTrace(
                        rendered_input_context={
                            "text": source_text,
                            "metadata": source_metadata,
                        },
                        previous_domain_context=previous_context,
                        structured_output=output,
                        output_domain_context=output_context,
                        model_metadata=compiled.model_metadata,
                    )
                    previous_context = output_context
                    await uow.job_items.update(job_item)

                job.mark_succeeded()
                await uow.jobs.update(job)
                await uow.commit()
        except Exception as exc:
            async with self.uow_factory() as uow:
                job = await uow.jobs.get(job_uuid)
                if job is not None:
                    job.mark_failed(str(exc))
                    await uow.jobs.update(job)
                    await uow.commit()
            raise

    def _load_source_text(self, source_item: SourceItem) -> tuple[str | None, dict]:
        text = source_item.text
        metadata = dict(source_item.metadata)
        text_uri = metadata.get("text_object_uri")
        if text is None and isinstance(text_uri, str):
            text = self.object_store.get_text(text_uri)
            metadata["loaded_from_storage"] = True
        else:
            metadata["loaded_from_storage"] = False
        return text, metadata

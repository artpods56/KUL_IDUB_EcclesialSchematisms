import os
from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends

from notarius_api.messaging import JobPublisher, create_job_publisher
from notarius_api.services.output_bundles import WorkflowRunOutputBundleService
from notarius_api.services.pdf_sources import PdfSourceIngestor
from notarius_api.services.validators import NameRequiredValidator
from notarius_api.services.workflow_execution import WorkflowRunExecutionService
from notarius_core.application.operators import builtin_node_specs
from notarius_core.application.workflows import NodeSpecRegistry
from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.models import (
    Artifact,
    ArtifactSequence,
    Experiment,
    Job,
    NodeRun,
    OutboxMessage,
    OutputSchema,
    Project,
    Recipe,
    Source,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)
from notarius_persistence.unit_of_work import create_sqlite_uow_factory
from notarius_storage import ArtifactPayloadStoragePort, LocalArtifactPayloadStorage
from notarius_worker.node_execution import NodeRunExecutor, NodeRunHandler
from notarius_worker.operators import builtin_node_handlers

_STORE = InMemoryDataStore()
_DATABASE_URL = os.getenv("NOTARIUS_DATABASE_URL")
_SQL_UOW_FACTORY = create_sqlite_uow_factory(_DATABASE_URL) if _DATABASE_URL else None


def get_store() -> InMemoryDataStore:
    return _STORE


def create_uow(
    store: Annotated[InMemoryDataStore, Depends(get_store)],
) -> StudioUnitOfWorkPort:
    if _SQL_UOW_FACTORY is not None:
        return _SQL_UOW_FACTORY()
    return InMemoryUnitOfWork(store)


def create_uow_factory(
    store: Annotated[InMemoryDataStore, Depends(get_store)],
) -> Callable[[], StudioUnitOfWorkPort]:
    if _SQL_UOW_FACTORY is not None:
        return _SQL_UOW_FACTORY
    return lambda: InMemoryUnitOfWork(store)


def get_job_publisher() -> JobPublisher:
    return create_job_publisher()


def get_node_spec_registry() -> NodeSpecRegistry:
    return builtin_node_specs()


def get_name_required_validator() -> NameRequiredValidator:
    return NameRequiredValidator()


def get_pdf_source_ingestor() -> PdfSourceIngestor:
    return PdfSourceIngestor()


def get_artifact_payload_storage() -> ArtifactPayloadStoragePort:
    storage_root = os.getenv(
        "NOTARIUS_ARTIFACT_PAYLOAD_DIR",
        os.getenv("NOTARIUS_OBJECT_STORAGE_DIR", ".notarius-artifacts"),
    )
    return LocalArtifactPayloadStorage(storage_root)


def get_node_run_handlers(
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(get_artifact_payload_storage),
    ],
) -> dict[tuple[str, str], NodeRunHandler]:
    return builtin_node_handlers(storage)


def create_node_run_executor(
    uow_factory: Annotated[
        Callable[[], StudioUnitOfWorkPort],
        Depends(create_uow_factory),
    ],
    handlers: Annotated[
        dict[tuple[str, str], NodeRunHandler],
        Depends(get_node_run_handlers),
    ],
    node_specs: Annotated[
        NodeSpecRegistry,
        Depends(get_node_spec_registry),
    ],
) -> NodeRunExecutor:
    return NodeRunExecutor(uow_factory, handlers, node_specs)


def create_workflow_run_execution_service(
    executor: Annotated[
        NodeRunExecutor,
        Depends(create_node_run_executor),
    ],
    uow_factory: Annotated[
        Callable[[], StudioUnitOfWorkPort],
        Depends(create_uow_factory),
    ],
) -> WorkflowRunExecutionService:
    return WorkflowRunExecutionService(executor, uow_factory)


def create_workflow_run_output_bundle_service(
    uow_factory: Annotated[
        Callable[[], StudioUnitOfWorkPort],
        Depends(create_uow_factory),
    ],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(get_artifact_payload_storage),
    ],
) -> WorkflowRunOutputBundleService:
    return WorkflowRunOutputBundleService(uow_factory, storage)


NameRequiredValidatorDependency = Annotated[
    NameRequiredValidator,
    Depends(get_name_required_validator),
]


async def get_project_or_404(uow: StudioUnitOfWorkPort, project_id: UUID) -> Project:
    project = await uow.projects.get(project_id)
    if project is None:
        raise NotFoundError("Project", str(project_id))
    return project


async def get_source_or_404(uow: StudioUnitOfWorkPort, source_id: UUID) -> Source:
    source = await uow.sources.get(source_id)
    if source is None:
        raise NotFoundError("Source", str(source_id))
    return source


async def get_schema_or_404(uow: StudioUnitOfWorkPort, schema_id: UUID) -> OutputSchema:
    schema = await uow.output_schemas.get(schema_id)
    if schema is None:
        raise NotFoundError("OutputSchema", str(schema_id))
    return schema


async def get_recipe_or_404(uow: StudioUnitOfWorkPort, recipe_id: UUID) -> Recipe:
    recipe = await uow.recipes.get(recipe_id)
    if recipe is None:
        raise NotFoundError("Recipe", str(recipe_id))
    return recipe


async def get_job_or_404(uow: StudioUnitOfWorkPort, job_id: UUID) -> Job:
    job = await uow.jobs.get(job_id)
    if job is None:
        raise NotFoundError("Job", str(job_id))
    return job


async def get_workflow_definition_or_404(
    uow: StudioUnitOfWorkPort,
    workflow_definition_id: UUID,
) -> WorkflowDefinition:
    definition = await uow.workflow_definitions.get(workflow_definition_id)
    if definition is None:
        raise NotFoundError("WorkflowDefinition", str(workflow_definition_id))
    return definition


async def get_workflow_version_or_404(
    uow: StudioUnitOfWorkPort,
    workflow_version_id: UUID,
) -> WorkflowVersion:
    version = await uow.workflow_versions.get(workflow_version_id)
    if version is None:
        raise NotFoundError("WorkflowVersion", str(workflow_version_id))
    return version


async def get_workflow_run_or_404(
    uow: StudioUnitOfWorkPort,
    workflow_run_id: UUID,
) -> WorkflowRun:
    run = await uow.workflow_runs.get(workflow_run_id)
    if run is None:
        raise NotFoundError("WorkflowRun", str(workflow_run_id))
    return run


async def get_experiment_or_404(
    uow: StudioUnitOfWorkPort,
    experiment_id: UUID,
) -> Experiment:
    experiment = await uow.experiments.get(experiment_id)
    if experiment is None:
        raise NotFoundError("Experiment", str(experiment_id))
    return experiment


async def get_node_run_or_404(
    uow: StudioUnitOfWorkPort,
    node_run_id: UUID,
) -> NodeRun:
    node_run = await uow.node_runs.get(node_run_id)
    if node_run is None:
        raise NotFoundError("NodeRun", str(node_run_id))
    return node_run


async def get_outbox_message_or_404(
    uow: StudioUnitOfWorkPort,
    outbox_message_id: UUID,
) -> OutboxMessage:
    message = await uow.outbox_messages.get(outbox_message_id)
    if message is None:
        raise NotFoundError("OutboxMessage", str(outbox_message_id))
    return message


async def get_artifact_or_404(
    uow: StudioUnitOfWorkPort,
    artifact_id: UUID,
) -> Artifact:
    artifact = await uow.artifacts.get(artifact_id)
    if artifact is None:
        raise NotFoundError("Artifact", str(artifact_id))
    return artifact


async def get_artifact_sequence_or_404(
    uow: StudioUnitOfWorkPort,
    sequence_id: UUID,
) -> ArtifactSequence:
    sequence = await uow.artifact_sequences.get(sequence_id)
    if sequence is None:
        raise NotFoundError("ArtifactSequence", str(sequence_id))
    return sequence

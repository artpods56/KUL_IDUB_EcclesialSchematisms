from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import Iterable, Self, final, override
from uuid import UUID

from notarius_core.domain.models.studio import (
    Job,
    JobItem,
    JobStatus,
    OutputSchema,
    Project,
    Recipe,
    Source,
    SourceItem,
)
from notarius_core.domain.models.platform import (
    Artifact,
    ArtifactSequence,
    Experiment,
    InputAssemblyTrace,
    InvocationTrace,
    NodeRun,
    NodeRunStatus,
    OutboxMessage,
    OutboxMessageStatus,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
)
from notarius_core.ports.repositories import (
    ArtifactRepositoryPort,
    ArtifactSequenceRepositoryPort,
    ExperimentRepositoryPort,
    InputAssemblyTraceRepositoryPort,
    InvocationTraceRepositoryPort,
    JobItemRepositoryPort,
    JobRepositoryPort,
    NodeRunRepositoryPort,
    OutboxMessageRepositoryPort,
    OutputSchemaRepositoryPort,
    ProjectRepositoryPort,
    RecipeRepositoryPort,
    SourceItemRepositoryPort,
    SourceRepositoryPort,
    WorkflowDefinitionRepositoryPort,
    WorkflowRunRepositoryPort,
    WorkflowVersionRepositoryPort,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort


def _clone[T](value: T) -> T:
    return deepcopy(value)


def _sorted_by_created_at[T](items: Iterable[T]) -> list[T]:
    return sorted(items, key=lambda item: (item.created_at, item.id))


def _outbox_payload_workflow_run_id(message: OutboxMessage) -> UUID | None:
    value = message.payload.get("workflow_run_id")
    if value is None:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


@dataclass(slots=True)
class InMemoryDataStore:
    projects: dict[UUID, Project] = field(default_factory=dict)
    sources: dict[UUID, Source] = field(default_factory=dict)
    source_items: dict[UUID, SourceItem] = field(default_factory=dict)
    output_schemas: dict[UUID, OutputSchema] = field(default_factory=dict)
    recipes: dict[UUID, Recipe] = field(default_factory=dict)
    jobs: dict[UUID, Job] = field(default_factory=dict)
    job_items: dict[UUID, JobItem] = field(default_factory=dict)
    workflow_definitions: dict[UUID, WorkflowDefinition] = field(default_factory=dict)
    workflow_versions: dict[UUID, WorkflowVersion] = field(default_factory=dict)
    workflow_runs: dict[UUID, WorkflowRun] = field(default_factory=dict)
    node_runs: dict[UUID, NodeRun] = field(default_factory=dict)
    artifacts: dict[UUID, Artifact] = field(default_factory=dict)
    artifact_sequences: dict[UUID, ArtifactSequence] = field(default_factory=dict)
    experiments: dict[UUID, Experiment] = field(default_factory=dict)
    input_assembly_traces: dict[UUID, InputAssemblyTrace] = field(default_factory=dict)
    invocation_traces: dict[UUID, InvocationTrace] = field(default_factory=dict)
    outbox_messages: dict[UUID, OutboxMessage] = field(default_factory=dict)

    def clone(self) -> Self:
        return _clone(self)

    def replace_with(self, other: Self) -> None:
        self.projects = _clone(other.projects)
        self.sources = _clone(other.sources)
        self.source_items = _clone(other.source_items)
        self.output_schemas = _clone(other.output_schemas)
        self.recipes = _clone(other.recipes)
        self.jobs = _clone(other.jobs)
        self.job_items = _clone(other.job_items)
        self.workflow_definitions = _clone(other.workflow_definitions)
        self.workflow_versions = _clone(other.workflow_versions)
        self.workflow_runs = _clone(other.workflow_runs)
        self.node_runs = _clone(other.node_runs)
        self.artifacts = _clone(other.artifacts)
        self.artifact_sequences = _clone(other.artifact_sequences)
        self.experiments = _clone(other.experiments)
        self.input_assembly_traces = _clone(other.input_assembly_traces)
        self.invocation_traces = _clone(other.invocation_traces)
        self.outbox_messages = _clone(other.outbox_messages)


@final
class InMemoryProjectRepository(ProjectRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, project: Project) -> None:
        self._store.projects[project.id] = project

    @override
    async def get(self, project_id: UUID) -> Project | None:
        return self._store.projects.get(project_id)

    @override
    async def list(self) -> list[Project]:
        return _sorted_by_created_at(self._store.projects.values())


@final
class InMemorySourceRepository(SourceRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, source: Source) -> None:
        self._store.sources[source.id] = source

    @override
    async def get(self, source_id: UUID) -> Source | None:
        return self._store.sources.get(source_id)

    @override
    async def list_for_project(self, project_id: UUID) -> list[Source]:
        return _sorted_by_created_at(
            source
            for source in self._store.sources.values()
            if source.project_id == project_id
        )


@final
class InMemorySourceItemRepository(SourceItemRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, item: SourceItem) -> None:
        self._store.source_items[item.id] = item

    @override
    async def add_batch(self, items: list[SourceItem]) -> None:
        for item in items:
            await self.add(item)

    @override
    async def get(self, item_id: UUID) -> SourceItem | None:
        return self._store.source_items.get(item_id)

    @override
    async def list_for_source(self, source_id: UUID) -> list[SourceItem]:
        return sorted(
            (
                item
                for item in self._store.source_items.values()
                if item.source_id == source_id
            ),
            key=lambda item: (item.order, item.created_at, item.id),
        )


@final
class InMemoryOutputSchemaRepository(OutputSchemaRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, schema: OutputSchema) -> None:
        self._store.output_schemas[schema.id] = schema

    @override
    async def get(self, schema_id: UUID) -> OutputSchema | None:
        return self._store.output_schemas.get(schema_id)

    @override
    async def list_for_project(self, project_id: UUID) -> list[OutputSchema]:
        return _sorted_by_created_at(
            schema
            for schema in self._store.output_schemas.values()
            if schema.project_id == project_id
        )


@final
class InMemoryRecipeRepository(RecipeRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, recipe: Recipe) -> None:
        self._store.recipes[recipe.id] = recipe

    @override
    async def get(self, recipe_id: UUID) -> Recipe | None:
        return self._store.recipes.get(recipe_id)

    @override
    async def list_for_project(self, project_id: UUID) -> list[Recipe]:
        return _sorted_by_created_at(
            recipe
            for recipe in self._store.recipes.values()
            if recipe.project_id == project_id
        )


@final
class InMemoryJobRepository(JobRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, job: Job) -> None:
        self._store.jobs[job.id] = job

    @override
    async def get(self, job_id: UUID) -> Job | None:
        return self._store.jobs.get(job_id)

    @override
    async def next_queued(self) -> Job | None:
        queued = await self.list_by_status(JobStatus.QUEUED)
        return queued[0] if queued else None

    @override
    async def update(self, job: Job) -> None:
        self._store.jobs[job.id] = job

    @override
    async def list_for_project(self, project_id: UUID) -> list[Job]:
        return _sorted_by_created_at(
            job for job in self._store.jobs.values() if job.project_id == project_id
        )

    @override
    async def list_by_status(self, status: JobStatus) -> list[Job]:
        return _sorted_by_created_at(
            job for job in self._store.jobs.values() if job.status == status
        )


@final
class InMemoryJobItemRepository(JobItemRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, job_item: JobItem) -> None:
        self._store.job_items[job_item.id] = job_item

    @override
    async def add_batch(self, job_items: list[JobItem]) -> None:
        for job_item in job_items:
            await self.add(job_item)

    @override
    async def get(self, job_item_id: UUID) -> JobItem | None:
        return self._store.job_items.get(job_item_id)

    @override
    async def list_for_job(self, job_id: UUID) -> list[JobItem]:
        return sorted(
            (item for item in self._store.job_items.values() if item.job_id == job_id),
            key=lambda item: (item.order, item.created_at, item.id),
        )

    @override
    async def update(self, job_item: JobItem) -> None:
        self._store.job_items[job_item.id] = job_item


@final
class InMemoryWorkflowDefinitionRepository(WorkflowDefinitionRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, definition: WorkflowDefinition) -> None:
        self._store.workflow_definitions[definition.id] = definition

    @override
    async def get(self, definition_id: UUID) -> WorkflowDefinition | None:
        return self._store.workflow_definitions.get(definition_id)

    @override
    async def update(self, definition: WorkflowDefinition) -> None:
        self._store.workflow_definitions[definition.id] = definition

    @override
    async def list(self) -> list[WorkflowDefinition]:
        return _sorted_by_created_at(self._store.workflow_definitions.values())


@final
class InMemoryWorkflowVersionRepository(WorkflowVersionRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, version: WorkflowVersion) -> None:
        self._store.workflow_versions[version.id] = version

    @override
    async def get(self, version_id: UUID) -> WorkflowVersion | None:
        return self._store.workflow_versions.get(version_id)

    @override
    async def list_for_definition(
        self,
        definition_id: UUID,
    ) -> list[WorkflowVersion]:
        return sorted(
            (
                version
                for version in self._store.workflow_versions.values()
                if version.workflow_definition_id == definition_id
            ),
            key=lambda version: (
                version.version_number,
                version.created_at,
                version.id,
            ),
        )

    @override
    async def latest_for_definition(
        self,
        definition_id: UUID,
    ) -> WorkflowVersion | None:
        versions = await self.list_for_definition(definition_id)
        return versions[-1] if versions else None


@final
class InMemoryWorkflowRunRepository(WorkflowRunRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, run: WorkflowRun) -> None:
        self._store.workflow_runs[run.id] = run

    @override
    async def get(self, run_id: UUID) -> WorkflowRun | None:
        return self._store.workflow_runs.get(run_id)

    @override
    async def update(self, run: WorkflowRun) -> None:
        self._store.workflow_runs[run.id] = run

    @override
    async def next_queued(self) -> WorkflowRun | None:
        queued = await self.list_by_status(WorkflowRunStatus.QUEUED)
        return queued[0] if queued else None

    @override
    async def list_for_version(self, version_id: UUID) -> list[WorkflowRun]:
        return sorted(
            (
                run
                for run in self._store.workflow_runs.values()
                if run.workflow_version_id == version_id
            ),
            key=lambda run: (run.queued_at, run.id),
        )

    @override
    async def list_by_status(self, status: WorkflowRunStatus) -> list[WorkflowRun]:
        return sorted(
            (run for run in self._store.workflow_runs.values() if run.status == status),
            key=lambda run: (run.queued_at, run.id),
        )


@final
class InMemoryNodeRunRepository(NodeRunRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, node_run: NodeRun) -> None:
        self._store.node_runs[node_run.id] = node_run

    @override
    async def add_batch(self, node_runs: list[NodeRun]) -> None:
        for node_run in node_runs:
            await self.add(node_run)

    @override
    async def get(self, node_run_id: UUID) -> NodeRun | None:
        return self._store.node_runs.get(node_run_id)

    @override
    async def update(self, node_run: NodeRun) -> None:
        self._store.node_runs[node_run.id] = node_run

    @override
    async def next_queued(self) -> NodeRun | None:
        queued = await self.list_by_status(NodeRunStatus.QUEUED)
        return queued[0] if queued else None

    @override
    async def list_for_workflow_run(self, workflow_run_id: UUID) -> list[NodeRun]:
        return sorted(
            (
                node_run
                for node_run in self._store.node_runs.values()
                if node_run.workflow_run_id == workflow_run_id
            ),
            key=lambda node_run: (node_run.queued_at, node_run.id),
        )

    @override
    async def list_by_status(self, status: NodeRunStatus) -> list[NodeRun]:
        return sorted(
            (
                node_run
                for node_run in self._store.node_runs.values()
                if node_run.status == status
            ),
            key=lambda node_run: (node_run.queued_at, node_run.id),
        )


@final
class InMemoryArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, artifact: Artifact) -> None:
        self._store.artifacts[artifact.id] = artifact

    @override
    async def get(self, artifact_id: UUID) -> Artifact | None:
        return self._store.artifacts.get(artifact_id)

    @override
    async def list_for_source(self, source_id: UUID) -> list[Artifact]:
        return sorted(
            (
                artifact
                for artifact in self._store.artifacts.values()
                if artifact.metadata.get("source_id") == str(source_id)
            ),
            key=lambda artifact: (artifact.created_at, artifact.id),
        )

    @override
    async def list_for_workflow_run(self, workflow_run_id: UUID) -> list[Artifact]:
        return sorted(
            (
                artifact
                for artifact in self._store.artifacts.values()
                if artifact.workflow_run_id == workflow_run_id
            ),
            key=lambda artifact: (artifact.created_at, artifact.id),
        )

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[Artifact]:
        return sorted(
            (
                artifact
                for artifact in self._store.artifacts.values()
                if artifact.producer_node_run_id == node_run_id
            ),
            key=lambda artifact: (artifact.created_at, artifact.id),
        )

    @override
    async def list_by_type(self, artifact_type: str) -> list[Artifact]:
        return sorted(
            (
                artifact
                for artifact in self._store.artifacts.values()
                if artifact.artifact_type == artifact_type
            ),
            key=lambda artifact: (artifact.created_at, artifact.id),
        )


@final
class InMemoryArtifactSequenceRepository(ArtifactSequenceRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, sequence: ArtifactSequence) -> None:
        self._store.artifact_sequences[sequence.id] = sequence

    @override
    async def get(self, sequence_id: UUID) -> ArtifactSequence | None:
        return self._store.artifact_sequences.get(sequence_id)

    @override
    async def list_for_source(self, source_id: UUID) -> list[ArtifactSequence]:
        return sorted(
            (
                sequence
                for sequence in self._store.artifact_sequences.values()
                if sequence.metadata.get("source_id") == str(source_id)
            ),
            key=lambda sequence: (sequence.created_at, sequence.id),
        )

    @override
    async def list_by_artifact_type(self, artifact_type: str) -> list[ArtifactSequence]:
        return sorted(
            (
                sequence
                for sequence in self._store.artifact_sequences.values()
                if sequence.artifact_type == artifact_type
            ),
            key=lambda sequence: (sequence.created_at, sequence.id),
        )


@final
class InMemoryExperimentRepository(ExperimentRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, experiment: Experiment) -> None:
        self._store.experiments[experiment.id] = experiment

    @override
    async def get(self, experiment_id: UUID) -> Experiment | None:
        return self._store.experiments.get(experiment_id)

    @override
    async def update(self, experiment: Experiment) -> None:
        self._store.experiments[experiment.id] = experiment

    @override
    async def list_for_workflow_version(
        self,
        workflow_version_id: UUID,
    ) -> list[Experiment]:
        return _sorted_by_created_at(
            experiment
            for experiment in self._store.experiments.values()
            if experiment.workflow_version_id == workflow_version_id
        )

    @override
    async def list(self) -> list[Experiment]:
        return _sorted_by_created_at(self._store.experiments.values())


@final
class InMemoryInputAssemblyTraceRepository(InputAssemblyTraceRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, trace: InputAssemblyTrace) -> None:
        self._store.input_assembly_traces[trace.id] = trace

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[InputAssemblyTrace]:
        return sorted(
            (
                trace
                for trace in self._store.input_assembly_traces.values()
                if trace.node_run_id == node_run_id
            ),
            key=lambda trace: (trace.created_at, trace.id),
        )


@final
class InMemoryInvocationTraceRepository(InvocationTraceRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, trace: InvocationTrace) -> None:
        self._store.invocation_traces[trace.id] = trace

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[InvocationTrace]:
        return sorted(
            (
                trace
                for trace in self._store.invocation_traces.values()
                if trace.node_run_id == node_run_id
            ),
            key=lambda trace: (trace.created_at, trace.id),
        )


@final
class InMemoryOutboxMessageRepository(OutboxMessageRepositoryPort):
    def __init__(self, store: InMemoryDataStore):
        self._store = store

    @override
    async def add(self, message: OutboxMessage) -> None:
        self._store.outbox_messages[message.id] = message

    @override
    async def get(self, message_id: UUID) -> OutboxMessage | None:
        return self._store.outbox_messages.get(message_id)

    @override
    async def update(self, message: OutboxMessage) -> None:
        self._store.outbox_messages[message.id] = message

    @override
    async def delete_many(self, message_ids: Iterable[UUID]) -> int:
        deleted_count = 0
        for message_id in message_ids:
            if message_id in self._store.outbox_messages:
                del self._store.outbox_messages[message_id]
                deleted_count += 1
        return deleted_count

    @override
    async def list_for_workflow_run(
        self,
        workflow_run_id: UUID,
    ) -> list[OutboxMessage]:
        return sorted(
            (
                message
                for message in self._store.outbox_messages.values()
                if _outbox_payload_workflow_run_id(message) == workflow_run_id
            ),
            key=lambda message: (message.created_at, message.id),
        )

    @override
    async def list_by_status(
        self,
        status: OutboxMessageStatus,
    ) -> list[OutboxMessage]:
        return sorted(
            (
                message
                for message in self._store.outbox_messages.values()
                if message.status == status
            ),
            key=lambda message: (message.created_at, message.id),
        )

    @override
    async def list_pending(self) -> list[OutboxMessage]:
        return await self.list_by_status(OutboxMessageStatus.PENDING)


@final
class InMemoryUnitOfWork(StudioUnitOfWorkPort):
    def __init__(self, store: InMemoryDataStore | None = None):
        self._store = store or InMemoryDataStore()
        self._working: InMemoryDataStore | None = None
        self._projects: ProjectRepositoryPort | None = None
        self._sources: SourceRepositoryPort | None = None
        self._source_items: SourceItemRepositoryPort | None = None
        self._output_schemas: OutputSchemaRepositoryPort | None = None
        self._recipes: RecipeRepositoryPort | None = None
        self._jobs: JobRepositoryPort | None = None
        self._job_items: JobItemRepositoryPort | None = None
        self._workflow_definitions: WorkflowDefinitionRepositoryPort | None = None
        self._workflow_versions: WorkflowVersionRepositoryPort | None = None
        self._workflow_runs: WorkflowRunRepositoryPort | None = None
        self._node_runs: NodeRunRepositoryPort | None = None
        self._artifacts: ArtifactRepositoryPort | None = None
        self._artifact_sequences: ArtifactSequenceRepositoryPort | None = None
        self._experiments: ExperimentRepositoryPort | None = None
        self._input_assembly_traces: InputAssemblyTraceRepositoryPort | None = None
        self._invocation_traces: InvocationTraceRepositoryPort | None = None
        self._outbox_messages: OutboxMessageRepositoryPort | None = None

    @property
    @override
    def projects(self) -> ProjectRepositoryPort:
        if self._projects is None:
            raise RuntimeError("Unit of work is not entered")
        return self._projects

    @property
    @override
    def sources(self) -> SourceRepositoryPort:
        if self._sources is None:
            raise RuntimeError("Unit of work is not entered")
        return self._sources

    @property
    @override
    def source_items(self) -> SourceItemRepositoryPort:
        if self._source_items is None:
            raise RuntimeError("Unit of work is not entered")
        return self._source_items

    @property
    @override
    def output_schemas(self) -> OutputSchemaRepositoryPort:
        if self._output_schemas is None:
            raise RuntimeError("Unit of work is not entered")
        return self._output_schemas

    @property
    @override
    def recipes(self) -> RecipeRepositoryPort:
        if self._recipes is None:
            raise RuntimeError("Unit of work is not entered")
        return self._recipes

    @property
    @override
    def jobs(self) -> JobRepositoryPort:
        if self._jobs is None:
            raise RuntimeError("Unit of work is not entered")
        return self._jobs

    @property
    @override
    def job_items(self) -> JobItemRepositoryPort:
        if self._job_items is None:
            raise RuntimeError("Unit of work is not entered")
        return self._job_items

    @property
    @override
    def workflow_definitions(self) -> WorkflowDefinitionRepositoryPort:
        if self._workflow_definitions is None:
            raise RuntimeError("Unit of work is not entered")
        return self._workflow_definitions

    @property
    @override
    def workflow_versions(self) -> WorkflowVersionRepositoryPort:
        if self._workflow_versions is None:
            raise RuntimeError("Unit of work is not entered")
        return self._workflow_versions

    @property
    @override
    def workflow_runs(self) -> WorkflowRunRepositoryPort:
        if self._workflow_runs is None:
            raise RuntimeError("Unit of work is not entered")
        return self._workflow_runs

    @property
    @override
    def node_runs(self) -> NodeRunRepositoryPort:
        if self._node_runs is None:
            raise RuntimeError("Unit of work is not entered")
        return self._node_runs

    @property
    @override
    def artifacts(self) -> ArtifactRepositoryPort:
        if self._artifacts is None:
            raise RuntimeError("Unit of work is not entered")
        return self._artifacts

    @property
    @override
    def artifact_sequences(self) -> ArtifactSequenceRepositoryPort:
        if self._artifact_sequences is None:
            raise RuntimeError("Unit of work is not entered")
        return self._artifact_sequences

    @property
    @override
    def experiments(self) -> ExperimentRepositoryPort:
        if self._experiments is None:
            raise RuntimeError("Unit of work is not entered")
        return self._experiments

    @property
    @override
    def input_assembly_traces(self) -> InputAssemblyTraceRepositoryPort:
        if self._input_assembly_traces is None:
            raise RuntimeError("Unit of work is not entered")
        return self._input_assembly_traces

    @property
    @override
    def invocation_traces(self) -> InvocationTraceRepositoryPort:
        if self._invocation_traces is None:
            raise RuntimeError("Unit of work is not entered")
        return self._invocation_traces

    @property
    @override
    def outbox_messages(self) -> OutboxMessageRepositoryPort:
        if self._outbox_messages is None:
            raise RuntimeError("Unit of work is not entered")
        return self._outbox_messages

    @override
    async def __aenter__(self) -> Self:
        self._working = self._store.clone()
        self._projects = InMemoryProjectRepository(self._working)
        self._sources = InMemorySourceRepository(self._working)
        self._source_items = InMemorySourceItemRepository(self._working)
        self._output_schemas = InMemoryOutputSchemaRepository(self._working)
        self._recipes = InMemoryRecipeRepository(self._working)
        self._jobs = InMemoryJobRepository(self._working)
        self._job_items = InMemoryJobItemRepository(self._working)
        self._workflow_definitions = InMemoryWorkflowDefinitionRepository(self._working)
        self._workflow_versions = InMemoryWorkflowVersionRepository(self._working)
        self._workflow_runs = InMemoryWorkflowRunRepository(self._working)
        self._node_runs = InMemoryNodeRunRepository(self._working)
        self._artifacts = InMemoryArtifactRepository(self._working)
        self._artifact_sequences = InMemoryArtifactSequenceRepository(self._working)
        self._experiments = InMemoryExperimentRepository(self._working)
        self._input_assembly_traces = InMemoryInputAssemblyTraceRepository(
            self._working
        )
        self._invocation_traces = InMemoryInvocationTraceRepository(self._working)
        self._outbox_messages = InMemoryOutboxMessageRepository(self._working)
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()
        self._working = None
        self._projects = None
        self._sources = None
        self._source_items = None
        self._output_schemas = None
        self._recipes = None
        self._jobs = None
        self._job_items = None
        self._workflow_definitions = None
        self._workflow_versions = None
        self._workflow_runs = None
        self._node_runs = None
        self._artifacts = None
        self._artifact_sequences = None
        self._experiments = None
        self._input_assembly_traces = None
        self._invocation_traces = None
        self._outbox_messages = None

    @override
    async def commit(self) -> None:
        if self._working is None:
            raise RuntimeError("Unit of work is not entered")
        self._store.replace_with(self._working)

    @override
    async def rollback(self) -> None:
        self._working = self._store.clone()

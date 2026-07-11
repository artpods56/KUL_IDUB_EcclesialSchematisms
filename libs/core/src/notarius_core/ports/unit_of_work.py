from types import TracebackType
from typing import Protocol, Self

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


class StudioUnitOfWorkPort(Protocol):
    @property
    def projects(self) -> ProjectRepositoryPort: ...

    @property
    def sources(self) -> SourceRepositoryPort: ...

    @property
    def source_items(self) -> SourceItemRepositoryPort: ...

    @property
    def output_schemas(self) -> OutputSchemaRepositoryPort: ...

    @property
    def recipes(self) -> RecipeRepositoryPort: ...

    @property
    def jobs(self) -> JobRepositoryPort: ...

    @property
    def job_items(self) -> JobItemRepositoryPort: ...

    @property
    def workflow_definitions(self) -> WorkflowDefinitionRepositoryPort: ...

    @property
    def workflow_versions(self) -> WorkflowVersionRepositoryPort: ...

    @property
    def workflow_runs(self) -> WorkflowRunRepositoryPort: ...

    @property
    def experiments(self) -> ExperimentRepositoryPort: ...

    @property
    def node_runs(self) -> NodeRunRepositoryPort: ...

    @property
    def artifacts(self) -> ArtifactRepositoryPort: ...

    @property
    def artifact_sequences(self) -> ArtifactSequenceRepositoryPort: ...

    @property
    def input_assembly_traces(self) -> InputAssemblyTraceRepositoryPort: ...

    @property
    def invocation_traces(self) -> InvocationTraceRepositoryPort: ...

    @property
    def outbox_messages(self) -> OutboxMessageRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

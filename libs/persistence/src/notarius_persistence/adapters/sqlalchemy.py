from collections.abc import Callable, Iterable
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from types import TracebackType
from typing import Any, Self, final, override
from uuid import UUID

from sqlalchemy import Connection, Engine, Select, create_engine, delete, insert, select, update

from notarius_core.domain.models.studio import (
    ContextTrace,
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
    ArtifactPortRef,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    Experiment,
    ExperimentParameter,
    ExperimentStatus,
    ExperimentVariant,
    InputAssemblyTrace,
    InvocationTrace,
    NodeRun,
    NodeRunStatus,
    OutboxMessage,
    OutboxMessageStatus,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
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
from notarius_persistence import schema


def _uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


def _optional_uuid(value: UUID | str | None) -> UUID | None:
    if value is None:
        return None
    return _uuid(value)


def _status(value: JobStatus | str) -> JobStatus:
    return value if isinstance(value, JobStatus) else JobStatus(value)


def _context_trace(value: ContextTrace | dict[str, Any] | None) -> ContextTrace | None:
    if value is None or isinstance(value, ContextTrace):
        return value
    return ContextTrace(**value)


def _datetime(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _optional_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


def _workflow_run_status(value: WorkflowRunStatus | str) -> WorkflowRunStatus:
    return value if isinstance(value, WorkflowRunStatus) else WorkflowRunStatus(value)


def _node_run_status(value: NodeRunStatus | str) -> NodeRunStatus:
    return value if isinstance(value, NodeRunStatus) else NodeRunStatus(value)


def _outbox_message_status(
    value: OutboxMessageStatus | str,
) -> OutboxMessageStatus:
    return (
        value if isinstance(value, OutboxMessageStatus) else OutboxMessageStatus(value)
    )


def _experiment_status(value: ExperimentStatus | str) -> ExperimentStatus:
    return value if isinstance(value, ExperimentStatus) else ExperimentStatus(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _source_id_from_metadata(metadata: object) -> str | None:
    if not isinstance(metadata, dict):
        return None
    source_id = metadata.get("source_id")
    return source_id if isinstance(source_id, str) else None


def _artifact_ref(value: ArtifactRef | dict[str, Any]) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    return ArtifactRef(
        artifact_id=_uuid(value["artifact_id"]),
        artifact_type=value["artifact_type"],
        schema_version=value["schema_version"],
        content_hash=value.get("content_hash"),
    )


def _artifact_refs(values: list[ArtifactRef | dict[str, Any]]) -> list[ArtifactRef]:
    return [_artifact_ref(value) for value in values]


def _artifact_sequence_ref(
    value: ArtifactSequenceRef | dict[str, Any],
) -> ArtifactSequenceRef:
    if isinstance(value, ArtifactSequenceRef):
        return value
    return ArtifactSequenceRef(
        sequence_id=_uuid(value["sequence_id"]),
        artifact_type=value["artifact_type"],
        schema_version=value["schema_version"],
    )


def _artifact_sequence_refs(
    values: list[ArtifactSequenceRef | dict[str, Any]],
) -> list[ArtifactSequenceRef]:
    return [_artifact_sequence_ref(value) for value in values]


def _artifact_port_ref(
    value: ArtifactRef | ArtifactSequenceRef | dict[str, Any],
) -> ArtifactRef | ArtifactSequenceRef:
    if isinstance(value, ArtifactRef | ArtifactSequenceRef):
        return value
    if "sequence_id" in value:
        return _artifact_sequence_ref(value)
    return _artifact_ref(value)


def _artifact_ref_or_list(
    value: (
        ArtifactRef
        | ArtifactSequenceRef
        | dict[str, Any]
        | list[ArtifactRef | dict[str, Any]]
    ),
) -> ArtifactPortRef:
    if isinstance(value, list):
        return _artifact_refs(value)
    return _artifact_port_ref(value)


def _artifact_ref_map(
    value: dict[
        str,
        ArtifactRef
        | ArtifactSequenceRef
        | dict[str, Any]
        | list[ArtifactRef | dict[str, Any]],
    ],
) -> dict[str, ArtifactPortRef]:
    return {key: _artifact_ref_or_list(item) for key, item in value.items()}


def _workflow_node(value: WorkflowNode | dict[str, Any]) -> WorkflowNode:
    if isinstance(value, WorkflowNode):
        return value
    return WorkflowNode(
        id=value["id"],
        operator_id=value["operator_id"],
        operator_version=value["operator_version"],
        config=value.get("config", {}),
        label=value.get("label"),
        ui_position=value.get("ui_position", {}),
    )


def _workflow_edge(value: WorkflowEdge | dict[str, Any]) -> WorkflowEdge:
    if isinstance(value, WorkflowEdge):
        return value
    return WorkflowEdge(
        from_node_id=value["from_node_id"],
        from_port=value["from_port"],
        to_node_id=value["to_node_id"],
        to_port=value["to_port"],
    )


def _port_spec(value: PortSpec | dict[str, Any]) -> PortSpec:
    if isinstance(value, PortSpec):
        return value
    return PortSpec(
        name=value["name"],
        artifact_type=value["artifact_type"],
        schema_version=value["schema_version"],
        sequence=value.get("sequence", False),
        required=value.get("required", True),
        description=value.get("description"),
    )


def _workflow_definition(
    value: WorkflowDefinition | dict[str, Any],
) -> WorkflowDefinition:
    if isinstance(value, WorkflowDefinition):
        return value
    return WorkflowDefinition(
        name=value["name"],
        nodes=[_workflow_node(node) for node in value.get("nodes", [])],
        edges=[_workflow_edge(edge) for edge in value.get("edges", [])],
        id=_uuid(value["id"]),
        description=value.get("description"),
        declared_inputs=[_port_spec(port) for port in value.get("declared_inputs", [])],
        metadata=value.get("metadata", {}),
        created_at=_datetime(value["created_at"]),
        updated_at=_datetime(value["updated_at"]),
    )


def _uuid_list(values: list[UUID | str]) -> list[UUID]:
    return [_uuid(value) for value in values]


def _experiment_parameter(
    value: ExperimentParameter | dict[str, Any],
) -> ExperimentParameter:
    if isinstance(value, ExperimentParameter):
        return value
    return ExperimentParameter(
        name=value["name"],
        node_id=value["node_id"],
        config_path=tuple(value["config_path"]),
        values=tuple(value["values"]),
        description=value.get("description"),
    )


def _experiment_parameters(
    values: list[ExperimentParameter | dict[str, Any]],
) -> list[ExperimentParameter]:
    return [_experiment_parameter(value) for value in values]


def _experiment_variant(value: ExperimentVariant | dict[str, Any]) -> ExperimentVariant:
    if isinstance(value, ExperimentVariant):
        return value
    return ExperimentVariant(
        key=value["key"],
        ordinal=value["ordinal"],
        parameter_values=value["parameter_values"],
        workflow_run_id=_uuid(value["workflow_run_id"]),
        id=_uuid(value["id"]),
        metadata=value.get("metadata", {}),
    )


def _experiment_variants(
    values: list[ExperimentVariant | dict[str, Any]],
) -> list[ExperimentVariant]:
    return [_experiment_variant(value) for value in values]


@final
class SqlProjectRepository(ProjectRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, project: Project) -> None:
        self.conn.execute(
            insert(schema.projects).values(
                id=str(project.id),
                name=project.name,
                description=project.description,
                created_at=project.created_at,
            )
        )

    @override
    async def get(self, project_id: UUID) -> Project | None:
        row = (
            self.conn.execute(
                select(schema.projects).where(schema.projects.c.id == str(project_id))
            )
            .mappings()
            .first()
        )
        return None if row is None else Project(**{**row, "id": _uuid(row["id"])})

    @override
    async def list(self) -> list[Project]:
        rows = self.conn.execute(
            select(schema.projects).order_by(
                schema.projects.c.created_at, schema.projects.c.id
            )
        ).mappings()
        return [Project(**{**row, "id": _uuid(row["id"])}) for row in rows]


@final
class SqlSourceRepository(SourceRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, source: Source) -> None:
        self.conn.execute(
            insert(schema.sources).values(
                id=str(source.id),
                project_id=str(source.project_id),
                name=source.name,
                description=source.description,
                created_at=source.created_at,
            )
        )

    @override
    async def get(self, source_id: UUID) -> Source | None:
        row = (
            self.conn.execute(
                select(schema.sources).where(schema.sources.c.id == str(source_id))
            )
            .mappings()
            .first()
        )
        return None if row is None else Source(**self._decode(row))

    @override
    async def list_for_project(self, project_id: UUID) -> list[Source]:
        rows = self.conn.execute(
            select(schema.sources)
            .where(schema.sources.c.project_id == str(project_id))
            .order_by(schema.sources.c.created_at, schema.sources.c.id)
        ).mappings()
        return [Source(**self._decode(row)) for row in rows]

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {**row, "id": _uuid(row["id"]), "project_id": _uuid(row["project_id"])}


@final
class SqlSourceItemRepository(SourceItemRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, item: SourceItem) -> None:
        self.conn.execute(insert(schema.source_items).values(**self._encode(item)))

    @override
    async def add_batch(self, items: list[SourceItem]) -> None:
        for item in items:
            await self.add(item)

    @override
    async def get(self, item_id: UUID) -> SourceItem | None:
        row = (
            self.conn.execute(
                select(schema.source_items).where(
                    schema.source_items.c.id == str(item_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else SourceItem(**self._decode(row))

    @override
    async def list_for_source(self, source_id: UUID) -> list[SourceItem]:
        rows = self.conn.execute(
            select(schema.source_items)
            .where(schema.source_items.c.source_id == str(source_id))
            .order_by(
                schema.source_items.c.order,
                schema.source_items.c.created_at,
                schema.source_items.c.id,
            )
        ).mappings()
        return [SourceItem(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(item: SourceItem) -> dict[str, Any]:
        return {
            "id": str(item.id),
            "source_id": str(item.source_id),
            "order": item.order,
            "text": item.text,
            "image_path": item.image_path,
            "metadata": item.metadata,
            "created_at": item.created_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {**row, "id": _uuid(row["id"]), "source_id": _uuid(row["source_id"])}


@final
class SqlOutputSchemaRepository(OutputSchemaRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, output_schema: OutputSchema) -> None:
        self.conn.execute(
            insert(schema.output_schemas).values(
                id=str(output_schema.id),
                project_id=str(output_schema.project_id),
                name=output_schema.name,
                description=output_schema.description,
                json_schema=output_schema.json_schema,
                created_at=output_schema.created_at,
            )
        )

    @override
    async def get(self, schema_id: UUID) -> OutputSchema | None:
        row = (
            self.conn.execute(
                select(schema.output_schemas).where(
                    schema.output_schemas.c.id == str(schema_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else OutputSchema(**self._decode(row))

    @override
    async def list_for_project(self, project_id: UUID) -> list[OutputSchema]:
        rows = self.conn.execute(
            select(schema.output_schemas)
            .where(schema.output_schemas.c.project_id == str(project_id))
            .order_by(schema.output_schemas.c.created_at, schema.output_schemas.c.id)
        ).mappings()
        return [OutputSchema(**self._decode(row)) for row in rows]

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {**row, "id": _uuid(row["id"]), "project_id": _uuid(row["project_id"])}


@final
class SqlRecipeRepository(RecipeRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, recipe: Recipe) -> None:
        self.conn.execute(
            insert(schema.recipes).values(
                id=str(recipe.id),
                project_id=str(recipe.project_id),
                schema_id=str(recipe.schema_id),
                name=recipe.name,
                description=recipe.description,
                config=recipe.config,
                created_at=recipe.created_at,
            )
        )

    @override
    async def get(self, recipe_id: UUID) -> Recipe | None:
        row = (
            self.conn.execute(
                select(schema.recipes).where(schema.recipes.c.id == str(recipe_id))
            )
            .mappings()
            .first()
        )
        return None if row is None else Recipe(**self._decode(row))

    @override
    async def list_for_project(self, project_id: UUID) -> list[Recipe]:
        rows = self.conn.execute(
            select(schema.recipes)
            .where(schema.recipes.c.project_id == str(project_id))
            .order_by(schema.recipes.c.created_at, schema.recipes.c.id)
        ).mappings()
        return [Recipe(**self._decode(row)) for row in rows]

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "project_id": _uuid(row["project_id"]),
            "schema_id": _uuid(row["schema_id"]),
        }


@final
class SqlJobRepository(JobRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, job: Job) -> None:
        self.conn.execute(insert(schema.jobs).values(**self._encode(job)))

    @override
    async def get(self, job_id: UUID) -> Job | None:
        row = (
            self.conn.execute(
                select(schema.jobs).where(schema.jobs.c.id == str(job_id))
            )
            .mappings()
            .first()
        )
        return None if row is None else Job(**self._decode(row))

    @override
    async def next_queued(self) -> Job | None:
        rows = await self.list_by_status(JobStatus.QUEUED)
        return rows[0] if rows else None

    @override
    async def update(self, job: Job) -> None:
        self.conn.execute(
            update(schema.jobs)
            .where(schema.jobs.c.id == str(job.id))
            .values(**self._encode(job))
        )

    @override
    async def list_for_project(self, project_id: UUID) -> list[Job]:
        rows = self.conn.execute(
            select(schema.jobs)
            .where(schema.jobs.c.project_id == str(project_id))
            .order_by(schema.jobs.c.created_at, schema.jobs.c.id)
        ).mappings()
        return [Job(**self._decode(row)) for row in rows]

    @override
    async def list_by_status(self, status: JobStatus) -> list[Job]:
        stmt: Select[Any] = (
            select(schema.jobs)
            .where(schema.jobs.c.status == status.value)
            .order_by(schema.jobs.c.created_at, schema.jobs.c.id)
        )
        rows = self.conn.execute(stmt).mappings()
        return [Job(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(job: Job) -> dict[str, Any]:
        return {
            "id": str(job.id),
            "project_id": str(job.project_id),
            "source_id": str(job.source_id),
            "recipe_id": str(job.recipe_id),
            "status": job.status.value,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "project_id": _uuid(row["project_id"]),
            "source_id": _uuid(row["source_id"]),
            "recipe_id": _uuid(row["recipe_id"]),
            "status": _status(row["status"]),
        }


@final
class SqlJobItemRepository(JobItemRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, job_item: JobItem) -> None:
        self.conn.execute(insert(schema.job_items).values(**self._encode(job_item)))

    @override
    async def add_batch(self, job_items: list[JobItem]) -> None:
        for job_item in job_items:
            await self.add(job_item)

    @override
    async def get(self, job_item_id: UUID) -> JobItem | None:
        row = (
            self.conn.execute(
                select(schema.job_items).where(
                    schema.job_items.c.id == str(job_item_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else JobItem(**self._decode(row))

    @override
    async def list_for_job(self, job_id: UUID) -> list[JobItem]:
        rows = self.conn.execute(
            select(schema.job_items)
            .where(schema.job_items.c.job_id == str(job_id))
            .order_by(
                schema.job_items.c.order,
                schema.job_items.c.created_at,
                schema.job_items.c.id,
            )
        ).mappings()
        return [JobItem(**self._decode(row)) for row in rows]

    @override
    async def update(self, job_item: JobItem) -> None:
        self.conn.execute(
            update(schema.job_items)
            .where(schema.job_items.c.id == str(job_item.id))
            .values(**self._encode(job_item))
        )

    @staticmethod
    def _encode(job_item: JobItem) -> dict[str, Any]:
        return {
            "id": str(job_item.id),
            "job_id": str(job_item.job_id),
            "source_item_id": str(job_item.source_item_id),
            "order": job_item.order,
            "status": job_item.status.value,
            "structured_output": job_item.structured_output,
            "context_trace": asdict(job_item.context_trace)
            if job_item.context_trace
            else None,
            "error": job_item.error,
            "created_at": job_item.created_at,
            "updated_at": job_item.updated_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "job_id": _uuid(row["job_id"]),
            "source_item_id": _uuid(row["source_item_id"]),
            "status": _status(row["status"]),
            "context_trace": _context_trace(row["context_trace"]),
        }


@final
class SqlWorkflowDefinitionRepository(WorkflowDefinitionRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, definition: WorkflowDefinition) -> None:
        self.conn.execute(
            insert(schema.workflow_definitions).values(**self._encode(definition))
        )

    @override
    async def get(self, definition_id: UUID) -> WorkflowDefinition | None:
        row = (
            self.conn.execute(
                select(schema.workflow_definitions).where(
                    schema.workflow_definitions.c.id == str(definition_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else WorkflowDefinition(**self._decode(row))

    @override
    async def update(self, definition: WorkflowDefinition) -> None:
        self.conn.execute(
            update(schema.workflow_definitions)
            .where(schema.workflow_definitions.c.id == str(definition.id))
            .values(**self._encode(definition))
        )

    @override
    async def list(self) -> list[WorkflowDefinition]:
        rows = self.conn.execute(
            select(schema.workflow_definitions).order_by(
                schema.workflow_definitions.c.created_at,
                schema.workflow_definitions.c.id,
            )
        ).mappings()
        return [WorkflowDefinition(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(definition: WorkflowDefinition) -> dict[str, Any]:
        return {
            "id": str(definition.id),
            "name": definition.name,
            "description": definition.description,
            "nodes": _json_ready(definition.nodes),
            "edges": _json_ready(definition.edges),
            "declared_inputs": _json_ready(definition.declared_inputs),
            "metadata": definition.metadata,
            "created_at": definition.created_at,
            "updated_at": definition.updated_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "nodes": [_workflow_node(node) for node in row["nodes"]],
            "edges": [_workflow_edge(edge) for edge in row["edges"]],
            "declared_inputs": [_port_spec(port) for port in row["declared_inputs"]],
            "created_at": _datetime(row["created_at"]),
            "updated_at": _datetime(row["updated_at"]),
        }


@final
class SqlWorkflowVersionRepository(WorkflowVersionRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, version: WorkflowVersion) -> None:
        self.conn.execute(
            insert(schema.workflow_versions).values(**self._encode(version))
        )

    @override
    async def get(self, version_id: UUID) -> WorkflowVersion | None:
        row = (
            self.conn.execute(
                select(schema.workflow_versions).where(
                    schema.workflow_versions.c.id == str(version_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else WorkflowVersion(**self._decode(row))

    @override
    async def list_for_definition(
        self,
        definition_id: UUID,
    ) -> list[WorkflowVersion]:
        rows = self.conn.execute(
            select(schema.workflow_versions)
            .where(
                schema.workflow_versions.c.workflow_definition_id == str(definition_id)
            )
            .order_by(
                schema.workflow_versions.c.version_number,
                schema.workflow_versions.c.created_at,
                schema.workflow_versions.c.id,
            )
        ).mappings()
        return [WorkflowVersion(**self._decode(row)) for row in rows]

    @override
    async def latest_for_definition(
        self,
        definition_id: UUID,
    ) -> WorkflowVersion | None:
        versions = await self.list_for_definition(definition_id)
        return versions[-1] if versions else None

    @staticmethod
    def _encode(version: WorkflowVersion) -> dict[str, Any]:
        return {
            "id": str(version.id),
            "workflow_definition_id": str(version.workflow_definition_id),
            "version_number": version.version_number,
            "definition_snapshot": _json_ready(version.definition_snapshot),
            "created_at": version.created_at,
            "created_by": version.created_by,
            "change_note": version.change_note,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "workflow_definition_id": _uuid(row["workflow_definition_id"]),
            "definition_snapshot": _workflow_definition(row["definition_snapshot"]),
            "created_at": _datetime(row["created_at"]),
        }


@final
class SqlWorkflowRunRepository(WorkflowRunRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, run: WorkflowRun) -> None:
        self.conn.execute(insert(schema.workflow_runs).values(**self._encode(run)))

    @override
    async def get(self, run_id: UUID) -> WorkflowRun | None:
        row = (
            self.conn.execute(
                select(schema.workflow_runs).where(
                    schema.workflow_runs.c.id == str(run_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else WorkflowRun(**self._decode(row))

    @override
    async def update(self, run: WorkflowRun) -> None:
        self.conn.execute(
            update(schema.workflow_runs)
            .where(schema.workflow_runs.c.id == str(run.id))
            .values(**self._encode(run))
        )

    @override
    async def next_queued(self) -> WorkflowRun | None:
        rows = await self.list_by_status(WorkflowRunStatus.QUEUED)
        return rows[0] if rows else None

    @override
    async def list_for_version(self, version_id: UUID) -> list[WorkflowRun]:
        rows = self.conn.execute(
            select(schema.workflow_runs)
            .where(schema.workflow_runs.c.workflow_version_id == str(version_id))
            .order_by(schema.workflow_runs.c.queued_at, schema.workflow_runs.c.id)
        ).mappings()
        return [WorkflowRun(**self._decode(row)) for row in rows]

    @override
    async def list_by_status(self, status: WorkflowRunStatus) -> list[WorkflowRun]:
        rows = self.conn.execute(
            select(schema.workflow_runs)
            .where(schema.workflow_runs.c.status == status.value)
            .order_by(schema.workflow_runs.c.queued_at, schema.workflow_runs.c.id)
        ).mappings()
        return [WorkflowRun(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(run: WorkflowRun) -> dict[str, Any]:
        return {
            "id": str(run.id),
            "workflow_version_id": str(run.workflow_version_id),
            "status": run.status.value,
            "input_artifact_refs": _json_ready(run.input_artifact_refs),
            "input_artifact_sequence_refs": _json_ready(
                run.input_artifact_sequence_refs
            ),
            "output_artifact_refs": _json_ready(run.output_artifact_refs),
            "metadata": run.metadata,
            "error": run.error,
            "queued_at": run.queued_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "workflow_version_id": _uuid(row["workflow_version_id"]),
            "status": _workflow_run_status(row["status"]),
            "input_artifact_refs": _artifact_refs(row["input_artifact_refs"]),
            "input_artifact_sequence_refs": _artifact_sequence_refs(
                row["input_artifact_sequence_refs"]
            ),
            "output_artifact_refs": _artifact_refs(row["output_artifact_refs"]),
            "queued_at": _datetime(row["queued_at"]),
            "started_at": _optional_datetime(row["started_at"]),
            "finished_at": _optional_datetime(row["finished_at"]),
        }


@final
class SqlExperimentRepository(ExperimentRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, experiment: Experiment) -> None:
        self.conn.execute(insert(schema.experiments).values(**self._encode(experiment)))

    @override
    async def get(self, experiment_id: UUID) -> Experiment | None:
        row = (
            self.conn.execute(
                select(schema.experiments).where(
                    schema.experiments.c.id == str(experiment_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else Experiment(**self._decode(row))

    @override
    async def update(self, experiment: Experiment) -> None:
        self.conn.execute(
            update(schema.experiments)
            .where(schema.experiments.c.id == str(experiment.id))
            .values(**self._encode(experiment))
        )

    @override
    async def list_for_workflow_version(
        self,
        workflow_version_id: UUID,
    ) -> list[Experiment]:
        rows = self.conn.execute(
            select(schema.experiments)
            .where(schema.experiments.c.workflow_version_id == str(workflow_version_id))
            .order_by(
                schema.experiments.c.created_at,
                schema.experiments.c.id,
            )
        ).mappings()
        return [Experiment(**self._decode(row)) for row in rows]

    @override
    async def list(self) -> list[Experiment]:
        rows = self.conn.execute(
            select(schema.experiments).order_by(
                schema.experiments.c.created_at,
                schema.experiments.c.id,
            )
        ).mappings()
        return [Experiment(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(experiment: Experiment) -> dict[str, Any]:
        return {
            "id": str(experiment.id),
            "name": experiment.name,
            "description": experiment.description,
            "workflow_version_id": str(experiment.workflow_version_id),
            "status": experiment.status.value,
            "parameters": _json_ready(experiment.parameters),
            "input_artifact_refs": _json_ready(experiment.input_artifact_refs),
            "input_artifact_sequence_refs": _json_ready(
                experiment.input_artifact_sequence_refs
            ),
            "variants": _json_ready(experiment.variants),
            "metadata": experiment.metadata,
            "created_at": experiment.created_at,
            "updated_at": experiment.updated_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "workflow_version_id": _uuid(row["workflow_version_id"]),
            "status": _experiment_status(row["status"]),
            "parameters": _experiment_parameters(row["parameters"]),
            "input_artifact_refs": _artifact_refs(row["input_artifact_refs"]),
            "input_artifact_sequence_refs": _artifact_sequence_refs(
                row["input_artifact_sequence_refs"]
            ),
            "variants": _experiment_variants(row["variants"]),
            "created_at": _datetime(row["created_at"]),
            "updated_at": _datetime(row["updated_at"]),
        }


@final
class SqlNodeRunRepository(NodeRunRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, node_run: NodeRun) -> None:
        self.conn.execute(insert(schema.node_runs).values(**self._encode(node_run)))

    @override
    async def add_batch(self, node_runs: list[NodeRun]) -> None:
        for node_run in node_runs:
            await self.add(node_run)

    @override
    async def get(self, node_run_id: UUID) -> NodeRun | None:
        row = (
            self.conn.execute(
                select(schema.node_runs).where(
                    schema.node_runs.c.id == str(node_run_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else NodeRun(**self._decode(row))

    @override
    async def update(self, node_run: NodeRun) -> None:
        self.conn.execute(
            update(schema.node_runs)
            .where(schema.node_runs.c.id == str(node_run.id))
            .values(**self._encode(node_run))
        )

    @override
    async def next_queued(self) -> NodeRun | None:
        rows = await self.list_by_status(NodeRunStatus.QUEUED)
        return rows[0] if rows else None

    @override
    async def list_for_workflow_run(self, workflow_run_id: UUID) -> list[NodeRun]:
        rows = self.conn.execute(
            select(schema.node_runs)
            .where(schema.node_runs.c.workflow_run_id == str(workflow_run_id))
            .order_by(schema.node_runs.c.queued_at, schema.node_runs.c.id)
        ).mappings()
        return [NodeRun(**self._decode(row)) for row in rows]

    @override
    async def list_by_status(self, status: NodeRunStatus) -> list[NodeRun]:
        rows = self.conn.execute(
            select(schema.node_runs)
            .where(schema.node_runs.c.status == status.value)
            .order_by(schema.node_runs.c.queued_at, schema.node_runs.c.id)
        ).mappings()
        return [NodeRun(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(node_run: NodeRun) -> dict[str, Any]:
        return {
            "id": str(node_run.id),
            "workflow_run_id": str(node_run.workflow_run_id),
            "workflow_node_id": node_run.workflow_node_id,
            "operator_id": node_run.operator_id,
            "operator_version": node_run.operator_version,
            "status": node_run.status.value,
            "input_artifact_refs": _json_ready(node_run.input_artifact_refs),
            "output_artifact_refs": _json_ready(node_run.output_artifact_refs),
            "attempt_count": node_run.attempt_count,
            "max_attempts": node_run.max_attempts,
            "metadata": node_run.metadata,
            "error": node_run.error,
            "queued_at": node_run.queued_at,
            "started_at": node_run.started_at,
            "finished_at": node_run.finished_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "workflow_run_id": _uuid(row["workflow_run_id"]),
            "status": _node_run_status(row["status"]),
            "input_artifact_refs": _artifact_ref_map(row["input_artifact_refs"]),
            "output_artifact_refs": _artifact_ref_map(row["output_artifact_refs"]),
            "queued_at": _datetime(row["queued_at"]),
            "started_at": _optional_datetime(row["started_at"]),
            "finished_at": _optional_datetime(row["finished_at"]),
        }


@final
class SqlArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, artifact: Artifact) -> None:
        self.conn.execute(insert(schema.artifacts).values(**self._encode(artifact)))

    @override
    async def get(self, artifact_id: UUID) -> Artifact | None:
        row = (
            self.conn.execute(
                select(schema.artifacts).where(
                    schema.artifacts.c.id == str(artifact_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else Artifact(**self._decode(row))

    @override
    async def list_for_source(self, source_id: UUID) -> list[Artifact]:
        rows = self.conn.execute(
            select(schema.artifacts).order_by(
                schema.artifacts.c.created_at,
                schema.artifacts.c.id,
            )
        ).mappings()
        source_id_value = str(source_id)
        return [
            Artifact(**self._decode(row))
            for row in rows
            if _source_id_from_metadata(row["metadata"]) == source_id_value
        ]

    @override
    async def list_for_workflow_run(self, workflow_run_id: UUID) -> list[Artifact]:
        rows = self.conn.execute(
            select(schema.artifacts)
            .where(schema.artifacts.c.workflow_run_id == str(workflow_run_id))
            .order_by(schema.artifacts.c.created_at, schema.artifacts.c.id)
        ).mappings()
        return [Artifact(**self._decode(row)) for row in rows]

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[Artifact]:
        rows = self.conn.execute(
            select(schema.artifacts)
            .where(schema.artifacts.c.producer_node_run_id == str(node_run_id))
            .order_by(schema.artifacts.c.created_at, schema.artifacts.c.id)
        ).mappings()
        return [Artifact(**self._decode(row)) for row in rows]

    @override
    async def list_by_type(self, artifact_type: str) -> list[Artifact]:
        rows = self.conn.execute(
            select(schema.artifacts)
            .where(schema.artifacts.c.artifact_type == artifact_type)
            .order_by(schema.artifacts.c.created_at, schema.artifacts.c.id)
        ).mappings()
        return [Artifact(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(artifact: Artifact) -> dict[str, Any]:
        return {
            "id": str(artifact.id),
            "artifact_type": artifact.artifact_type,
            "schema_version": artifact.schema_version,
            "workflow_run_id": (
                str(artifact.workflow_run_id)
                if artifact.workflow_run_id is not None
                else None
            ),
            "producer_node_run_id": (
                str(artifact.producer_node_run_id)
                if artifact.producer_node_run_id is not None
                else None
            ),
            "payload_ref": artifact.payload_ref,
            "producer_operator_id": artifact.producer_operator_id,
            "producer_operator_version": artifact.producer_operator_version,
            "input_artifact_ids": [str(value) for value in artifact.input_artifact_ids],
            "content_hash": artifact.content_hash,
            "preview_ref": artifact.preview_ref,
            "metadata": artifact.metadata,
            "created_at": artifact.created_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        producer_node_run_id = row["producer_node_run_id"]
        return {
            **row,
            "id": _uuid(row["id"]),
            "workflow_run_id": _optional_uuid(row["workflow_run_id"]),
            "producer_node_run_id": (
                _uuid(producer_node_run_id)
                if producer_node_run_id is not None
                else None
            ),
            "input_artifact_ids": _uuid_list(row["input_artifact_ids"]),
            "created_at": _datetime(row["created_at"]),
        }


@final
class SqlArtifactSequenceRepository(ArtifactSequenceRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, sequence: ArtifactSequence) -> None:
        self.conn.execute(
            insert(schema.artifact_sequences).values(**self._encode(sequence))
        )

    @override
    async def get(self, sequence_id: UUID) -> ArtifactSequence | None:
        row = (
            self.conn.execute(
                select(schema.artifact_sequences).where(
                    schema.artifact_sequences.c.id == str(sequence_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else ArtifactSequence(**self._decode(row))

    @override
    async def list_for_source(self, source_id: UUID) -> list[ArtifactSequence]:
        rows = self.conn.execute(
            select(schema.artifact_sequences).order_by(
                schema.artifact_sequences.c.created_at,
                schema.artifact_sequences.c.id,
            )
        ).mappings()
        source_id_value = str(source_id)
        return [
            ArtifactSequence(**self._decode(row))
            for row in rows
            if _source_id_from_metadata(row["metadata"]) == source_id_value
        ]

    @override
    async def list_by_artifact_type(self, artifact_type: str) -> list[ArtifactSequence]:
        rows = self.conn.execute(
            select(schema.artifact_sequences)
            .where(schema.artifact_sequences.c.artifact_type == artifact_type)
            .order_by(
                schema.artifact_sequences.c.created_at,
                schema.artifact_sequences.c.id,
            )
        ).mappings()
        return [ArtifactSequence(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(sequence: ArtifactSequence) -> dict[str, Any]:
        return {
            "id": str(sequence.id),
            "artifact_type": sequence.artifact_type,
            "item_refs": _json_ready(sequence.item_refs),
            "schema_version": sequence.schema_version,
            "ordered": sequence.ordered,
            "index_key": sequence.index_key,
            "metadata": sequence.metadata,
            "created_at": sequence.created_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "item_refs": _artifact_refs(row["item_refs"]),
            "created_at": _datetime(row["created_at"]),
        }


@final
class SqlInputAssemblyTraceRepository(InputAssemblyTraceRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, trace: InputAssemblyTrace) -> None:
        self.conn.execute(
            insert(schema.input_assembly_traces).values(**self._encode(trace))
        )

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[InputAssemblyTrace]:
        rows = self.conn.execute(
            select(schema.input_assembly_traces)
            .where(schema.input_assembly_traces.c.node_run_id == str(node_run_id))
            .order_by(
                schema.input_assembly_traces.c.created_at,
                schema.input_assembly_traces.c.id,
            )
        ).mappings()
        return [InputAssemblyTrace(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(trace: InputAssemblyTrace) -> dict[str, Any]:
        return {
            "id": str(trace.id),
            "node_run_id": str(trace.node_run_id),
            "selected_inputs": _json_ready(trace.selected_inputs),
            "omitted_inputs": trace.omitted_inputs,
            "policies": trace.policies,
            "metadata": trace.metadata,
            "created_at": trace.created_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "node_run_id": _uuid(row["node_run_id"]),
            "selected_inputs": _artifact_ref_map(row["selected_inputs"]),
            "created_at": _datetime(row["created_at"]),
        }


@final
class SqlInvocationTraceRepository(InvocationTraceRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, trace: InvocationTrace) -> None:
        self.conn.execute(
            insert(schema.invocation_traces).values(**self._encode(trace))
        )

    @override
    async def list_for_node_run(self, node_run_id: UUID) -> list[InvocationTrace]:
        rows = self.conn.execute(
            select(schema.invocation_traces)
            .where(schema.invocation_traces.c.node_run_id == str(node_run_id))
            .order_by(
                schema.invocation_traces.c.created_at,
                schema.invocation_traces.c.id,
            )
        ).mappings()
        return [InvocationTrace(**self._decode(row)) for row in rows]

    @staticmethod
    def _encode(trace: InvocationTrace) -> dict[str, Any]:
        return {
            "id": str(trace.id),
            "node_run_id": str(trace.node_run_id),
            "invocation_type": trace.invocation_type,
            "input_artifact_refs": _json_ready(trace.input_artifact_refs),
            "output_artifact_refs": _json_ready(trace.output_artifact_refs),
            "provider": trace.provider,
            "model": trace.model,
            "request_ref": trace.request_ref,
            "response_ref": trace.response_ref,
            "runtime": trace.runtime,
            "metadata": trace.metadata,
            "error": trace.error,
            "created_at": trace.created_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "node_run_id": _uuid(row["node_run_id"]),
            "input_artifact_refs": _artifact_refs(row["input_artifact_refs"]),
            "output_artifact_refs": _artifact_refs(row["output_artifact_refs"]),
            "created_at": _datetime(row["created_at"]),
        }


@final
class SqlOutboxMessageRepository(OutboxMessageRepositoryPort):
    def __init__(self, conn: Connection):
        self.conn = conn

    @override
    async def add(self, message: OutboxMessage) -> None:
        self.conn.execute(
            insert(schema.outbox_messages).values(**self._encode(message))
        )

    @override
    async def get(self, message_id: UUID) -> OutboxMessage | None:
        row = (
            self.conn.execute(
                select(schema.outbox_messages).where(
                    schema.outbox_messages.c.id == str(message_id)
                )
            )
            .mappings()
            .first()
        )
        return None if row is None else OutboxMessage(**self._decode(row))

    @override
    async def update(self, message: OutboxMessage) -> None:
        self.conn.execute(
            update(schema.outbox_messages)
            .where(schema.outbox_messages.c.id == str(message.id))
            .values(**self._encode(message))
        )

    @override
    async def delete_many(self, message_ids: Iterable[UUID]) -> int:
        encoded_ids = [str(message_id) for message_id in message_ids]
        if not encoded_ids:
            return 0
        result = self.conn.execute(
            delete(schema.outbox_messages).where(
                schema.outbox_messages.c.id.in_(encoded_ids)
            )
        )
        return int(result.rowcount or 0)

    @override
    async def list_for_workflow_run(
        self,
        workflow_run_id: UUID,
    ) -> list[OutboxMessage]:
        rows = self.conn.execute(
            select(schema.outbox_messages).order_by(
                schema.outbox_messages.c.created_at,
                schema.outbox_messages.c.id,
            )
        ).mappings()
        messages = [OutboxMessage(**self._decode(row)) for row in rows]
        return [
            message
            for message in messages
            if self._payload_workflow_run_id(message) == workflow_run_id
        ]

    @override
    async def list_by_status(
        self,
        status: OutboxMessageStatus,
    ) -> list[OutboxMessage]:
        rows = self.conn.execute(
            select(schema.outbox_messages)
            .where(schema.outbox_messages.c.status == status.value)
            .order_by(
                schema.outbox_messages.c.created_at,
                schema.outbox_messages.c.id,
            )
        ).mappings()
        return [OutboxMessage(**self._decode(row)) for row in rows]

    @override
    async def list_pending(self) -> list[OutboxMessage]:
        return await self.list_by_status(OutboxMessageStatus.PENDING)

    @staticmethod
    def _encode(message: OutboxMessage) -> dict[str, Any]:
        return {
            "id": str(message.id),
            "subject": message.subject,
            "message_type": message.message_type,
            "payload": message.payload,
            "status": message.status.value,
            "attempts": message.attempts,
            "error": message.error,
            "created_at": message.created_at,
            "published_at": message.published_at,
        }

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        return {
            **row,
            "id": _uuid(row["id"]),
            "status": _outbox_message_status(row["status"]),
            "created_at": _datetime(row["created_at"]),
            "published_at": _optional_datetime(row["published_at"]),
        }

    @staticmethod
    def _payload_workflow_run_id(message: OutboxMessage) -> UUID | None:
        value = message.payload.get("workflow_run_id")
        if value is None:
            return None
        try:
            return UUID(str(value))
        except ValueError:
            return None


@final
class SqlAlchemyUnitOfWork(StudioUnitOfWorkPort):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.conn: Connection | None = None
        self.transaction: Any | None = None
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
        self._experiments: ExperimentRepositoryPort | None = None
        self._node_runs: NodeRunRepositoryPort | None = None
        self._artifacts: ArtifactRepositoryPort | None = None
        self._artifact_sequences: ArtifactSequenceRepositoryPort | None = None
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
    def experiments(self) -> ExperimentRepositoryPort:
        if self._experiments is None:
            raise RuntimeError("Unit of work is not entered")
        return self._experiments

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
        self.conn = self.engine.connect()
        self.transaction = self.conn.begin()
        self._projects = SqlProjectRepository(self.conn)
        self._sources = SqlSourceRepository(self.conn)
        self._source_items = SqlSourceItemRepository(self.conn)
        self._output_schemas = SqlOutputSchemaRepository(self.conn)
        self._recipes = SqlRecipeRepository(self.conn)
        self._jobs = SqlJobRepository(self.conn)
        self._job_items = SqlJobItemRepository(self.conn)
        self._workflow_definitions = SqlWorkflowDefinitionRepository(self.conn)
        self._workflow_versions = SqlWorkflowVersionRepository(self.conn)
        self._workflow_runs = SqlWorkflowRunRepository(self.conn)
        self._experiments = SqlExperimentRepository(self.conn)
        self._node_runs = SqlNodeRunRepository(self.conn)
        self._artifacts = SqlArtifactRepository(self.conn)
        self._artifact_sequences = SqlArtifactSequenceRepository(self.conn)
        self._input_assembly_traces = SqlInputAssemblyTraceRepository(self.conn)
        self._invocation_traces = SqlInvocationTraceRepository(self.conn)
        self._outbox_messages = SqlOutboxMessageRepository(self.conn)
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
        if self.conn is not None:
            self.conn.close()
        self.conn = None
        self.transaction = None
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
        self._experiments = None
        self._node_runs = None
        self._artifacts = None
        self._artifact_sequences = None
        self._input_assembly_traces = None
        self._invocation_traces = None
        self._outbox_messages = None

    @override
    async def commit(self) -> None:
        if self.transaction is None:
            raise RuntimeError("Unit of work is not entered")
        self.transaction.commit()
        self.transaction = self.conn.begin() if self.conn is not None else None

    @override
    async def rollback(self) -> None:
        if self.transaction is None:
            raise RuntimeError("Unit of work is not entered")
        self.transaction.rollback()
        self.transaction = self.conn.begin() if self.conn is not None else None


def create_sqlite_uow_factory(database_url: str) -> Callable[[], SqlAlchemyUnitOfWork]:
    engine = create_engine(database_url, future=True)
    schema.metadata.create_all(engine)
    return lambda: SqlAlchemyUnitOfWork(engine)

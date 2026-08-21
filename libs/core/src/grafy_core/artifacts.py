from asyncio import Lock
from collections.abc import Collection
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    final,
    override,
    Any,
    Callable,
)
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from grafy_core.domain.collaboration import GraphActiveExecutionSlot
    from grafy_core.domain.invocation_cache import InvocationCacheEntry
    from grafy_core.domain.execution_history import (
        GraphExecution,
        GraphExecutionCursor,
        GraphExecutionDetail,
        GraphExecutionNodeResult,
        GraphExecutionPage,
        GraphExecutionStatus,
    )
    from grafy_core.domain.materialized_outputs import MaterializedNodeOutputs
    from grafy_core.domain.staged_uploads import StagedUpload
    from grafy_core.ports.invocation_cache import InvocationCacheRepositoryPort
    from grafy_core.ports.execution_history import (
        GraphExecutionHistoryRepositoryPort,
    )
    from grafy_core.ports.materialized_outputs import (
        MaterializedNodeOutputsRepositoryPort,
    )
    from grafy_core.ports.staged_uploads import StagedUploadRepositoryPort
    from grafy_core.plugins import PluginRuntimeContext
    from grafy_core.runtime.persistence import ArtifactOutputWriter
    from grafy_core.runtime.resolvers import Resolver

JsonObject: TypeAlias = dict[str, object]
MaterializedJsonType: TypeAlias = Literal["string", "integer"]


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class NodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoConfig(NodeConfig):
    pass


def _clone[T](value: T) -> T:
    return deepcopy(value)


def artifact_id() -> UUID:
    return uuid4()


def sequence_id() -> UUID:
    return uuid4()


@dataclass(frozen=True, slots=True)
class ArtifactTypeKey:
    id: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class ArtifactFieldProjection:
    path: tuple[str, ...]
    target: ArtifactTypeKey
    title: str


@dataclass(frozen=True, slots=True)
class ArtifactExportFormat:
    """One downloadable rendering of an artifact type, beyond the universal JSON.

    `json` is always available for any artifact with a payload (it is the
    canonical whole-artifact document). `export_formats` declares the additional
    formats a type can be rendered into, e.g. bare text for text scalars.
    """

    format: str
    content_type: str
    filename: str


@dataclass(frozen=True, slots=True)
class ArtifactTypeSpec:
    key: ArtifactTypeKey
    title: str
    payload_schema: JsonObject = field(default_factory=dict)
    field_projections: tuple[ArtifactFieldProjection, ...] = ()
    materialized_json_type: MaterializedJsonType | None = None
    export_formats: tuple[ArtifactExportFormat, ...] = ()


@dataclass(frozen=True, slots=True)
class Artifact:
    spec: ArtifactTypeSpec
    resolver: "Callable[[PluginRuntimeContext], Resolver[Any]]"
    writer: "Callable[[PluginRuntimeContext], ArtifactOutputWriter]"


class ArtifactRef(BaseModel):
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_hash: str | None = None

    @classmethod
    def from_key(
        cls,
        *,
        artifact_id: UUID,
        key: ArtifactTypeKey,
        content_hash: str | None = None,
    ) -> Self:
        return cls(
            artifact_id=artifact_id,
            artifact_type=key.id,
            schema_version=key.schema_version,
            content_hash=content_hash,
        )

    def key(self) -> ArtifactTypeKey:
        return ArtifactTypeKey(self.artifact_type, self.schema_version)


class ArtifactRefSequence(BaseModel):
    sequence_id: UUID = Field(default_factory=sequence_id)
    artifact_type: str
    schema_version: int
    item_refs: list[ArtifactRef]
    ordered: bool = True
    index_key: str = "order_index"
    metadata: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_key(
        cls,
        *,
        key: ArtifactTypeKey,
        item_refs: list[ArtifactRef],
        metadata: JsonObject | None = None,
    ) -> Self:
        return cls(
            artifact_type=key.id,
            schema_version=key.schema_version,
            item_refs=item_refs,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def validate_item_refs(self) -> Self:
        for item_ref in self.item_refs:
            if item_ref.artifact_type != self.artifact_type:
                message = (
                    f"ArtifactSequence item type mismatch: expected "
                    f"{self.artifact_type}, got {item_ref.artifact_type}"
                )
                raise ValueError(message)
            if item_ref.schema_version != self.schema_version:
                message = (
                    f"ArtifactSequence schema version mismatch: expected "
                    f"{self.schema_version}, got {item_ref.schema_version}"
                )
                raise ValueError(message)
        return self


@dataclass
class ArtifactObject:
    workspace_id: UUID
    artifact_type: str
    schema_version: int
    content_type: str
    id: UUID = field(default_factory=artifact_id)
    storage_backend: str = "local"
    bucket: str | None = None
    object_key: str | None = None
    inline_payload: JsonObject | None = None
    byte_size: int | None = None
    sha256: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def ref(self) -> ArtifactRef:
        return ArtifactRef.from_key(
            artifact_id=self.id,
            key=ArtifactTypeKey(self.artifact_type, self.schema_version),
            content_hash=self.sha256,
        )


class ArtifactRepositoryPort(Protocol):
    async def add(self, artifact: ArtifactObject) -> None: ...

    async def get(
        self,
        workspace_id: UUID,
        artifact_id: UUID,
    ) -> ArtifactObject | None: ...

    async def get_many(
        self,
        workspace_id: UUID,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]: ...

    async def remove(self, workspace_id: UUID, artifact: ArtifactObject) -> None: ...

    async def list_by_type(
        self,
        workspace_id: UUID,
        key: ArtifactTypeKey,
    ) -> list[ArtifactObject]: ...


@dataclass(slots=True)
class InMemoryDataStore:
    artifacts: dict[UUID, ArtifactObject] = field(default_factory=dict)
    materialized_outputs: dict[
        tuple[UUID, UUID, int, str],
        "MaterializedNodeOutputs",
    ] = field(default_factory=dict)
    invocation_cache: dict[tuple[UUID, str], "InvocationCacheEntry"] = field(
        default_factory=dict
    )
    staged_uploads: dict[tuple[UUID, str], "StagedUpload"] = field(default_factory=dict)
    graph_executions: dict[
        UUID,
        "GraphExecution",
    ] = field(default_factory=dict)
    graph_execution_node_results: dict[
        tuple[UUID, UUID, str],
        "GraphExecutionNodeResult",
    ] = field(default_factory=dict)
    active_execution_slots: dict[
        tuple[UUID, UUID],
        "GraphActiveExecutionSlot",
    ] = field(default_factory=dict)

    def clone(self) -> Self:
        return _clone(self)

    def replace_with(self, other: Self) -> None:
        self.artifacts = _clone(other.artifacts)
        self.materialized_outputs = _clone(other.materialized_outputs)
        self.invocation_cache = _clone(other.invocation_cache)
        self.staged_uploads = _clone(other.staged_uploads)
        self.graph_executions = _clone(other.graph_executions)
        self.graph_execution_node_results = _clone(other.graph_execution_node_results)
        self.active_execution_slots = _clone(other.active_execution_slots)


class UnitOfWorkPort(Protocol):
    @property
    def artifacts(self) -> ArtifactRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


@final
class InMemoryArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    @override
    async def add(self, artifact: ArtifactObject) -> None:
        if artifact.id in self._store.artifacts:
            raise ObjectAlreadyExistsError(f"Artifact already exists: {artifact.id}")
        self._store.artifacts[artifact.id] = artifact

    @override
    async def get(
        self,
        workspace_id: UUID,
        artifact_id: UUID,
    ) -> ArtifactObject | None:
        artifact = self._store.artifacts.get(artifact_id)
        if artifact is None or artifact.workspace_id != workspace_id:
            return None
        return artifact

    @override
    async def get_many(
        self,
        workspace_id: UUID,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]:
        return {
            artifact_id: artifact
            for artifact_id in artifact_ids
            if (artifact := self._store.artifacts.get(artifact_id)) is not None
            and artifact.workspace_id == workspace_id
        }

    @override
    async def remove(self, workspace_id: UUID, artifact: ArtifactObject) -> None:
        if artifact.workspace_id != workspace_id:
            return
        stored = self._store.artifacts.get(artifact.id)
        if stored is not None and stored.workspace_id == workspace_id:
            self._store.artifacts.pop(artifact.id, None)

    @override
    async def list_by_type(
        self,
        workspace_id: UUID,
        key: ArtifactTypeKey,
    ) -> list[ArtifactObject]:
        return [
            artifact
            for artifact in self._store.artifacts.values()
            if artifact.workspace_id == workspace_id
            and artifact.artifact_type == key.id
            and artifact.schema_version == key.schema_version
        ]


@final
class InMemoryMaterializedNodeOutputsRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def upsert(self, value: "MaterializedNodeOutputs") -> None:
        key = (
            value.workspace_id,
            value.graph_id,
            value.graph_revision,
            value.node_id,
        )
        self._store.materialized_outputs[key] = _clone(value)

    async def get(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> "MaterializedNodeOutputs | None":
        value = self._store.materialized_outputs.get(
            (workspace_id, graph_id, graph_revision, node_id)
        )
        return _clone(value) if value is not None else None

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
    ) -> list["MaterializedNodeOutputs"]:
        values = [
            value
            for (saved_workspace_id, saved_graph_id, saved_revision, _), value in (
                self._store.materialized_outputs.items()
            )
            if (
                saved_workspace_id == workspace_id
                and saved_graph_id == graph_id
                and saved_revision == graph_revision
            )
        ]
        return _clone(sorted(values, key=lambda value: value.node_id))


@final
class InMemoryInvocationCacheRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def get(
        self,
        workspace_id: UUID,
        key_sha256: str,
    ) -> "InvocationCacheEntry | None":
        entry = self._store.invocation_cache.get((workspace_id, key_sha256))
        return _clone(entry) if entry is not None else None

    async def put_if_absent(self, entry: "InvocationCacheEntry") -> bool:
        key = (entry.workspace_id, entry.key_sha256)
        if key in self._store.invocation_cache:
            return False
        self._store.invocation_cache[key] = _clone(entry)
        return True

    async def remove_if_current(
        self,
        workspace_id: UUID,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        entry = self._store.invocation_cache.get((workspace_id, key_sha256))
        if entry is None or entry.generation != generation:
            return False
        del self._store.invocation_cache[(workspace_id, key_sha256)]
        return True


@final
class InMemoryGraphExecutionHistoryRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def add(self, execution: "GraphExecution") -> None:
        execution_key = execution.execution_id
        if execution_key in self._store.graph_executions:
            raise ObjectAlreadyExistsError(
                f"Graph execution already exists: {execution.execution_id}"
            )
        self._store.graph_executions[execution_key] = _clone(execution)

    async def update(self, execution: "GraphExecution") -> None:
        current = self._store.graph_executions.get(execution.execution_id)
        if current is None:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        if current.workspace_id != execution.workspace_id:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        if (
            current.graph_id != execution.graph_id
            or current.graph_revision != execution.graph_revision
            or current.scope != execution.scope
            or current.requested_node_ids != execution.requested_node_ids
            or current.created_at != execution.created_at
        ):
            raise ValueError(
                f"Graph execution {execution.execution_id} identity and request "
                "fields are immutable"
            )
        self._store.graph_executions[execution.execution_id] = _clone(execution)

    async def add_node_result(self, result: "GraphExecutionNodeResult") -> None:
        execution = self._store.graph_executions.get(result.execution_id)
        if execution is None:
            raise NotFoundError("Graph execution", str(result.execution_id))
        if execution.workspace_id != result.workspace_id:
            raise NotFoundError("Graph execution", str(result.execution_id))
        if result.node_id not in execution.requested_node_ids:
            raise ValueError(
                f"Graph execution {result.execution_id} did not request node "
                f"{result.node_id!r}"
            )
        key = (result.workspace_id, result.execution_id, result.node_id)
        if key in self._store.graph_execution_node_results:
            raise ObjectAlreadyExistsError(
                "Graph execution node result already exists: "
                f"{result.execution_id}/{result.node_id}"
            )
        if any(
            stored.workspace_id == result.workspace_id
            and stored.execution_id == result.execution_id
            and stored.position == result.position
            for stored in self._store.graph_execution_node_results.values()
        ):
            raise ObjectAlreadyExistsError(
                "Graph execution node result position already exists: "
                f"{result.execution_id}/{result.position}"
            )
        self._store.graph_execution_node_results[key] = _clone(result)

    async def get(
        self,
        workspace_id: UUID,
        execution_id: UUID,
    ) -> "GraphExecutionDetail | None":
        from grafy_core.domain.execution_history import GraphExecutionDetail

        execution = self._store.graph_executions.get(execution_id)
        if execution is None or execution.workspace_id != workspace_id:
            return None
        node_results = sorted(
            (
                result
                for result in self._store.graph_execution_node_results.values()
                if (
                    result.workspace_id == workspace_id
                    and result.execution_id == execution_id
                )
            ),
            key=lambda result: (result.position, result.node_id),
        )
        return GraphExecutionDetail(
            execution=_clone(execution),
            node_results=tuple(_clone(node_results)),
        )

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        limit: int,
        cursor: "GraphExecutionCursor | None" = None,
        graph_revision: int | None = None,
        status: "GraphExecutionStatus | None" = None,
        node_id: str | None = None,
    ) -> "GraphExecutionPage":
        from grafy_core.domain.execution_history import (
            GraphExecutionCursor,
            GraphExecutionListItem,
            GraphExecutionPage,
        )

        if limit < 1:
            raise ValueError("Graph execution page limit must be at least 1")
        if graph_revision is not None and graph_revision < 1:
            raise ValueError("Graph execution revision filter must be at least 1")
        normalized_node_id = None
        if node_id is not None:
            normalized_node_id = node_id.strip()
            if normalized_node_id == "":
                raise ValueError("Graph execution node filter must not be blank")

        values = [
            execution
            for execution in self._store.graph_executions.values()
            if execution.workspace_id == workspace_id
            and execution.graph_id == graph_id
            and (graph_revision is None or execution.graph_revision == graph_revision)
            and (status is None or execution.status == status)
            and (
                normalized_node_id is None
                or normalized_node_id in execution.requested_node_ids
            )
        ]
        if cursor is not None:
            values = [
                execution
                for execution in values
                if (execution.created_at, execution.execution_id.int)
                < (cursor.created_at, cursor.execution_id.int)
            ]
        values.sort(
            key=lambda execution: (
                execution.created_at,
                execution.execution_id.int,
            ),
            reverse=True,
        )
        has_more = len(values) > limit
        page_values = values[:limit]
        items: list[GraphExecutionListItem] = []
        for execution in page_values:
            results = [
                result
                for result in self._store.graph_execution_node_results.values()
                if (
                    result.workspace_id == workspace_id
                    and result.execution_id == execution.execution_id
                )
            ]
            items.append(
                GraphExecutionListItem(
                    execution=_clone(execution),
                    node_count=len(results),
                    artifact_count=sum(result.artifact_count for result in results),
                )
            )
        next_cursor = None
        if has_more and page_values:
            last = page_values[-1]
            next_cursor = GraphExecutionCursor(
                created_at=last.created_at,
                execution_id=last.execution_id,
            )
        return GraphExecutionPage(items=tuple(items), next_cursor=next_cursor)

    async def interrupt_all_active(
        self,
        *,
        finished_at: datetime,
        error: str,
    ) -> int:
        if finished_at.tzinfo is None:
            raise ValueError(
                "Graph execution interruption timestamp must be timezone-aware"
            )
        interrupted = 0
        for execution_key, execution in list(self._store.graph_executions.items()):
            if execution.status not in {"queued", "running", "cancelling"}:
                continue
            self._store.graph_executions[execution_key] = replace(
                execution,
                status="failed",
                finished_at=finished_at,
                error=error,
            )
            interrupted += 1
        return interrupted


@final
class InMemoryActiveExecutionSlotRepository:
    """Minimal slot store for in-memory execution history tests."""

    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def get_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> "GraphActiveExecutionSlot | None":
        slot = self._store.active_execution_slots.get((workspace_id, graph_id))
        if slot is None:
            return None
        return _clone(slot)

    async def acquire_active_execution_slot(
        self,
        slot: "GraphActiveExecutionSlot",
    ) -> bool:
        key = (slot.workspace_id, slot.graph_id)
        if key in self._store.active_execution_slots:
            return False
        self._store.active_execution_slots[key] = _clone(slot)
        return True

    async def clear_active_execution_slot(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        execution_id: UUID | None = None,
    ) -> None:
        key = (workspace_id, graph_id)
        existing = self._store.active_execution_slots.get(key)
        if existing is None:
            return
        if execution_id is not None and existing.execution_id != execution_id:
            return
        self._store.active_execution_slots.pop(key, None)

    async def clear_all_active_execution_slots(self) -> int:
        count = len(self._store.active_execution_slots)
        self._store.active_execution_slots.clear()
        return count


@final
class InMemoryStagedUploadRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def add(self, upload: "StagedUpload") -> None:
        key = (upload.workspace_id, upload.upload_key)
        if key in self._store.staged_uploads:
            raise ObjectAlreadyExistsError(
                f"Staged upload already exists: {upload.workspace_id}/{upload.upload_key}"
            )
        self._store.staged_uploads[key] = _clone(upload)

    async def get(
        self,
        workspace_id: UUID,
        upload_key: str,
    ) -> "StagedUpload | None":
        upload = self._store.staged_uploads.get((workspace_id, upload_key))
        if upload is None:
            return None
        return _clone(upload)

    async def list_for_workspace(self, workspace_id: UUID) -> list["StagedUpload"]:
        uploads = [
            _clone(upload)
            for (stored_workspace_id, _), upload in self._store.staged_uploads.items()
            if stored_workspace_id == workspace_id
        ]
        uploads.sort(key=lambda upload: (upload.created_at, upload.upload_key))
        return uploads

    async def remove(self, workspace_id: UUID, upload_key: str) -> None:
        self._store.staged_uploads.pop((workspace_id, upload_key), None)


@dataclass(frozen=True, slots=True)
class _InMemoryUnitOfWorkState:
    working_store: InMemoryDataStore
    artifacts: ArtifactRepositoryPort
    materialized_outputs: "MaterializedNodeOutputsRepositoryPort"
    invocation_cache: "InvocationCacheRepositoryPort"
    staged_uploads: "StagedUploadRepositoryPort"
    execution_history: "GraphExecutionHistoryRepositoryPort"
    collaboration: InMemoryActiveExecutionSlotRepository


class InMemoryUnitOfWork(UnitOfWorkPort):
    def __init__(self, store: InMemoryDataStore | None = None) -> None:
        self._store = store or InMemoryDataStore()
        self._transaction_lock = Lock()
        self._state: ContextVar[_InMemoryUnitOfWorkState | None] = ContextVar(
            "grafy_in_memory_unit_of_work_state",
            default=None,
        )
        self.commit_count = 0
        self.rollback_count = 0

    @property
    @override
    def artifacts(self) -> ArtifactRepositoryPort:
        return self._entered_state().artifacts

    @property
    def materialized_outputs(self) -> "MaterializedNodeOutputsRepositoryPort":
        return self._entered_state().materialized_outputs

    @property
    def invocation_cache(self) -> "InvocationCacheRepositoryPort":
        return self._entered_state().invocation_cache

    @property
    def staged_uploads(self) -> "StagedUploadRepositoryPort":
        return self._entered_state().staged_uploads

    @property
    def execution_history(self) -> "GraphExecutionHistoryRepositoryPort":
        return self._entered_state().execution_history

    @property
    def collaboration(self) -> InMemoryActiveExecutionSlotRepository:
        return self._entered_state().collaboration

    @override
    async def __aenter__(self) -> Self:
        if self._state.get() is not None:
            raise RuntimeError("Unit of work is already entered in this task")

        await self._transaction_lock.acquire()
        try:
            working_store = self._store.clone()
            self._state.set(
                _InMemoryUnitOfWorkState(
                    working_store=working_store,
                    artifacts=InMemoryArtifactRepository(working_store),
                    materialized_outputs=InMemoryMaterializedNodeOutputsRepository(
                        working_store
                    ),
                    invocation_cache=InMemoryInvocationCacheRepository(working_store),
                    staged_uploads=InMemoryStagedUploadRepository(working_store),
                    execution_history=InMemoryGraphExecutionHistoryRepository(
                        working_store
                    ),
                    collaboration=InMemoryActiveExecutionSlotRepository(working_store),
                )
            )
        except BaseException:
            self._transaction_lock.release()
            raise
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        state = self._state.get()
        if state is None:
            return
        try:
            if exc_type is not None:
                await self.rollback()
        finally:
            self._state.set(None)
            self._transaction_lock.release()

    @override
    async def commit(self) -> None:
        state = self._entered_state()
        self._store.replace_with(state.working_store)
        self.commit_count += 1

    @override
    async def rollback(self) -> None:
        state = self._entered_state()
        state.working_store.replace_with(self._store)
        self.rollback_count += 1

    def _entered_state(self) -> _InMemoryUnitOfWorkState:
        state = self._state.get()
        if state is None:
            raise RuntimeError("Unit of work is not entered")
        return state


from grafy_core.domain.errors import (  # noqa: E402  # domain package imports artifacts
    NotFoundError,
    ObjectAlreadyExistsError,
)

from typing import Protocol
from uuid import UUID

from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.ports.invocation_cache import InvocationCacheRepositoryPort
from notarius_core.ports.execution_history import ExecutionHistoryUnitOfWorkPort
from notarius_core.ports.staged_uploads import StagedUploadRepositoryPort


class MaterializedNodeOutputsRepositoryPort(Protocol):
    async def upsert(self, value: MaterializedNodeOutputs) -> None: ...

    async def get(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> MaterializedNodeOutputs | None: ...

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
    ) -> list[MaterializedNodeOutputs]: ...


class WorkbenchUnitOfWorkPort(ExecutionHistoryUnitOfWorkPort, Protocol):
    @property
    def materialized_outputs(self) -> MaterializedNodeOutputsRepositoryPort: ...

    @property
    def invocation_cache(self) -> InvocationCacheRepositoryPort: ...

    @property
    def staged_uploads(self) -> StagedUploadRepositoryPort: ...

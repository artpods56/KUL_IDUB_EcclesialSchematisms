from typing import Protocol
from uuid import UUID

from notarius_core.artifacts import UnitOfWorkPort
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.ports.invocation_cache import InvocationCacheRepositoryPort


class MaterializedNodeOutputsRepositoryPort(Protocol):
    async def upsert(self, value: MaterializedNodeOutputs) -> None: ...

    async def get(
        self,
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> MaterializedNodeOutputs | None: ...

    async def list_for_graph(
        self,
        graph_id: UUID,
        graph_revision: int,
    ) -> list[MaterializedNodeOutputs]: ...


class WorkbenchUnitOfWorkPort(UnitOfWorkPort, Protocol):
    @property
    def materialized_outputs(self) -> MaterializedNodeOutputsRepositoryPort: ...

    @property
    def invocation_cache(self) -> InvocationCacheRepositoryPort: ...

from typing import Protocol
from uuid import UUID

from grafy_core.domain.invocation_cache import InvocationCacheEntry


class InvocationCacheRepositoryPort(Protocol):
    async def get(
        self,
        workspace_id: UUID,
        key_sha256: str,
    ) -> InvocationCacheEntry | None: ...

    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool: ...

    async def remove_if_current(
        self,
        workspace_id: UUID,
        key_sha256: str,
        generation: UUID,
    ) -> bool: ...

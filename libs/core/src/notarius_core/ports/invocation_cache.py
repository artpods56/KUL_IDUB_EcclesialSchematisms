from typing import Protocol
from uuid import UUID

from notarius_core.domain.invocation_cache import InvocationCacheEntry


class InvocationCacheRepositoryPort(Protocol):
    async def get(self, key_sha256: str) -> InvocationCacheEntry | None: ...

    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool: ...

    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool: ...

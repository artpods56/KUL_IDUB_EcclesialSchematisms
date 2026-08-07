from datetime import UTC, datetime
from uuid import UUID

import pytest

from notarius_core.artifacts import InMemoryDataStore, InMemoryInvocationCacheRepository
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.staged_uploads import (
    InMemoryStagedUploadRepository,
    StagedUpload,
)


WORKSPACE_ONE = UUID("00000000-0000-0000-0000-000000000101")
WORKSPACE_TWO = UUID("00000000-0000-0000-0000-000000000102")


@pytest.mark.asyncio
async def test_in_memory_cache_partitions_same_key_by_workspace() -> None:
    repository = InMemoryInvocationCacheRepository(InMemoryDataStore())
    first = InvocationCacheEntry(
        workspace_id=WORKSPACE_ONE, key_sha256="a" * 64, outputs={}
    )
    second = InvocationCacheEntry(
        workspace_id=WORKSPACE_TWO, key_sha256="a" * 64, outputs={}
    )

    assert await repository.put_if_absent(first)
    assert await repository.put_if_absent(second)
    assert await repository.get(WORKSPACE_ONE, first.key_sha256) == first
    assert await repository.get(WORKSPACE_TWO, second.key_sha256) == second


@pytest.mark.asyncio
async def test_in_memory_staged_upload_identity_is_workspace_qualified() -> None:
    repository = InMemoryStagedUploadRepository()
    upload = StagedUpload(
        workspace_id=WORKSPACE_ONE,
        upload_key="legacy-key",
        original_filename="input.csv",
        byte_size=12,
        created_at=datetime(2026, 8, 7, tzinfo=UTC),
    )
    await repository.add(upload)

    assert await repository.get(WORKSPACE_ONE, upload.upload_key) == upload
    assert await repository.get(WORKSPACE_TWO, upload.upload_key) is None


def test_staged_upload_rejects_unbounded_or_negative_metadata() -> None:
    with pytest.raises(ValueError, match="byte size"):
        StagedUpload(
            workspace_id=WORKSPACE_ONE,
            upload_key="key",
            original_filename="input.csv",
            byte_size=-1,
        )
    with pytest.raises(ValueError, match="at most 255"):
        StagedUpload(
            workspace_id=WORKSPACE_ONE,
            upload_key="key",
            original_filename="x" * 256,
            byte_size=0,
        )

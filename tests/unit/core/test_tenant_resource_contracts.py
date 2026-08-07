from uuid import UUID

import pytest

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactTypeKey,
    InMemoryDataStore,
    InMemoryInvocationCacheRepository,
)
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.staged_uploads import StagedUpload


WORKSPACE_ONE = UUID("00000000-0000-0000-0000-000000000101")
WORKSPACE_TWO = UUID("00000000-0000-0000-0000-000000000102")


@pytest.mark.asyncio
async def test_in_memory_cache_partitions_same_key_by_workspace() -> None:
    repository = InMemoryInvocationCacheRepository(InMemoryDataStore())
    first = InvocationCacheEntry(
        workspace_id=WORKSPACE_ONE,
        key_sha256="a" * 64,
        generation=UUID("00000000-0000-0000-0000-000000000111"),
        outputs={
            "workspace": ArtifactRef.from_key(
                artifact_id=UUID("00000000-0000-0000-0000-000000000113"),
                key=ArtifactTypeKey("test.value", 1),
            )
        },
    )
    second = InvocationCacheEntry(
        workspace_id=WORKSPACE_TWO,
        key_sha256="a" * 64,
        generation=UUID("00000000-0000-0000-0000-000000000112"),
        outputs={
            "workspace": ArtifactRef.from_key(
                artifact_id=UUID("00000000-0000-0000-0000-000000000114"),
                key=ArtifactTypeKey("test.value", 1),
            )
        },
    )

    assert await repository.put_if_absent(first)
    assert await repository.put_if_absent(second)
    assert await repository.get(WORKSPACE_ONE, first.key_sha256) == first
    assert await repository.get(WORKSPACE_TWO, second.key_sha256) == second


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
    for invalid_key in ("", ".", "..", "../escape", r"..\escape", "\x00key"):
        with pytest.raises(ValueError, match="non-path opaque"):
            StagedUpload(
                workspace_id=WORKSPACE_ONE,
                upload_key=invalid_key,
                original_filename="input.csv",
                byte_size=0,
            )
    with pytest.raises(ValueError, match="at most 1024"):
        StagedUpload(
            workspace_id=WORKSPACE_ONE,
            upload_key="x" * 1025,
            original_filename="input.csv",
            byte_size=0,
        )

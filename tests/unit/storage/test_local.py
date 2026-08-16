from io import BytesIO
from pathlib import Path

import pytest

from grafy_core.ports.storage import SaveFileCommand
from grafy_storage import LocalFileObjectStore


@pytest.mark.asyncio
async def test_local_object_store_stats_and_loads_bounded_byte_range(
    tmp_path: Path,
) -> None:
    storage = LocalFileObjectStore(tmp_path)
    await storage.save(
        SaveFileCommand(
            bucket="artifacts",
            path="runs/output.bin",
            stream=BytesIO(b"0123456789"),
            content_type="application/octet-stream",
            metadata={"source": "unit-test"},
        )
    )

    info = await storage.stat("artifacts", "runs/output.bin")
    content = await storage.load_range("artifacts", "runs/output.bin", 2, 7)

    assert info is not None
    assert info.bucket == "artifacts"
    assert info.path == "runs/output.bin"
    assert info.byte_size == 10
    assert info.etag is None
    assert info.version_id is None
    assert content == b"23456"


@pytest.mark.asyncio
async def test_local_object_store_range_validation_and_missing_object_context(
    tmp_path: Path,
) -> None:
    storage = LocalFileObjectStore(tmp_path)

    assert await storage.stat("artifacts", "runs/missing.bin") is None
    assert await storage.load_range("artifacts", "runs/missing.bin", 3, 3) == b""

    with pytest.raises(ValueError, match="nonnegative.*start=-1"):
        await storage.load_range("artifacts", "runs/missing.bin", -1, 2)
    with pytest.raises(ValueError, match="must not precede.*start=3"):
        await storage.load_range("artifacts", "runs/missing.bin", 3, 2)
    with pytest.raises(FileNotFoundError, match="artifacts/runs/missing.bin"):
        await storage.load_range("artifacts", "runs/missing.bin", 0, 1)

from collections.abc import Iterator
from hashlib import sha256
from io import BytesIO
from typing import BinaryIO

import pytest
from obstore.exceptions import AlreadyExistsError

from notarius_core.domain.errors import ObjectAlreadyExistsError
from notarius_core.ports.storage import SaveFileCommand
from notarius_storage.adapters import s3 as s3_module
from notarius_storage.adapters.s3 import S3ObjectStore


class _FakeS3Store:
    instances: list["_FakeS3Store"] = []

    def __init__(
        self,
        bucket: str,
        *,
        config: dict[str, object],
        client_options: dict[str, object] | None,
    ) -> None:
        self.bucket = bucket
        self.config = config
        self.client_options = client_options
        self.objects: dict[str, bytes] = {}
        self.attributes: dict[str, dict[str, str]] = {}
        self.put_modes: list[str] = []
        self.instances.append(self)

    def put(
        self,
        path: str,
        stream: BinaryIO,
        *,
        attributes: dict[str, str],
        mode: str,
    ) -> dict[str, str]:
        self.put_modes.append(mode)
        if mode == "create" and path in self.objects:
            raise AlreadyExistsError
        self.objects[path] = stream.read()
        self.attributes[path] = attributes
        return {"e_tag": '"etag"', "version": "version-1"}

    def copy(self, source: str, destination: str, *, overwrite: bool) -> None:
        if not overwrite and destination in self.objects:
            raise AlreadyExistsError
        self.objects[destination] = self.objects[source]

    def get(self, path: str) -> Iterator[bytes]:
        content = self.objects[path]
        return iter((content[:2], content[2:]))

    def head(self, path: str) -> dict[str, str]:
        if path not in self.objects:
            raise FileNotFoundError(path)
        return {"path": path}

    def delete(self, path: str) -> None:
        self.objects.pop(path, None)

    async def delete_async(self, path: str) -> None:
        self.delete(path)


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> type[_FakeS3Store]:
    _FakeS3Store.instances.clear()
    monkeypatch.setattr(s3_module, "S3Store", _FakeS3Store)
    return _FakeS3Store


def test_s3_object_store_configures_minio_endpoint_and_reuses_bucket_store(
    fake_s3: type[_FakeS3Store],
) -> None:
    storage = S3ObjectStore(
        endpoint_url="http://minio:9000",
        region="eu-central-1",
        access_key_id="access-key",
        secret_access_key="secret-key",
        force_path_style=True,
    )

    assert storage.exists("artifacts", "missing") is False
    assert storage.exists("artifacts", "still-missing") is False

    assert len(fake_s3.instances) == 1
    store = fake_s3.instances[0]
    assert store.bucket == "artifacts"
    assert store.config == {
        "region": "eu-central-1",
        "virtual_hosted_style_request": False,
        "endpoint": "http://minio:9000",
        "access_key_id": "access-key",
        "secret_access_key": "secret-key",
    }
    assert store.client_options == {"allow_http": True}


@pytest.mark.asyncio
async def test_s3_object_store_creates_then_explicitly_overwrites(
    fake_s3: type[_FakeS3Store],
) -> None:
    storage = _storage()
    first = await storage.save(_command(b"first"))

    assert first.etag == '"etag"'
    assert first.version_id == "version-1"
    assert first.byte_size == 5
    assert first.sha256 == sha256(b"first").hexdigest()

    store = fake_s3.instances[0]
    assert store.objects["runs/output.bin"] == b"first"
    assert store.attributes["runs/output.bin"] == {
        "source": "unit-test",
        "Content-Type": "application/octet-stream",
    }
    assert store.put_modes == ["create"]

    with pytest.raises(ObjectAlreadyExistsError, match="artifacts/runs/output.bin"):
        await storage.save(_command(b"rejected"))

    await storage.save(_command(b"replacement", allow_overwrite=True))

    assert store.objects["runs/output.bin"] == b"replacement"
    assert store.put_modes == ["create", "create", "overwrite"]


@pytest.mark.asyncio
async def test_s3_object_store_loads_moves_and_deletes(
    fake_s3: type[_FakeS3Store],
) -> None:
    storage = _storage()
    await storage.save(_command(b"content"))

    loaded = await storage.load("artifacts", "runs/output.bin")
    try:
        assert loaded.read() == b"content"
    finally:
        loaded.close()

    await storage.move("artifacts", "runs/output.bin", "runs/final.bin")
    assert storage.exists("artifacts", "runs/output.bin") is False
    assert storage.exists("artifacts", "runs/final.bin") is True

    await storage.move("artifacts", "runs/output.bin", "runs/final.bin")

    await storage.delete("artifacts", "runs/final.bin")
    assert storage.exists("artifacts", "runs/final.bin") is False
    await storage.delete("artifacts", "runs/final.bin")

    assert len(fake_s3.instances) == 1


@pytest.mark.asyncio
async def test_s3_object_store_move_rejects_missing_source_and_destination(
    fake_s3: type[_FakeS3Store],
) -> None:
    storage = _storage()

    with pytest.raises(FileNotFoundError, match="artifacts/runs/missing.bin"):
        await storage.move("artifacts", "runs/missing.bin", "runs/final.bin")

    assert len(fake_s3.instances) == 1


def _storage() -> S3ObjectStore:
    return S3ObjectStore(
        endpoint_url=None,
        region="eu-central-1",
        access_key_id=None,
        secret_access_key=None,
        force_path_style=False,
    )


def _command(content: bytes, *, allow_overwrite: bool = False) -> SaveFileCommand:
    return SaveFileCommand(
        bucket="artifacts",
        path="runs/output.bin",
        stream=BytesIO(content),
        content_type="application/octet-stream",
        metadata={"source": "unit-test", "original_filename": None},
        allow_overwrite=allow_overwrite,
    )

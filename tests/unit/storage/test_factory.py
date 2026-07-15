from pathlib import Path

from notarius_storage import LocalFileObjectStore, S3ObjectStore
from notarius_storage.factory import create_file_storage


def test_create_file_storage_builds_local_adapter(tmp_path: Path) -> None:
    storage = create_file_storage(backend="local", local_root=tmp_path)

    assert isinstance(storage, LocalFileObjectStore)


def test_create_file_storage_builds_s3_adapter(tmp_path: Path) -> None:
    storage = create_file_storage(
        backend="s3",
        local_root=tmp_path,
        s3_endpoint_url="http://minio:9000",
        s3_region="eu-central-1",
        s3_access_key_id="access-key",
        s3_secret_access_key="secret-key",
        s3_force_path_style=True,
    )

    assert isinstance(storage, S3ObjectStore)

import hashlib

import pytest

from notarius_storage import (
    ArtifactPayloadLocation,
    LocalArtifactPayloadStorage,
    SaveArtifactPayloadCommand,
    StoredArtifactPayload,
    artifact_payload_ref,
    parse_artifact_payload_ref,
)


def test_save_persists_payload_metadata_and_loads_it(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    payload = b"payload bytes"

    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket="artifacts",
            key="nested/payload.bin",
            payload=payload,
        )
    )

    expected = StoredArtifactPayload(
        bucket="artifacts",
        key="nested/payload.bin",
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )
    assert stored == expected
    assert (tmp_path / "artifacts" / "nested" / "payload.bin").read_bytes() == payload
    assert storage.load("artifacts", "nested/payload.bin") == expected


def test_save_refuses_to_overwrite_existing_payload_by_default(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    bucket = "artifacts"
    key = "payload.bin"
    original_payload = b"original"

    storage.save(
        SaveArtifactPayloadCommand(
            bucket=bucket,
            key=key,
            payload=original_payload,
        )
    )

    with pytest.raises(FileExistsError, match="already exists"):
        storage.save(
            SaveArtifactPayloadCommand(
                bucket=bucket,
                key=key,
                payload=b"replacement",
            )
        )

    assert storage.load(bucket, key).payload == original_payload
    assert sorted(path.name for path in (tmp_path / bucket).iterdir()) == [
        "payload.bin"
    ]


def test_save_can_overwrite_existing_payload_when_requested(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)

    storage.save(
        SaveArtifactPayloadCommand(
            bucket="artifacts",
            key="payload.bin",
            payload=b"original",
        )
    )
    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket="artifacts",
            key="payload.bin",
            payload=b"replacement",
            overwrite=True,
        )
    )

    assert stored.payload == b"replacement"
    assert storage.load("artifacts", "payload.bin").payload == b"replacement"


def test_delete_removes_payload(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)

    storage.save(
        SaveArtifactPayloadCommand(
            bucket="artifacts",
            key="payload.bin",
            payload=b"payload",
        )
    )

    assert storage.exists("artifacts", "payload.bin")
    storage.delete("artifacts", "payload.bin")

    assert not storage.exists("artifacts", "payload.bin")
    with pytest.raises(FileNotFoundError, match="not found"):
        storage.load("artifacts", "payload.bin")


def test_artifact_payload_ref_round_trips_location() -> None:
    payload_ref = artifact_payload_ref(
        bucket="source-page-images",
        key="projects/demo/page-1.png",
    )

    assert payload_ref == "artifact://source-page-images/projects/demo/page-1.png"
    assert parse_artifact_payload_ref(payload_ref) == ArtifactPayloadLocation(
        bucket="source-page-images",
        key="projects/demo/page-1.png",
    )
    assert parse_artifact_payload_ref(payload_ref).ref == payload_ref


def test_parse_artifact_payload_ref_rejects_unsupported_refs() -> None:
    with pytest.raises(ValueError, match="Unsupported artifact payload ref"):
        parse_artifact_payload_ref("s3://bucket/key")


@pytest.mark.parametrize(
    ("bucket", "key"),
    [
        ("", "payload.bin"),
        (".", "payload.bin"),
        ("..", "payload.bin"),
        ("bad/bucket", "payload.bin"),
        ("bad\\bucket", "payload.bin"),
        ("C:", "payload.bin"),
        ("artifacts", ""),
        ("artifacts", "/payload.bin"),
        ("artifacts", "C:/payload.bin"),
        ("artifacts", "../payload.bin"),
        ("artifacts", "nested/../payload.bin"),
        ("artifacts", "nested//payload.bin"),
        ("artifacts", "nested\\payload.bin"),
    ],
)
def test_rejects_unsafe_bucket_and_key_values(tmp_path, bucket: str, key: str) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.save(
            SaveArtifactPayloadCommand(
                bucket=bucket,
                key=key,
                payload=b"payload",
            )
        )

    assert not (tmp_path.parent / "payload.bin").exists()

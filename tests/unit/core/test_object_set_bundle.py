from io import BytesIO
from hashlib import sha256
from pathlib import Path
import tarfile

from pydantic import ValidationError
import pytest

from grafy_core.runtime.object_set_bundle import (
    ObjectSetBundleError,
    ObjectSetBundleManifest,
    ObjectSetFile,
    PortableArtifactBundleMetadata,
    PortableArtifactFile,
    PortableMetadataReference,
    load_object_set_bundle,
    object_set_manifest,
    write_object_set_bundle,
)


def _portable() -> tuple[ObjectSetBundleManifest, dict[str, bytes]]:
    root = "workspaces/workspace-1/geo.raster_scan/v1"
    contents = {
        f"{root}/source.tif": b"canonical-cog",
        f"{root}/projections/cog/xyz/0/0/0.png": b"tile-zero",
        f"{root}/projections/cog/xyz/1/0/0.png": b"tile-one",
    }
    files = tuple(
        PortableArtifactFile(
            object_key=object_key,
            byte_size=len(content),
            sha256=sha256(content).hexdigest(),
            content_type="image/tiff" if object_key.endswith(".tif") else "image/png",
        )
        for object_key, content in contents.items()
    )
    metadata = {
        "source_name": "Historical map",
        "original_filename": "map.tif",
        "raster_projection": {
            "bucket": "guest-outputs",
            "prefix": f"{root}/projections/cog/xyz",
        },
        "_portable_bundle": PortableArtifactBundleMetadata(
            files=files,
            references=(
                PortableMetadataReference(
                    path=("raster_projection", "bucket"),
                    kind="bucket",
                ),
                PortableMetadataReference(
                    path=("raster_projection", "prefix"),
                    kind="prefix",
                ),
            ),
        ).as_metadata_value(),
    }
    manifest = object_set_manifest(
        content_type="image/tiff",
        primary_object_key=f"{root}/source.tif",
        logical_byte_size=len(contents[f"{root}/source.tif"]),
        logical_sha256=files[0].sha256,
        metadata=metadata,
        portable=PortableArtifactBundleMetadata.model_validate(
            metadata["_portable_bundle"]
        ),
        object_prefix=root,
    )
    bundle_contents = {
        descriptor.relative_path: contents[source.object_key]
        for source, descriptor in zip(files, manifest.files, strict=True)
    }
    return manifest, bundle_contents


def test_object_set_bundle_round_trips_exact_files_and_typed_references(
    tmp_path: Path,
) -> None:
    manifest, contents = _portable()
    path = tmp_path / "artifact.objects.tar"
    write_object_set_bundle(path, manifest, contents)

    restored_manifest, restored_contents = load_object_set_bundle(
        path,
        max_bytes=1_000_000,
        max_files=10,
    )
    restored_metadata = restored_manifest.restored_metadata(
        bucket="artifacts",
        paths={
            relative_path: f"minted/{relative_path.removeprefix('files/')}"
            for relative_path in restored_contents
        },
    )

    assert restored_manifest == manifest
    assert restored_contents == contents
    assert restored_metadata["original_filename"] == "map.tif"
    assert restored_metadata["raster_projection"] == {
        "bucket": "artifacts",
        "prefix": "minted/projections/cog/xyz",
    }
    assert "_portable_bundle" not in restored_metadata


def test_object_set_contract_rejects_traversal_and_undeclared_archive_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="normalized|relative"):
        ObjectSetFile(
            relative_path="files/../secret",
            byte_size=1,
            sha256="a" * 64,
            content_type="application/octet-stream",
        )

    manifest, contents = _portable()
    path = tmp_path / "malicious.objects.tar"
    write_object_set_bundle(path, manifest, contents)
    with tarfile.open(path, mode="a") as archive:
        info = tarfile.TarInfo("files/undeclared")
        info.size = 1
        archive.addfile(info, BytesIO(b"x"))

    with pytest.raises(ObjectSetBundleError, match="missing or undeclared"):
        load_object_set_bundle(path, max_bytes=1_000_000, max_files=10)

    symlink_path = tmp_path / "symlink.objects.tar"
    write_object_set_bundle(symlink_path, manifest, contents)
    with tarfile.open(symlink_path, mode="a") as archive:
        info = tarfile.TarInfo("files/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../secret"
        archive.addfile(info)
    with pytest.raises(ObjectSetBundleError, match="non-regular file"):
        load_object_set_bundle(symlink_path, max_bytes=1_000_000, max_files=10)


def test_object_set_rejects_metadata_references_outside_declared_inventory() -> None:
    manifest, _contents = _portable()
    root = "workspaces/workspace-1/geo.raster_scan/v1"
    portable = PortableArtifactBundleMetadata(
        files=tuple(
            PortableArtifactFile(
                object_key=f"{root}/{file.relative_path.removeprefix('files/')}",
                byte_size=file.byte_size,
                sha256=file.sha256,
                content_type=file.content_type,
            )
            for file in manifest.files
        ),
        references=(
            PortableMetadataReference(
                path=("projection", "object_key"),
                kind="object",
            ),
        ),
    )

    with pytest.raises(ObjectSetBundleError, match="absent from the inventory"):
        object_set_manifest(
            content_type=manifest.content_type,
            primary_object_key=f"{root}/source.tif",
            logical_byte_size=manifest.logical_byte_size,
            logical_sha256=manifest.logical_sha256,
            metadata={
                "projection": {
                    "object_key": f"{root}/undeclared.pmtiles",
                }
            },
            portable=portable,
            object_prefix=root,
        )


def test_object_set_bundle_enforces_expanded_byte_and_file_limits(
    tmp_path: Path,
) -> None:
    manifest, contents = _portable()
    path = tmp_path / "limited.objects.tar"
    write_object_set_bundle(path, manifest, contents)

    with pytest.raises(ObjectSetBundleError, match="file limit"):
        load_object_set_bundle(path, max_bytes=1_000_000, max_files=2)
    with pytest.raises(ObjectSetBundleError, match="byte limit"):
        load_object_set_bundle(path, max_bytes=path.stat().st_size - 1, max_files=10)

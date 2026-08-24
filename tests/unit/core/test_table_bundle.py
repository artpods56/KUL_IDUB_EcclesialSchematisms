from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from grafy_core.operators.tables import (
    Table,
    TableColumn,
    TableValueType,
    iter_table_chunks,
)
from grafy_core.runtime.table_bundle import (
    TABLE_BUNDLE_MANIFEST_PATH,
    TableBundleError,
    TableBundleManifest,
    file_identity,
    iter_table_bundle_chunks,
    load_table_bundle,
    table_bundle_manifest,
    validate_table_bundle,
    write_table_bundle,
    write_table_bundle_archive,
)


def _table(*, row_count: int = 101) -> Table:
    return Table(
        columns=[
            TableColumn(
                id="name",
                title="Name",
                value_type=TableValueType.TEXT,
            ),
            TableColumn(
                id="count",
                title="Count",
                value_type=TableValueType.INTEGER,
            ),
        ],
        rows=[{"name": f"row-{index}", "count": index} for index in range(row_count)],
    )


def _limits(path: Path) -> dict[str, int]:
    return {
        "max_bytes": path.stat().st_size,
        "max_files": 10,
        "max_rows": 1_000,
        "max_columns": 10,
        "max_chunks": 10,
    }


def _archive_contents(path: Path) -> list[tuple[str, bytes]]:
    with tarfile.open(path, mode="r:") as archive:
        contents: list[tuple[str, bytes]] = []
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            assert stream is not None
            contents.append((member.name, stream.read()))
        return contents


def _write_archive(
    path: Path,
    contents: list[tuple[str, bytes]],
    *,
    link_name: str | None = None,
) -> None:
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, content in contents:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
        if link_name is not None:
            info = tarfile.TarInfo(link_name)
            info.type = tarfile.SYMTYPE
            info.linkname = TABLE_BUNDLE_MANIFEST_PATH
            archive.addfile(info)


def test_table_bundle_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    table = _table()
    first_path = tmp_path / "first.table.tar"
    second_path = tmp_path / "second.table.tar"

    first_manifest = write_table_bundle(first_path, table)
    second_manifest = write_table_bundle(second_path, table)

    logical_content = table.model_dump_json().encode("utf-8")
    assert first_manifest == second_manifest
    assert first_manifest.logical_byte_size == len(logical_content)
    assert first_manifest.logical_sha256 == sha256(logical_content).hexdigest()
    assert first_path.read_bytes() == second_path.read_bytes()
    assert file_identity(first_path).byte_size == first_path.stat().st_size
    assert (
        file_identity(first_path).sha256 == sha256(first_path.read_bytes()).hexdigest()
    )
    assert load_table_bundle(first_path, **_limits(first_path)) == table


def test_table_bundle_uses_the_canonical_table_chunking(tmp_path: Path) -> None:
    table = _table()
    path = tmp_path / "table.tar"
    manifest = write_table_bundle(path, table)

    expected_chunks = list(iter_table_chunks(table))
    archived_chunks = list(iter_table_bundle_chunks(path, manifest))

    assert [descriptor.offset for descriptor in manifest.chunks] == [
        chunk.offset for chunk in expected_chunks
    ]
    assert [content for _descriptor, content in archived_chunks] == [
        chunk.model_dump_json().encode("utf-8") for chunk in expected_chunks
    ]


def test_table_bundle_writer_rejects_missing_chunk_content(tmp_path: Path) -> None:
    manifest = table_bundle_manifest(_table(row_count=1))

    with pytest.raises(TableBundleError, match="chunk count"):
        write_table_bundle_archive(tmp_path / "incomplete.tar", manifest, ())


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [
        ("max_bytes", 1, "byte limit"),
        ("max_files", 2, "file-count limit"),
        ("max_rows", 100, "row limit"),
        ("max_columns", 1, "column limit"),
        ("max_chunks", 1, "chunk limit"),
    ],
)
def test_table_bundle_enforces_every_resource_limit(
    tmp_path: Path,
    limit: str,
    value: int,
    message: str,
) -> None:
    path = tmp_path / "table.tar"
    write_table_bundle(path, _table())
    limits = _limits(path)
    limits[limit] = value

    with pytest.raises(TableBundleError, match=message):
        validate_table_bundle(path, **limits)


def test_table_bundle_rejects_tampered_chunk(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    tampered = tmp_path / "tampered.tar"
    manifest = write_table_bundle(source, _table())
    contents = _archive_contents(source)
    chunk_path = manifest.chunks[0].relative_path
    rewritten = [
        (name, content + b" ") if name == chunk_path else (name, content)
        for name, content in contents
    ]
    _write_archive(tampered, rewritten)

    with pytest.raises(TableBundleError, match="failed validation|oversized"):
        validate_table_bundle(tampered, **_limits(tampered))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "symlink"])
def test_table_bundle_rejects_undeclared_or_non_regular_files(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source.tar"
    mutated = tmp_path / f"{mutation}.tar"
    manifest = write_table_bundle(source, _table())
    contents = _archive_contents(source)
    first_chunk_path = manifest.chunks[0].relative_path

    if mutation == "missing":
        contents = [
            (name, content) for name, content in contents if name != first_chunk_path
        ]
        _write_archive(mutated, contents)
    elif mutation == "extra":
        contents.append(("chunks/undeclared.json", b"{}"))
        _write_archive(mutated, contents)
    elif mutation == "duplicate":
        contents.append(contents[-1])
        _write_archive(mutated, contents)
    else:
        _write_archive(mutated, contents, link_name="chunks/link.json")

    with pytest.raises(TableBundleError):
        validate_table_bundle(mutated, **_limits(mutated))


def test_table_bundle_rejects_reordered_chunk_descriptors(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    reordered = tmp_path / "reordered.tar"
    write_table_bundle(source, _table())
    contents = _archive_contents(source)
    manifest_content = next(
        content for name, content in contents if name == TABLE_BUNDLE_MANIFEST_PATH
    )
    manifest = TableBundleManifest.model_validate_json(manifest_content)
    payload = manifest.model_dump(mode="json")
    payload["chunks"] = list(reversed(payload["chunks"]))
    rewritten_manifest = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    rewritten = [
        (name, rewritten_manifest)
        if name == TABLE_BUNDLE_MANIFEST_PATH
        else (name, content)
        for name, content in contents
    ]
    _write_archive(reordered, rewritten)

    with pytest.raises(TableBundleError, match="schema is invalid"):
        validate_table_bundle(reordered, **_limits(reordered))


def test_table_bundle_rejects_unsafe_manifest_chunk_path(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    unsafe = tmp_path / "unsafe.tar"
    write_table_bundle(source, _table(row_count=1))
    contents = _archive_contents(source)
    manifest_content = next(
        content for name, content in contents if name == TABLE_BUNDLE_MANIFEST_PATH
    )
    manifest = TableBundleManifest.model_validate_json(manifest_content)
    payload = manifest.model_dump(mode="json")
    payload["chunks"][0]["relative_path"] = "../chunk.json"
    rewritten_manifest = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    rewritten = [
        (name, rewritten_manifest)
        if name == TABLE_BUNDLE_MANIFEST_PATH
        else (name, content)
        for name, content in contents
    ]
    _write_archive(unsafe, rewritten)

    with pytest.raises(TableBundleError, match="schema is invalid"):
        validate_table_bundle(unsafe, **_limits(unsafe))

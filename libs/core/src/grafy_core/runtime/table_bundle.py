"""Portable, deterministic Table artifact bundles for Plugin invocation."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from grafy_core.operators.tables import (
    Table,
    TableChunk,
    TableColumn,
    TableValue,
    iter_table_chunks,
)


TABLE_BUNDLE_FORMAT = "grafy.plugin.table-bundle.v1"
TABLE_BUNDLE_MANIFEST_PATH = "manifest.json"
TABLE_BUNDLE_MANIFEST_MAX_BYTES = 2 * 1_024 * 1_024

_COLUMNS_ADAPTER = TypeAdapter(list[TableColumn])
_ROW_ADAPTER = TypeAdapter(dict[str, TableValue])


class TableBundleError(ValueError):
    """A Table bundle is malformed, unsafe, or inconsistent."""


class _TableBundleValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class TableBundleChunkDescriptor(_TableBundleValue):
    offset: int = Field(ge=0, strict=True)
    row_count: int = Field(ge=1, strict=True)
    relative_path: str = Field(min_length=1, max_length=1_024)
    byte_size: int = Field(ge=1, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or re.fullmatch(r"chunks/[0-9]{6}\.json", self.relative_path) is None
        ):
            raise ValueError("Table bundle chunk path is unsafe or non-canonical")
        return self


class TableBundleManifest(_TableBundleValue):
    format: str = Field(
        default=TABLE_BUNDLE_FORMAT, pattern=r"^grafy\.plugin\.table-bundle\.v1$"
    )
    columns: tuple[TableColumn, ...] = Field(max_length=10_000)
    row_count: int = Field(ge=0, strict=True)
    logical_byte_size: int = Field(ge=0, strict=True)
    logical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[TableBundleChunkDescriptor, ...]

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        expected_offset = 0
        for index, chunk in enumerate(self.chunks):
            expected_path = f"chunks/{index:06d}.json"
            if chunk.relative_path != expected_path:
                raise ValueError(
                    f"Table bundle chunk {index} must use {expected_path!r}"
                )
            if chunk.offset != expected_offset:
                raise ValueError(
                    f"Table bundle chunk at offset {chunk.offset} must start at "
                    f"{expected_offset}"
                )
            expected_offset += chunk.row_count
        if expected_offset != self.row_count:
            raise ValueError(
                f"Table bundle chunks cover {expected_offset} rows, expected "
                f"{self.row_count}"
            )
        return self


@dataclass(frozen=True, slots=True)
class TableBundleIdentity:
    byte_size: int
    sha256: str


class _LogicalTableDigest:
    def __init__(self, columns: tuple[TableColumn, ...] | list[TableColumn]) -> None:
        self._digest = sha256()
        self.byte_size = 0
        self.row_count = 0
        self._write(b'{"columns":')
        self._write(_COLUMNS_ADAPTER.dump_json(list(columns)))
        self._write(b',"rows":[')

    def add_rows(self, rows: list[dict[str, TableValue]]) -> None:
        for row in rows:
            if self.row_count:
                self._write(b",")
            self._write(_ROW_ADAPTER.dump_json(row))
            self.row_count += 1

    def finish(self) -> TableBundleIdentity:
        self._write(b"]}")
        return TableBundleIdentity(
            byte_size=self.byte_size,
            sha256=self._digest.hexdigest(),
        )

    def _write(self, content: bytes) -> None:
        self._digest.update(content)
        self.byte_size += len(content)


def table_bundle_manifest(table: Table) -> TableBundleManifest:
    descriptors: list[TableBundleChunkDescriptor] = []
    logical = _LogicalTableDigest(table.columns)
    for index, chunk in enumerate(iter_table_chunks(table)):
        content = chunk.model_dump_json().encode("utf-8")
        logical.add_rows(chunk.rows)
        descriptors.append(
            TableBundleChunkDescriptor(
                offset=chunk.offset,
                row_count=len(chunk.rows),
                relative_path=f"chunks/{index:06d}.json",
                byte_size=len(content),
                sha256=sha256(content).hexdigest(),
            )
        )
    identity = logical.finish()
    return TableBundleManifest(
        columns=tuple(table.columns),
        row_count=len(table.rows),
        logical_byte_size=identity.byte_size,
        logical_sha256=identity.sha256,
        chunks=tuple(descriptors),
    )


def write_table_bundle(path: Path, table: Table) -> TableBundleManifest:
    manifest = table_bundle_manifest(table)
    write_table_bundle_archive(
        path,
        manifest,
        (
            (descriptor, chunk.model_dump_json().encode("utf-8"))
            for descriptor, chunk in zip(
                manifest.chunks,
                iter_table_chunks(table),
                strict=True,
            )
        ),
    )
    return manifest


def write_table_bundle_archive(
    path: Path,
    manifest: TableBundleManifest,
    chunks: Iterable[tuple[TableBundleChunkDescriptor, bytes]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_content = manifest.model_dump_json().encode("utf-8")
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(
            _tar_info(TABLE_BUNDLE_MANIFEST_PATH, len(manifest_content)),
            BytesIO(manifest_content),
        )
        chunk_count = 0
        try:
            for expected, (descriptor, content) in zip(
                manifest.chunks,
                chunks,
                strict=True,
            ):
                if descriptor != expected:
                    raise TableBundleError("Table bundle chunks are out of order")
                _validate_chunk_content(descriptor, content, manifest.columns)
                archive.addfile(
                    _tar_info(descriptor.relative_path, len(content)),
                    BytesIO(content),
                )
                chunk_count += 1
        except TableBundleError:
            raise
        except ValueError as exc:
            raise TableBundleError(
                "Table bundle chunk count does not match its manifest"
            ) from exc
        if chunk_count != len(manifest.chunks):
            raise TableBundleError("Table bundle is missing chunk content")


def validate_table_bundle(
    path: Path,
    *,
    max_bytes: int,
    max_files: int,
    max_rows: int,
    max_columns: int,
    max_chunks: int,
) -> TableBundleManifest:
    manifest, _rows = _inspect_table_bundle(
        path,
        max_bytes=max_bytes,
        max_files=max_files,
        max_rows=max_rows,
        max_columns=max_columns,
        max_chunks=max_chunks,
        collect_rows=False,
    )
    return manifest


def load_table_bundle(
    path: Path,
    *,
    max_bytes: int,
    max_files: int,
    max_rows: int,
    max_columns: int,
    max_chunks: int,
) -> Table:
    _manifest, table = load_table_bundle_with_manifest(
        path,
        max_bytes=max_bytes,
        max_files=max_files,
        max_rows=max_rows,
        max_columns=max_columns,
        max_chunks=max_chunks,
    )
    return table


def load_table_bundle_with_manifest(
    path: Path,
    *,
    max_bytes: int,
    max_files: int,
    max_rows: int,
    max_columns: int,
    max_chunks: int,
) -> tuple[TableBundleManifest, Table]:
    manifest, rows = _inspect_table_bundle(
        path,
        max_bytes=max_bytes,
        max_files=max_files,
        max_rows=max_rows,
        max_columns=max_columns,
        max_chunks=max_chunks,
        collect_rows=True,
    )
    return manifest, Table(columns=list(manifest.columns), rows=rows)


def iter_table_bundle_chunks(
    path: Path,
    manifest: TableBundleManifest,
) -> Iterator[tuple[TableBundleChunkDescriptor, bytes]]:
    with tarfile.open(path, mode="r:") as archive:
        for descriptor in manifest.chunks:
            member = archive.getmember(descriptor.relative_path)
            content = _read_member(archive, member, descriptor.byte_size)
            _validate_chunk_content(descriptor, content, manifest.columns)
            yield descriptor, content


def file_identity(path: Path) -> TableBundleIdentity:
    digest = sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(1 * 1_024 * 1_024):
            digest.update(chunk)
            byte_size += len(chunk)
    return TableBundleIdentity(byte_size=byte_size, sha256=digest.hexdigest())


def _inspect_table_bundle(
    path: Path,
    *,
    max_bytes: int,
    max_files: int,
    max_rows: int,
    max_columns: int,
    max_chunks: int,
    collect_rows: bool,
) -> tuple[TableBundleManifest, list[dict[str, TableValue]]]:
    if path.is_symlink() or not path.is_file():
        raise TableBundleError("Table bundle must be a regular file")
    if path.stat().st_size > max_bytes:
        raise TableBundleError("Table bundle exceeds its byte limit")
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise TableBundleError("Table bundle contains duplicate paths")
            if len(members) > max_files:
                raise TableBundleError("Table bundle exceeds its file-count limit")
            if any(not member.isfile() for member in members):
                raise TableBundleError("Table bundle contains a non-regular file")
            if TABLE_BUNDLE_MANIFEST_PATH not in names:
                raise TableBundleError("Table bundle has no manifest")
            manifest_member = archive.getmember(TABLE_BUNDLE_MANIFEST_PATH)
            manifest_content = _read_member(
                archive,
                manifest_member,
                TABLE_BUNDLE_MANIFEST_MAX_BYTES,
            )
            manifest = TableBundleManifest.model_validate_json(manifest_content)
            if len(manifest.columns) > max_columns:
                raise TableBundleError("Table bundle exceeds its column limit")
            if manifest.row_count > max_rows:
                raise TableBundleError("Table bundle exceeds its row limit")
            if len(manifest.chunks) > max_chunks:
                raise TableBundleError("Table bundle exceeds its chunk limit")
            expected_names = {
                TABLE_BUNDLE_MANIFEST_PATH,
                *(chunk.relative_path for chunk in manifest.chunks),
            }
            if set(names) != expected_names:
                raise TableBundleError(
                    "Table bundle contains missing or undeclared files"
                )
            logical = _LogicalTableDigest(manifest.columns)
            rows: list[dict[str, TableValue]] = []
            for descriptor in manifest.chunks:
                member = archive.getmember(descriptor.relative_path)
                content = _read_member(archive, member, descriptor.byte_size)
                chunk = _validate_chunk_content(
                    descriptor,
                    content,
                    manifest.columns,
                )
                logical.add_rows(chunk.rows)
                if collect_rows:
                    rows.extend(chunk.rows)
            identity = logical.finish()
            if logical.row_count != manifest.row_count:
                raise TableBundleError("Table bundle row count is inconsistent")
            if (
                identity.byte_size != manifest.logical_byte_size
                or identity.sha256 != manifest.logical_sha256
            ):
                raise TableBundleError("Table bundle logical content is inconsistent")
            return manifest, rows
    except ValidationError as exc:
        raise TableBundleError("Table bundle schema is invalid") from exc
    except (tarfile.TarError, KeyError) as exc:
        raise TableBundleError("Table bundle is not a valid archive") from exc


def _validate_chunk_content(
    descriptor: TableBundleChunkDescriptor,
    content: bytes,
    columns: tuple[TableColumn, ...],
) -> TableChunk:
    if (
        len(content) != descriptor.byte_size
        or sha256(content).hexdigest() != descriptor.sha256
    ):
        raise TableBundleError(
            f"Table bundle chunk {descriptor.relative_path!r} failed validation"
        )
    chunk = TableChunk.model_validate_json(content)
    if chunk.offset != descriptor.offset or len(chunk.rows) != descriptor.row_count:
        raise TableBundleError(
            f"Table bundle chunk {descriptor.relative_path!r} has wrong coverage"
        )
    Table(columns=list(columns), rows=chunk.rows)
    if chunk.model_dump_json().encode("utf-8") != content:
        raise TableBundleError(
            f"Table bundle chunk {descriptor.relative_path!r} is not canonical"
        )
    return chunk


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    max_bytes: int,
) -> bytes:
    if member.size > max_bytes:
        raise TableBundleError(f"Table bundle member {member.name!r} is oversized")
    stream = archive.extractfile(member)
    if stream is None:
        raise TableBundleError(f"Table bundle member {member.name!r} is unreadable")
    content = stream.read(max_bytes + 1)
    if len(content) != member.size:
        raise TableBundleError(f"Table bundle member {member.name!r} is truncated")
    return content


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o400
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


__all__ = [
    "TABLE_BUNDLE_FORMAT",
    "TABLE_BUNDLE_MANIFEST_PATH",
    "TableBundleChunkDescriptor",
    "TableBundleError",
    "TableBundleIdentity",
    "TableBundleManifest",
    "file_identity",
    "iter_table_bundle_chunks",
    "load_table_bundle",
    "load_table_bundle_with_manifest",
    "table_bundle_manifest",
    "validate_table_bundle",
    "write_table_bundle",
    "write_table_bundle_archive",
]

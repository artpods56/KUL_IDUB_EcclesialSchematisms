"""Canonical portable bundles for artifacts backed by an exact file set."""

from collections.abc import Iterator, Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import tarfile
from typing import ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from grafy_core.artifacts import JsonObject


OBJECT_SET_BUNDLE_FORMAT = "grafy.plugin.object-set.v1"
OBJECT_SET_BUNDLE_MANIFEST_PATH = "manifest.json"
OBJECT_SET_BUNDLE_MANIFEST_MAX_BYTES = 2 * 1_024 * 1_024
PORTABLE_BUNDLE_METADATA_KEY = "_portable_bundle"


class ObjectSetBundleError(ValueError):
    """An object-set bundle is malformed, unsafe, or inconsistent."""


class _ObjectSetValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _normalized_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("Object-set paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        value == ""
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("Object-set paths must be normalized and relative")
    return path


class PortableArtifactFile(_ObjectSetValue):
    object_key: str = Field(min_length=1, max_length=2_048)
    byte_size: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_object_key(self) -> Self:
        _normalized_path(self.object_key)
        return self


class PortableMetadataReference(_ObjectSetValue):
    path: tuple[str, ...] = Field(min_length=1, max_length=32)
    kind: Literal["bucket", "object", "prefix"]

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if any(part == "" or len(part) > 255 for part in self.path):
            raise ValueError("Portable metadata reference paths must be non-empty")
        return self


class PortableArtifactBundleMetadata(_ObjectSetValue):
    files: tuple[PortableArtifactFile, ...] = Field(min_length=1)
    references: tuple[PortableMetadataReference, ...] = ()

    @model_validator(mode="after")
    def validate_unique_identity(self) -> Self:
        object_keys = [file.object_key for file in self.files]
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("Portable artifact files must have unique object keys")
        paths = [reference.path for reference in self.references]
        if len(paths) != len(set(paths)):
            raise ValueError("Portable metadata references must have unique paths")
        return self

    def as_metadata_value(self) -> JsonObject:
        return cast(JsonObject, self.model_dump(mode="json"))


class ObjectSetFile(_ObjectSetValue):
    relative_path: str = Field(min_length=1, max_length=2_048)
    byte_size: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        path = _normalized_path(self.relative_path)
        if not path.is_relative_to(PurePosixPath("files")):
            raise ValueError("Object-set files must be beneath files/")
        return self


class ObjectSetMetadataReference(_ObjectSetValue):
    path: tuple[str, ...] = Field(min_length=1, max_length=32)
    kind: Literal["bucket", "object", "prefix"]
    relative_path: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if any(part == "" or len(part) > 255 for part in self.path):
            raise ValueError("Object-set metadata reference paths must be non-empty")
        if self.kind == "bucket":
            if self.relative_path is not None:
                raise ValueError("Bucket references cannot target a file path")
        else:
            if self.relative_path is None:
                raise ValueError("Object and prefix references require a path")
            path = _normalized_path(self.relative_path)
            if not path.is_relative_to(PurePosixPath("files")):
                raise ValueError("Object-set references must target files/")
        return self


class ObjectSetBundleManifest(_ObjectSetValue):
    format: Literal["grafy.plugin.object-set.v1"] = OBJECT_SET_BUNDLE_FORMAT
    content_type: str = Field(min_length=1, max_length=255)
    primary_path: str = Field(min_length=1, max_length=2_048)
    logical_byte_size: int = Field(ge=0, strict=True)
    logical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: JsonObject
    files: tuple[ObjectSetFile, ...] = Field(min_length=1)
    references: tuple[ObjectSetMetadataReference, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        paths = [file.relative_path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Object-set files must have unique paths")
        if self.primary_path not in set(paths):
            raise ValueError("Object-set primary path must name a declared file")
        reference_paths = [reference.path for reference in self.references]
        if len(reference_paths) != len(set(reference_paths)):
            raise ValueError("Object-set metadata references must have unique paths")
        file_paths = set(paths)
        for reference in self.references:
            if reference.kind == "object" and reference.relative_path not in file_paths:
                raise ValueError("Object-set object reference targets an unknown file")
            if reference.kind == "prefix" and not any(
                path.startswith(f"{reference.relative_path}/") for path in file_paths
            ):
                raise ValueError("Object-set prefix reference targets no files")
            _metadata_value(self.metadata, reference.path)
        return self

    def restored_metadata(
        self,
        *,
        bucket: str,
        paths: Mapping[str, str],
    ) -> JsonObject:
        restored = cast(JsonObject, _deep_copy_json(self.metadata))
        for reference in self.references:
            if reference.kind == "bucket":
                value = bucket
            elif reference.kind == "object":
                assert reference.relative_path is not None
                value = paths[reference.relative_path]
            else:
                assert reference.relative_path is not None
                prefix = f"{reference.relative_path}/"
                matching = [
                    (relative_path, destination)
                    for relative_path, destination in paths.items()
                    if relative_path.startswith(prefix)
                ]
                if not matching:
                    raise ObjectSetBundleError(
                        "Object-set prefix reference has no destination files"
                    )
                destination_prefixes = {
                    destination.removesuffix(relative_path.removeprefix(prefix)).rstrip(
                        "/"
                    )
                    for relative_path, destination in matching
                }
                if len(destination_prefixes) != 1:
                    raise ObjectSetBundleError(
                        "Object-set prefix files do not share one destination"
                    )
                value = destination_prefixes.pop()
            _set_metadata_value(restored, reference.path, value)
        return restored


def object_set_manifest(
    *,
    content_type: str,
    primary_object_key: str,
    logical_byte_size: int,
    logical_sha256: str,
    metadata: JsonObject,
    portable: PortableArtifactBundleMetadata,
    object_prefix: str,
) -> ObjectSetBundleManifest:
    normalized_prefix = _normalized_path(object_prefix).as_posix().rstrip("/") + "/"
    files: list[ObjectSetFile] = []
    paths_by_object: dict[str, str] = {}
    for source in portable.files:
        if not source.object_key.startswith(normalized_prefix):
            raise ObjectSetBundleError(
                f"Portable object {source.object_key!r} is outside its artifact root"
            )
        suffix = source.object_key.removeprefix(normalized_prefix)
        if suffix == "":
            raise ObjectSetBundleError("Portable object cannot equal its root")
        relative_path = f"files/{suffix}"
        files.append(
            ObjectSetFile(
                relative_path=relative_path,
                byte_size=source.byte_size,
                sha256=source.sha256,
                content_type=source.content_type,
            )
        )
        paths_by_object[source.object_key] = relative_path
    primary_path = paths_by_object.get(primary_object_key)
    if primary_path is None:
        raise ObjectSetBundleError("Portable inventory omits the primary object")
    portable_metadata = cast(JsonObject, _deep_copy_json(metadata))
    portable_metadata.pop(PORTABLE_BUNDLE_METADATA_KEY, None)
    references: list[ObjectSetMetadataReference] = []
    for declaration in portable.references:
        source_value = _metadata_value(metadata, declaration.path)
        relative_path: str | None = None
        if declaration.kind == "bucket":
            if not isinstance(source_value, str):
                raise ObjectSetBundleError("Portable bucket reference is not text")
        elif declaration.kind == "object":
            if not isinstance(source_value, str):
                raise ObjectSetBundleError("Portable object reference is not text")
            relative_path = paths_by_object.get(source_value)
            if relative_path is None:
                raise ObjectSetBundleError(
                    "Portable object reference is absent from the inventory"
                )
        else:
            if not isinstance(source_value, str):
                raise ObjectSetBundleError("Portable prefix reference is not text")
            prefix = source_value.rstrip("/") + "/"
            matching = [
                path for key, path in paths_by_object.items() if key.startswith(prefix)
            ]
            if not matching:
                raise ObjectSetBundleError(
                    "Portable prefix reference has no inventoried objects"
                )
            suffixes = [
                key.removeprefix(prefix)
                for key in paths_by_object
                if key.startswith(prefix)
            ]
            common_relative = (
                matching[suffixes.index(min(suffixes))]
                .removesuffix(min(suffixes))
                .rstrip("/")
            )
            relative_path = common_relative
        token = f"grafy-object-set:{declaration.kind}"
        if relative_path is not None:
            token = f"{token}:{relative_path}"
        _set_metadata_value(portable_metadata, declaration.path, token)
        references.append(
            ObjectSetMetadataReference(
                path=declaration.path,
                kind=declaration.kind,
                relative_path=relative_path,
            )
        )
    return ObjectSetBundleManifest(
        content_type=content_type,
        primary_path=primary_path,
        logical_byte_size=logical_byte_size,
        logical_sha256=logical_sha256,
        metadata=portable_metadata,
        files=tuple(files),
        references=tuple(references),
    )


def write_object_set_bundle(
    path: Path,
    manifest: ObjectSetBundleManifest,
    contents: Mapping[str, bytes],
) -> None:
    expected = {file.relative_path for file in manifest.files}
    if set(contents) != expected:
        raise ObjectSetBundleError("Object-set content does not match its manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_content = manifest.model_dump_json().encode("utf-8")
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(
            _tar_info(OBJECT_SET_BUNDLE_MANIFEST_PATH, len(manifest_content)),
            BytesIO(manifest_content),
        )
        for descriptor in manifest.files:
            content = contents[descriptor.relative_path]
            _validate_file(descriptor, content)
            archive.addfile(
                _tar_info(descriptor.relative_path, len(content)),
                BytesIO(content),
            )


def load_object_set_bundle(
    path: Path,
    *,
    max_bytes: int,
    max_files: int,
) -> tuple[ObjectSetBundleManifest, dict[str, bytes]]:
    if path.is_symlink() or not path.is_file():
        raise ObjectSetBundleError("Object-set bundle must be a regular file")
    if path.stat().st_size > max_bytes:
        raise ObjectSetBundleError("Object-set bundle exceeds its byte limit")
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ObjectSetBundleError("Object-set bundle has duplicate paths")
            if len(members) > max_files:
                raise ObjectSetBundleError("Object-set bundle exceeds its file limit")
            if any(not member.isfile() for member in members):
                raise ObjectSetBundleError("Object-set bundle has a non-regular file")
            manifest_member = archive.getmember(OBJECT_SET_BUNDLE_MANIFEST_PATH)
            manifest_content = _read_member(
                archive,
                manifest_member,
                OBJECT_SET_BUNDLE_MANIFEST_MAX_BYTES,
            )
            manifest = ObjectSetBundleManifest.model_validate_json(manifest_content)
            expected = {
                OBJECT_SET_BUNDLE_MANIFEST_PATH,
                *(file.relative_path for file in manifest.files),
            }
            if set(names) != expected:
                raise ObjectSetBundleError(
                    "Object-set bundle contains missing or undeclared files"
                )
            total = len(manifest_content)
            contents: dict[str, bytes] = {}
            for descriptor in manifest.files:
                content = _read_member(
                    archive,
                    archive.getmember(descriptor.relative_path),
                    descriptor.byte_size,
                )
                _validate_file(descriptor, content)
                total += len(content)
                if total > max_bytes:
                    raise ObjectSetBundleError(
                        "Object-set expanded content exceeds its byte limit"
                    )
                contents[descriptor.relative_path] = content
            return manifest, contents
    except ValidationError as exc:
        raise ObjectSetBundleError("Object-set manifest schema is invalid") from exc
    except (tarfile.TarError, KeyError) as exc:
        raise ObjectSetBundleError("Object-set bundle is not a valid archive") from exc


def iter_object_set_files(
    manifest: ObjectSetBundleManifest,
    contents: Mapping[str, bytes],
) -> Iterator[tuple[ObjectSetFile, bytes]]:
    for descriptor in manifest.files:
        content = contents[descriptor.relative_path]
        _validate_file(descriptor, content)
        yield descriptor, content


def portable_metadata(metadata: JsonObject) -> PortableArtifactBundleMetadata:
    value = metadata.get(PORTABLE_BUNDLE_METADATA_KEY)
    if not isinstance(value, dict):
        raise ObjectSetBundleError("Artifact lacks portable object-set metadata")
    return PortableArtifactBundleMetadata.model_validate(value)


def _metadata_value(metadata: JsonObject, path: tuple[str, ...]) -> object:
    current: object = metadata
    for part in path:
        if not isinstance(current, dict) or part not in current:
            raise ObjectSetBundleError(
                f"Portable metadata reference {'/'.join(path)!r} does not exist"
            )
        current = current[part]
    return current


def _set_metadata_value(
    metadata: JsonObject,
    path: tuple[str, ...],
    value: object,
) -> None:
    current: JsonObject = metadata
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ObjectSetBundleError(
                f"Portable metadata reference {'/'.join(path)!r} is invalid"
            )
        current = child
    if path[-1] not in current:
        raise ObjectSetBundleError(
            f"Portable metadata reference {'/'.join(path)!r} does not exist"
        )
    current[path[-1]] = value


def _deep_copy_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _deep_copy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy_json(item) for item in value]
    return value


def _validate_file(descriptor: ObjectSetFile, content: bytes) -> None:
    if (
        len(content) != descriptor.byte_size
        or sha256(content).hexdigest() != descriptor.sha256
    ):
        raise ObjectSetBundleError(
            f"Object-set file {descriptor.relative_path!r} failed validation"
        )


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    max_bytes: int,
) -> bytes:
    if member.size > max_bytes:
        raise ObjectSetBundleError(f"Object-set member {member.name!r} is oversized")
    stream = archive.extractfile(member)
    if stream is None:
        raise ObjectSetBundleError(f"Object-set member {member.name!r} is unreadable")
    content = stream.read(max_bytes + 1)
    if len(content) != member.size:
        raise ObjectSetBundleError(f"Object-set member {member.name!r} is truncated")
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
    "OBJECT_SET_BUNDLE_FORMAT",
    "ObjectSetBundleError",
    "ObjectSetBundleManifest",
    "ObjectSetFile",
    "ObjectSetMetadataReference",
    "PORTABLE_BUNDLE_METADATA_KEY",
    "PortableArtifactBundleMetadata",
    "PortableArtifactFile",
    "PortableMetadataReference",
    "iter_object_set_files",
    "load_object_set_bundle",
    "object_set_manifest",
    "portable_metadata",
    "write_object_set_bundle",
]

"""Host authorization, artifact staging, and atomic Plugin output import."""

import asyncio
from io import BytesIO
import json
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast, final, override
from uuid import uuid4

from grafy_core.artifacts import (
    ArtifactObject,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    JsonObject,
    UnitOfWorkPort,
)
from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeKey,
    PluginPortContract,
    plugin_protocol_digest,
)
from grafy_core.domain.errors import ObjectAlreadyExistsError
from grafy_core.operators.tables import (
    TABLE_DATA,
    Table,
    TableChunk,
    TableChunkDescriptor,
    TableManifest,
    load_table_manifest,
)
from grafy_core.ports.storage import (
    FileMetadata,
    FileStoragePort,
    SaveFileCommand,
    StoredFile,
)
from grafy_core.runtime.plugin_invocation import (
    PluginInvocationError,
    PluginInvocationRequest,
    PluginInvocationResult,
    PluginInvoker,
)
from grafy_core.runtime.plugin_protocol import (
    PluginArtifactShape,
    PluginFailureCode,
    PluginInputArtifactBundle,
    PluginInputArtifactGroup,
    PluginInputBinding,
    PluginInvocationArtifactTypeBinding,
    PluginInvocationEnvelope,
    PluginInvocationLimits,
    PluginInvocationRelease,
    PluginInvocationResultEnvelope,
    PluginOutputDeclaration,
)
from grafy_core.runtime.table_bundle import (
    TableBundleChunkDescriptor,
    TableBundleError,
    TableBundleManifest,
    file_identity,
    iter_table_bundle_chunks,
    validate_table_bundle,
    write_table_bundle,
    write_table_bundle_archive,
)


_RESULT_MANIFEST_MAX_BYTES = 1 * 1_024 * 1_024


class PluginGuestRunner(Protocol):
    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None: ...


class PluginInvocationScratch(Protocol):
    def root_for(self, request: PluginInvocationRequest, /) -> Path: ...


class PluginGuestRunError(RuntimeError):
    def __init__(self, code: PluginFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PluginInvocationCapacityDiagnostics:
    max_active_invocations: int
    active_invocations: int
    waiting_invocations: int
    total_invocations: int
    completed_invocations: int
    failed_invocations: int


class _BoundedLogError(RuntimeError):
    pass


async def _read_bounded_log(
    stream: asyncio.StreamReader,
    max_bytes: int,
) -> None:
    byte_count = 0
    while chunk := await stream.read(64 * 1_024):
        byte_count += len(chunk)
        if byte_count > max_bytes:
            raise _BoundedLogError(f"Plugin guest log exceeded {max_bytes} bytes")


async def _wait_for_process(process: asyncio.subprocess.Process) -> None:
    _ = await process.wait()


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()


@final
class SubprocessPluginGuestRunner(PluginGuestRunner):
    """Run the guest protocol entrypoint in a bounded local subprocess.

    This adapter is useful for protocol contract tests and development. It is
    not an operating-system sandbox; the Docker adapter in the next slice owns
    production isolation.
    """

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("Plugin guest subprocess command must not be empty")
        self._command = command
        self._environment = {
            "PYTHONHASHSEED": "0",
            "PYTHONUTF8": "1",
        }
        if environment is not None:
            self._environment.update(environment)

    @override
    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        process = await asyncio.create_subprocess_exec(
            *self._command,
            str(invocation_root),
            cwd=invocation_root,
            env=self._environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            await _kill_process(process)
            raise PluginGuestRunError(
                PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                "Plugin guest subprocess did not expose bounded log streams",
            )
        wait_task = asyncio.create_task(_wait_for_process(process))
        stdout_task = asyncio.create_task(
            _read_bounded_log(process.stdout, limits.max_log_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_bounded_log(process.stderr, limits.max_log_bytes)
        )
        tasks = {wait_task, stdout_task, stderr_task}
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=limits.wall_time_seconds,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            log_failure = next(
                (
                    task.exception()
                    for task in done
                    if not task.cancelled()
                    and isinstance(task.exception(), _BoundedLogError)
                ),
                None,
            )
            if log_failure is not None:
                await _kill_process(process)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise PluginGuestRunError(
                    PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                    str(log_failure),
                ) from log_failure
            if pending:
                await _kill_process(process)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise PluginGuestRunError(
                    PluginFailureCode.TIMEOUT,
                    f"Plugin guest exceeded {limits.wall_time_seconds} seconds",
                )
            wait_task.result()
            stdout_task.result()
            stderr_task.result()
            return_code = process.returncode
            if return_code != 0:
                raise PluginGuestRunError(
                    PluginFailureCode.INTERNAL_ADAPTER_FAILURE,
                    f"Plugin guest subprocess exited with status {return_code}",
                )
        except asyncio.CancelledError:
            await _kill_process(process)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


@dataclass(frozen=True, slots=True)
class _ValidatedInlineOutputArtifact:
    payload: JsonObject
    byte_count: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedTableOutputArtifact:
    path: Path
    manifest: TableBundleManifest
    byte_count: int
    content_sha256: str


type _ValidatedOutputArtifact = (
    _ValidatedInlineOutputArtifact | _ValidatedTableOutputArtifact
)


def _canonical_payload_bytes(payload: JsonObject) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _resolved_artifact_type(
    port: PluginPortContract,
    bindings: Mapping[str, ArtifactTypeKey],
) -> ArtifactTypeKey:
    if port.artifact_type is not None:
        return ArtifactTypeKey(
            port.artifact_type.id,
            port.artifact_type.schema_version,
        )
    variable = port.artifact_type_variable
    if variable is None or variable not in bindings:
        raise PluginInvocationError(
            f"Workspace Plugin port {port.name!r} has no concrete artifact type"
        )
    return bindings[variable]


def _input_groups(
    port: PluginPortContract,
    value: object,
) -> list[ArtifactRef | ArtifactRefSequence]:
    if port.instance_plugs or port.variadic:
        if not isinstance(value, list):
            raise PluginInvocationError(
                f"Workspace Plugin input {port.name!r} expected one artifact "
                "container per incoming edge"
            )
        groups = cast(list[object], value)
        if any(
            not isinstance(group, ArtifactRef | ArtifactRefSequence) for group in groups
        ):
            raise PluginInvocationError(
                f"Workspace Plugin input {port.name!r} contains a non-reference "
                "input group"
            )
        return cast(list[ArtifactRef | ArtifactRefSequence], groups)
    if not isinstance(value, ArtifactRef | ArtifactRefSequence):
        raise PluginInvocationError(
            f"Workspace Plugin input {port.name!r} is not an artifact reference "
            "container"
        )
    return [value]


def _group_refs(
    port: PluginPortContract,
    group: ArtifactRef | ArtifactRefSequence,
    expected: ArtifactTypeKey,
) -> tuple[PluginArtifactShape, list[ArtifactRef]]:
    if isinstance(group, ArtifactRef):
        if port.shape != "one":
            raise PluginInvocationError(
                f"Workspace Plugin input {port.name!r} expected cardinality many"
            )
        refs = [group]
        shape = "one"
    else:
        if port.shape != "many":
            raise PluginInvocationError(
                f"Workspace Plugin input {port.name!r} expected cardinality one"
            )
        refs = list(group.item_refs)
        shape = "many"
    if any(ref.key() != expected for ref in refs):
        raise PluginInvocationError(
            f"Workspace Plugin input {port.name!r} does not match "
            f"{expected.id}@{expected.schema_version}"
        )
    return shape, refs


def _bundle_path(invocation_root: Path, relative_path: str) -> Path:
    root = invocation_root.resolve()
    path = invocation_root.joinpath(*relative_path.split("/"))
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise PluginInvocationError("Plugin bundle path escapes invocation scratch")
    return path


@final
class ArtifactBundlePluginInvoker(PluginInvoker):
    """Authorize refs, exchange inline bundles, and mint host output refs."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkPort,
        runner: PluginGuestRunner,
        limits: PluginInvocationLimits | None = None,
        scratch_root: Path | None = None,
        scratch: PluginInvocationScratch | None = None,
        storage: FileStoragePort | None = None,
        bucket: str = "artifacts",
        storage_backend: str = "local",
        max_concurrent_invocations: int = 4,
    ) -> None:
        if scratch_root is not None and scratch is not None:
            raise ValueError(
                "Plugin invocation scratch path and provider are mutually exclusive"
            )
        if max_concurrent_invocations < 1:
            raise ValueError("Maximum concurrent Plugin invocations must be positive")
        self._unit_of_work = unit_of_work
        self._runner = runner
        self._limits = limits or PluginInvocationLimits()
        self._scratch_root = scratch_root
        self._scratch = scratch
        self._storage = storage
        self._bucket = bucket
        self._storage_backend = storage_backend
        self._max_concurrent_invocations = max_concurrent_invocations
        self._invocation_capacity = asyncio.Semaphore(max_concurrent_invocations)
        self._active_invocations = 0
        self._waiting_invocations = 0
        self._total_invocations = 0
        self._completed_invocations = 0
        self._failed_invocations = 0

    @override
    async def invoke(
        self,
        request: PluginInvocationRequest,
        /,
    ) -> PluginInvocationResult:
        self._total_invocations += 1
        self._waiting_invocations += 1
        try:
            await self._invocation_capacity.acquire()
        except BaseException:
            self._waiting_invocations -= 1
            self._failed_invocations += 1
            raise
        self._waiting_invocations -= 1
        self._active_invocations += 1
        try:
            result = await self._invoke_with_capacity(request)
        except BaseException:
            self._failed_invocations += 1
            raise
        else:
            self._completed_invocations += 1
            return result
        finally:
            self._active_invocations -= 1
            self._invocation_capacity.release()

    def diagnostics(self) -> PluginInvocationCapacityDiagnostics:
        return PluginInvocationCapacityDiagnostics(
            max_active_invocations=self._max_concurrent_invocations,
            active_invocations=self._active_invocations,
            waiting_invocations=self._waiting_invocations,
            total_invocations=self._total_invocations,
            completed_invocations=self._completed_invocations,
            failed_invocations=self._failed_invocations,
        )

    async def _invoke_with_capacity(
        self,
        request: PluginInvocationRequest,
    ) -> PluginInvocationResult:
        if request.release.protocol_digest != plugin_protocol_digest():
            raise PluginInvocationError(
                f"Workspace Plugin {request.release.slug!r} revision "
                f"{request.release.revision} uses an unsupported invocation "
                "protocol"
            )
        scratch_parent = None
        resolved_scratch_root = self._scratch_root
        if self._scratch is not None:
            resolved_scratch_root = self._scratch.root_for(request)
        if resolved_scratch_root is not None:
            resolved_scratch_root.mkdir(parents=True, exist_ok=True)
            scratch_parent = str(resolved_scratch_root)
        with TemporaryDirectory(
            prefix="grafy-plugin-invocation-",
            dir=scratch_parent,
        ) as temporary_directory:
            invocation_root = Path(temporary_directory)
            (invocation_root / "inputs").mkdir(mode=0o700)
            (invocation_root / "outputs").mkdir(mode=0o700)
            envelope = await self._stage_inputs(request, invocation_root)
            (invocation_root / "invocation.json").write_bytes(
                envelope.canonical_json_bytes()
            )
            try:
                await self._runner.run(invocation_root, self._limits)
            except PluginGuestRunError as exc:
                raise PluginInvocationError(
                    f"Workspace Plugin {request.release.slug!r} revision "
                    f"{request.release.revision} {exc.code.value} during "
                    f"invocation {envelope.invocation_id} for workflow "
                    f"{request.workflow_run_id}, node {request.node_id!r}, "
                    f"MAP index {request.invocation_index}: {exc}"
                ) from exc
            result = self._read_result(invocation_root, envelope)
            validated_outputs = self._validate_outputs(
                invocation_root,
                envelope,
                result,
            )
            return await self._import_outputs(request, validated_outputs)

    async def _stage_table_input(
        self,
        artifact: ArtifactObject,
        path: Path,
    ) -> tuple[int, str, int]:
        try:
            if artifact.inline_payload is not None:
                table = Table.model_validate(artifact.inline_payload)
                if len(table.rows) > self._limits.max_table_rows:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input exceeds its row limit"
                    )
                if len(table.columns) > self._limits.max_table_columns:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input exceeds its column limit"
                    )
                manifest = write_table_bundle(path, table)
            else:
                if self._storage is None:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input requires artifact storage"
                    )
                source_manifest = await load_table_manifest(artifact, self._storage)
                if artifact.bucket is None:
                    raise PluginInvocationError(
                        f"Workspace Plugin Table artifact {artifact.id} has no bucket"
                    )
                if artifact.byte_size is None or artifact.sha256 is None:
                    raise PluginInvocationError(
                        f"Workspace Plugin Table artifact {artifact.id} has no "
                        "logical content identity"
                    )
                if source_manifest.row_count > self._limits.max_table_rows:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input exceeds its row limit"
                    )
                if len(source_manifest.columns) > self._limits.max_table_columns:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input exceeds its column limit"
                    )
                if len(source_manifest.chunks) > self._limits.max_table_chunks:
                    raise PluginInvocationError(
                        "Workspace Plugin Table input exceeds its chunk limit"
                    )
                canonical_root = (
                    f"workspaces/{artifact.workspace_id}/{TABLE_DATA.key.id}/"
                    f"v{TABLE_DATA.key.schema_version}"
                )
                manifest_sha256 = artifact.metadata.get("manifest_sha256")
                if (
                    not isinstance(manifest_sha256, str)
                    or artifact.object_key
                    != f"{canonical_root}/manifests/{manifest_sha256}.json"
                ):
                    raise PluginInvocationError(
                        f"Workspace Plugin Table artifact {artifact.id} has a "
                        "non-canonical manifest object"
                    )
                manifest = TableBundleManifest(
                    columns=tuple(source_manifest.columns),
                    row_count=source_manifest.row_count,
                    logical_byte_size=artifact.byte_size,
                    logical_sha256=artifact.sha256,
                    chunks=tuple(
                        TableBundleChunkDescriptor(
                            offset=descriptor.offset,
                            row_count=descriptor.row_count,
                            relative_path=f"chunks/{index:06d}.json",
                            byte_size=descriptor.byte_size,
                            sha256=descriptor.sha256,
                        )
                        for index, descriptor in enumerate(source_manifest.chunks)
                    ),
                )
                with TemporaryDirectory(
                    prefix="grafy-table-export-",
                    dir=path.parent,
                ) as staging_directory:
                    staged_paths: list[Path] = []
                    staged_byte_size = 0
                    for index, descriptor in enumerate(source_manifest.chunks):
                        expected_object_key = (
                            f"{canonical_root}/chunks/{descriptor.sha256}.json"
                        )
                        if descriptor.object_key != expected_object_key:
                            raise PluginInvocationError(
                                f"Workspace Plugin Table artifact {artifact.id} "
                                f"references a non-canonical chunk at offset "
                                f"{descriptor.offset}"
                            )
                        staged_byte_size += descriptor.byte_size
                        if staged_byte_size > self._limits.max_input_bytes:
                            raise PluginInvocationError(
                                "Workspace Plugin Table input exceeds its byte limit"
                            )
                        stream = await self._storage.load(
                            artifact.bucket,
                            descriptor.object_key,
                        )
                        try:
                            content = stream.read(descriptor.byte_size + 1)
                        finally:
                            stream.close()
                        if (
                            len(content) != descriptor.byte_size
                            or sha256(content).hexdigest() != descriptor.sha256
                        ):
                            raise PluginInvocationError(
                                f"Workspace Plugin Table artifact {artifact.id} "
                                f"chunk at offset {descriptor.offset} failed "
                                "stored content validation"
                            )
                        chunk = TableChunk.model_validate_json(content)
                        if (
                            chunk.offset != descriptor.offset
                            or len(chunk.rows) != descriptor.row_count
                            or chunk.model_dump_json().encode("utf-8") != content
                        ):
                            raise PluginInvocationError(
                                f"Workspace Plugin Table artifact {artifact.id} "
                                f"chunk at offset {descriptor.offset} does not "
                                "match its manifest"
                            )
                        staged_path = Path(staging_directory) / f"{index:06d}.json"
                        staged_path.write_bytes(content)
                        staged_paths.append(staged_path)
                    write_table_bundle_archive(
                        path,
                        manifest,
                        (
                            (descriptor, staged_path.read_bytes())
                            for descriptor, staged_path in zip(
                                manifest.chunks,
                                staged_paths,
                                strict=True,
                            )
                        ),
                    )

            validated_manifest = validate_table_bundle(
                path,
                max_bytes=self._limits.max_input_bytes,
                max_files=self._limits.max_files,
                max_rows=self._limits.max_table_rows,
                max_columns=self._limits.max_table_columns,
                max_chunks=self._limits.max_table_chunks,
            )
            if (
                artifact.byte_size is not None
                and artifact.byte_size != validated_manifest.logical_byte_size
            ) or (
                artifact.sha256 is not None
                and artifact.sha256 != validated_manifest.logical_sha256
            ):
                raise PluginInvocationError(
                    f"Workspace Plugin Table artifact {artifact.id} logical "
                    "content metadata is stale"
                )
            identity = file_identity(path)
            path.chmod(0o400)
            return (
                identity.byte_size,
                identity.sha256,
                1 + len(validated_manifest.chunks),
            )
        except PluginInvocationError:
            raise
        except Exception as exc:
            raise PluginInvocationError(
                f"Workspace Plugin Table artifact {artifact.id} could not be staged"
            ) from exc

    async def _save_content_addressed_output(
        self,
        command: SaveFileCommand,
        *,
        expected_byte_size: int,
        expected_sha256: str,
    ) -> tuple[StoredFile, bool]:
        storage = self._storage
        if storage is None:
            raise PluginInvocationError(
                "Workspace Plugin Table output requires artifact storage"
            )
        try:
            stored = await storage.save(command)
        except ObjectAlreadyExistsError:
            stream = await storage.load(command.bucket, command.path)
            try:
                content = stream.read(expected_byte_size + 1)
            finally:
                stream.close()
            if (
                len(content) != expected_byte_size
                or sha256(content).hexdigest() != expected_sha256
            ):
                raise PluginInvocationError(
                    "Workspace Plugin Table output collided with invalid "
                    f"content-addressed object {command.bucket}/{command.path}"
                )
            return (
                StoredFile(
                    bucket=command.bucket,
                    path=command.path,
                    etag=None,
                    version_id=None,
                    byte_size=expected_byte_size,
                    sha256=expected_sha256,
                ),
                False,
            )
        if stored.byte_size != expected_byte_size or stored.sha256 != expected_sha256:
            try:
                await storage.delete(stored.bucket, stored.path)
            except Exception as cleanup_exc:
                raise PluginInvocationError(
                    "Workspace Plugin Table output storage changed a new "
                    "content-addressed object and cleanup failed"
                ) from cleanup_exc
            raise PluginInvocationError(
                "Workspace Plugin Table output storage changed a new "
                "content-addressed object"
            )
        return stored, True

    async def _stage_inputs(
        self,
        request: PluginInvocationRequest,
        invocation_root: Path,
    ) -> PluginInvocationEnvelope:
        ports_by_name = {port.name: port for port in request.contract.inputs}
        unknown_inputs = sorted(set(request.inputs) - set(ports_by_name))
        if unknown_inputs:
            raise PluginInvocationError(
                f"Workspace Plugin invocation contains unknown inputs: "
                f"{', '.join(unknown_inputs)}"
            )
        missing_inputs = sorted(
            port.name
            for port in request.contract.inputs
            if port.required and port.name not in request.inputs
        )
        if missing_inputs:
            raise PluginInvocationError(
                f"Workspace Plugin invocation is missing required inputs: "
                f"{', '.join(missing_inputs)}"
            )

        refs: list[ArtifactRef] = []
        staged_groups: dict[
            str,
            list[tuple[PluginArtifactShape, list[ArtifactRef]]],
        ] = {}
        for port in request.contract.inputs:
            if port.name not in request.inputs:
                continue
            expected = _resolved_artifact_type(
                port,
                request.artifact_type_bindings,
            )
            groups: list[tuple[PluginArtifactShape, list[ArtifactRef]]] = []
            for group in _input_groups(port, request.inputs[port.name]):
                shape, group_refs = _group_refs(port, group, expected)
                groups.append((shape, group_refs))
                refs.extend(group_refs)
            staged_groups[port.name] = groups

        artifact_ids = {ref.artifact_id for ref in refs}
        async with self._unit_of_work as unit_of_work:
            artifacts = await unit_of_work.artifacts.get_many(
                request.workspace_id,
                artifact_ids,
            )

        bindings: list[PluginInputBinding] = []
        total_bytes = 0
        total_files = 0
        for port_index, port in enumerate(request.contract.inputs):
            if port.name not in staged_groups:
                continue
            expected = _resolved_artifact_type(
                port,
                request.artifact_type_bindings,
            )
            protocol_groups: list[PluginInputArtifactGroup] = []
            for group_index, (shape, group_refs) in enumerate(staged_groups[port.name]):
                bundles: list[PluginInputArtifactBundle] = []
                for artifact_index, ref in enumerate(group_refs):
                    artifact = artifacts.get(ref.artifact_id)
                    if artifact is None:
                        raise PluginInvocationError(
                            f"Workspace Plugin input {port.name!r} references an "
                            "inaccessible or missing artifact"
                        )
                    if artifact.ref() != ref:
                        raise PluginInvocationError(
                            f"Workspace Plugin input {port.name!r} contains a stale "
                            f"or type-mismatched ref for artifact {ref.artifact_id}"
                        )
                    if expected == TABLE_DATA.key:
                        relative_path = (
                            f"inputs/p{port_index:04d}/g{group_index:04d}/"
                            f"a{artifact_index:06d}.table.tar"
                        )
                        path = _bundle_path(invocation_root, relative_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        (
                            byte_count,
                            content_digest,
                            file_count,
                        ) = await self._stage_table_input(artifact, path)
                    else:
                        payload = artifact.inline_payload
                        if payload is None:
                            raise PluginInvocationError(
                                f"Workspace Plugin input {port.name!r} artifact "
                                f"{ref.artifact_id} is not supported by the inline "
                                "JSON protocol"
                            )
                        content = _canonical_payload_bytes(payload)
                        content_digest = sha256(content).hexdigest()
                        if (
                            artifact.byte_size != len(content)
                            or artifact.sha256 != content_digest
                            or ref.content_hash != content_digest
                        ):
                            raise PluginInvocationError(
                                f"Workspace Plugin input {port.name!r} artifact "
                                f"{ref.artifact_id} content metadata is stale"
                            )
                        relative_path = (
                            f"inputs/p{port_index:04d}/g{group_index:04d}/"
                            f"a{artifact_index:06d}.json"
                        )
                        path = _bundle_path(invocation_root, relative_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                        path.chmod(0o400)
                        byte_count = len(content)
                        file_count = 1
                    total_bytes += byte_count
                    total_files += file_count
                    if total_bytes > self._limits.max_input_bytes:
                        raise PluginInvocationError(
                            "Workspace Plugin aggregate input byte limit exceeded"
                        )
                    if total_files > self._limits.max_files:
                        raise PluginInvocationError(
                            "Workspace Plugin aggregate input file-count limit exceeded"
                        )
                    bundles.append(
                        PluginInputArtifactBundle(
                            artifact_id=ref.artifact_id,
                            relative_path=relative_path,
                            byte_count=byte_count,
                            content_sha256=content_digest,
                        )
                    )
                protocol_groups.append(
                    PluginInputArtifactGroup(
                        shape=shape,
                        artifacts=tuple(bundles),
                    )
                )
            bindings.append(
                PluginInputBinding(
                    port=port.name,
                    artifact_type=PluginArtifactTypeKey.from_key(expected),
                    groups=tuple(protocol_groups),
                )
            )

        declarations: list[PluginOutputDeclaration] = []
        for port in request.contract.outputs:
            shape: PluginArtifactShape = "many" if port.shape == "many" else "one"
            declarations.append(
                PluginOutputDeclaration(
                    port=port.name,
                    artifact_type=PluginArtifactTypeKey.from_key(
                        _resolved_artifact_type(
                            port,
                            request.artifact_type_bindings,
                        )
                    ),
                    shape=shape,
                    required=port.required,
                )
            )
        invocation_id = uuid4()
        execution_scope_id = request.workflow_run_id or invocation_id
        return PluginInvocationEnvelope(
            invocation_id=invocation_id,
            execution_scope_id=execution_scope_id,
            workspace_id=request.workspace_id,
            workflow_run_id=request.workflow_run_id,
            node_id=request.node_id,
            invocation_index=request.invocation_index,
            release=PluginInvocationRelease(
                slug=request.release.slug,
                revision=request.release.revision,
                source_digest=request.release.source_digest,
                contract_digest=request.release.contract_digest,
                protocol_digest=request.release.protocol_digest,
            ),
            operator_id=request.contract.operator_id,
            operator_version=request.contract.operator_version,
            artifact_type_bindings=tuple(
                PluginInvocationArtifactTypeBinding(
                    variable=variable,
                    artifact_type=PluginArtifactTypeKey.from_key(artifact_type),
                )
                for variable, artifact_type in sorted(
                    request.artifact_type_bindings.items()
                )
            ),
            config=request.config,
            inputs=tuple(bindings),
            outputs=tuple(declarations),
            limits=self._limits,
        )

    def _read_result(
        self,
        invocation_root: Path,
        request: PluginInvocationEnvelope,
    ) -> PluginInvocationResultEnvelope:
        result_path = invocation_root / "result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise PluginInvocationError(
                f"Workspace Plugin {request.release.slug!r} revision "
                f"{request.release.revision} returned no regular result manifest"
            )
        result_size = result_path.stat().st_size
        if result_size > _RESULT_MANIFEST_MAX_BYTES:
            raise PluginInvocationError(
                "Workspace Plugin result manifest exceeds the protocol limit"
            )
        try:
            result = PluginInvocationResultEnvelope.from_json_bytes(
                result_path.read_bytes()
            )
        except Exception as exc:
            raise PluginInvocationError(
                f"Workspace Plugin {request.release.slug!r} revision "
                f"{request.release.revision} returned an invalid result manifest"
            ) from exc
        if result.invocation_id != request.invocation_id:
            raise PluginInvocationError(
                "Workspace Plugin result invocation identity does not match its request"
            )
        if result.status == "failed":
            failure = result.failure
            if failure is None:
                raise PluginInvocationError(
                    "Workspace Plugin returned a failed result without context"
                )
            if (
                failure.release_slug != request.release.slug
                or failure.release_revision != request.release.revision
                or failure.operator_id != request.operator_id
                or failure.operator_version != request.operator_version
                or failure.node_id != request.node_id
                or failure.invocation_index != request.invocation_index
            ):
                raise PluginInvocationError(
                    "Workspace Plugin failure context does not match its request"
                )
            raise PluginInvocationError(
                f"Workspace Plugin {failure.release_slug!r} revision "
                f"{failure.release_revision} operator {failure.operator_id}@"
                f"{failure.operator_version} {failure.code.value}: "
                f"{failure.message}"
            )
        return result

    def _validate_outputs(
        self,
        invocation_root: Path,
        request: PluginInvocationEnvelope,
        result: PluginInvocationResultEnvelope,
    ) -> dict[str, list[_ValidatedOutputArtifact]]:
        declarations = {
            declaration.port: declaration for declaration in request.outputs
        }
        bindings = {binding.port: binding for binding in result.outputs}
        extra_ports = sorted(set(bindings) - set(declarations))
        if extra_ports:
            raise PluginInvocationError(
                f"Workspace Plugin returned undeclared outputs: "
                f"{', '.join(extra_ports)}"
            )
        missing_ports = sorted(
            declaration.port
            for declaration in request.outputs
            if declaration.required and declaration.port not in bindings
        )
        if missing_ports:
            raise PluginInvocationError(
                f"Workspace Plugin omitted required outputs: {', '.join(missing_ports)}"
            )

        declared_paths = {
            artifact.relative_path
            for binding in result.outputs
            for artifact in binding.artifacts
        }
        actual_paths: set[str] = set()
        output_dir = invocation_root / "outputs"
        for path in output_dir.rglob("*"):
            if path.is_symlink():
                raise PluginInvocationError(
                    "Workspace Plugin output bundles must not contain symlinks"
                )
            if path.is_file():
                actual_paths.add(path.relative_to(invocation_root).as_posix())
        if actual_paths != declared_paths:
            raise PluginInvocationError(
                "Workspace Plugin output directory must contain exactly the "
                "declared bundle files"
            )

        validated: dict[str, list[_ValidatedOutputArtifact]] = {}
        total_bytes = 0
        total_files = 0
        for port, binding in bindings.items():
            declaration = declarations[port]
            if (
                binding.artifact_type != declaration.artifact_type
                or binding.shape != declaration.shape
            ):
                raise PluginInvocationError(
                    f"Workspace Plugin output {port!r} does not match its declared "
                    "type or cardinality"
                )
            artifacts: list[_ValidatedOutputArtifact] = []
            for bundle in binding.artifacts:
                total_bytes += bundle.byte_count
                if total_bytes > request.limits.max_output_bytes:
                    raise PluginInvocationError(
                        "Workspace Plugin aggregate output byte limit exceeded"
                    )
                path = _bundle_path(invocation_root, bundle.relative_path)
                if path.is_symlink() or not path.is_file():
                    raise PluginInvocationError(
                        f"Workspace Plugin output bundle "
                        f"{bundle.relative_path!r} is not a regular file"
                    )
                identity = file_identity(path)
                if (
                    identity.byte_size != bundle.byte_count
                    or identity.sha256 != bundle.content_sha256
                ):
                    raise PluginInvocationError(
                        f"Workspace Plugin output bundle "
                        f"{bundle.relative_path!r} failed size or digest validation"
                    )
                is_table = (
                    declaration.artifact_type.id == TABLE_DATA.key.id
                    and declaration.artifact_type.schema_version
                    == TABLE_DATA.key.schema_version
                )
                if is_table:
                    try:
                        manifest = validate_table_bundle(
                            path,
                            max_bytes=request.limits.max_output_bytes,
                            max_files=request.limits.max_files,
                            max_rows=request.limits.max_table_rows,
                            max_columns=request.limits.max_table_columns,
                            max_chunks=request.limits.max_table_chunks,
                        )
                    except TableBundleError as exc:
                        raise PluginInvocationError(
                            f"Workspace Plugin output Table bundle "
                            f"{bundle.relative_path!r} is invalid"
                        ) from exc
                    total_files += 1 + len(manifest.chunks)
                    artifacts.append(
                        _ValidatedTableOutputArtifact(
                            path=path,
                            manifest=manifest,
                            byte_count=bundle.byte_count,
                            content_sha256=bundle.content_sha256,
                        )
                    )
                else:
                    content = path.read_bytes()
                    value = json.loads(content)
                    if not isinstance(value, dict):
                        raise PluginInvocationError(
                            f"Workspace Plugin output bundle "
                            f"{bundle.relative_path!r} must contain one JSON object"
                        )
                    raw_payload = cast(dict[object, object], value)
                    if any(not isinstance(key, str) for key in raw_payload):
                        raise PluginInvocationError(
                            f"Workspace Plugin output bundle "
                            f"{bundle.relative_path!r} must contain one JSON object"
                        )
                    payload = cast(JsonObject, dict(raw_payload))
                    if _canonical_payload_bytes(payload) != content:
                        raise PluginInvocationError(
                            f"Workspace Plugin output bundle "
                            f"{bundle.relative_path!r} is not canonical inline JSON"
                        )
                    total_files += 1
                    artifacts.append(
                        _ValidatedInlineOutputArtifact(
                            payload=payload,
                            byte_count=len(content),
                            content_sha256=bundle.content_sha256,
                        )
                    )
                if total_files > request.limits.max_files:
                    raise PluginInvocationError(
                        "Workspace Plugin aggregate output file-count limit exceeded"
                    )
            validated[port] = artifacts
        return validated

    async def _import_outputs(
        self,
        request: PluginInvocationRequest,
        validated: Mapping[str, list[_ValidatedOutputArtifact]],
    ) -> PluginInvocationResult:
        artifacts_by_port: dict[str, list[ArtifactObject]] = {}
        created_objects: list[tuple[str, str]] = []
        ports_by_name = {port.name: port for port in request.contract.inputs}
        input_provenance: dict[str, object] = {}
        for name, value in request.inputs.items():
            artifact_ids: list[str] = []
            for group in _input_groups(ports_by_name[name], value):
                refs = (
                    group.item_refs
                    if isinstance(group, ArtifactRefSequence)
                    else [group]
                )
                artifact_ids.extend(str(ref.artifact_id) for ref in refs)
            input_provenance[name] = artifact_ids
        try:
            for port in request.contract.outputs:
                bundles = validated.get(port.name)
                if bundles is None:
                    continue
                key = _resolved_artifact_type(port, request.artifact_type_bindings)
                artifacts: list[ArtifactObject] = []
                for bundle in bundles:
                    artifact_metadata: JsonObject = {
                        "producer_node_id": request.node_id,
                        "provenance": input_provenance,
                        "plugin_release": {
                            "slug": request.release.slug,
                            "revision": request.release.revision,
                            "source_digest": request.release.source_digest,
                            "contract_digest": request.release.contract_digest,
                        },
                    }
                    if isinstance(bundle, _ValidatedInlineOutputArtifact):
                        artifact = ArtifactObject(
                            workspace_id=request.workspace_id,
                            artifact_type=key.id,
                            schema_version=key.schema_version,
                            content_type="application/json",
                            storage_backend="inline",
                            inline_payload=bundle.payload,
                            byte_size=bundle.byte_count,
                            sha256=bundle.content_sha256,
                            metadata=artifact_metadata,
                        )
                    else:
                        if self._storage is None:
                            raise PluginInvocationError(
                                "Workspace Plugin Table output requires artifact "
                                "storage"
                            )
                        stored_byte_size = 0
                        chunk_descriptors: list[TableChunkDescriptor] = []
                        for descriptor, content in iter_table_bundle_chunks(
                            bundle.path,
                            bundle.manifest,
                        ):
                            object_key = (
                                f"workspaces/{request.workspace_id}/"
                                f"{TABLE_DATA.key.id}/"
                                f"v{TABLE_DATA.key.schema_version}/chunks/"
                                f"{descriptor.sha256}.json"
                            )
                            metadata: FileMetadata = {
                                "artifact_kind": TABLE_DATA.key.id,
                                "sha256": descriptor.sha256,
                            }
                            if request.node_id is not None:
                                metadata["job_id"] = request.node_id
                            stored, created = await self._save_content_addressed_output(
                                SaveFileCommand(
                                    bucket=self._bucket,
                                    path=object_key,
                                    stream=BytesIO(content),
                                    content_type="application/json",
                                    metadata=metadata,
                                    allow_overwrite=False,
                                ),
                                expected_byte_size=descriptor.byte_size,
                                expected_sha256=descriptor.sha256,
                            )
                            if created:
                                created_objects.append((stored.bucket, stored.path))
                            stored_byte_size += stored.byte_size
                            chunk_descriptors.append(
                                TableChunkDescriptor(
                                    offset=descriptor.offset,
                                    row_count=descriptor.row_count,
                                    object_key=stored.path,
                                    byte_size=stored.byte_size,
                                    sha256=stored.sha256,
                                )
                            )
                        table_manifest = TableManifest(
                            columns=list(bundle.manifest.columns),
                            row_count=bundle.manifest.row_count,
                            chunks=chunk_descriptors,
                        )
                        manifest_content = table_manifest.model_dump_json().encode(
                            "utf-8"
                        )
                        manifest_sha256 = sha256(manifest_content).hexdigest()
                        manifest_path = (
                            f"workspaces/{request.workspace_id}/"
                            f"{TABLE_DATA.key.id}/"
                            f"v{TABLE_DATA.key.schema_version}/manifests/"
                            f"{manifest_sha256}.json"
                        )
                        (
                            stored_manifest,
                            created,
                        ) = await self._save_content_addressed_output(
                            SaveFileCommand(
                                bucket=self._bucket,
                                path=manifest_path,
                                stream=BytesIO(manifest_content),
                                content_type="application/json",
                                metadata={
                                    "artifact_kind": TABLE_DATA.key.id,
                                    "sha256": manifest_sha256,
                                },
                                allow_overwrite=False,
                            ),
                            expected_byte_size=len(manifest_content),
                            expected_sha256=manifest_sha256,
                        )
                        if created:
                            created_objects.append(
                                (stored_manifest.bucket, stored_manifest.path)
                            )
                        artifact_metadata.update(
                            {
                                "storage_format": table_manifest.format,
                                "row_count": table_manifest.row_count,
                                "column_count": len(table_manifest.columns),
                                "chunk_count": len(table_manifest.chunks),
                                "logical_byte_size": (
                                    bundle.manifest.logical_byte_size
                                ),
                                "storage_byte_size": (
                                    stored_byte_size + stored_manifest.byte_size
                                ),
                                "manifest_byte_size": stored_manifest.byte_size,
                                "manifest_sha256": stored_manifest.sha256,
                            }
                        )
                        artifact = ArtifactObject(
                            workspace_id=request.workspace_id,
                            artifact_type=key.id,
                            schema_version=key.schema_version,
                            content_type="application/json",
                            storage_backend=self._storage_backend,
                            bucket=stored_manifest.bucket,
                            object_key=stored_manifest.path,
                            byte_size=bundle.manifest.logical_byte_size,
                            sha256=bundle.manifest.logical_sha256,
                            metadata=artifact_metadata,
                        )
                    artifacts.append(artifact)
                artifacts_by_port[port.name] = artifacts

            async with self._unit_of_work as unit_of_work:
                for artifacts in artifacts_by_port.values():
                    for artifact in artifacts:
                        await unit_of_work.artifacts.add(artifact)
                await unit_of_work.commit()
        except Exception as exc:
            cleanup_failures: list[str] = []
            if self._storage is not None:
                for bucket, object_key in reversed(created_objects):
                    try:
                        await self._storage.delete(bucket, object_key)
                    except Exception as cleanup_exc:
                        cleanup_failures.append(
                            f"{bucket}/{object_key}: {type(cleanup_exc).__name__}"
                        )
            cleanup_context = ""
            if cleanup_failures:
                cleanup_context = "; partial object cleanup failed for " + ", ".join(
                    cleanup_failures
                )
            raise PluginInvocationError(
                f"Workspace Plugin {request.release.slug!r} revision "
                f"{request.release.revision} outputs could not be committed "
                f"atomically{cleanup_context}"
            ) from exc

        outputs: dict[str, ArtifactRef | ArtifactRefSequence] = {}
        for port in request.contract.outputs:
            output_artifacts = artifacts_by_port.get(port.name)
            if output_artifacts is None:
                continue
            refs = [artifact.ref() for artifact in output_artifacts]
            key = _resolved_artifact_type(port, request.artifact_type_bindings)
            if port.shape == "many":
                outputs[port.name] = ArtifactRefSequence.from_key(
                    key=key,
                    item_refs=refs,
                )
            else:
                outputs[port.name] = refs[0]
        return PluginInvocationResult(outputs=outputs)


__all__ = [
    "ArtifactBundlePluginInvoker",
    "PluginInvocationCapacityDiagnostics",
    "PluginGuestRunError",
    "PluginGuestRunner",
    "PluginInvocationScratch",
    "SubprocessPluginGuestRunner",
]

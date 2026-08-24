"""Guest-side execution of one scalar Workspace Plugin artifact invocation."""

import asyncio
import json
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, cast, final, override
from uuid import UUID

from pydantic import BaseModel, TypeAdapter

from grafy_core.artifacts import (
    ArtifactObject,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    InMemoryUnitOfWork,
    JsonObject,
    UnitOfWorkPort,
)
from grafy_core.domain.plugin_releases import (
    PLUGIN_INVOCATION_PROTOCOL,
    PluginCatalogManifest,
    plugin_contract_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import (
    InputContract,
    Node,
    NodeExecutionContext,
    OutputContract,
    resolve_node_contracts,
)
from grafy_core.operators.tables import TABLE_DATA, Table
from grafy_core.plugins import Plugin, PluginRuntimeContext, PluginUnitOfWorkPort
from grafy_core.ports.storage import (
    FileStoragePort,
    FileStreamProtocol,
    SaveFileCommand,
    StoredFile,
    StoredObjectInfo,
)
from grafy_core.runtime.materialization import InputMaterializer, MaterializationError
from grafy_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
    ArtifactWriterRegistry,
    OutputPersister,
    PersistedNodeOutput,
)
from grafy_core.runtime.plugin_protocol import (
    PluginFailureCode,
    PluginFailureEnvelope,
    PluginInvocationEnvelope,
    PluginInvocationResultEnvelope,
    PluginOutputArtifactBundle,
    PluginOutputBinding,
)
from grafy_core.runtime.resolvers import Resolver, ResolverRegistry
from grafy_core.runtime.table_bundle import (
    TableBundleError,
    file_identity,
    load_table_bundle_with_manifest,
    write_table_bundle,
)


class PluginGuestError(RuntimeError):
    """The staged invocation cannot be executed by the guest runtime."""


@final
class _UnavailableGuestStorage(FileStoragePort):
    """Fail closed if an inline-only invocation asks for durable storage."""

    @override
    async def save(self, command: SaveFileCommand) -> StoredFile:
        del command
        raise PluginGuestError("Guest durable storage is unavailable")

    @override
    async def move(
        self,
        bucket: str,
        source_path: str,
        destination_path: str,
    ) -> None:
        del bucket, source_path, destination_path
        raise PluginGuestError("Guest durable storage is unavailable")

    @override
    async def load(self, bucket: str, path: str) -> FileStreamProtocol:
        del bucket, path
        raise PluginGuestError("Guest durable storage is unavailable")

    @override
    async def stat(self, bucket: str, path: str) -> StoredObjectInfo | None:
        del bucket, path
        raise PluginGuestError("Guest durable storage is unavailable")

    @override
    async def load_range(
        self,
        bucket: str,
        path: str,
        start: int,
        end_exclusive: int,
    ) -> bytes:
        del bucket, path, start, end_exclusive
        raise PluginGuestError("Guest durable storage is unavailable")

    @override
    async def delete(self, bucket: str, path: str) -> None:
        del bucket, path
        raise PluginGuestError("Guest durable storage is unavailable")


@final
class _GuestInlineResolver(Resolver[object]):
    def __init__(
        self,
        *,
        source: ArtifactTypeKey,
        target: type[object],
        unit_of_work: UnitOfWorkPort,
    ) -> None:
        self.source = source
        self.target = target
        self._unit_of_work = unit_of_work

    @override
    async def resolve(self, ref: ArtifactRef, workspace_id: UUID) -> object:
        async with self._unit_of_work as unit_of_work:
            artifact = await unit_of_work.artifacts.get(workspace_id, ref.artifact_id)
        if artifact is None or artifact.ref() != ref:
            raise PluginGuestError(
                f"Staged input artifact {ref.artifact_id} is unavailable"
            )
        payload = artifact.inline_payload
        if payload is None:
            raise PluginGuestError(
                f"Staged input artifact {ref.artifact_id} is not inline JSON"
            )
        if issubclass(self.target, BaseModel):
            return self.target.model_validate(payload)
        if set(payload) != {"value"}:
            raise PluginGuestError(
                f"Inline scalar {ref.artifact_type}@{ref.schema_version} must "
                "contain exactly one 'value' field"
            )
        return TypeAdapter(self.target).validate_python(payload["value"], strict=True)


@final
class _GuestInlineOutputWriter(ArtifactOutputWriter):
    def __init__(
        self,
        *,
        artifact_type: ArtifactTypeKey,
        unit_of_work: UnitOfWorkPort,
    ) -> None:
        self.artifact_type = artifact_type
        self._unit_of_work = unit_of_work

    @override
    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        if isinstance(value, BaseModel):
            payload = cast(JsonObject, value.model_dump(mode="json", by_alias=True))
        elif isinstance(value, dict):
            raw_payload = cast(dict[object, object], value)
            if any(not isinstance(key, str) for key in raw_payload):
                raise PluginGuestError("Inline JSON output keys must be strings")
            payload = cast(JsonObject, dict(raw_payload))
        else:
            payload = {"value": value}
        payload_bytes = _canonical_payload_bytes(payload)
        artifact = ArtifactObject(
            workspace_id=context.node_context.workspace_id,
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type="application/json",
            storage_backend="inline",
            inline_payload=payload,
            byte_size=len(payload_bytes),
            sha256=sha256(payload_bytes).hexdigest(),
            metadata={"producer_node_id": context.node_context.node_id},
        )
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.artifacts.add(artifact)
            await unit_of_work.commit()
        return artifact.ref()


def _canonical_payload_bytes(payload: JsonObject) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _bundle_path(invocation_root: Path, relative_path: str) -> Path:
    root = invocation_root.resolve()
    path = invocation_root.joinpath(*relative_path.split("/"))
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise PluginGuestError("Plugin bundle path escapes the invocation root")
    return path


def _read_bundle(
    invocation_root: Path,
    relative_path: str,
    expected_bytes: int,
    expected_sha256: str,
) -> JsonObject:
    path = _bundle_path(invocation_root, relative_path)
    if path.is_symlink() or not path.is_file():
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} must be a regular file"
        )
    content = path.read_bytes()
    if len(content) != expected_bytes:
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} byte count does not match its manifest"
        )
    if sha256(content).hexdigest() != expected_sha256:
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} digest does not match its manifest"
        )
    value = json.loads(content)
    if not isinstance(value, dict):
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} must contain one JSON object"
        )
    raw_payload = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw_payload):
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} must contain one JSON object"
        )
    payload = cast(JsonObject, dict(raw_payload))
    if _canonical_payload_bytes(payload) != content:
        raise PluginGuestError(
            f"Plugin bundle {relative_path!r} is not canonical inline JSON"
        )
    return payload


def _declared_input_files(request: PluginInvocationEnvelope) -> set[str]:
    return {
        artifact.relative_path
        for binding in request.inputs
        for group in binding.groups
        for artifact in group.artifacts
    }


def _require_exact_input_files(
    invocation_root: Path,
    request: PluginInvocationEnvelope,
) -> None:
    inputs_dir = invocation_root / "inputs"
    actual: set[str] = set()
    for path in inputs_dir.rglob("*"):
        if path.is_symlink():
            raise PluginGuestError("Plugin input bundles must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(invocation_root).as_posix())
    if actual != _declared_input_files(request):
        raise PluginGuestError(
            "Plugin input directory must contain exactly the declared bundle files"
        )


async def _stage_input_artifacts(
    invocation_root: Path,
    request: PluginInvocationEnvelope,
    unit_of_work: InMemoryUnitOfWork,
    input_contract: InputContract[Any],
) -> dict[str, object]:
    _require_exact_input_files(invocation_root, request)
    artifacts_by_id: dict[UUID, ArtifactObject] = {}
    inputs: dict[str, object] = {}
    total_bytes = 0
    total_files = 0
    for binding in request.inputs:
        groups: list[ArtifactRef | ArtifactRefSequence] = []
        key = ArtifactTypeKey(
            binding.artifact_type.id,
            binding.artifact_type.schema_version,
        )
        for group in binding.groups:
            refs: list[ArtifactRef] = []
            for bundle in group.artifacts:
                total_bytes += bundle.byte_count
                if total_bytes > request.limits.max_input_bytes:
                    raise PluginGuestError("Plugin input byte limit exceeded")
                if key == TABLE_DATA.key:
                    path = _bundle_path(invocation_root, bundle.relative_path)
                    if path.is_symlink() or not path.is_file():
                        raise PluginGuestError(
                            f"Plugin Table bundle {bundle.relative_path!r} must be "
                            "a regular file"
                        )
                    identity = file_identity(path)
                    if (
                        identity.byte_size != bundle.byte_count
                        or identity.sha256 != bundle.content_sha256
                    ):
                        raise PluginGuestError(
                            f"Plugin Table bundle {bundle.relative_path!r} failed "
                            "size or digest validation"
                        )
                    try:
                        manifest, table = load_table_bundle_with_manifest(
                            path,
                            max_bytes=request.limits.max_input_bytes,
                            max_files=request.limits.max_files,
                            max_rows=request.limits.max_table_rows,
                            max_columns=request.limits.max_table_columns,
                            max_chunks=request.limits.max_table_chunks,
                        )
                    except TableBundleError as exc:
                        raise PluginGuestError(
                            f"Plugin Table bundle {bundle.relative_path!r} is invalid"
                        ) from exc
                    payload = cast(
                        JsonObject,
                        table.model_dump(mode="json", by_alias=True),
                    )
                    artifact_byte_size = manifest.logical_byte_size
                    artifact_sha256 = manifest.logical_sha256
                    total_files += 1 + len(manifest.chunks)
                else:
                    payload = _read_bundle(
                        invocation_root,
                        bundle.relative_path,
                        bundle.byte_count,
                        bundle.content_sha256,
                    )
                    artifact_byte_size = bundle.byte_count
                    artifact_sha256 = bundle.content_sha256
                    total_files += 1
                if total_files > request.limits.max_files:
                    raise PluginGuestError("Plugin input file-count limit exceeded")
                artifact = ArtifactObject(
                    workspace_id=request.workspace_id,
                    id=bundle.artifact_id,
                    artifact_type=key.id,
                    schema_version=key.schema_version,
                    content_type="application/json",
                    storage_backend="inline",
                    inline_payload=payload,
                    byte_size=artifact_byte_size,
                    sha256=artifact_sha256,
                )
                existing = artifacts_by_id.get(artifact.id)
                if existing is not None and existing != artifact:
                    raise PluginGuestError(
                        f"Input artifact {artifact.id} has conflicting bundles"
                    )
                artifacts_by_id[artifact.id] = artifact
                refs.append(artifact.ref())
            if group.shape == "one":
                groups.append(refs[0])
            else:
                groups.append(ArtifactRefSequence.from_key(key=key, item_refs=refs))
        spec = input_contract.ports[binding.port]
        if spec.instance_plugs or spec.variadic:
            inputs[binding.port] = groups
        else:
            inputs[binding.port] = groups[0]
    async with unit_of_work as entered:
        for artifact in artifacts_by_id.values():
            await entered.artifacts.add(artifact)
        await entered.commit()
    return inputs


def _build_node(
    plugin: Plugin,
    request: PluginInvocationEnvelope,
    context: PluginRuntimeContext,
) -> Node[Any, Any, Any]:
    for registration in plugin.nodes:
        if registration.key != (request.operator_id, request.operator_version):
            continue
        if registration.factory is not None:
            return registration.factory(context)
        return registration.node_class()
    raise PluginGuestError(
        f"Installed Plugin does not declare {request.operator_id}@"
        f"{request.operator_version}"
    )


def _artifact_type_bindings(
    request: PluginInvocationEnvelope,
) -> dict[str, ArtifactTypeKey]:
    return {
        binding.variable: ArtifactTypeKey(
            binding.artifact_type.id,
            binding.artifact_type.schema_version,
        )
        for binding in request.artifact_type_bindings
    }


def _validate_request_contract(
    request: PluginInvocationEnvelope,
    node: Node[Any, Any, Any],
) -> tuple[InputContract[Any], OutputContract[Any]]:
    resolved = resolve_node_contracts(node, _artifact_type_bindings(request))
    input_specs = resolved.input_contract.ports
    provided_input_names = {binding.port for binding in request.inputs}
    missing = sorted(
        name
        for name, spec in input_specs.items()
        if spec.required and name not in provided_input_names
    )
    if missing:
        raise PluginGuestError(
            f"Invocation is missing required input ports: {', '.join(missing)}"
        )
    extra = sorted(provided_input_names - set(input_specs))
    if extra:
        raise PluginGuestError(
            f"Invocation declares unknown input ports: {', '.join(extra)}"
        )
    for binding in request.inputs:
        spec = input_specs[binding.port]
        if not isinstance(spec.accepts, ArtifactTypeKey):
            raise PluginGuestError(
                f"Input port {binding.port!r} kept an unresolved artifact type"
            )
        if (
            binding.artifact_type.id != spec.accepts.id
            or binding.artifact_type.schema_version != spec.accepts.schema_version
        ):
            raise PluginGuestError(
                f"Input port {binding.port!r} artifact type does not match the "
                "installed Plugin"
            )
        expected_shape = spec.shape.value
        if any(group.shape != expected_shape for group in binding.groups):
            raise PluginGuestError(
                f"Input port {binding.port!r} cardinality does not match the "
                "installed Plugin"
            )
        if not spec.instance_plugs and not spec.variadic and len(binding.groups) != 1:
            raise PluginGuestError(
                f"Input port {binding.port!r} does not accept multiple groups"
            )

    output_specs = resolved.output_contract.ports
    if set(output_specs) != {declaration.port for declaration in request.outputs}:
        raise PluginGuestError(
            "Invocation output declarations do not match the installed Plugin"
        )
    for declaration in request.outputs:
        spec = output_specs[declaration.port]
        if not isinstance(spec.produces, ArtifactTypeKey):
            raise PluginGuestError(
                f"Output port {declaration.port!r} kept an unresolved artifact type"
            )
        if (
            declaration.artifact_type.id != spec.produces.id
            or declaration.artifact_type.schema_version != spec.produces.schema_version
            or declaration.shape != spec.shape.value
            or declaration.required != spec.required
        ):
            raise PluginGuestError(
                f"Output port {declaration.port!r} contract does not match the "
                "installed Plugin"
            )
    return resolved.input_contract, resolved.output_contract


def _plugin_runtime(
    plugin: Plugin,
    input_contract: InputContract[Any],
    output_contract: OutputContract[Any],
    context: PluginRuntimeContext,
    unit_of_work: InMemoryUnitOfWork,
) -> tuple[InputMaterializer, OutputPersister]:
    resolvers = [factory(context) for factory in plugin.resolver_factories]
    resolver_keys = {(resolver.source, resolver.target) for resolver in resolvers}
    for spec in input_contract.ports.values():
        if not isinstance(spec.accepts, ArtifactTypeKey) or spec.target_type is None:
            continue
        key = (spec.accepts, spec.target_type)
        if key in resolver_keys:
            continue
        resolvers.append(
            _GuestInlineResolver(
                source=spec.accepts,
                target=spec.target_type,
                unit_of_work=unit_of_work,
            )
        )
        resolver_keys.add(key)

    writers = [factory(context) for factory in plugin.writer_factories]
    writer_keys = {writer.artifact_type for writer in writers}
    for spec in output_contract.ports.values():
        if not isinstance(spec.produces, ArtifactTypeKey):
            continue
        if spec.produces in writer_keys:
            continue
        writers.append(
            _GuestInlineOutputWriter(
                artifact_type=spec.produces,
                unit_of_work=unit_of_work,
            )
        )
        writer_keys.add(spec.produces)
    return (
        InputMaterializer(ResolverRegistry(resolvers)),
        OutputPersister(ArtifactWriterRegistry(writers)),
    )


async def _write_output_bundles(
    invocation_root: Path,
    request: PluginInvocationEnvelope,
    persisted: PersistedNodeOutput,
    unit_of_work: InMemoryUnitOfWork,
) -> tuple[PluginOutputBinding, ...]:
    output_dir = invocation_root / "outputs"
    if any(output_dir.iterdir()):
        raise PluginGuestError("Plugin output directory must be empty before execution")
    bindings: list[PluginOutputBinding] = []
    total_bytes = 0
    total_files = 0
    for output_index, declaration in enumerate(request.outputs):
        value = persisted.values.get(declaration.port)
        if value is None:
            if declaration.required:
                raise PluginGuestError(
                    f"Plugin returned no value for output {declaration.port!r}"
                )
            continue
        refs = (
            value.item_refs
            if isinstance(value, ArtifactRefSequence)
            else [value]
            if isinstance(value, ArtifactRef)
            else []
        )
        if not refs:
            raise PluginGuestError(
                f"Plugin output {declaration.port!r} did not persist artifact refs"
            )
        bundles: list[PluginOutputArtifactBundle] = []
        for artifact_index, ref in enumerate(refs):
            async with unit_of_work as entered:
                artifact = await entered.artifacts.get(
                    request.workspace_id,
                    ref.artifact_id,
                )
            if artifact is None or artifact.ref() != ref:
                raise PluginGuestError(
                    f"Plugin output {declaration.port!r} returned an unknown artifact"
                )
            payload = artifact.inline_payload
            if payload is None:
                raise PluginGuestError(
                    f"Plugin output {declaration.port!r} is not inline JSON"
                )
            is_table = (
                declaration.artifact_type.id == TABLE_DATA.key.id
                and declaration.artifact_type.schema_version
                == TABLE_DATA.key.schema_version
            )
            if is_table:
                table = Table.model_validate(payload)
                if len(table.rows) > request.limits.max_table_rows:
                    raise PluginGuestError("Plugin Table output exceeds its row limit")
                if len(table.columns) > request.limits.max_table_columns:
                    raise PluginGuestError(
                        "Plugin Table output exceeds its column limit"
                    )
                relative_path = (
                    f"outputs/o{output_index:04d}/a{artifact_index:06d}.table.tar"
                )
                path = _bundle_path(invocation_root, relative_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                manifest = write_table_bundle(path, table)
                if len(manifest.chunks) > request.limits.max_table_chunks:
                    raise PluginGuestError(
                        "Plugin Table output exceeds its chunk limit"
                    )
                identity = file_identity(path)
                byte_count = identity.byte_size
                content_sha256 = identity.sha256
                file_count = 1 + len(manifest.chunks)
            else:
                content = _canonical_payload_bytes(payload)
                relative_path = (
                    f"outputs/o{output_index:04d}/a{artifact_index:06d}.json"
                )
                path = _bundle_path(invocation_root, relative_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                byte_count = len(content)
                content_sha256 = sha256(content).hexdigest()
                file_count = 1
            total_bytes += byte_count
            total_files += file_count
            if total_bytes > request.limits.max_output_bytes:
                raise PluginGuestError("Plugin output byte limit exceeded")
            if total_files > request.limits.max_files:
                raise PluginGuestError("Plugin output file-count limit exceeded")
            bundles.append(
                PluginOutputArtifactBundle(
                    relative_path=relative_path,
                    byte_count=byte_count,
                    content_sha256=content_sha256,
                )
            )
        bindings.append(
            PluginOutputBinding(
                port=declaration.port,
                artifact_type=declaration.artifact_type,
                shape=declaration.shape,
                artifacts=tuple(bundles),
            )
        )
    return tuple(bindings)


def _failure(
    request: PluginInvocationEnvelope,
    code: PluginFailureCode,
    message: str,
) -> PluginInvocationResultEnvelope:
    return PluginInvocationResultEnvelope(
        invocation_id=request.invocation_id,
        status="failed",
        failure=PluginFailureEnvelope(
            code=code,
            message=message,
            release_slug=request.release.slug,
            release_revision=request.release.revision,
            operator_id=request.operator_id,
            operator_version=request.operator_version,
            node_id=request.node_id,
            invocation_index=request.invocation_index,
        ),
    )


def _clear_output_directory(invocation_root: Path) -> None:
    output_dir = invocation_root / "outputs"
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


async def execute_plugin_invocation(invocation_root: Path) -> None:
    root = invocation_root.resolve()
    request = PluginInvocationEnvelope.from_json_bytes(
        (root / "invocation.json").read_bytes()
    )
    result_path = root / "result.json"
    if result_path.exists():
        raise PluginGuestError("Plugin invocation result manifest already exists")
    if request.release.protocol_digest != plugin_protocol_digest():
        result = _failure(
            request,
            PluginFailureCode.CONTRACT_FAILURE,
            f"Release protocol is incompatible with {PLUGIN_INVOCATION_PROTOCOL}",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return

    try:
        module = import_module("grafy_plugin")
        plugin = getattr(module, "PLUGIN", None)
        if not isinstance(plugin, Plugin):
            raise PluginGuestError("Installed project must export grafy_plugin.PLUGIN")
        catalog = PluginCatalogManifest.from_plugin(plugin)
        if (
            plugin.slug != request.release.slug
            or plugin_contract_digest(catalog) != request.release.contract_digest
        ):
            raise PluginGuestError(
                "Installed Plugin contract does not match the exact release"
            )
        unit_of_work = InMemoryUnitOfWork()
        plugin_context = PluginRuntimeContext(
            workspace=root,
            uploads_dir=root / "inputs",
            storage=_UnavailableGuestStorage(),
            uow=cast(PluginUnitOfWorkPort, unit_of_work),
            bucket="guest-unavailable",
            storage_backend="guest-inline",
        )
        node = _build_node(plugin, request, plugin_context)
        input_contract, output_contract = _validate_request_contract(request, node)
    except Exception:
        result = _failure(
            request,
            PluginFailureCode.CONTRACT_FAILURE,
            f"Plugin contract validation failed for {request.operator_id}@"
            f"{request.operator_version}",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return

    try:
        inputs = await _stage_input_artifacts(
            root,
            request,
            unit_of_work,
            input_contract,
        )
        materializer, persister = _plugin_runtime(
            plugin,
            input_contract,
            output_contract,
            plugin_context,
            unit_of_work,
        )
        materialized, provenance = await materializer.materialize(
            input_contract,
            inputs,
            request.workspace_id,
        )
    except MaterializationError as exc:
        result = _failure(
            request,
            PluginFailureCode.MATERIALIZATION_FAILURE,
            f"Input materialization failed for port {exc.port_name!r}",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return
    except Exception:
        result = _failure(
            request,
            PluginFailureCode.MATERIALIZATION_FAILURE,
            "Input artifact materialization failed",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return

    try:
        config = node.config_contract.model.model_validate(request.config)
    except Exception:
        result = _failure(
            request,
            PluginFailureCode.CONTRACT_FAILURE,
            f"Configuration validation failed for {request.operator_id}@"
            f"{request.operator_version}",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return

    node_context = NodeExecutionContext(
        workspace_id=request.workspace_id,
        workflow_run_id=request.workflow_run_id,
        node_id=request.node_id,
        invocation_index=request.invocation_index,
    )
    try:
        output = await node.run(node_context, config, materialized)
    except Exception:
        result = _failure(
            request,
            PluginFailureCode.OPERATOR_FAILURE,
            f"Operator {request.operator_id}@{request.operator_version} failed",
        )
        result_path.write_bytes(result.canonical_json_bytes())
        return

    try:
        persisted = await persister.persist(
            output_contract,
            node_context,
            output,
            provenance,
        )
        if not isinstance(persisted, PersistedNodeOutput):
            raise PluginGuestError("Plugin operator produced no artifact output ports")
        bindings = await _write_output_bundles(
            root,
            request,
            persisted,
            unit_of_work,
        )
        result = PluginInvocationResultEnvelope(
            invocation_id=request.invocation_id,
            status="succeeded",
            outputs=bindings,
        )
    except Exception:
        _clear_output_directory(root)
        result = _failure(
            request,
            PluginFailureCode.OUTPUT_VALIDATION,
            f"Output validation failed for {request.operator_id}@"
            f"{request.operator_version}",
        )
    result_path.write_bytes(result.canonical_json_bytes())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m grafy_core.runtime.plugin_guest ROOT")
    try:
        asyncio.run(execute_plugin_invocation(Path(sys.argv[1])))
    except Exception as exc:
        raise SystemExit(f"Plugin guest runtime failed: {type(exc).__name__}") from exc


if __name__ == "__main__":
    main()


__all__ = ["PluginGuestError", "execute_plugin_invocation"]

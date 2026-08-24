import asyncio
import json
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest

from grafy_core.artifacts import (
    ArtifactObject,
    ArtifactRef,
    ArtifactTypeKey,
    InMemoryUnitOfWork,
    JsonObject,
)
from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeContract,
    PluginArtifactTypeKey,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginNodeContract,
    PluginPortContract,
    PluginRelease,
    PluginReleaseIdentity,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import PortShape
from grafy_core.nodes import NodeExecutionContext
from grafy_core.operators.tables import (
    TABLE_DATA,
    Table,
    TableArtifactResolver,
    TableArtifactWriter,
    TableColumn,
    TableValueType,
)
from grafy_core.ports.storage import (
    FileStreamProtocol,
    SaveFileCommand,
    StoredFile,
    StoredObjectInfo,
)
from grafy_core.runtime.materialization import MaterializationProvenance
from grafy_core.runtime.persistence import ArtifactWriteContext
from grafy_core.runtime.plugin_invocation import (
    PluginInvocationError,
    PluginInvocationRequest,
)
from grafy_core.runtime.plugin_protocol import (
    PluginFailureCode,
    PluginFailureEnvelope,
    PluginInputBinding,
    PluginInvocationEnvelope,
    PluginInvocationLimits,
    PluginInvocationResultEnvelope,
    PluginOutputArtifactBundle,
    PluginOutputBinding,
)
from grafy_core.runtime.table_bundle import (
    load_table_bundle,
    write_table_bundle,
)
from grafy_storage import LocalFileObjectStore

from grafy_api.v1.routes.executions.runtime.plugin_artifacts import (
    ArtifactBundlePluginInvoker,
    PluginGuestRunError,
    PluginGuestRunner,
    SubprocessPluginGuestRunner,
)


WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000401")
OTHER_WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000402")
SUMMARY = ArtifactTypeKey("notes.summary", 1)
TEXT = ArtifactTypeKey("scalar.text", 1)
TABLE = TABLE_DATA.key


def _canonical(payload: JsonObject) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _port(
    name: str,
    direction: Literal["input", "output"],
    artifact_type: ArtifactTypeKey,
    shape: PortShape = PortShape.ONE,
) -> PluginPortContract:
    return PluginPortContract(
        name=name,
        direction=direction,
        artifact_type=PluginArtifactTypeKey.from_key(artifact_type),
        shape=shape,
        accepted_shapes=(shape,),
    )


def _release(
    *,
    two_outputs: bool = False,
    many_output: bool = False,
    two_inputs: bool = False,
    protocol_digest: str | None = None,
) -> PluginRelease:
    output_shape = PortShape.MANY if many_output else PortShape.ONE
    outputs = [_port("text", "output", TEXT, output_shape)]
    if two_outputs:
        outputs.append(_port("second", "output", TEXT))
    inputs = [_port("summary", "input", SUMMARY)]
    if two_inputs:
        inputs.append(_port("other_summary", "input", SUMMARY))
    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        artifact_types=(
            PluginArtifactTypeContract(
                key=PluginArtifactTypeKey.from_key(SUMMARY),
                title="Summary",
                payload_schema={"type": "object"},
            ),
        ),
        nodes=(
            PluginNodeContract(
                operator_id="notes.render",
                operator_version=1,
                title="Render",
                description="Render a summary",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=tuple(inputs),
                outputs=tuple(outputs),
            ),
        ),
    )
    capabilities = PluginCapabilityManifest()
    return PluginRelease(
        workspace_id=WORKSPACE_ID,
        slug="notes",
        revision=4,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=protocol_digest or plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/notes/source.tar.gz",
        source_digest="a" * 64,
        lock_digest="b" * 64,
        runtime_profile="python-uv",
    )


def _table_release() -> PluginRelease:
    catalog = PluginCatalogManifest(
        slug="tables",
        title="Tables",
        nodes=(
            PluginNodeContract(
                operator_id="tables.copy",
                operator_version=1,
                title="Copy Table",
                description="Copy one Table through the Plugin boundary",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(_port("source", "input", TABLE),),
                outputs=(_port("result", "output", TABLE),),
            ),
        ),
    )
    capabilities = PluginCapabilityManifest()
    return PluginRelease(
        workspace_id=WORKSPACE_ID,
        slug="tables",
        revision=1,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/tables/source.tar.gz",
        source_digest="c" * 64,
        lock_digest="d" * 64,
        runtime_profile="python-uv",
    )


async def _seed_inline_artifact(
    unit_of_work: InMemoryUnitOfWork,
    *,
    workspace_id: UUID = WORKSPACE_ID,
) -> ArtifactRef:
    payload: JsonObject = {
        "row_count": 2,
        "column_count": 1,
        "column_ids": ["name"],
    }
    content = _canonical(payload)
    artifact = ArtifactObject(
        workspace_id=workspace_id,
        artifact_type=SUMMARY.id,
        schema_version=SUMMARY.schema_version,
        content_type="application/json",
        storage_backend="inline",
        inline_payload=payload,
        byte_size=len(content),
        sha256=sha256(content).hexdigest(),
    )
    async with unit_of_work as entered:
        await entered.artifacts.add(artifact)
        await entered.commit()
    return artifact.ref()


def _request(
    release: PluginRelease,
    ref: ArtifactRef,
    *,
    second_ref: ArtifactRef | None = None,
) -> PluginInvocationRequest:
    inputs: dict[str, object] = {"summary": ref}
    if second_ref is not None:
        inputs["other_summary"] = second_ref
    return PluginInvocationRequest(
        release=PluginReleaseIdentity.from_release(release),
        contract=release.catalog.nodes[0],
        artifact_type_bindings={},
        config={},
        inputs=inputs,
        workspace_id=WORKSPACE_ID,
        node_id="render",
    )


RunnerMode = Literal[
    "valid",
    "bad_second_digest",
    "extra_file",
    "symlink",
    "wrong_type",
    "wrong_cardinality",
    "missing_required",
    "many_files",
    "mismatched_failure",
]


class ManifestRunner(PluginGuestRunner):
    def __init__(self, mode: RunnerMode = "valid", *, payload_size: int = 0) -> None:
        self.mode = mode
        self.payload_size = payload_size
        self.calls = 0
        self.observed_inputs: tuple[PluginInputBinding, ...] = ()

    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        del limits
        self.calls += 1
        request = PluginInvocationEnvelope.from_json_bytes(
            (invocation_root / "invocation.json").read_bytes()
        )
        self.observed_inputs = request.inputs
        if self.mode == "mismatched_failure":
            result = PluginInvocationResultEnvelope(
                invocation_id=request.invocation_id,
                status="failed",
                failure=PluginFailureEnvelope(
                    code=PluginFailureCode.OPERATOR_FAILURE,
                    message="Operator failed",
                    release_slug="other",
                    release_revision=request.release.revision,
                    operator_id=request.operator_id,
                    operator_version=request.operator_version,
                    node_id=request.node_id,
                    invocation_index=request.invocation_index,
                ),
            )
            (invocation_root / "result.json").write_bytes(result.canonical_json_bytes())
            return
        bindings: list[PluginOutputBinding] = []
        for output_index, declaration in enumerate(request.outputs):
            if self.mode == "missing_required" and output_index == 0:
                continue
            artifact_count = 2 if self.mode == "many_files" else 1
            bundles: list[PluginOutputArtifactBundle] = []
            for artifact_index in range(artifact_count):
                payload: JsonObject = {
                    "value": (
                        "x" * self.payload_size
                        if self.payload_size
                        else f"rendered-{artifact_index}"
                    )
                }
                content = _canonical(payload)
                relative_path = (
                    f"outputs/o{output_index:04d}/a{artifact_index:06d}.json"
                )
                path = invocation_root.joinpath(*relative_path.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                if self.mode == "symlink" and output_index == 0:
                    target = invocation_root / "symlink-target.json"
                    target.write_bytes(content)
                    path.symlink_to(target)
                else:
                    path.write_bytes(content)
                digest = sha256(content).hexdigest()
                if self.mode == "bad_second_digest" and output_index == 1:
                    digest = "0" * 64
                bundles.append(
                    PluginOutputArtifactBundle(
                        relative_path=relative_path,
                        byte_count=len(content),
                        content_sha256=digest,
                    )
                )
            artifact_type = declaration.artifact_type
            if self.mode == "wrong_type" and output_index == 0:
                artifact_type = PluginArtifactTypeKey(
                    id="scalar.integer",
                    schema_version=1,
                )
            shape = declaration.shape
            if self.mode == "wrong_cardinality" and output_index == 0:
                shape = "many"
            bindings.append(
                PluginOutputBinding(
                    port=declaration.port,
                    artifact_type=artifact_type,
                    shape=shape,
                    artifacts=tuple(bundles),
                )
            )
        if self.mode == "extra_file":
            (invocation_root / "outputs" / "undeclared.json").write_text("{}")
        result = PluginInvocationResultEnvelope(
            invocation_id=request.invocation_id,
            status="succeeded",
            outputs=tuple(bindings),
        )
        (invocation_root / "result.json").write_bytes(result.canonical_json_bytes())


class FailingGuestRunner(PluginGuestRunner):
    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        del invocation_root, limits
        raise PluginGuestRunError(
            PluginFailureCode.TIMEOUT,
            "controlled timeout",
        )


class CapacityObservingRunner(PluginGuestRunner):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            await self.release.wait()
            await ManifestRunner().run(invocation_root, limits)
        finally:
            self.active -= 1


class TableCopyRunner(PluginGuestRunner):
    async def run(
        self,
        invocation_root: Path,
        limits: PluginInvocationLimits,
    ) -> None:
        request = PluginInvocationEnvelope.from_json_bytes(
            (invocation_root / "invocation.json").read_bytes()
        )
        input_bundle = request.inputs[0].groups[0].artifacts[0]
        input_path = invocation_root.joinpath(*input_bundle.relative_path.split("/"))
        table = load_table_bundle(
            input_path,
            max_bytes=limits.max_input_bytes,
            max_files=limits.max_files,
            max_rows=limits.max_table_rows,
            max_columns=limits.max_table_columns,
            max_chunks=limits.max_table_chunks,
        )
        table.rows.append({"name": "guest", "count": 999})
        relative_path = "outputs/o0000/a000000.table.tar"
        output_path = invocation_root.joinpath(*relative_path.split("/"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_table_bundle(output_path, table)
        content = output_path.read_bytes()
        result = PluginInvocationResultEnvelope(
            invocation_id=request.invocation_id,
            status="succeeded",
            outputs=(
                PluginOutputBinding(
                    port="result",
                    artifact_type=PluginArtifactTypeKey.from_key(TABLE),
                    shape="one",
                    artifacts=(
                        PluginOutputArtifactBundle(
                            relative_path=relative_path,
                            byte_count=len(content),
                            content_sha256=sha256(content).hexdigest(),
                        ),
                    ),
                ),
            ),
        )
        (invocation_root / "result.json").write_bytes(result.canonical_json_bytes())


@pytest.mark.asyncio
async def test_plugin_invocation_capacity_is_process_wide_and_bounded(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    release = _release()
    runner = CapacityObservingRunner()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        scratch_root=tmp_path / "scratch",
        max_concurrent_invocations=1,
    )
    request = _request(release, input_ref)

    first = asyncio.create_task(invoker.invoke(request))
    second = asyncio.create_task(invoker.invoke(request))
    await runner.entered.wait()
    await asyncio.sleep(0)

    assert runner.calls == 1
    assert runner.max_active == 1
    active_diagnostics = invoker.diagnostics()
    assert active_diagnostics.max_active_invocations == 1
    assert active_diagnostics.active_invocations == 1
    assert active_diagnostics.waiting_invocations == 1
    assert active_diagnostics.total_invocations == 2

    runner.release.set()
    await asyncio.gather(first, second)

    assert runner.calls == 2
    assert runner.max_active == 1
    terminal_diagnostics = invoker.diagnostics()
    assert terminal_diagnostics.active_invocations == 0
    assert terminal_diagnostics.waiting_invocations == 0
    assert terminal_diagnostics.completed_invocations == 2
    assert terminal_diagnostics.failed_invocations == 0


class FailingTableManifestStorage:
    def __init__(self, delegate: LocalFileObjectStore) -> None:
        self._delegate = delegate
        self.deleted: list[tuple[str, str]] = []

    async def save(self, command: SaveFileCommand) -> StoredFile:
        if "/manifests/" in command.path:
            raise OSError("controlled manifest save failure")
        return await self._delegate.save(command)

    async def move(
        self,
        bucket: str,
        source_path: str,
        destination_path: str,
    ) -> None:
        await self._delegate.move(bucket, source_path, destination_path)

    async def load(self, bucket: str, path: str) -> FileStreamProtocol:
        return await self._delegate.load(bucket, path)

    async def stat(self, bucket: str, path: str) -> StoredObjectInfo | None:
        return await self._delegate.stat(bucket, path)

    async def load_range(
        self,
        bucket: str,
        path: str,
        start: int,
        end_exclusive: int,
    ) -> bytes:
        return await self._delegate.load_range(bucket, path, start, end_exclusive)

    async def delete(self, bucket: str, path: str) -> None:
        self.deleted.append((bucket, path))
        await self._delegate.delete(bucket, path)


@pytest.mark.asyncio
async def test_table_input_and_output_cross_the_bundle_boundary(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "objects")
    table = Table(
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
        rows=[{"name": f"row-{index}", "count": index} for index in range(101)],
    )
    input_ref = await TableArtifactWriter(
        storage=storage,
        uow=unit_of_work,
        bucket="artifacts",
        storage_backend="local",
    ).write(
        table,
        ArtifactWriteContext(
            node_context=NodeExecutionContext(
                workspace_id=WORKSPACE_ID,
                node_id="source",
            ),
            provenance=MaterializationProvenance(refs_by_input={}),
        ),
    )
    release = _table_release()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=TableCopyRunner(),
        scratch_root=tmp_path / "scratch",
        storage=storage,
        bucket="artifacts",
        storage_backend="local",
    )
    request = PluginInvocationRequest(
        release=PluginReleaseIdentity.from_release(release),
        contract=release.catalog.nodes[0],
        artifact_type_bindings={},
        config={},
        inputs={"source": input_ref},
        workspace_id=WORKSPACE_ID,
        node_id="copy",
    )

    result = await invoker.invoke(request)

    output_ref = result.outputs["result"]
    assert isinstance(output_ref, ArtifactRef)
    resolved = await TableArtifactResolver(
        uow=unit_of_work,
        storage=storage,
    ).resolve(output_ref, WORKSPACE_ID)
    assert resolved.rows == [*table.rows, {"name": "guest", "count": 999}]
    async with unit_of_work as entered:
        output = await entered.artifacts.get(WORKSPACE_ID, output_ref.artifact_id)
    assert output is not None
    assert output.inline_payload is None
    assert output.metadata["row_count"] == 102
    assert output.metadata["chunk_count"] == 2
    assert output.metadata["plugin_release"] == {
        "slug": "tables",
        "revision": 1,
        "source_digest": "c" * 64,
        "contract_digest": release.contract_digest,
    }
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.asyncio
async def test_table_from_another_workspace_fails_before_staging(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "objects")
    foreign_ref = await TableArtifactWriter(
        storage=storage,
        uow=unit_of_work,
        bucket="artifacts",
        storage_backend="local",
    ).write(
        Table(
            columns=[
                TableColumn(
                    id="name",
                    title="Name",
                    value_type=TableValueType.TEXT,
                )
            ],
            rows=[{"name": "private"}],
        ),
        ArtifactWriteContext(
            node_context=NodeExecutionContext(
                workspace_id=OTHER_WORKSPACE_ID,
                node_id="foreign-source",
            ),
            provenance=MaterializationProvenance(refs_by_input={}),
        ),
    )
    runner = ManifestRunner()
    release = _table_release()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        scratch_root=tmp_path / "scratch",
        storage=storage,
        bucket="artifacts",
        storage_backend="local",
    )

    with pytest.raises(PluginInvocationError, match="inaccessible or missing"):
        await invoker.invoke(
            PluginInvocationRequest(
                release=PluginReleaseIdentity.from_release(release),
                contract=release.catalog.nodes[0],
                artifact_type_bindings={},
                config={},
                inputs={"source": foreign_ref},
                workspace_id=WORKSPACE_ID,
                node_id="copy",
            )
        )

    assert runner.calls == 0


@pytest.mark.asyncio
async def test_failed_table_import_removes_new_objects_and_exposes_no_output(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    durable_storage = LocalFileObjectStore(tmp_path / "objects")
    table = Table(
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
        rows=[{"name": "host", "count": 1}],
    )
    input_ref = await TableArtifactWriter(
        storage=durable_storage,
        uow=unit_of_work,
        bucket="artifacts",
        storage_backend="local",
    ).write(
        table,
        ArtifactWriteContext(
            node_context=NodeExecutionContext(
                workspace_id=WORKSPACE_ID,
                node_id="source",
            ),
            provenance=MaterializationProvenance(refs_by_input={}),
        ),
    )
    failing_storage = FailingTableManifestStorage(durable_storage)
    release = _table_release()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=TableCopyRunner(),
        scratch_root=tmp_path / "scratch",
        storage=failing_storage,
        bucket="artifacts",
        storage_backend="local",
    )
    request = PluginInvocationRequest(
        release=PluginReleaseIdentity.from_release(release),
        contract=release.catalog.nodes[0],
        artifact_type_bindings={},
        config={},
        inputs={"source": input_ref},
        workspace_id=WORKSPACE_ID,
        node_id="copy",
    )

    with pytest.raises(PluginInvocationError, match="committed atomically"):
        await invoker.invoke(request)

    assert failing_storage.deleted
    for bucket, object_key in failing_storage.deleted:
        assert await durable_storage.stat(bucket, object_key) is None
    async with unit_of_work as entered:
        tables = await entered.artifacts.list_by_type(WORKSPACE_ID, TABLE)
    assert [artifact.id for artifact in tables] == [input_ref.artifact_id]
    assert (
        await TableArtifactResolver(
            uow=unit_of_work,
            storage=durable_storage,
        ).resolve(input_ref, WORKSPACE_ID)
        == table
    )


@pytest.mark.asyncio
async def test_host_authorizes_stages_and_atomically_mints_output_refs(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    release = _release()
    runner = ManifestRunner()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        scratch_root=tmp_path,
    )

    result = await invoker.invoke(_request(release, input_ref))

    assert runner.calls == 1
    assert runner.observed_inputs[0].groups[0].artifacts[0].artifact_id == (
        input_ref.artifact_id
    )
    output_ref = result.outputs["text"]
    assert isinstance(output_ref, ArtifactRef)
    assert output_ref.artifact_id != input_ref.artifact_id
    async with unit_of_work as entered:
        output = await entered.artifacts.get(WORKSPACE_ID, output_ref.artifact_id)
        original = await entered.artifacts.get(WORKSPACE_ID, input_ref.artifact_id)
    assert output is not None
    assert output.inline_payload == {"value": "rendered-0"}
    assert output.object_key is None
    assert output.metadata["plugin_release"] == {
        "slug": "notes",
        "revision": 4,
        "source_digest": "a" * 64,
        "contract_digest": release.contract_digest,
    }
    assert original is not None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_guest_failure_preserves_release_and_invocation_context(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    workflow_run_id = uuid4()
    request = replace(
        _request(_release(), input_ref),
        workflow_run_id=workflow_run_id,
        invocation_index=7,
    )
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=FailingGuestRunner(),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError) as captured:
        await invoker.invoke(request)

    message = str(captured.value)
    assert "Plugin 'notes' revision 4 timeout" in message
    assert f"workflow {workflow_run_id}" in message
    assert "node 'render'" in message
    assert "MAP index 7" in message
    assert "controlled timeout" in message


@pytest.mark.asyncio
async def test_cross_workspace_and_stale_refs_fail_before_guest_execution(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    other_ref = await _seed_inline_artifact(
        unit_of_work,
        workspace_id=OTHER_WORKSPACE_ID,
    )
    runner = ManifestRunner()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        scratch_root=tmp_path,
    )
    release = _release()

    with pytest.raises(PluginInvocationError, match="inaccessible or missing"):
        await invoker.invoke(_request(release, other_ref))

    own_ref = await _seed_inline_artifact(unit_of_work)
    stale = own_ref.model_copy(update={"content_hash": "f" * 64})
    with pytest.raises(PluginInvocationError, match="stale or type-mismatched"):
        await invoker.invoke(_request(release, stale))
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_incompatible_release_protocol_fails_before_guest_execution(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    runner = ManifestRunner()
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match="unsupported invocation protocol"):
        await invoker.invoke(_request(_release(protocol_digest="0" * 64), input_ref))

    assert runner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["bytes", "files"])
async def test_input_limits_fail_before_guest_execution(
    tmp_path: Path,
    limit: str,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    second_ref = await _seed_inline_artifact(unit_of_work)
    runner = ManifestRunner()
    limits = (
        PluginInvocationLimits(max_input_bytes=1)
        if limit == "bytes"
        else PluginInvocationLimits(max_files=1)
    )
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=runner,
        limits=limits,
        scratch_root=tmp_path,
    )

    with pytest.raises(
        PluginInvocationError,
        match=f"input {limit[:-1] if limit.endswith('s') else limit}",
    ):
        await invoker.invoke(
            _request(
                _release(two_inputs=True),
                input_ref,
                second_ref=second_ref,
            )
        )

    assert runner.calls == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("extra_file", "exactly the declared"),
        ("symlink", "must not contain symlinks"),
        ("wrong_type", "type or cardinality"),
        ("wrong_cardinality", "type or cardinality"),
        ("missing_required", "omitted required"),
    ],
)
async def test_untrusted_output_shape_and_files_fail_before_import(
    tmp_path: Path,
    mode: RunnerMode,
    message: str,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=ManifestRunner(mode),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match=message):
        await invoker.invoke(_request(_release(), input_ref))

    async with unit_of_work as entered:
        outputs = await entered.artifacts.list_by_type(WORKSPACE_ID, TEXT)
    assert outputs == []


@pytest.mark.asyncio
async def test_invalid_second_output_leaves_no_first_output_visible(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=ManifestRunner("bad_second_digest"),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match="digest validation"):
        await invoker.invoke(_request(_release(two_outputs=True), input_ref))

    async with unit_of_work as entered:
        outputs = await entered.artifacts.list_by_type(WORKSPACE_ID, TEXT)
    assert outputs == []


@pytest.mark.asyncio
async def test_guest_failure_context_must_match_request(tmp_path: Path) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=ManifestRunner("mismatched_failure"),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match="context does not match"):
        await invoker.invoke(_request(_release(), input_ref))


@pytest.mark.asyncio
async def test_output_byte_limit_fails_predictably_before_import(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=ManifestRunner(payload_size=128),
        limits=PluginInvocationLimits(max_output_bytes=32),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match="output byte limit"):
        await invoker.invoke(_request(_release(), input_ref))


@pytest.mark.asyncio
async def test_output_file_count_limit_fails_predictably_before_import(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    input_ref = await _seed_inline_artifact(unit_of_work)
    invoker = ArtifactBundlePluginInvoker(
        unit_of_work=unit_of_work,
        runner=ManifestRunner("many_files"),
        limits=PluginInvocationLimits(max_files=1),
        scratch_root=tmp_path,
    )

    with pytest.raises(PluginInvocationError, match="file-count limit"):
        await invoker.invoke(_request(_release(many_output=True), input_ref))


@pytest.mark.asyncio
async def test_subprocess_runner_bounds_logs_and_wall_time(tmp_path: Path) -> None:
    log_runner = SubprocessPluginGuestRunner(
        (sys.executable, "-c", "print('x' * 1000)"),
    )
    with pytest.raises(PluginGuestRunError, match="log exceeded") as log_error:
        await log_runner.run(
            tmp_path,
            PluginInvocationLimits(max_log_bytes=32),
        )
    assert log_error.value.code.value == "internal_adapter_failure"

    timeout_runner = SubprocessPluginGuestRunner(
        (sys.executable, "-c", "import time; time.sleep(5)"),
    )
    with pytest.raises(PluginGuestRunError, match="exceeded 1 seconds") as timeout:
        await timeout_runner.run(
            tmp_path,
            PluginInvocationLimits(wall_time_seconds=1),
        )
    assert timeout.value.code.value == "timeout"

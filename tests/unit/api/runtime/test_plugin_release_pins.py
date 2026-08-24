"""Compiler contracts for exact Workspace Plugin release pins."""

from collections.abc import Mapping
from pathlib import Path
from typing import Never, cast
from uuid import UUID, uuid4

import pytest

from grafy_core.application.saved_graphs import SavedGraphService
from grafy_core.artifacts import ArtifactRef, ArtifactTypeKey, InMemoryUnitOfWork
from grafy_core.domain.modules import GraphModuleDefinition
from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeKey,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginNodeContract,
    PluginPortContract,
    PluginPortDirection,
    PluginRelease,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import NodeExecutionContext, PortShape
from grafy_core.operators.text import TextValueOutputWriter, TextValueResolver
from grafy_core.plugins import PluginRuntimeContext
from grafy_core.ports.node_secrets import UnavailableNodeSecretResolver
from grafy_core.ports.modules import GraphModuleExecutionResult
from grafy_core.runtime.invocation import InvocationMode
from grafy_core.runtime.execution import NodeRuntime
from grafy_core.runtime.materialization import InputMaterializer
from grafy_core.runtime.persistence import ArtifactWriterRegistry, OutputPersister
from grafy_core.runtime.plugin_invocation import (
    PluginInvocationRequest,
    PluginInvocationResult,
    WorkspacePluginReleaseNode,
)
from grafy_core.runtime.resolvers import ResolverRegistry
from grafy_storage import LocalFileObjectStore

from grafy_api.builtins import builtin_plugins
from grafy_api.plugin_discovery import build_plugin_registry
from grafy_api.v1.routes.artifacts.services import ArtifactService
from grafy_api.v1.routes.executions.models import (
    RunEdgeRequest,
    RunNodeRequest,
    RunRequest,
)
from grafy_api.v1.routes.executions.runtime.coordinator import (
    GraphExecutionCoordinator,
)
from grafy_api.v1.routes.catalog.services import GraphModuleCatalog
from grafy_api.v1.models import PluginReleasePinModel
from grafy_api.v1.routes.executions.runtime.compiler import GraphCompiler
from grafy_api.v1.routes.executions.runtime.errors import GraphExecutionError
from grafy_api.v1.routes.executions.runtime.edge_values import EdgeValueResolver
from grafy_api.v1.routes.executions.runtime.models import PreparedGraphExecution
from grafy_api.v1.routes.executions.runtime.node_execution import (
    NodeExecutionService,
)


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000871")
OTHER_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000872")
TEXT = PluginArtifactTypeKey(id="scalar.text", schema_version=1)


def _text_port(
    name: str,
    direction: PluginPortDirection,
) -> PluginPortContract:
    return PluginPortContract(
        name=name,
        title=name.title(),
        direction=direction,
        artifact_type=TEXT,
        shape=PortShape.ONE,
        accepted_shapes=(PortShape.ONE,),
    )


def _echo_contract(operator_version: int = 1) -> PluginNodeContract:
    return PluginNodeContract(
        operator_id="notes.echo",
        operator_version=operator_version,
        title="Echo",
        description="Echoes one text reference.",
        config_schema={"type": "object"},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        inputs=(_text_port("text", "input"),),
        outputs=(_text_port("text", "output"),),
    )


def _release(
    revision: int,
    nodes: tuple[PluginNodeContract, ...] = (_echo_contract(),),
    *,
    workspace_id: UUID = WORKSPACE_ID,
) -> PluginRelease:
    catalog = PluginCatalogManifest(slug="notes", title="Notes", nodes=nodes)
    capabilities = PluginCapabilityManifest()
    return PluginRelease(
        workspace_id=workspace_id,
        slug=catalog.slug,
        revision=revision,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key=f"plugin-releases/notes/r{revision}.tar.gz",
        source_digest=f"{revision}" * 64,
        lock_digest="9" * 64,
        runtime_profile="python-uv",
    )


class RecordingReleaseLookup:
    def __init__(self, *releases: PluginRelease) -> None:
        self._releases = releases

    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginRelease | None:
        for release in self._releases:
            if (
                release.workspace_id == workspace_id
                and release.slug == slug
                and release.revision == revision
            ):
                return release
        return None


class NoopInvoker:
    def __init__(self) -> None:
        self.requests: list[PluginInvocationRequest] = []

    async def invoke(
        self,
        request: PluginInvocationRequest,
        /,
    ) -> PluginInvocationResult:
        self.requests.append(request)
        return PluginInvocationResult(outputs={})


class _UnusedModuleExecutor:
    async def execute_module(
        self,
        _definition: GraphModuleDefinition,
        _context: NodeExecutionContext,
        _inputs: Mapping[str, object],
        /,
    ) -> GraphModuleExecutionResult:
        raise AssertionError("Compiler test unexpectedly executed a graph module")


def _unused_saved_graph_uow() -> Never:
    raise AssertionError("Compiler test unexpectedly queried saved graphs")


def _compiler(
    lookup: RecordingReleaseLookup | None = None,
    invoker: NoopInvoker | None = None,
) -> GraphCompiler:
    registry = build_plugin_registry(builtin_plugins(), external_plugins=())
    unit_of_work = InMemoryUnitOfWork()
    workbench = Path("/tmp/grafy-plugin-pin-tests")
    uploads_dir = workbench / "uploads"
    plugin_context = PluginRuntimeContext(
        workspace=workbench,
        uploads_dir=uploads_dir,
        storage=LocalFileObjectStore(workbench / "objects"),
        uow=unit_of_work,
        bucket="test-artifacts",
    )
    saved_graphs = SavedGraphService(_unused_saved_graph_uow, registry)
    return GraphCompiler(
        plugin_registry=registry,
        plugin_context=plugin_context,
        module_catalog=GraphModuleCatalog(saved_graphs, registry),
        plugin_release_lookup=lookup or RecordingReleaseLookup(),
        plugin_invoker=invoker or NoopInvoker(),
    )


def _pinned_node(
    node_id: str,
    pin: PluginReleasePinModel | None,
    *,
    operator_id: str = "notes.echo",
) -> RunNodeRequest:
    return RunNodeRequest(
        id=node_id,
        operator_id=operator_id,
        operator_version=1,
        config={"scale": 2},
        plugin_release=pin,
    )


def _echo_run_request(
    pin_revision: int,
) -> RunRequest:
    """One pinned echo node fed by a pinned upstream output edge."""
    from grafy_core.artifacts import ArtifactRef, ArtifactTypeKey
    from grafy_api.v1.routes.executions.models import (
        PinnedOutputRequest,
        RunEdgeRequest,
    )

    return RunRequest(
        nodes=[
            _pinned_node(
                "echo", PluginReleasePinModel(slug="notes", revision=pin_revision)
            )
        ],
        edges=[
            RunEdgeRequest(
                from_node="upstream",
                from_port="text",
                to_node="echo",
                to_port="text",
            )
        ],
        pinned_outputs=[
            PinnedOutputRequest(
                from_node="upstream",
                from_port="text",
                value=ArtifactRef.from_key(
                    artifact_id=uuid4(),
                    key=ArtifactTypeKey(TEXT.id, TEXT.schema_version),
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_compile_imports_no_plugin_source_modules() -> None:
    """Compilation resolves persisted contracts only; Plugin Python never loads."""
    import sys

    modules_before = frozenset(sys.modules)

    compiled = await _compiler(
        RecordingReleaseLookup(_release(1)),
        NoopInvoker(),
    ).compile(
        _echo_run_request(pin_revision=1),
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )

    assert isinstance(compiled.nodes[0].node, WorkspacePluginReleaseNode)
    new_module_names = [name for name in sys.modules if name not in modules_before]
    assert [name for name in new_module_names if name.startswith("grafy_plugin")] == []


@pytest.mark.asyncio
async def test_graph_pinned_to_revision_one_stays_on_it_after_two_is_published() -> (
    None
):
    lookup = RecordingReleaseLookup(_release(1), _release(2))

    compiled = await _compiler(lookup).compile(
        _echo_run_request(pin_revision=1),
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )

    pinned = compiled.nodes[0]
    assert isinstance(pinned.node, WorkspacePluginReleaseNode)
    assert pinned.node is not None
    assert pinned.node.release.revision == 1  # type: ignore[attr-defined]
    assert pinned.registration is None
    assert pinned.plugin_release is not None
    assert pinned.plugin_release.slug == "notes"
    assert pinned.plugin_release.revision == 1
    assert pinned.plugin_release.source_digest == "1" * 64


@pytest.mark.asyncio
async def test_same_operator_in_two_releases_compiles_to_distinct_identities() -> None:
    first = _release(1)
    second = _release(2)

    compiled_first = await _compiler(RecordingReleaseLookup(first, second)).compile(
        _echo_run_request(pin_revision=1),
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )
    compiled_second = await _compiler(RecordingReleaseLookup(first, second)).compile(
        _echo_run_request(pin_revision=2),
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )

    identity_first = compiled_first.nodes[0].plugin_release
    identity_second = compiled_second.nodes[0].plugin_release
    assert identity_first is not None and identity_second is not None
    assert identity_first != identity_second
    assert identity_first.revision == 1
    assert identity_second.revision == 2
    assert (
        identity_first.fingerprint_document() != identity_second.fingerprint_document()
    )


@pytest.mark.asyncio
async def test_pin_to_another_workspace_fails_without_disclosing_the_release() -> None:
    foreign = _release(4, workspace_id=OTHER_WORKSPACE_ID)
    request = RunRequest(
        nodes=[_pinned_node("echo", PluginReleasePinModel(slug="notes", revision=4))]
    )

    with pytest.raises(GraphExecutionError, match="does not exist in this workspace"):
        await _compiler(RecordingReleaseLookup(foreign)).compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_host_node_cannot_carry_a_plugin_release_pin() -> None:
    request = RunRequest(
        nodes=[
            _pinned_node(
                "input",
                PluginReleasePinModel(slug="notes", revision=1),
                operator_id="text.input",
            )
        ]
    )

    with pytest.raises(GraphExecutionError, match="host nodes cannot carry"):
        await _compiler().compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_graph_module_cannot_carry_a_plugin_release_pin() -> None:
    request = RunRequest(
        nodes=[
            _pinned_node(
                "module",
                PluginReleasePinModel(slug="notes", revision=1),
                operator_id=f"graph.module.{uuid4()}",
            )
        ]
    )

    with pytest.raises(GraphExecutionError, match="modules cannot carry"):
        await _compiler().compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_workspace_plugin_node_without_a_pin_fails_closed() -> None:
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="echo",
                operator_id="notes.echo",
                operator_version=1,
                config={"scale": 2},
            )
        ]
    )

    with pytest.raises(GraphExecutionError, match="must pin one exact Plugin release"):
        await _compiler().compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_missing_pinned_release_blocks_compilation_with_a_clear_error() -> None:
    published = _release(1)
    request = RunRequest(
        nodes=[_pinned_node("echo", PluginReleasePinModel(slug="notes", revision=7))]
    )

    with pytest.raises(GraphExecutionError, match="revision 7, which does not exist"):
        await _compiler(RecordingReleaseLookup(published)).compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_release_that_does_not_declare_the_operator_is_rejected() -> None:
    published = _release(3, nodes=(_echo_contract(),))
    request = RunRequest(
        nodes=[
            _pinned_node(
                "echo",
                PluginReleasePinModel(slug="notes", revision=3),
                operator_id="notes.other",
            )
        ]
    )

    with pytest.raises(GraphExecutionError, match="does not declare operator"):
        await _compiler(RecordingReleaseLookup(published)).compile(
            request,
            _UnusedModuleExecutor(),
            workspace_id=WORKSPACE_ID,
        )


@pytest.mark.asyncio
async def test_pinned_plugin_participates_in_ordinary_map_semantics() -> None:
    """Projections, cardinality, and MAP derivation stay on the shared path."""

    lookup = RecordingReleaseLookup(_release(1), _release(2))
    from grafy_core.artifacts import ArtifactRef, ArtifactTypeKey
    from grafy_api.v1.routes.executions.models import (
        PinnedOutputRequest,
        RunEdgeRequest,
    )

    source = RunNodeRequest(
        id="source",
        operator_id="text.split",
        operator_version=1,
        config={"separator": "|"},
    )
    echo = _pinned_node("echo", PluginReleasePinModel(slug="notes", revision=1))
    request = RunRequest(
        nodes=[source, echo],
        edges=[
            RunEdgeRequest(
                from_node="upstream",
                from_port="text",
                to_node="source",
                to_port="text",
            ),
            RunEdgeRequest(
                from_node="source",
                from_port="parts",
                to_node="echo",
                to_port="text",
                collection_mode="map",
            ),
        ],
        pinned_outputs=[
            PinnedOutputRequest(
                from_node="upstream",
                from_port="text",
                value=ArtifactRef.from_key(
                    artifact_id=uuid4(),
                    key=ArtifactTypeKey(TEXT.id, TEXT.schema_version),
                ),
            )
        ],
    )

    compiled = await _compiler(lookup).compile(
        request,
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )

    echo_compiled = next(node for node in compiled.nodes if node.request.id == "echo")
    assert echo_compiled.invocation.mode is InvocationMode.MAP
    assert echo_compiled.invocation.map_input == "text"
    assert echo_compiled.resolved_contracts.input_contract.ports["text"].accepts == (  # noqa: E501
        ArtifactTypeKey(TEXT.id, TEXT.schema_version)
    )


class OutputInvoker:
    """Invoker stub that returns one fixed host-minted output ref."""

    def __init__(self, output: ArtifactRef) -> None:
        self._output = output
        self.requests: list[PluginInvocationRequest] = []

    async def invoke(
        self,
        request: PluginInvocationRequest,
        /,
    ) -> PluginInvocationResult:
        self.requests.append(request)
        return PluginInvocationResult(outputs={"text": self._output})


class _CountingWriter:
    artifact_type = ArtifactTypeKey("test.release_pin.value", 1)

    def __init__(self) -> None:
        self.calls = 0

    async def write(self, value: object, context: object) -> ArtifactRef:
        self.calls += 1
        raise AssertionError("Plugin-owned values must not reach a host writer")


class _MemoryInvocationCache:
    def __init__(self) -> None:
        self.puts = 0

    async def get(self, workspace_id: UUID, key_sha256: str) -> object:
        return None

    async def put_if_absent(self, entry: object) -> bool:
        self.puts += 1
        return True

    async def remove_if_current(
        self,
        workspace_id: UUID,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        return False


class _FixedEdgeValues:
    def __init__(self, inputs: Mapping[str, object]) -> None:
        self._inputs = inputs

    async def assemble_inputs(
        self,
        compiled_node: object,
        incoming_edges: object,
        outputs: object,
        workflow_run_id: UUID,
        workspace_id: UUID,
    ) -> dict[str, object]:
        del compiled_node, incoming_edges, outputs, workflow_run_id, workspace_id
        return dict(self._inputs)


@pytest.mark.asyncio
async def test_release_node_executes_with_caching_disabled_and_refs_untouched() -> None:
    """An EXACT-style release path still runs fail-closed without invocation
    caching, and host-minted output refs pass through the persister."""

    from grafy_core.artifacts import ArtifactTypeSpec
    from grafy_core.plugins import NodeCachePolicy, NodeRegistration
    from grafy_core.runtime.invocation_cache import InvocationCachePort
    from grafy_api.v1.routes.executions.runtime.models import (
        CompiledGraph,
        CompiledNode,
    )

    value_key = ArtifactTypeKey(TEXT.id, TEXT.schema_version)
    outgoing = ArtifactRef.from_key(artifact_id=uuid4(), key=value_key)
    invoker = OutputInvoker(outgoing)
    lookup = RecordingReleaseLookup(_release(1), _release(2))

    compiled = await _compiler(lookup, cast(NoopInvoker | None, invoker)).compile(
        _echo_run_request(pin_revision=1),
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )
    pinned = compiled.nodes[0]
    assert isinstance(pinned.node, WorkspacePluginReleaseNode)

    # Even a registration that would normally earn caching stays fail-closed.
    forced_registration = NodeRegistration(
        node_class=type(pinned.node),
        factory=None,
        cache_policy=NodeCachePolicy.EXACT,
    )
    recompiled = CompiledNode(
        request=pinned.request,
        node=pinned.node,
        registration=forced_registration,
        resolved_contracts=pinned.resolved_contracts,
        invocation=pinned.invocation,
        artifact_type_bindings=pinned.artifact_type_bindings,
        plugin_release=pinned.plugin_release,
    )
    plan = CompiledGraph(nodes=(recompiled,), edges=(), pinned_outputs={})

    incoming_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=value_key)
    writer = _CountingWriter()
    cache = _MemoryInvocationCache()
    runtime = NodeRuntime(
        materializer=InputMaterializer(ResolverRegistry()),
        persister=OutputPersister(ArtifactWriterRegistry([writer])),
        invocation_cache=cast(InvocationCachePort, cache),
    )
    coordinator = GraphExecutionCoordinator(
        node_execution=NodeExecutionService(
            runtime=runtime,
            edge_values=cast(
                EdgeValueResolver, _FixedEdgeValues({"text": incoming_ref})
            ),
            node_secrets=UnavailableNodeSecretResolver(),
        )
    )

    def _execution() -> PreparedGraphExecution:
        return PreparedGraphExecution(
            workspace_id=WORKSPACE_ID,
            plan=plan,
            initial_outputs={},
            graph_id=None,
            graph_revision=None,
            secret_graph_id=None,
            secret_graph_revision=None,
            secret_node_ids=frozenset(),
            module_path=(),
            raise_node_errors=False,
        )

    first = await coordinator.execute(_execution())
    second = await coordinator.execute(_execution())

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    # The invoker ran for both executions; no cache entry was consulted or set.
    assert len(invoker.requests) == 2
    assert cache.puts == 0
    for request_record in invoker.requests:
        assert request_record.inputs["text"] is incoming_ref
    # Host-minted output refs pass through untouched; the writer never ran.
    first_output = first.node_results[0].outputs["text"]
    second_output = second.node_results[0].outputs["text"]
    assert isinstance(first_output, ArtifactRef)
    assert first_output == outgoing
    assert second_output == outgoing
    assert writer.calls == 0
    del value_key, ArtifactTypeSpec


@pytest.mark.asyncio
async def test_host_node_output_feeds_pinned_workspace_plugin_in_same_graph(
    tmp_path: Path,
) -> None:
    """Host and release nodes share the ordinary artifact-ref graph boundary."""

    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "objects")
    registry = build_plugin_registry(builtin_plugins(), external_plugins=())
    plugin_context = PluginRuntimeContext(
        workspace=tmp_path,
        uploads_dir=tmp_path / "uploads",
        storage=storage,
        uow=unit_of_work,
        bucket="test-artifacts",
    )
    outgoing = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=ArtifactTypeKey(TEXT.id, TEXT.schema_version),
    )
    invoker = OutputInvoker(outgoing)
    saved_graphs = SavedGraphService(_unused_saved_graph_uow, registry)
    compiler = GraphCompiler(
        plugin_registry=registry,
        plugin_context=plugin_context,
        module_catalog=GraphModuleCatalog(saved_graphs, registry),
        plugin_release_lookup=RecordingReleaseLookup(_release(1), _release(2)),
        plugin_invoker=invoker,
    )
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="host-input",
                operator_id="text.input",
                operator_version=1,
                config={"text": "from host"},
            ),
            _pinned_node(
                "plugin-echo",
                PluginReleasePinModel(slug="notes", revision=1),
            ),
        ],
        edges=[
            RunEdgeRequest(
                from_node="host-input",
                from_port="text",
                to_node="plugin-echo",
                to_port="text",
            )
        ],
    )
    plan = await compiler.compile(
        request,
        _UnusedModuleExecutor(),
        workspace_id=WORKSPACE_ID,
    )

    writer_registry = ArtifactWriterRegistry([TextValueOutputWriter(uow=unit_of_work)])
    resolver_registry = ResolverRegistry([TextValueResolver(uow=unit_of_work)])
    artifacts = ArtifactService(
        unit_of_work,
        storage,
        artifact_types={
            (spec.key.id, spec.key.schema_version): spec
            for spec in registry.artifact_types
        },
    )
    coordinator = GraphExecutionCoordinator(
        node_execution=NodeExecutionService(
            runtime=NodeRuntime(
                materializer=InputMaterializer(resolver_registry),
                persister=OutputPersister(writer_registry),
            ),
            edge_values=EdgeValueResolver(
                resolvers=resolver_registry,
                writers=writer_registry,
                artifacts=artifacts,
            ),
            node_secrets=UnavailableNodeSecretResolver(),
        )
    )
    result = await coordinator.execute(
        PreparedGraphExecution(
            workspace_id=WORKSPACE_ID,
            plan=plan,
            initial_outputs={},
            graph_id=None,
            graph_revision=None,
            secret_graph_id=None,
            secret_graph_revision=None,
            secret_node_ids=frozenset(),
            module_path=(),
            raise_node_errors=True,
        )
    )

    assert result.status == "succeeded"
    assert [node.status for node in result.node_results] == [
        "succeeded",
        "succeeded",
    ]
    assert result.node_results[0].plugin_release is None
    assert result.node_results[1].plugin_release is not None
    assert result.node_results[1].plugin_release.revision == 1
    assert len(invoker.requests) == 1
    host_ref = invoker.requests[0].inputs["text"]
    assert isinstance(host_ref, ArtifactRef)
    assert await TextValueResolver(uow=unit_of_work).resolve(
        host_ref, WORKSPACE_ID
    ) == ("from host")
    assert result.outputs["plugin-echo"]["text"] == outgoing

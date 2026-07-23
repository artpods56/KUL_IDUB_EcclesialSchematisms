from typing import override
from uuid import UUID

import pytest

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.artifacts import (
    ArtifactTypeKey,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphArtifactTypeBinding,
    SavedGraphConversion,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphInputPlug,
    SavedGraphNode,
    SavedGraphProjection,
    SavedGraphRevision,
)
from notarius_core.nodes import Node, NodeExecutionContext
from notarius_core.plugins import NodeSecretInput, Plugin, PluginRegistry

from notarius_api.v1.models import ArtifactTypeBindingModel, ArtifactTypeKeyResponse
from notarius_api.v1.routes.executions.models import (
    ArtifactConversionRequest,
    FieldProjectionRequest,
    RunEdgeRequest,
    RunInputPlugRequest,
    RunNodeRequest,
    RunRequest,
)
from notarius_api.v1.routes.executions.runtime.errors import GraphExecutionError
from notarius_api.v1.routes.executions.runtime.preflight import GraphRunPreflight


class SecretConfig(NodeConfig):
    base_url: str
    model: str = "default-model"


class EmptyInput(NodeInput):
    pass


class EmptyOutput(NodeOutput):
    pass


PREFLIGHT_PLUGIN = Plugin(
    slug="test.graph-preflight",
    title="Graph preflight test",
)


@PREFLIGHT_PLUGIN.node(
    operator_id="test.graph-preflight.secret",
    version=1,
    title="Secret node",
    secret_inputs=(
        NodeSecretInput(
            name="api_key",
            title="API key",
            config_dependencies=("base_url",),
        ),
    ),
)
class SecretNode(Node[SecretConfig, EmptyInput, EmptyOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: SecretConfig,
        _inputs: EmptyInput,
        /,
    ) -> EmptyOutput:
        raise AssertionError("Preflight tests must not execute nodes")


@PREFLIGHT_PLUGIN.node(
    operator_id="test.graph-preflight.plain",
    version=1,
    title="Plain node",
)
class PlainNode(Node[NodeConfig, EmptyInput, EmptyOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NodeConfig,
        _inputs: EmptyInput,
        /,
    ) -> EmptyOutput:
        raise AssertionError("Preflight tests must not execute nodes")


PLUGIN_REGISTRY = PluginRegistry()
PLUGIN_REGISTRY.install(PREFLIGHT_PLUGIN)
PLUGIN_REGISTRY.freeze()


class _RecordingSavedGraphs(SavedGraphService):
    def __init__(self, *revisions: SavedGraphRevision) -> None:
        self._revisions = {
            (revision.id, revision.revision): revision for revision in revisions
        }
        self.calls: list[tuple[UUID, int]] = []

    @override
    async def get_revision(
        self,
        graph_id: UUID,
        revision: int,
    ) -> SavedGraphRevision:
        self.calls.append((graph_id, revision))
        return self._revisions[(graph_id, revision)]


def _fragment_case() -> tuple[SavedGraphRevision, RunRequest]:
    bound_type = ArtifactTypeKey("test.graph-preflight.value", 1)
    saved_input = SavedGraphEdge(
        id="source-to-target",
        from_node="source",
        from_port="response",
        to_node="target",
        to_port="items",
        to_plug="payload",
        collection_mode="map",
        projection=SavedGraphProjection(path=("payload",)),
        conversion_path=(SavedGraphConversion(id="test.convert", version=2),),
    )
    graph = SavedGraph(
        name="Saved fragment",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="source",
                    operator_id="test.graph-preflight.plain",
                    operator_version=1,
                    config={},
                    position=GraphPoint(x=0, y=0),
                ),
                SavedGraphNode(
                    id="target",
                    operator_id="test.graph-preflight.plain",
                    operator_version=1,
                    config={"nested": {"strict": True}},
                    position=GraphPoint(x=100, y=0),
                    input_plugs=(SavedGraphInputPlug(id="payload", port="items"),),
                    artifact_type_bindings=(
                        SavedGraphArtifactTypeBinding(
                            variable="T",
                            artifact_type=bound_type,
                        ),
                    ),
                ),
                SavedGraphNode(
                    id="downstream",
                    operator_id="test.graph-preflight.plain",
                    operator_version=1,
                    config={},
                    position=GraphPoint(x=200, y=0),
                ),
            ),
            edges=(
                saved_input,
                SavedGraphEdge(
                    id="target-to-downstream",
                    from_node="target",
                    from_port="result",
                    to_node="downstream",
                    to_port="value",
                ),
            ),
        ),
    ).snapshot()
    request = RunRequest(
        graph_id=graph.id,
        graph_revision=graph.revision,
        nodes=[
            RunNodeRequest(
                id="target",
                operator_id="test.graph-preflight.plain",
                operator_version=1,
                config={"nested": {"strict": True}},
                input_plugs=[RunInputPlugRequest(id="payload", port="items")],
                artifact_type_bindings=[
                    ArtifactTypeBindingModel(
                        variable="T",
                        artifact_type=ArtifactTypeKeyResponse(
                            id=bound_type.id,
                            schema_version=bound_type.schema_version,
                        ),
                    )
                ],
            )
        ],
        edges=[
            RunEdgeRequest(
                from_node=saved_input.from_node,
                from_port=saved_input.from_port,
                to_node=saved_input.to_node,
                to_port=saved_input.to_port,
                to_plug=saved_input.to_plug,
                collection_mode=saved_input.collection_mode,
                projection=FieldProjectionRequest(path=["payload"]),
                conversion_path=[
                    ArtifactConversionRequest(id="test.convert", version=2)
                ],
            )
        ],
    )
    return graph, request


def _secret_revision() -> SavedGraphRevision:
    return SavedGraph(
        name="Saved secret binding",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="secret",
                    operator_id="test.graph-preflight.secret",
                    operator_version=1,
                    config={
                        "base_url": "https://provider.example/v1",
                        "model": "saved-model",
                    },
                    position=GraphPoint(x=0, y=0),
                ),
            )
        ),
    ).snapshot()


def _module_input_fragment(*, required: bool) -> tuple[SavedGraphRevision, RunRequest]:
    graph = SavedGraph(
        name="Module input fragment",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="module-input",
                    operator_id="module.input",
                    operator_version=1,
                    config={"public_name": "value", "required": required},
                    position=GraphPoint(x=0, y=0),
                ),
                SavedGraphNode(
                    id="target",
                    operator_id="test.graph-preflight.plain",
                    operator_version=1,
                    config={},
                    position=GraphPoint(x=100, y=0),
                ),
            ),
            edges=(
                SavedGraphEdge(
                    id="input-to-target",
                    from_node="module-input",
                    from_port="value",
                    to_node="target",
                    to_port="value",
                ),
            ),
        ),
    ).snapshot()
    request = RunRequest(
        graph_id=graph.id,
        graph_revision=graph.revision,
        nodes=[
            RunNodeRequest(
                id="target",
                operator_id="test.graph-preflight.plain",
                operator_version=1,
            )
        ],
    )
    return graph, request


@pytest.mark.asyncio
async def test_saved_fragment_matches_exact_node_and_incoming_edge_state() -> None:
    graph, request = _fragment_case()
    saved_graphs = _RecordingSavedGraphs(graph)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=saved_graphs,
    )

    context = await preflight.validate(request)

    assert context.secret_node_ids == frozenset()
    assert saved_graphs.calls == [(graph.id, graph.revision)]


@pytest.mark.asyncio
async def test_saved_fragment_rejects_node_and_edge_drift() -> None:
    graph, matching = _fragment_case()
    saved_graphs = _RecordingSavedGraphs(graph)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=saved_graphs,
    )
    changed_node = matching.nodes[0].model_copy(
        update={"config": {"nested": {"strict": False}}}
    )
    changed_edge = matching.edges[0].model_copy(update={"conversion_path": []})

    with pytest.raises(GraphExecutionError, match="does not match saved graph"):
        await preflight.validate(matching.model_copy(update={"nodes": [changed_node]}))
    with pytest.raises(
        GraphExecutionError,
        match="1 missing and 1 unexpected or duplicated",
    ):
        await preflight.validate(matching.model_copy(update={"edges": [changed_edge]}))


@pytest.mark.asyncio
async def test_saved_fragment_ignores_disabled_incoming_edges() -> None:
    graph, submitted = _fragment_case()
    disabled_document = graph.document.model_copy(
        update={
            "edges": (
                graph.document.edges[0].model_copy(update={"enabled": False}),
                graph.document.edges[1],
            )
        }
    )
    disabled_graph = SavedGraphRevision(
        graph_id=graph.graph_id,
        revision=graph.revision,
        name=graph.name,
        document=disabled_document,
        created_at=graph.created_at,
    )
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=_RecordingSavedGraphs(disabled_graph),
    )
    active_node = submitted.nodes[0].model_copy(update={"input_plugs": []})
    active_request = submitted.model_copy(update={"nodes": [active_node], "edges": []})

    await preflight.validate(active_request)

    with pytest.raises(
        GraphExecutionError,
        match="0 missing and 1 unexpected or duplicated",
    ):
        await preflight.validate(
            active_request.model_copy(update={"edges": submitted.edges})
        )


@pytest.mark.asyncio
async def test_optional_unpinned_module_input_edge_may_be_omitted() -> None:
    graph, request = _module_input_fragment(required=False)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=_RecordingSavedGraphs(graph),
    )

    context = await preflight.validate(request)

    assert context.secret_node_ids == frozenset()


@pytest.mark.asyncio
async def test_required_unpinned_module_input_edge_must_not_be_omitted() -> None:
    graph, request = _module_input_fragment(required=True)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=_RecordingSavedGraphs(graph),
    )

    with pytest.raises(
        GraphExecutionError,
        match="1 missing and 0 unexpected or duplicated",
    ):
        await preflight.validate(request)


@pytest.mark.asyncio
async def test_dirty_secret_context_checks_only_saved_secret_dependencies() -> None:
    graph = _secret_revision()
    saved_graphs = _RecordingSavedGraphs(graph)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=saved_graphs,
    )
    request = RunRequest(
        secret_graph_id=graph.id,
        secret_graph_revision=graph.revision,
        nodes=[
            RunNodeRequest(
                id="secret",
                operator_id="test.graph-preflight.secret",
                operator_version=1,
                config={
                    "base_url": "https://provider.example/v1",
                    "model": "dirty-unsaved-model",
                },
            ),
            RunNodeRequest(
                id="unsaved-plain",
                operator_id="test.graph-preflight.plain",
                operator_version=1,
            ),
        ],
    )

    context = await preflight.validate(request)

    assert context.secret_node_ids == frozenset({"secret"})
    assert saved_graphs.calls == [(graph.id, graph.revision)]


@pytest.mark.asyncio
async def test_secret_context_rejects_changed_dependency_binding() -> None:
    graph = _secret_revision()
    saved_graphs = _RecordingSavedGraphs(graph)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=saved_graphs,
    )

    with pytest.raises(
        GraphExecutionError,
        match="saved configuration required by secret input 'api_key'",
    ):
        await preflight.validate(
            RunRequest(
                secret_graph_id=graph.id,
                secret_graph_revision=graph.revision,
                nodes=[
                    RunNodeRequest(
                        id="secret",
                        operator_id="test.graph-preflight.secret",
                        operator_version=1,
                        config={
                            "base_url": "https://changed.example/v1",
                            "model": "saved-model",
                        },
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_secret_nodes_require_explicit_secret_context_before_graph_lookup() -> (
    None
):
    graph = _secret_revision()
    saved_graphs = _RecordingSavedGraphs(graph)
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=saved_graphs,
    )

    with pytest.raises(
        GraphExecutionError,
        match="saved secret graph context.*'alpha', 'zeta'",
    ):
        await preflight.validate(
            RunRequest(
                graph_id=graph.id,
                graph_revision=graph.revision,
                nodes=[
                    RunNodeRequest(
                        id=node_id,
                        operator_id="test.graph-preflight.secret",
                        operator_version=1,
                        config={"base_url": "https://provider.example/v1"},
                    )
                    for node_id in ("zeta", "alpha")
                ],
            )
        )
    assert saved_graphs.calls == []


@pytest.mark.asyncio
async def test_saved_context_requires_configured_saved_graph_service() -> None:
    graph = _secret_revision()
    preflight = GraphRunPreflight(
        plugin_registry=PLUGIN_REGISTRY,
        saved_graphs=None,
    )

    with pytest.raises(
        GraphExecutionError,
        match="Saved graph context is not configured for this workbench",
    ):
        await preflight.validate(
            RunRequest(
                graph_id=graph.id,
                graph_revision=graph.revision,
                nodes=[],
            )
        )

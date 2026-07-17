from collections.abc import Mapping
from pathlib import Path
from typing import Never
from uuid import uuid4

import pytest

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.artifacts import ArtifactRef, ArtifactTypeKey, InMemoryUnitOfWork
from notarius_core.domain.modules import GraphModuleDefinition
from notarius_core.nodes import NodeExecutionContext
from notarius_core.plugins import PluginRuntimeContext
from notarius_core.ports.modules import GraphModuleExecutionResult
from notarius_core.runtime.invocation import InvocationMode
from notarius_storage import LocalFileObjectStore

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.schemas.workbench import (
    ArtifactConversionRequest,
    PinnedOutputRequest,
    RunEdgeRequest,
    RunNodeRequest,
    RunRequest,
)
from notarius_api.services.execution.compiler import GraphCompiler
from notarius_api.services.modules import GraphModuleCatalog


class _UnusedModuleExecutor:
    async def execute_module(
        self,
        _definition: GraphModuleDefinition,
        _context: NodeExecutionContext,
        _inputs: Mapping[str, ArtifactRef],
        /,
    ) -> GraphModuleExecutionResult:
        raise AssertionError("Compiler test unexpectedly executed a graph module")


def _unused_saved_graph_uow() -> Never:
    raise AssertionError("Compiler test unexpectedly queried saved graphs")


def _compiler(tmp_path: Path) -> GraphCompiler:
    registry = build_plugin_registry(builtin_plugins(), external_plugins=())
    unit_of_work = InMemoryUnitOfWork()
    workspace = tmp_path / "workbench"
    uploads_dir = workspace / "uploads"
    uploads_dir.mkdir(parents=True)
    plugin_context = PluginRuntimeContext(
        workspace=workspace,
        uploads_dir=uploads_dir,
        storage=LocalFileObjectStore(workspace / "objects"),
        uow=unit_of_work,
        bucket="test-artifacts",
    )
    saved_graphs = SavedGraphService(_unused_saved_graph_uow, registry)
    return GraphCompiler(
        plugin_registry=registry,
        plugin_context=plugin_context,
        module_catalog=GraphModuleCatalog(saved_graphs, registry),
    )


@pytest.mark.asyncio
async def test_compiler_orders_nodes_and_resolves_declared_conversions(
    tmp_path: Path,
) -> None:
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="replace",
                operator_id="text.replace",
                operator_version=1,
                config={"search": "1", "replacement": "one"},
            ),
            RunNodeRequest(
                id="number",
                operator_id="arithmetic.number",
                operator_version=1,
                config={"value": 12},
            ),
        ],
        edges=[
            RunEdgeRequest(
                from_node="number",
                from_port="value",
                to_node="replace",
                to_port="text",
                conversion_path=[
                    ArtifactConversionRequest(
                        id="builtin.scalar.integer_to_text",
                        version=1,
                    )
                ],
            )
        ],
    )

    compiled = await _compiler(tmp_path).compile(request, _UnusedModuleExecutor())

    assert [node.request.id for node in compiled.nodes] == ["number", "replace"]
    assert compiled.nodes[0].registration is not None
    assert compiled.nodes[1].resolved_contracts.input_contract.ports[
        "text"
    ].accepts == (ArtifactTypeKey("scalar.text", 1))
    assert len(compiled.edges) == 1
    assert compiled.edges[0].projection is None
    assert [conversion.key.id for conversion in compiled.edges[0].conversion_path] == [
        "builtin.scalar.integer_to_text"
    ]


@pytest.mark.asyncio
async def test_compiler_derives_map_invocation_from_the_incoming_edge(
    tmp_path: Path,
) -> None:
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="replace",
                operator_id="text.replace",
                operator_version=1,
                config={"search": "a", "replacement": "A"},
            ),
            RunNodeRequest(
                id="split",
                operator_id="text.split",
                operator_version=1,
                config={"separator": "|"},
            ),
            RunNodeRequest(
                id="source",
                operator_id="text.input",
                operator_version=1,
                config={"text": "a|ba"},
            ),
        ],
        edges=[
            RunEdgeRequest(
                from_node="source",
                from_port="text",
                to_node="split",
                to_port="text",
            ),
            RunEdgeRequest(
                from_node="split",
                from_port="parts",
                to_node="replace",
                to_port="text",
                collection_mode="map",
            ),
        ],
    )

    compiled = await _compiler(tmp_path).compile(request, _UnusedModuleExecutor())

    replace = next(node for node in compiled.nodes if node.request.id == "replace")
    assert replace.invocation.mode is InvocationMode.MAP
    assert replace.invocation.map_input == "text"
    assert compiled.edges[1].request.collection_mode == "map"


@pytest.mark.asyncio
async def test_compiler_accepts_an_external_edge_only_with_its_exact_pin(
    tmp_path: Path,
) -> None:
    pinned_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=ArtifactTypeKey("scalar.text", 1),
    )
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="replace",
                operator_id="text.replace",
                operator_version=1,
                config={"search": "a", "replacement": "A"},
            )
        ],
        edges=[
            RunEdgeRequest(
                from_node="upstream",
                from_port="text",
                to_node="replace",
                to_port="text",
            )
        ],
        pinned_outputs=[
            PinnedOutputRequest(
                from_node="upstream",
                from_port="text",
                value=pinned_ref,
            )
        ],
    )

    compiled = await _compiler(tmp_path).compile(request, _UnusedModuleExecutor())

    assert compiled.pinned_outputs == {("upstream", "text"): pinned_ref}
    assert [node.request.id for node in compiled.nodes] == ["replace"]

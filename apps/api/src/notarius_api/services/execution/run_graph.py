"""Application use case for top-level and nested graph execution."""

from collections.abc import Mapping

from notarius_core.artifacts import ArtifactRef
from notarius_core.domain.modules import MODULE_BOUNDARY_PORT, GraphModuleDefinition
from notarius_core.nodes import NodeExecutionContext
from notarius_core.ports.modules import GraphModuleExecutionResult

from notarius_api.schemas.workbench import (
    ArtifactConversionRequest,
    ArtifactTypeBindingModel,
    ArtifactTypeKeyResponse,
    FieldProjectionRequest,
    PinnedOutputRequest,
    RunEdgeRequest,
    RunInputPlugRequest,
    RunNodeRequest,
    RunRequest,
)
from notarius_api.services.execution.compiler import GraphCompiler
from notarius_api.services.execution.engine import (
    GraphExecutionEngine,
    PreparedGraphExecution,
)
from notarius_api.services.execution.errors import GraphExecutionError
from notarius_api.services.execution.models import GraphExecutionResult
from notarius_api.services.execution.preflight import GraphRunPreflight
from notarius_api.services.materializations import MaterializationService


class RunGraph:
    """Coordinate the collaborators required to execute a graph run."""

    def __init__(
        self,
        *,
        preflight: GraphRunPreflight,
        compiler: GraphCompiler,
        engine: GraphExecutionEngine,
        materializations: MaterializationService,
    ) -> None:
        self._preflight = preflight
        self._compiler = compiler
        self._engine = engine
        self._materializations = materializations

    async def run(self, request: RunRequest) -> GraphExecutionResult:
        return await self._execute(
            request,
            module_path=(),
            persist_materializations=True,
            validate_materialized_pins=True,
            raise_node_errors=False,
        )

    async def execute_module(
        self,
        definition: GraphModuleDefinition,
        context: NodeExecutionContext,
        inputs: Mapping[str, ArtifactRef],
        /,
    ) -> GraphModuleExecutionResult:
        graph_id = definition.reference.graph_id
        graph_revision = definition.reference.revision
        graph_path_item = definition.reference.module_path_item
        if graph_path_item in context.module_path:
            rendered_path = " -> ".join((*context.module_path, graph_path_item))
            raise GraphExecutionError(
                f"Graph module cycle detected while entering {definition.name!r} "
                f"at revision {graph_revision}: {rendered_path}"
            )

        input_boundary_ids = {port.boundary_node_id for port in definition.input_ports}
        executed_nodes = [
            node
            for node in definition.document.nodes
            if node.id not in input_boundary_ids
        ]
        executed_node_ids = {node.id for node in executed_nodes}
        request = RunRequest(
            nodes=[
                RunNodeRequest(
                    id=node.id,
                    operator_id=node.operator_id,
                    operator_version=node.operator_version,
                    config=node.config_dict(),
                    input_plugs=[
                        RunInputPlugRequest(id=plug.id, port=plug.port)
                        for plug in node.input_plugs
                    ],
                    artifact_type_bindings=[
                        ArtifactTypeBindingModel(
                            variable=binding.variable,
                            artifact_type=ArtifactTypeKeyResponse(
                                id=binding.artifact_type.id,
                                schema_version=binding.artifact_type.schema_version,
                            ),
                        )
                        for binding in node.artifact_type_bindings
                    ],
                )
                for node in executed_nodes
            ],
            edges=[
                RunEdgeRequest(
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    to_node=edge.to_node,
                    to_port=edge.to_port,
                    to_plug=edge.to_plug,
                    projection=(
                        FieldProjectionRequest(path=list(edge.projection.path))
                        if edge.projection is not None
                        else None
                    ),
                    conversion_path=[
                        ArtifactConversionRequest(
                            id=conversion.id,
                            version=conversion.version,
                        )
                        for conversion in edge.conversion_path
                    ],
                    collection_mode=edge.collection_mode,
                )
                for edge in definition.document.edges
                if edge.to_node in executed_node_ids
            ],
            pinned_outputs=[
                PinnedOutputRequest(
                    from_node=port.boundary_node_id,
                    from_port=MODULE_BOUNDARY_PORT,
                    value=inputs[port.name],
                )
                for port in definition.input_ports
            ],
            graph_id=graph_id,
            graph_revision=graph_revision,
            secret_graph_id=graph_id,
            secret_graph_revision=graph_revision,
        )
        execution = await self._execute(
            request,
            module_path=(*context.module_path, graph_path_item),
            persist_materializations=False,
            validate_materialized_pins=False,
            raise_node_errors=True,
        )

        outputs: dict[str, ArtifactRef] = {}
        for port in definition.output_ports:
            boundary_outputs = execution.outputs.get(port.boundary_node_id)
            value = (
                boundary_outputs.get(MODULE_BOUNDARY_PORT)
                if boundary_outputs is not None
                else None
            )
            if value is None:
                raise GraphExecutionError(
                    f"Graph module {definition.name!r} revision {graph_revision} "
                    f"did not produce public output {port.name!r} at boundary "
                    f"node {port.boundary_node_id!r}"
                )
            if not isinstance(value, ArtifactRef):
                raise GraphExecutionError(
                    f"Graph module {definition.name!r} revision {graph_revision} "
                    f"public output {port.name!r} produced a sequence; module "
                    "boundary ports must be scalar"
                )
            outputs[port.name] = value
        return GraphModuleExecutionResult(outputs=outputs)

    async def _execute(
        self,
        request: RunRequest,
        *,
        module_path: tuple[str, ...],
        persist_materializations: bool,
        validate_materialized_pins: bool,
        raise_node_errors: bool,
    ) -> GraphExecutionResult:
        run_context = await self._preflight.validate(request)
        plan = await self._compiler.compile(request, self)
        if (
            validate_materialized_pins
            and request.graph_id is not None
            and request.graph_revision is not None
        ):
            await self._materializations.validate_latest_pins(
                request.graph_id,
                request.graph_revision,
                plan.pinned_outputs,
            )
        initial_outputs = await self._materializations.resolve_pinned_outputs(
            plan.pinned_outputs
        )
        execution = await self._engine.execute(
            PreparedGraphExecution(
                plan=plan,
                initial_outputs=initial_outputs,
                graph_id=request.graph_id,
                graph_revision=request.graph_revision,
                secret_graph_id=request.secret_graph_id,
                secret_graph_revision=request.secret_graph_revision,
                secret_node_ids=frozenset(run_context.secret_node_ids),
                module_path=module_path,
                raise_node_errors=raise_node_errors,
            )
        )
        if (
            persist_materializations
            and request.graph_id is not None
            and request.graph_revision is not None
        ):
            await self._materializations.persist_execution(
                request.graph_id,
                request.graph_revision,
                execution,
            )
        return execution


__all__ = ["RunGraph"]

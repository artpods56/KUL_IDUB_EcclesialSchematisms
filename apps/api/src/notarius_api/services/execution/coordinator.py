"""Framework-neutral coordination of one prepared graph execution."""

from uuid import UUID

from notarius_core.domain.artifact_outputs import ArtifactOutputValue

from notarius_api.services.execution.engine import (
    ExecutionTaskRunner,
    PreparedGraphExecution,
)
from notarius_api.services.execution.errors import GraphExecutionError
from notarius_api.services.execution.models import (
    GraphExecutionResult,
    NodeExecutionResult,
)
from notarius_api.services.execution.node_execution import NodeExecutionService


class GraphExecutionCoordinator:
    """Visit compiled nodes and assemble their graph-level execution result."""

    def __init__(self, *, node_execution: NodeExecutionService) -> None:
        self._node_execution = node_execution

    async def execute(
        self,
        execution: PreparedGraphExecution,
        task_runner: ExecutionTaskRunner,
        /,
    ) -> GraphExecutionResult:
        plan = execution.plan
        outputs: dict[str, dict[str, ArtifactOutputValue]] = {
            node_id: dict(node_outputs)
            for node_id, node_outputs in execution.initial_outputs.items()
        }
        incoming_edges = {
            compiled_node.request.id: tuple(
                edge
                for edge in plan.edges
                if edge.request.to_node == compiled_node.request.id
            )
            for compiled_node in plan.nodes
        }
        failed: set[str] = set()
        succeeded: set[str] = set()
        node_results: list[NodeExecutionResult] = []

        for compiled_node in plan.nodes:
            if execution.control is not None:
                execution.control.check_cancelled()
            node_request = compiled_node.request
            node_edges = incoming_edges[node_request.id]
            upstream_node_ids = {edge.request.from_node for edge in node_edges}
            if upstream_node_ids & failed:
                failed.add(node_request.id)
                node_results.append(
                    NodeExecutionResult(
                        node_id=node_request.id,
                        status="skipped",
                        error=None,
                        outputs={},
                    )
                )
                continue

            async def execute_node(
                node_run_id: UUID,
            ) -> dict[str, ArtifactOutputValue]:
                return await self._node_execution.execute(
                    execution=execution,
                    compiled_node=compiled_node,
                    incoming_edges=node_edges,
                    outputs=outputs,
                    task_runner=task_runner,
                    node_run_id=node_run_id,
                )

            control = execution.control
            tracks_outer_progress = not execution.module_path
            if control is not None and tracks_outer_progress:
                control.start_outer_node(node_request.id)
            try:
                node_outputs = await task_runner.run_node(
                    compiled_node,
                    frozenset(upstream_node_ids & succeeded),
                    execute_node,
                )
            except Exception as exc:
                if execution.raise_node_errors:
                    graph_context = "nested graph"
                    if execution.graph_id is not None:
                        graph_context = (
                            f"graph {execution.graph_id}@{execution.graph_revision}"
                        )
                    raise GraphExecutionError(
                        f"{graph_context} node {node_request.id!r} "
                        f"({node_request.operator_id}@"
                        f"{node_request.operator_version}) failed"
                    ) from exc
                failed.add(node_request.id)
                node_results.append(
                    NodeExecutionResult(
                        node_id=node_request.id,
                        status="failed",
                        error=_render_exception_chain(exc),
                        outputs={},
                    )
                )
                continue
            finally:
                if control is not None and tracks_outer_progress:
                    control.finish_outer_node(node_request.id)

            copied_node_outputs = dict(node_outputs)
            outputs[node_request.id] = copied_node_outputs
            succeeded.add(node_request.id)
            node_results.append(
                NodeExecutionResult(
                    node_id=node_request.id,
                    status="succeeded",
                    error=None,
                    outputs=copied_node_outputs,
                )
            )

        status = "failed" if failed else "succeeded"
        return GraphExecutionResult(
            workflow_run_id=task_runner.workflow_run_id,
            status=status,
            node_results=tuple(node_results),
            outputs=outputs,
        )


def _render_exception_chain(exception: BaseException) -> str:
    rendered: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen and len(rendered) < 12:
        seen.add(id(current))
        rendered.append(f"{type(current).__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = None if current.__suppress_context__ else current.__context__
    return " <- caused by ".join(rendered)


__all__ = ["GraphExecutionCoordinator"]

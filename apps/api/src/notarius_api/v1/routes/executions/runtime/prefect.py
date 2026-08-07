"""Local Prefect adapter for prepared graph execution."""

import asyncio
from collections.abc import Awaitable, Mapping
from typing import cast, final
from uuid import UUID

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.client.schemas.objects import State
from prefect.context import FlowRunContext, TaskRunContext
from prefect.states import Cancelled, Failed

from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.runtime.invocation import InvocationMode
from notarius_core.runtime.persistence import PersistedNodeOutput

from .coordinator import GraphExecutionCoordinator
from .engine import (
    MapItemExecutionOperation,
    NodeExecutionOperation,
    PreparedGraphExecution,
)
from .errors import GraphExecutionError
from .models import (
    CompiledNode,
    GraphExecutionResult,
)


@final
class PrefectExecutionTaskRunner:
    """Represent logical nodes and scalar MAP items as local Prefect tasks."""

    def __init__(
        self,
        *,
        workflow_run_id: UUID,
        task_retries: int,
        task_retry_delay_seconds: float,
    ) -> None:
        if task_retries < 0:
            raise ValueError("Prefect task retries must not be negative")
        if task_retry_delay_seconds < 0:
            raise ValueError("Prefect task retry delay must not be negative")
        self._workflow_run_id = workflow_run_id
        self._task_retries = task_retries
        self._task_retry_delay_seconds = task_retry_delay_seconds
        self._node_states: dict[
            str,
            State[Mapping[str, ArtifactOutputValue]],
        ] = {}

    @property
    def workflow_run_id(self) -> UUID:
        return self._workflow_run_id

    async def run_node(
        self,
        compiled_node: CompiledNode,
        upstream_node_ids: frozenset[str],
        operation: NodeExecutionOperation,
        /,
    ) -> Mapping[str, ArtifactOutputValue]:
        node_request = compiled_node.request
        task_retries = self._task_retries
        if compiled_node.invocation.mode is InvocationMode.MAP:
            task_retries = 0

        @task(
            name=(
                f"node:{node_request.id}:"
                f"{node_request.operator_id}@{node_request.operator_version}"
            ),
            persist_result=False,
            cache_policy=NO_CACHE,
            retries=task_retries,
            retry_delay_seconds=self._task_retry_delay_seconds,
        )
        async def execute_node() -> Mapping[str, ArtifactOutputValue]:
            task_context = TaskRunContext.get()
            if task_context is None:
                raise GraphExecutionError(
                    f"Prefect task context is unavailable for node {node_request.id!r}"
                )
            return await operation(task_context.task_run.id)

        upstream_states = [
            self._node_states[node_id]
            for node_id in upstream_node_ids
            if node_id in self._node_states
        ]
        pending_state = execute_node(
            return_state=True,
            wait_for=upstream_states,
        )
        node_state = await cast(
            Awaitable[State[Mapping[str, ArtifactOutputValue]]],
            pending_state,
        )
        self._node_states[node_request.id] = node_state
        result = await node_state.aresult()
        if isinstance(result, Exception):
            raise result
        return result

    async def run_map_item(
        self,
        compiled_node: CompiledNode,
        index: int,
        operation: MapItemExecutionOperation,
        /,
    ) -> PersistedNodeOutput:
        node_request = compiled_node.request

        @task(
            name=(
                f"node:{node_request.id}:map:{index}:"
                f"{node_request.operator_id}@{node_request.operator_version}"
            ),
            persist_result=False,
            cache_policy=NO_CACHE,
            retries=self._task_retries,
            retry_delay_seconds=self._task_retry_delay_seconds,
        )
        async def execute_map_item() -> PersistedNodeOutput:
            task_context = TaskRunContext.get()
            if task_context is None:
                raise GraphExecutionError(
                    f"Prefect task context is unavailable for node "
                    f"{node_request.id!r} MAP item {index}"
                )
            return await operation(task_context.task_run.id)

        return await execute_map_item()


@final
class PrefectExecutionEngine:
    """Execute one prepared graph as a local Prefect flow."""

    def __init__(
        self,
        *,
        coordinator: GraphExecutionCoordinator,
        task_retries: int = 0,
        task_retry_delay_seconds: float = 0,
    ) -> None:
        if task_retries < 0:
            raise ValueError("Prefect task retries must not be negative")
        if task_retry_delay_seconds < 0:
            raise ValueError("Prefect task retry delay must not be negative")
        self._coordinator = coordinator
        self._task_retries = task_retries
        self._task_retry_delay_seconds = task_retry_delay_seconds

    async def execute(
        self,
        execution: PreparedGraphExecution,
        /,
    ) -> GraphExecutionResult:
        flow_run_name = "draft-graph"
        if execution.graph_id is not None:
            flow_run_name = f"graph-{execution.graph_id}@{execution.graph_revision}"

        # A zero-parameter callback is intentional: Prefect records orchestration
        # state, while live runtime collaborators and secret-bearing node state stay
        # inside this process and never enter Prefect parameters or persisted results.
        @flow(
            name="notarius-graph-execution",
            flow_run_name=flow_run_name,
            validate_parameters=False,
            persist_result=False,
            retries=0,
        )
        async def execute_graph_flow() -> (
            GraphExecutionResult | State[GraphExecutionResult]
        ):
            flow_context = FlowRunContext.get()
            if flow_context is None or flow_context.flow_run is None:
                raise GraphExecutionError(
                    "Prefect flow context is unavailable for graph execution"
                )
            task_runner = PrefectExecutionTaskRunner(
                workflow_run_id=flow_context.flow_run.id,
                task_retries=self._task_retries,
                task_retry_delay_seconds=self._task_retry_delay_seconds,
            )
            try:
                result = await self._coordinator.execute(execution, task_runner)
            except asyncio.CancelledError:
                control = execution.control
                if control is None or not control.cancel_requested:
                    raise
                return cast(
                    State[GraphExecutionResult],
                    Cancelled(message="Notarius graph execution was cancelled"),
                )
            if result.status == "failed":
                return cast(
                    State[GraphExecutionResult],
                    Failed(
                        message="One or more Notarius graph nodes failed",
                        data=result,
                    ),
                )
            return result

        pending_flow_state = execute_graph_flow(return_state=True)
        flow_state = await cast(
            Awaitable[State[GraphExecutionResult]],
            pending_flow_state,
        )
        result = cast(
            object,
            await flow_state.aresult(raise_on_failure=False),
        )
        if isinstance(result, GraphExecutionResult):
            return result
        if isinstance(result, Exception):
            graph_context = "draft graph"
            if execution.graph_id is not None:
                graph_context = f"graph {execution.graph_id}@{execution.graph_revision}"
            raise GraphExecutionError(
                f"Local Prefect execution for {graph_context} failed before "
                "producing a graph result"
            ) from result
        raise GraphExecutionError(
            "Local Prefect execution produced an unexpected flow result of "
            f"type {type(result).__name__}"
        )


__all__ = ["PrefectExecutionEngine", "PrefectExecutionTaskRunner"]

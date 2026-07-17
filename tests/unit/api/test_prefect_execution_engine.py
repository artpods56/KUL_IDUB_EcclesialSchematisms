import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Annotated, ClassVar, final, override
from uuid import UUID

import pytest
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.client.schemas.objects import TaskRun
from prefect.settings import (
    PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY,
    PREFECT_HOME,
    PREFECT_SERVER_ANALYTICS_ENABLED,
    temporary_settings,
)
from prefect.testing.utilities import prefect_test_harness
from pydantic import Field, StrictInt, StrictStr

from notarius_core.artifacts import (
    ArtifactRefSequence,
    InMemoryUnitOfWork,
    NoConfig,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.arithmetic import INTEGER_VALUE, IntegerValueResolver
from notarius_core.plugins import Plugin

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.schemas.workbench import (
    RunEdgeRequest,
    RunNodeRequest,
    RunRequest,
)
from notarius_api.services.composition import (
    WorkbenchComponents,
    build_workbench_components,
)


@dataclass(frozen=True, slots=True)
class ObservedInvocation:
    workflow_run_id: UUID
    node_run_id: UUID
    node_id: str
    invocation_index: int | None


class ControlledIntegerConfig(NodeConfig):
    failures_before_success: StrictInt = Field(default=0, ge=0)
    opaque_marker: StrictStr = ""


class ControlledIntegerInput(NodeInput):
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class ControlledIntegerOutput(NodeOutput):
    result: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


class BlockingIntegerInput(NodeInput):
    pass


class BlockingIntegerOutput(NodeOutput):
    result: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


class ConcurrentIntegerInput(NodeInput):
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class ConcurrentIntegerOutput(NodeOutput):
    result: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


PREFECT_BEHAVIOR_PLUGIN = Plugin(
    slug="test.prefect-behavior",
    title="Prefect behavior test plugin",
)


@PREFECT_BEHAVIOR_PLUGIN.node(
    operator_id="test.prefect.controlled_integer",
    version=1,
    title="Controlled integer",
)
@final
class ControlledIntegerNode(
    Node[
        ControlledIntegerConfig,
        ControlledIntegerInput,
        ControlledIntegerOutput,
    ]
):
    invocations: ClassVar[list[ObservedInvocation]] = []
    attempts: ClassVar[dict[tuple[str, int | None], int]] = {}

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        config: ControlledIntegerConfig,
        inputs: ControlledIntegerInput,
        /,
    ) -> ControlledIntegerOutput:
        assert context.workflow_run_id is not None
        assert context.node_run_id is not None
        assert context.node_id is not None
        invocation = ObservedInvocation(
            workflow_run_id=context.workflow_run_id,
            node_run_id=context.node_run_id,
            node_id=context.node_id,
            invocation_index=context.invocation_index,
        )
        self.invocations.append(invocation)
        attempt_key = (invocation.node_id, invocation.invocation_index)
        attempt = self.attempts.get(attempt_key, 0) + 1
        self.attempts[attempt_key] = attempt
        if attempt <= config.failures_before_success:
            raise RuntimeError(
                f"Controlled failure for {invocation.node_id!r} on attempt {attempt}"
            )
        return ControlledIntegerOutput(result=inputs.left + inputs.right)


@PREFECT_BEHAVIOR_PLUGIN.node(
    operator_id="test.prefect.blocking_integer",
    version=1,
    title="Blocking integer",
)
@final
class BlockingIntegerNode(
    Node[NoConfig, BlockingIntegerInput, BlockingIntegerOutput]
):
    started: ClassVar[asyncio.Event | None] = None
    workflow_run_id: ClassVar[UUID | None] = None

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: BlockingIntegerInput,
        /,
    ) -> BlockingIntegerOutput:
        started = self.started
        assert started is not None
        assert context.workflow_run_id is not None
        type(self).workflow_run_id = context.workflow_run_id
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@PREFECT_BEHAVIOR_PLUGIN.node(
    operator_id="test.prefect.concurrent_integer",
    version=1,
    title="Concurrent integer",
)
@final
class ConcurrentIntegerNode(
    Node[NoConfig, ConcurrentIntegerInput, ConcurrentIntegerOutput]
):
    release_gates: ClassVar[tuple[asyncio.Event, ...]] = ()
    started: ClassVar[tuple[asyncio.Event, ...]] = ()
    completed: ClassVar[tuple[asyncio.Event, ...]] = ()
    completed_indexes: ClassVar[list[int]] = []
    active_count: ClassVar[int] = 0
    max_active_count: ClassVar[int] = 0

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ConcurrentIntegerInput,
        /,
    ) -> ConcurrentIntegerOutput:
        index = context.invocation_index
        assert index is not None
        node_type = type(self)
        node_type.active_count += 1
        node_type.max_active_count = max(
            node_type.max_active_count,
            node_type.active_count,
        )
        node_type.started[index].set()
        try:
            await node_type.release_gates[index].wait()
            node_type.completed_indexes.append(index)
            node_type.completed[index].set()
            return ConcurrentIntegerOutput(result=inputs.left + inputs.right)
        finally:
            node_type.active_count -= 1


@pytest.fixture(scope="module", autouse=True)
def prefect_harness(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    prefect_home = tmp_path_factory.mktemp("prefect-home")
    with temporary_settings(
        updates={
            PREFECT_HOME: prefect_home,
            PREFECT_SERVER_ANALYTICS_ENABLED: False,
            PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY: False,
        }
    ):
        with prefect_test_harness(server_startup_timeout=60):
            yield


@pytest.fixture(autouse=True)
def reset_controlled_integer_node() -> Iterator[None]:
    ControlledIntegerNode.invocations.clear()
    ControlledIntegerNode.attempts.clear()
    BlockingIntegerNode.started = None
    BlockingIntegerNode.workflow_run_id = None
    ConcurrentIntegerNode.release_gates = ()
    ConcurrentIntegerNode.started = ()
    ConcurrentIntegerNode.completed = ()
    ConcurrentIntegerNode.completed_indexes.clear()
    ConcurrentIntegerNode.active_count = 0
    ConcurrentIntegerNode.max_active_count = 0
    yield
    ControlledIntegerNode.invocations.clear()
    ControlledIntegerNode.attempts.clear()
    BlockingIntegerNode.started = None
    BlockingIntegerNode.workflow_run_id = None
    ConcurrentIntegerNode.release_gates = ()
    ConcurrentIntegerNode.started = ()
    ConcurrentIntegerNode.completed = ()
    ConcurrentIntegerNode.completed_indexes.clear()
    ConcurrentIntegerNode.active_count = 0
    ConcurrentIntegerNode.max_active_count = 0


def build_prefect_components(
    workspace: Path,
    *,
    map_max_concurrency: int = 4,
    task_retries: int = 0,
) -> tuple[WorkbenchComponents, InMemoryUnitOfWork]:
    unit_of_work = InMemoryUnitOfWork()
    registry = build_plugin_registry(
        (*builtin_plugins(), PREFECT_BEHAVIOR_PLUGIN),
        external_plugins=(),
    )
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="prefect",
        map_max_concurrency=map_max_concurrency,
        prefect_task_retries=task_retries,
        prefect_task_retry_delay_seconds=0,
        workspace=workspace,
        unit_of_work=unit_of_work,
    )
    return components, unit_of_work


async def read_settled_task_runs(
    workflow_run_id: UUID,
    *,
    expected_count: int,
) -> list[TaskRun]:
    deadline = monotonic() + 10
    task_runs: list[TaskRun] = []
    async with get_client(sync_client=False) as client:
        while monotonic() < deadline:
            task_runs = await client.read_task_runs(
                flow_run_filter=FlowRunFilter(
                    id=FlowRunFilterId(any_=[workflow_run_id])
                )
            )
            if len(task_runs) == expected_count and all(
                task_run.state is not None and task_run.state.is_final()
                for task_run in task_runs
            ):
                return task_runs
            await asyncio.sleep(0.05)
    return task_runs


@pytest.mark.asyncio
async def test_prefect_map_items_overlap_and_reduce_in_source_order(
    tmp_path: Path,
) -> None:
    release_gates = (asyncio.Event(), asyncio.Event())
    ConcurrentIntegerNode.release_gates = release_gates
    ConcurrentIntegerNode.started = (asyncio.Event(), asyncio.Event())
    ConcurrentIntegerNode.completed = (asyncio.Event(), asyncio.Event())
    components, unit_of_work = build_prefect_components(
        tmp_path / "workbench",
        map_max_concurrency=2,
    )
    run_task = asyncio.create_task(
        components.run_graph.run(
            RunRequest(
                nodes=[
                    RunNodeRequest(
                        id="sequence",
                        operator_id="arithmetic.integer_sequence",
                        operator_version=1,
                        config={"start": 1, "count": 2, "step": 1},
                    ),
                    RunNodeRequest(
                        id="ten",
                        operator_id="arithmetic.number",
                        operator_version=1,
                        config={"value": 10},
                    ),
                    RunNodeRequest(
                        id="mapped",
                        operator_id="test.prefect.concurrent_integer",
                        operator_version=1,
                    ),
                ],
                edges=[
                    RunEdgeRequest(
                        from_node="sequence",
                        from_port="values",
                        to_node="mapped",
                        to_port="left",
                        collection_mode="map",
                    ),
                    RunEdgeRequest(
                        from_node="ten",
                        from_port="value",
                        to_node="mapped",
                        to_port="right",
                    ),
                ],
            )
        )
    )
    try:
        async with asyncio.timeout(10):
            await ConcurrentIntegerNode.started[0].wait()
            await ConcurrentIntegerNode.started[1].wait()
        assert ConcurrentIntegerNode.active_count == 2

        release_gates[1].set()
        async with asyncio.timeout(10):
            await ConcurrentIntegerNode.completed[1].wait()
        release_gates[0].set()
        result = await run_task
    finally:
        for gate in release_gates:
            gate.set()
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)

    assert result.status == "succeeded"
    assert ConcurrentIntegerNode.max_active_count == 2
    assert ConcurrentIntegerNode.completed_indexes == [1, 0]
    mapped_output = result.outputs["mapped"]["result"]
    assert isinstance(mapped_output, ArtifactRefSequence)
    resolver = IntegerValueResolver(uow=unit_of_work)
    assert [
        await resolver.resolve(item_ref) for item_ref in mapped_output.item_refs
    ] == [11, 12]


@pytest.mark.asyncio
async def test_prefect_execution_preserves_map_order_and_uses_prefect_run_ids(
    tmp_path: Path,
) -> None:
    sensitive_marker = "prefect-must-not-store-this-value"
    components, unit_of_work = build_prefect_components(tmp_path / "workbench")
    result = await components.run_graph.run(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="sequence",
                    operator_id="arithmetic.integer_sequence",
                    operator_version=1,
                    config={"start": 1, "count": 3, "step": 1},
                ),
                RunNodeRequest(
                    id="ten",
                    operator_id="arithmetic.number",
                    operator_version=1,
                    config={"value": 10},
                ),
                RunNodeRequest(
                    id="once",
                    operator_id="test.prefect.controlled_integer",
                    operator_version=1,
                    config={"opaque_marker": sensitive_marker},
                ),
                RunNodeRequest(
                    id="mapped",
                    operator_id="test.prefect.controlled_integer",
                    operator_version=1,
                    config={"opaque_marker": sensitive_marker},
                ),
            ],
            edges=[
                RunEdgeRequest(
                    from_node="ten",
                    from_port="value",
                    to_node="once",
                    to_port="left",
                ),
                RunEdgeRequest(
                    from_node="ten",
                    from_port="value",
                    to_node="once",
                    to_port="right",
                ),
                RunEdgeRequest(
                    from_node="sequence",
                    from_port="values",
                    to_node="mapped",
                    to_port="left",
                    collection_mode="map",
                ),
                RunEdgeRequest(
                    from_node="once",
                    from_port="result",
                    to_node="mapped",
                    to_port="right",
                ),
            ],
        )
    )

    assert result.status == "succeeded"
    mapped_output = result.outputs["mapped"]["result"]
    assert isinstance(mapped_output, ArtifactRefSequence)
    integer_resolver = IntegerValueResolver(uow=unit_of_work)
    assert [
        await integer_resolver.resolve(item_ref)
        for item_ref in mapped_output.item_refs
    ] == [21, 22, 23]

    once_invocations = [
        invocation
        for invocation in ControlledIntegerNode.invocations
        if invocation.node_id == "once"
    ]
    mapped_invocations = [
        invocation
        for invocation in ControlledIntegerNode.invocations
        if invocation.node_id == "mapped"
    ]
    assert [invocation.invocation_index for invocation in once_invocations] == [None]
    assert [
        invocation.invocation_index for invocation in mapped_invocations
    ] == [0, 1, 2]
    assert all(
        invocation.workflow_run_id == result.workflow_run_id
        for invocation in ControlledIntegerNode.invocations
    )
    observed_node_run_ids = {
        invocation.node_run_id for invocation in ControlledIntegerNode.invocations
    }
    assert len(observed_node_run_ids) == 4

    all_task_runs = await read_settled_task_runs(
        result.workflow_run_id,
        expected_count=7,
    )
    task_runs_by_id = {task_run.id: task_run for task_run in all_task_runs}
    async with get_client(sync_client=False) as client:
        flow_run = await client.read_flow_run(result.workflow_run_id)

    assert flow_run.id == result.workflow_run_id
    assert flow_run.state is not None
    assert flow_run.state.is_completed()
    assert observed_node_run_ids <= task_runs_by_id.keys()
    assert all(
        task_runs_by_id[node_run_id].flow_run_id == result.workflow_run_id
        for node_run_id in observed_node_run_ids
    )
    orchestration_payload = repr(
        (
            flow_run.parameters,
            flow_run.state.data,
            [task_run.task_inputs for task_run in all_task_runs],
            [
                task_run.state.data if task_run.state is not None else None
                for task_run in all_task_runs
            ],
        )
    )
    assert sensitive_marker not in orchestration_payload


@pytest.mark.asyncio
async def test_failed_node_fails_prefect_flow_and_skips_downstream(
    tmp_path: Path,
) -> None:
    components, _unit_of_work = build_prefect_components(tmp_path / "workbench")
    result = await components.run_graph.run(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="left",
                    operator_id="arithmetic.number",
                    operator_version=1,
                    config={"value": 7},
                ),
                RunNodeRequest(
                    id="right",
                    operator_id="arithmetic.number",
                    operator_version=1,
                    config={"value": 5},
                ),
                RunNodeRequest(
                    id="failed",
                    operator_id="test.prefect.controlled_integer",
                    operator_version=1,
                    config={"failures_before_success": 1},
                ),
                RunNodeRequest(
                    id="downstream",
                    operator_id="test.prefect.controlled_integer",
                    operator_version=1,
                ),
            ],
            edges=[
                RunEdgeRequest(
                    from_node="left",
                    from_port="value",
                    to_node="failed",
                    to_port="left",
                ),
                RunEdgeRequest(
                    from_node="right",
                    from_port="value",
                    to_node="failed",
                    to_port="right",
                ),
                RunEdgeRequest(
                    from_node="failed",
                    from_port="result",
                    to_node="downstream",
                    to_port="left",
                ),
                RunEdgeRequest(
                    from_node="right",
                    from_port="value",
                    to_node="downstream",
                    to_port="right",
                ),
            ],
        )
    )

    assert result.status == "failed"
    assert {
        node_result.node_id: node_result.status
        for node_result in result.node_results
    } == {
        "left": "succeeded",
        "right": "succeeded",
        "failed": "failed",
        "downstream": "skipped",
    }
    assert [
        invocation.node_id for invocation in ControlledIntegerNode.invocations
    ] == ["failed"]
    failed_node_run_id = ControlledIntegerNode.invocations[0].node_run_id

    task_runs = await read_settled_task_runs(
        result.workflow_run_id,
        expected_count=3,
    )
    task_runs_by_id = {task_run.id: task_run for task_run in task_runs}
    async with get_client(sync_client=False) as client:
        flow_run = await client.read_flow_run(result.workflow_run_id)
    failed_task_run = task_runs_by_id[failed_node_run_id]

    assert flow_run.state is not None
    assert flow_run.state.is_failed()
    assert failed_task_run.state is not None
    assert failed_task_run.state.is_failed()
    assert len(task_runs) == 3


@pytest.mark.asyncio
async def test_prefect_task_retry_reuses_task_run_and_records_attempt_count(
    tmp_path: Path,
) -> None:
    components, _unit_of_work = build_prefect_components(
        tmp_path / "workbench",
        task_retries=1,
    )
    result = await components.run_graph.run(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="left",
                    operator_id="arithmetic.number",
                    operator_version=1,
                    config={"value": 7},
                ),
                RunNodeRequest(
                    id="right",
                    operator_id="arithmetic.number",
                    operator_version=1,
                    config={"value": 5},
                ),
                RunNodeRequest(
                    id="retried",
                    operator_id="test.prefect.controlled_integer",
                    operator_version=1,
                    config={"failures_before_success": 1},
                ),
            ],
            edges=[
                RunEdgeRequest(
                    from_node="left",
                    from_port="value",
                    to_node="retried",
                    to_port="left",
                ),
                RunEdgeRequest(
                    from_node="right",
                    from_port="value",
                    to_node="retried",
                    to_port="right",
                ),
            ],
        )
    )

    assert result.status == "succeeded"
    assert len(ControlledIntegerNode.invocations) == 2
    assert ControlledIntegerNode.attempts == {("retried", None): 2}
    node_run_ids = {
        invocation.node_run_id for invocation in ControlledIntegerNode.invocations
    }
    assert len(node_run_ids) == 1
    retried_node_run_id = next(iter(node_run_ids))

    task_runs = await read_settled_task_runs(
        result.workflow_run_id,
        expected_count=3,
    )
    task_runs_by_id = {task_run.id: task_run for task_run in task_runs}
    async with get_client(sync_client=False) as client:
        flow_run = await client.read_flow_run(result.workflow_run_id)
    retried_task_run = task_runs_by_id[retried_node_run_id]

    assert flow_run.state is not None
    assert flow_run.state.is_completed()
    assert retried_task_run.state is not None
    assert retried_task_run.state.is_completed()
    assert retried_task_run.run_count == 2


@pytest.mark.asyncio
async def test_prefect_managed_execution_cancels_active_node(tmp_path: Path) -> None:
    components, _unit_of_work = build_prefect_components(tmp_path / "workbench")
    BlockingIntegerNode.started = asyncio.Event()
    execution = await components.execution_manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="blocking",
                    operator_id="test.prefect.blocking_integer",
                    operator_version=1,
                )
            ]
        )
    )
    await asyncio.wait_for(BlockingIntegerNode.started.wait(), timeout=10)
    running = await components.execution_manager.get(execution.execution_id)
    assert running.status == "running"
    assert running.active_node_id == "blocking"

    await components.execution_manager.cancel(execution.execution_id)
    async with asyncio.timeout(10):
        while True:
            terminal = await components.execution_manager.get(execution.execution_id)
            if terminal.status == "cancelled":
                break
            await asyncio.sleep(0.01)

    assert terminal.active_node_id is None
    assert terminal.result is None
    assert terminal.error is None
    workflow_run_id = BlockingIntegerNode.workflow_run_id
    assert workflow_run_id is not None
    async with get_client(sync_client=False) as client:
        flow_run = await client.read_flow_run(workflow_run_id)
    assert flow_run.state is not None
    assert flow_run.state.is_cancelled()
    await components.execution_manager.shutdown()

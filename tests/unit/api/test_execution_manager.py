import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, final, override
from uuid import UUID, uuid4

import pytest
from pydantic import StrictInt, ValidationError

from notarius_core.artifacts import InMemoryUnitOfWork, NoConfig, NodeInput, NodeOutput
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionScope,
    GraphExecutionStatus,
)
from notarius_core.domain.errors import NotFoundError
from notarius_core.nodes import (
    MAX_NODE_PROGRESS_COUNTER,
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.operators.arithmetic import INTEGER_VALUE
from notarius_core.plugins import Plugin

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.v1.routes.executions.models import (
    ExecutionStatusEvent,
    NodeProgressEvent,
    NodeStatusEvent,
    RunEdgeRequest,
    RunNodeRequest,
    RunRequest,
)
from notarius_api.services.composition import build_workbench_components
from notarius_api.v1.routes.executions.runtime.control import RunExecutionControl
from notarius_api.v1.routes.executions.runtime.manager import (
    RunExecutionManager,
    RunExecutionSnapshot,
)
from notarius_api.v1.routes.executions.runtime.models import GraphExecutionResult
from notarius_api.v1.routes.executions.runtime.run_graph import RunGraph
from notarius_api.v1.routes.executions.services import ExecutionHistoryService


EXECUTION_TEST_PLUGIN = Plugin(
    slug="test.execution-control",
    title="Execution control test plugin",
)
_started: dict[str, asyncio.Event] = {}
_release: dict[str, asyncio.Event] = {}
_downstream_calls: list[str] = []
_progress_contexts: list[NodeExecutionContext] = []


async def _wait_at_gate(context: NodeExecutionContext) -> None:
    node_id = context.node_id
    assert node_id is not None
    _started[node_id].set()
    await _release[node_id].wait()


class FirstGateInput(NodeInput):
    pass


class FirstGateOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@EXECUTION_TEST_PLUGIN.node(
    operator_id="test.execution.first_gate",
    version=1,
    title="First gate",
)
@final
class FirstGateNode(Node[NoConfig, FirstGateInput, FirstGateOutput]):
    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: FirstGateInput,
        /,
    ) -> FirstGateOutput:
        await _wait_at_gate(context)
        return FirstGateOutput(value=1)


class SecondGateInput(NodeInput):
    value: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class SecondGateOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@EXECUTION_TEST_PLUGIN.node(
    operator_id="test.execution.second_gate",
    version=1,
    title="Second gate",
)
@final
class SecondGateNode(Node[NoConfig, SecondGateInput, SecondGateOutput]):
    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: SecondGateInput,
        /,
    ) -> SecondGateOutput:
        await _wait_at_gate(context)
        return SecondGateOutput(value=inputs.value + 1)


class RecordingInput(NodeInput):
    value: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class RecordingOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@EXECUTION_TEST_PLUGIN.node(
    operator_id="test.execution.recording",
    version=1,
    title="Recording downstream",
)
@final
class RecordingNode(Node[NoConfig, RecordingInput, RecordingOutput]):
    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: RecordingInput,
        /,
    ) -> RecordingOutput:
        assert context.node_id is not None
        _downstream_calls.append(context.node_id)
        return RecordingOutput(value=inputs.value + 1)


class ProgressInput(NodeInput):
    value: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class ProgressOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@EXECUTION_TEST_PLUGIN.function_node(
    operator_id="test.execution.progress",
    version=1,
    title="Progress reporter",
)
async def report_progress(
    context: NodeExecutionContext,
    _config: NoConfig,
    inputs: ProgressInput,
) -> ProgressOutput:
    _progress_contexts.append(context)
    invocation_index = context.invocation_index
    await context.progress(
        "Preparing mapped item",
        current=None if invocation_index is None else invocation_index + 1,
        total=3,
    )
    return ProgressOutput(value=inputs.value)


class FailingInput(NodeInput):
    pass


class FailingOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@EXECUTION_TEST_PLUGIN.node(
    operator_id="test.execution.failure",
    version=1,
    title="Failing node",
)
@final
class FailingNode(Node[NoConfig, FailingInput, FailingOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: FailingInput,
        /,
    ) -> FailingOutput:
        raise RuntimeError("controlled node failure")


class CancellationWrappingRunGraph(RunGraph):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    @override
    async def run(
        self,
        request: RunRequest,
        control: RunExecutionControl | None = None,
    ) -> GraphExecutionResult:
        del request, control
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise RuntimeError("wrapped execution cancellation") from exc
        raise AssertionError("unreachable")


class ControlledRunGraph(RunGraph):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @override
    async def run(
        self,
        request: RunRequest,
        control: RunExecutionControl | None = None,
    ) -> GraphExecutionResult:
        del request, control
        self.started.set()
        await self.release.wait()
        return GraphExecutionResult(
            workflow_run_id=uuid4(),
            status="succeeded",
            node_results=(),
            outputs={},
        )


class ControlledExecutionHistory(ExecutionHistoryService):
    def __init__(self) -> None:
        super().__init__(InMemoryUnitOfWork(), None)
        self.transitions: list[str] = []
        self.cancelling_entered = asyncio.Event()
        self.allow_cancelling = asyncio.Event()
        self.allow_cancelling.set()
        self.complete_entered = asyncio.Event()
        self.allow_complete = asyncio.Event()
        self.allow_complete.set()
        self.complete_failures_remaining = 0
        self.complete_calls = 0

    @override
    async def create_queued(
        self,
        *,
        execution_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        scope: GraphExecutionScope,
        requested_node_ids: tuple[str, ...],
    ) -> GraphExecution:
        self.transitions.append("queued")
        return GraphExecution(
            execution_id=execution_id,
            graph_id=graph_id,
            graph_revision=graph_revision,
            scope=scope,
            status="queued",
            requested_node_ids=requested_node_ids,
        )

    @override
    async def mark_running(self, execution: GraphExecution) -> None:
        execution.status = "running"
        execution.started_at = datetime.now(UTC)
        self.transitions.append("running")

    @override
    async def mark_cancelling(self, execution: GraphExecution) -> None:
        self.cancelling_entered.set()
        await self.allow_cancelling.wait()
        execution.status = "cancelling"
        self.transitions.append("cancelling")

    @override
    async def complete(
        self,
        execution: GraphExecution,
        *,
        status: GraphExecutionStatus,
        result: GraphExecutionResult | None,
        error: str | None,
    ) -> None:
        del result
        self.complete_calls += 1
        self.complete_entered.set()
        await self.allow_complete.wait()
        if self.complete_failures_remaining > 0:
            self.complete_failures_remaining -= 1
            raise RuntimeError("transient execution history failure")
        execution.status = status
        execution.finished_at = datetime.now(UTC)
        execution.error = error
        self.transitions.append(status)


def _manager(
    workspace: Path,
    *,
    terminal_retention: int = 100,
    event_capacity: int = 256,
) -> RunExecutionManager:
    registry = build_plugin_registry(
        (*builtin_plugins(), EXECUTION_TEST_PLUGIN),
        external_plugins=(),
    )
    components = build_workbench_components(
        plugin_registry=registry,
        execution_backend="inline",
        workspace=workspace,
    )
    return RunExecutionManager(
        components.run_graph,
        terminal_retention=terminal_retention,
        event_capacity=event_capacity,
    )


def _gate(node_id: str) -> None:
    _started[node_id] = asyncio.Event()
    _release[node_id] = asyncio.Event()


async def _terminal(
    manager: RunExecutionManager,
    execution_id: UUID,
) -> RunExecutionSnapshot:
    async with asyncio.timeout(3):
        while True:
            execution = await manager.get(execution_id)
            if execution.status in {"cancelled", "succeeded", "failed"}:
                return execution
            await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def reset_execution_test_state() -> None:
    _started.clear()
    _release.clear()
    _downstream_calls.clear()
    _progress_contexts.clear()


@pytest.mark.asyncio
async def test_manager_reports_exact_node_and_cancellation_stops_downstream(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "workbench")
    _gate("first")
    _gate("second")
    request = RunRequest(
        nodes=[
            RunNodeRequest(
                id="first",
                operator_id="test.execution.first_gate",
                operator_version=1,
            ),
            RunNodeRequest(
                id="second",
                operator_id="test.execution.second_gate",
                operator_version=1,
            ),
            RunNodeRequest(
                id="third",
                operator_id="test.execution.recording",
                operator_version=1,
            ),
        ],
        edges=[
            RunEdgeRequest(
                from_node="first",
                from_port="value",
                to_node="second",
                to_port="value",
            ),
            RunEdgeRequest(
                from_node="second",
                from_port="value",
                to_node="third",
                to_port="value",
            ),
        ],
    )

    execution = await manager.start(request)
    await asyncio.wait_for(_started["first"].wait(), timeout=3)
    assert (await manager.get(execution.execution_id)).active_node_id == "first"
    observed = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=0,
        timeout=0,
    )
    quiet = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=observed.events[-1].sequence,
        timeout=0,
    )
    assert quiet.events == ()
    assert quiet.terminal is False

    _release["first"].set()
    await asyncio.wait_for(_started["second"].wait(), timeout=3)
    assert (await manager.get(execution.execution_id)).active_node_id == "second"

    cancelling = await manager.cancel(execution.execution_id)
    assert cancelling.status == "cancelling"
    cancelled = await _terminal(manager, execution.execution_id)
    assert cancelled.status == "cancelled"
    assert cancelled.active_node_id is None
    assert cancelled.result is None
    assert cancelled.error is None
    assert _downstream_calls == []
    terminal_events = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=0,
        timeout=0,
    )
    lifecycle = [
        event.status
        for event in terminal_events.events
        if isinstance(event, ExecutionStatusEvent)
    ]
    assert "cancelling" in lifecycle
    assert lifecycle[-1] == "cancelled"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_is_idempotent_and_handles_cancel_before_task_start(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "workbench")
    execution = await manager.start(RunRequest(nodes=[]))

    first = await manager.cancel(execution.execution_id)
    second = await manager.cancel(execution.execution_id)
    assert first.status == "cancelling"
    assert second.status == "cancelling"
    terminal = await _terminal(manager, execution.execution_id)
    assert terminal.status == "cancelled"
    assert (await manager.cancel(execution.execution_id)).status == "cancelled"

    missing_id = uuid4()
    with pytest.raises(NotFoundError, match=str(missing_id)):
        await manager.get(missing_id)
    with pytest.raises(NotFoundError, match=str(missing_id)):
        await manager.cancel(missing_id)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_isolates_concurrent_execution_progress(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "workbench")
    _gate("run-a")
    _gate("run-b")
    first = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="run-a",
                    operator_id="test.execution.first_gate",
                    operator_version=1,
                )
            ]
        )
    )
    second = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="run-b",
                    operator_id="test.execution.first_gate",
                    operator_version=1,
                )
            ]
        )
    )

    await asyncio.gather(_started["run-a"].wait(), _started["run-b"].wait())
    assert (await manager.get(first.execution_id)).active_node_id == "run-a"
    assert (await manager.get(second.execution_id)).active_node_id == "run-b"

    _release["run-a"].set()
    assert (await _terminal(manager, first.execution_id)).status == "succeeded"
    still_running = await manager.get(second.execution_id)
    assert still_running.status == "running"
    assert still_running.active_node_id == "run-b"
    await manager.cancel(second.execution_id)
    assert (await _terminal(manager, second.execution_id)).status == "cancelled"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_and_awaits_active_tasks(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "workbench")
    _gate("run-a")
    _gate("run-b")
    first = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="run-a",
                    operator_id="test.execution.first_gate",
                    operator_version=1,
                )
            ]
        )
    )
    second = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="run-b",
                    operator_id="test.execution.first_gate",
                    operator_version=1,
                )
            ]
        )
    )
    await asyncio.gather(_started["run-a"].wait(), _started["run-b"].wait())

    await manager.shutdown()

    assert (await manager.get(first.execution_id)).status == "cancelled"
    assert (await manager.get(second.execution_id)).status == "cancelled"
    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.start(RunRequest(nodes=[]))


@pytest.mark.asyncio
async def test_manager_preserves_failed_graph_result(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "workbench")
    execution = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="failure",
                    operator_id="test.execution.failure",
                    operator_version=1,
                ),
                RunNodeRequest(
                    id="skipped",
                    operator_id="test.execution.recording",
                    operator_version=1,
                ),
            ],
            edges=[
                RunEdgeRequest(
                    from_node="failure",
                    from_port="value",
                    to_node="skipped",
                    to_port="value",
                )
            ],
        )
    )

    failed = await _terminal(manager, execution.execution_id)

    assert failed.status == "failed"
    assert failed.result is not None
    assert failed.result.status == "failed"
    assert "controlled node failure" in (failed.result.node_results[0].error or "")
    batch = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=0,
        timeout=0,
    )
    node_transitions = [
        (event.node_id, event.status)
        for event in batch.events
        if isinstance(event, NodeStatusEvent)
    ]
    assert node_transitions == [
        ("failure", "running"),
        ("failure", "failed"),
        ("skipped", "skipped"),
    ]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_replays_lifecycle_and_mapped_progress_events(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "workbench")
    execution = await manager.start(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="sequence",
                    operator_id="arithmetic.integer_sequence",
                    operator_version=1,
                    config={"start": 1, "count": 3, "step": 1},
                ),
                RunNodeRequest(
                    id="progress",
                    operator_id="test.execution.progress",
                    operator_version=1,
                ),
            ],
            edges=[
                RunEdgeRequest(
                    from_node="sequence",
                    from_port="values",
                    to_node="progress",
                    to_port="value",
                    collection_mode="map",
                )
            ],
        )
    )
    assert (await _terminal(manager, execution.execution_id)).status == "succeeded"

    batch = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=0,
        timeout=0,
    )
    progress_events = [
        event for event in batch.events if isinstance(event, NodeProgressEvent)
    ]
    status_events = [
        event for event in batch.events if isinstance(event, ExecutionStatusEvent)
    ]

    assert batch.terminal is True
    assert [event.sequence for event in batch.events] == list(
        range(1, len(batch.events) + 1)
    )
    assert [event.status for event in status_events[:2]] == ["queued", "running"]
    assert status_events[-1].status == "succeeded"
    assert [event.invocation_index for event in progress_events] == [0, 1, 2]
    assert [event.invocation_path for event in progress_events] == [[0], [1], [2]]
    assert [event.current for event in progress_events] == [1, 2, 3]
    assert all(event.total == 3 for event in progress_events)
    assert all(event.node_path == ["progress"] for event in progress_events)
    assert all(event.node_run_id is not None for event in progress_events)

    terminal_sequence = batch.events[-1].sequence
    await _progress_contexts[0].progress("Too late")
    after_terminal = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=terminal_sequence,
        timeout=0,
    )
    assert after_terminal.events == ()
    assert after_terminal.terminal is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_bounds_event_replay_and_detects_terminal_delivery(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "workbench", event_capacity=2)
    execution = await manager.start(RunRequest(nodes=[]))
    assert (await _terminal(manager, execution.execution_id)).status == "succeeded"

    replay = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=0,
        timeout=0,
    )
    after_terminal = await manager.wait_for_events(
        execution.execution_id,
        after_sequence=replay.events[-1].sequence,
        timeout=0,
    )

    assert [event.sequence for event in replay.events] == [2, 3]
    assert [event.kind for event in replay.events] == [
        "execution.status",
        "execution.status",
    ]
    assert replay.terminal is True
    assert after_terminal.events == ()
    assert after_terminal.terminal is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancellation_intent_wins_when_executor_wraps_cancelled_error() -> None:
    run_graph = CancellationWrappingRunGraph()
    manager = RunExecutionManager(run_graph)
    execution = await manager.start(RunRequest(nodes=[]))
    await run_graph.started.wait()

    await manager.cancel(execution.execution_id)
    terminal = await _terminal(manager, execution.execution_id)

    assert terminal.status == "cancelled"
    assert terminal.error is None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_bounds_terminal_execution_retention(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "workbench", terminal_retention=2)
    execution_ids: list[UUID] = []
    first_subscription = None
    for _ in range(3):
        execution = await manager.start(RunRequest(nodes=[]))
        execution_ids.append(execution.execution_id)
        assert (await _terminal(manager, execution.execution_id)).status == "succeeded"
        if first_subscription is None:
            first_subscription = await manager.subscribe_events(execution.execution_id)

    with pytest.raises(NotFoundError, match=str(execution_ids[0])):
        await manager.get(execution_ids[0])
    assert (await manager.get(execution_ids[1])).status == "succeeded"
    assert (await manager.get(execution_ids[2])).status == "succeeded"
    assert first_subscription is not None
    retained_events = await first_subscription.wait(after_sequence=0, timeout=0)
    assert retained_events.terminal is True
    assert retained_events.events
    assert all(
        event.execution_id == execution_ids[0] for event in retained_events.events
    )
    await manager.shutdown()


def test_node_execution_event_identity_is_bounded() -> None:
    event_fields = {
        "sequence": 1,
        "execution_id": uuid4(),
        "occurred_at": datetime.now(UTC),
        "node_id": "node",
        "node_path": ["node"],
        "node_run_id": uuid4(),
        "message": "Working",
    }

    normalized = NodeProgressEvent.model_validate(
        {
            **event_fields,
            "node_id": " node ",
            "node_path": [" parent ", " node "],
        }
    )
    assert normalized.node_id == "node"
    assert normalized.node_path == ["parent", "node"]

    boundary = NodeProgressEvent.model_validate(
        {
            **event_fields,
            "current": MAX_NODE_PROGRESS_COUNTER,
            "total": MAX_NODE_PROGRESS_COUNTER,
        }
    )
    assert boundary.current == MAX_NODE_PROGRESS_COUNTER
    assert boundary.total == MAX_NODE_PROGRESS_COUNTER

    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate({**event_fields, "node_id": "x" * 256})
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate({**event_fields, "node_path": ["x" * 256]})
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate({**event_fields, "node_path": ["  "]})
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate({**event_fields, "node_path": ["node"] * 65})
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate(
            {**event_fields, "invocation_path": list(range(65))}
        )
    with pytest.raises(ValidationError):
        ExecutionStatusEvent(
            sequence=1,
            execution_id=uuid4(),
            occurred_at=datetime.now(UTC),
            status="running",
            active_node_id="x" * 256,
        )
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate(
            {**event_fields, "current": MAX_NODE_PROGRESS_COUNTER + 1}
        )
    with pytest.raises(ValidationError):
        NodeProgressEvent.model_validate(
            {**event_fields, "total": MAX_NODE_PROGRESS_COUNTER + 1}
        )


@pytest.mark.asyncio
async def test_natural_completion_wins_cancel_race_without_durable_downgrade() -> None:
    run_graph = ControlledRunGraph()
    history = ControlledExecutionHistory()
    history.allow_complete.clear()
    manager = RunExecutionManager(run_graph, execution_history=history)
    execution = await manager.start(
        RunRequest(
            nodes=[],
            graph_id=uuid4(),
            graph_revision=1,
        )
    )
    await run_graph.started.wait()
    run_graph.release.set()
    await history.complete_entered.wait()

    cancelling = asyncio.create_task(manager.cancel(execution.execution_id))
    await asyncio.sleep(0)
    assert not cancelling.done()
    history.allow_complete.set()

    cancel_result = await cancelling
    terminal = await _terminal(manager, execution.execution_id)
    assert cancel_result.status == "succeeded"
    assert terminal.status == "succeeded"
    assert history.transitions == ["queued", "running", "succeeded"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_transition_wins_race_and_persists_one_terminal_result() -> None:
    run_graph = ControlledRunGraph()
    history = ControlledExecutionHistory()
    history.allow_cancelling.clear()
    manager = RunExecutionManager(run_graph, execution_history=history)
    execution = await manager.start(
        RunRequest(
            nodes=[],
            graph_id=uuid4(),
            graph_revision=1,
        )
    )
    await run_graph.started.wait()

    cancelling = asyncio.create_task(manager.cancel(execution.execution_id))
    await history.cancelling_entered.wait()
    run_graph.release.set()
    await asyncio.sleep(0)
    history.allow_cancelling.set()

    assert (await cancelling).status == "cancelling"
    terminal = await _terminal(manager, execution.execution_id)
    assert terminal.status == "cancelled"
    assert history.transitions == ["queued", "running", "cancelling", "cancelled"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_terminal_history_write_retries_before_exposing_terminal() -> None:
    run_graph = ControlledRunGraph()
    history = ControlledExecutionHistory()
    history.complete_failures_remaining = 1
    manager = RunExecutionManager(run_graph, execution_history=history)
    execution = await manager.start(
        RunRequest(
            nodes=[],
            graph_id=uuid4(),
            graph_revision=1,
        )
    )
    await run_graph.started.wait()

    run_graph.release.set()
    terminal = await _terminal(manager, execution.execution_id)

    assert terminal.status == "succeeded"
    assert terminal.error is None
    assert history.complete_calls == 2
    assert history.transitions == ["queued", "running", "succeeded"]
    await manager.shutdown()

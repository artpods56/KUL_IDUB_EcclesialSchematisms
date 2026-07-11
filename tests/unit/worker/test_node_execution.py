from uuid import uuid4

import pytest

from notarius_core.application.operators import (
    OCR_EXTRACT_PAGES_OPERATOR_ID,
    OCR_EXTRACT_PAGES_OPERATOR_VERSION,
)
from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    ExecutionMode,
    InputAssemblyTrace,
    InvocationTrace,
    NodeSpec,
    NodeRun,
    NodeRunStatus,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
)
from notarius_messaging.subjects import (
    ARTIFACT_CREATED_EVENT_SUBJECT,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    NODE_RUN_FAILED_PERMANENT_EVENT_SUBJECT,
    NODE_RUN_QUEUED_EVENT_SUBJECT,
    NODE_RUN_RUNNING_EVENT_SUBJECT,
    NODE_RUN_SUCCEEDED_EVENT_SUBJECT,
    WORKFLOW_RUN_RUNNING_EVENT_SUBJECT,
    WORKFLOW_RUN_SUCCEEDED_EVENT_SUBJECT,
)
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)
from notarius_storage import (
    LocalArtifactPayloadStorage,
    SaveArtifactPayloadCommand,
    artifact_payload_ref,
)
from notarius_worker.node_execution import (
    ArtifactSequenceInput,
    NodeExecutionRequest,
    NodeExecutionResult,
    NodeRunExecutionError,
    NodeRunExecutor,
)
from notarius_worker.operators import (
    OcrExtractPagesHandler,
    OcrPageEngineError,
    OcrPageInput,
    OcrPageResultPayload,
)


class SuccessfulHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="s3://notarius/runs/one/text/page-1.md",
            metadata={"handler": "successful"},
        )
        input_trace = InputAssemblyTrace(
            node_run_id=request.node_run.id,
            selected_inputs={
                key: value.ref()
                if isinstance(value, Artifact)
                else [item.ref() for item in value]
                for key, value in request.input_artifacts.items()
            },
        )
        invocation_trace = InvocationTrace(
            node_run_id=request.node_run.id,
            invocation_type="test",
            output_artifact_refs=[artifact.ref()],
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            input_assembly_traces=[input_trace],
            invocation_traces=[invocation_trace],
            output_artifact_refs={"markdown": artifact.ref()},
        )


class WrongOutputTypeHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="debug.text",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://wrong-output/text.txt",
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"markdown": artifact.ref()},
        )


class MissingRequiredOutputHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://missing-output/page.md",
        )
        return NodeExecutionResult(artifacts=[artifact], output_artifact_refs={})


class RetryableFailureHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raise NodeRunExecutionError("provider unavailable", retryable=True)


class UnexpectedHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raise AssertionError("handler should not run")


class RetryableOcrPageEngine:
    engine_id = "provider.retryable"

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload:
        raise OcrPageEngineError("provider unavailable", retryable=True)


class CancellingSuccessHandler:
    def __init__(self, store: InMemoryDataStore):
        self.store = store

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        async with InMemoryUnitOfWork(self.store) as uow:
            node_run = await uow.node_runs.get(request.node_run.id)
            assert node_run is not None
            node_run.mark_cancelled()
            await uow.node_runs.update(node_run)
            await uow.commit()

        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://cancelled/should-not-persist.md",
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"markdown": artifact.ref()},
        )


class WorkflowCancellingSuccessHandler:
    def __init__(self, store: InMemoryDataStore):
        self.store = store

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        async with InMemoryUnitOfWork(self.store) as uow:
            workflow_run = await uow.workflow_runs.get(request.node_run.workflow_run_id)
            assert workflow_run is not None
            workflow_run.mark_cancelled()
            await uow.workflow_runs.update(workflow_run)
            await uow.commit()

        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://workflow-cancelled/should-not-persist.md",
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"markdown": artifact.ref()},
        )


class SuccessfulOcrHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="ocr.page_result",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="s3://notarius/runs/one/ocr/page-1.json",
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"ocr_pages": [artifact.ref()]},
        )


class SequenceReadingHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        pages = request.input_artifacts["pages"]
        assert isinstance(pages, ArtifactSequenceInput)
        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://sequence/summary.md",
            metadata={
                "ordered_payload_refs": [
                    artifact.payload_ref for artifact in pages.artifacts
                ],
                "sequence_id": str(pages.sequence.id),
                "index_key": pages.sequence.index_key,
            },
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"markdown": artifact.ref()},
        )


class SequenceOutputHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="text.markdown",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://sequence/page-1.md",
        )
        sequence = ArtifactSequence(
            artifact_type="text.markdown",
            schema_version=1,
            item_refs=[artifact.ref()],
            metadata={"handler": "sequence-output"},
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            artifact_sequences=[sequence],
            output_artifact_refs={"markdown_pages": sequence.ref()},
        )


class OcrSequenceOutputHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        artifact = Artifact(
            artifact_type="ocr.page_result",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref=f"memory://ocr/{request.node_run.workflow_node_id}/page-1.json",
        )
        sequence = ArtifactSequence(
            artifact_type="ocr.page_result",
            schema_version=1,
            item_refs=[artifact.ref()],
            index_key="page_number",
            metadata={"source_node": request.node_run.workflow_node_id},
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            artifact_sequences=[sequence],
            output_artifact_refs={"ocr_pages": sequence.ref()},
        )


class ExportSequenceReadingHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        records = request.input_artifacts["records"]
        assert isinstance(records, ArtifactSequenceInput)
        artifact = Artifact(
            artifact_type="export.dataset",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            payload_ref="memory://export/dataset.json",
            metadata={
                "ordered_ocr_artifact_ids": [
                    str(artifact.id) for artifact in records.artifacts
                ],
                "input_sequence_id": str(records.sequence.id),
            },
        )
        return NodeExecutionResult(
            artifacts=[artifact],
            output_artifact_refs={"dataset": artifact.ref()},
        )


@pytest.mark.asyncio
async def test_node_run_executor_persists_outputs_and_marks_success() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    input_artifact = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=None,
        payload_ref="s3://notarius/source/page-1.png",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
        input_artifact_refs={"page": input_artifact.ref()},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.artifacts.add(input_artifact)
        await uow.node_runs.add(node_run)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): SuccessfulHandler()},
    )

    processed_id = await executor.execute_next_node_run()

    async with InMemoryUnitOfWork(store) as uow:
        processed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        processed_node_run = await uow.node_runs.get(node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)
        input_traces = await uow.input_assembly_traces.list_for_node_run(node_run.id)
        invocation_traces = await uow.invocation_traces.list_for_node_run(node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert processed_id == node_run.id
    assert processed_workflow_run.status == WorkflowRunStatus.SUCCEEDED
    assert processed_workflow_run.output_artifact_refs == [artifacts[0].ref()]
    assert processed_node_run.status == NodeRunStatus.SUCCEEDED
    assert processed_node_run.attempt_count == 1
    assert processed_node_run.output_artifact_refs == {"markdown": artifacts[0].ref()}
    assert artifacts[0].metadata == {"handler": "successful"}
    assert input_traces[0].selected_inputs == {"page": input_artifact.ref()}
    assert invocation_traces[0].output_artifact_refs == [artifacts[0].ref()]
    assert [message.subject for message in pending_messages] == [
        WORKFLOW_RUN_RUNNING_EVENT_SUBJECT,
        NODE_RUN_RUNNING_EVENT_SUBJECT,
        ARTIFACT_CREATED_EVENT_SUBJECT,
        NODE_RUN_SUCCEEDED_EVENT_SUBJECT,
        WORKFLOW_RUN_SUCCEEDED_EVENT_SUBJECT,
    ]


@pytest.mark.asyncio
async def test_node_run_executor_rejects_output_artifact_type_mismatch() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
    )
    spec = NodeSpec(
        id="text.markdown",
        version="1.0.0",
        inputs=(),
        outputs=(
            PortSpec(
                name="markdown",
                artifact_type="text.markdown",
                schema_version=1,
            ),
        ),
        execution_mode=ExecutionMode.SINGLE,
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): WrongOutputTypeHandler()},
        {("text.markdown", "1.0.0"): spec},
    )

    with pytest.raises(NodeRunExecutionError, match="output artifact type mismatch"):
        await executor.execute_next_node_run()

    async with InMemoryUnitOfWork(store) as uow:
        processed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        processed_node_run = await uow.node_runs.get(node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)

    assert processed_workflow_run is not None
    assert processed_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert processed_node_run is not None
    assert processed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert "output artifact type mismatch" in (processed_node_run.error or "")
    assert artifacts == []


@pytest.mark.asyncio
async def test_node_run_executor_rejects_missing_required_output_port() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
    )
    spec = NodeSpec(
        id="text.markdown",
        version="1.0.0",
        inputs=(),
        outputs=(
            PortSpec(
                name="markdown",
                artifact_type="text.markdown",
                schema_version=1,
            ),
        ),
        execution_mode=ExecutionMode.SINGLE,
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): MissingRequiredOutputHandler()},
        {("text.markdown", "1.0.0"): spec},
    )

    with pytest.raises(NodeRunExecutionError, match="required output port markdown"):
        await executor.execute_next_node_run()

    async with InMemoryUnitOfWork(store) as uow:
        processed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        processed_node_run = await uow.node_runs.get(node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)

    assert processed_workflow_run is not None
    assert processed_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert processed_node_run is not None
    assert processed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert "required output port markdown" in (processed_node_run.error or "")
    assert artifacts == []


@pytest.mark.asyncio
async def test_node_run_executor_queues_ready_downstream_node_runs() -> None:
    store = InMemoryDataStore()
    definition = WorkflowDefinition(
        name="OCR export",
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.provider",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="export",
                operator_id="export.dataset",
                operator_version="1.0.0",
            ),
        ],
        edges=[
            WorkflowEdge("ocr", "ocr_pages", "export", "records"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    workflow_run = WorkflowRun(workflow_version_id=version.id)
    input_artifact = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=None,
        payload_ref="s3://notarius/source/page-1.png",
    )
    ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
        input_artifact_refs={"pages": [input_artifact.ref()]},
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
        },
    )
    export_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="export",
        operator_id="export.dataset",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [str(ocr_node_run.id)],
        },
    )
    export_node_run.mark_blocked()
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(workflow_run)
        await uow.artifacts.add(input_artifact)
        await uow.node_runs.add(ocr_node_run)
        await uow.node_runs.add(export_node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("ocr.provider", "1.0.0"): SuccessfulOcrHandler()},
    )

    await executor.execute_node_run(ocr_node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        completed_ocr_run = await uow.node_runs.get(ocr_node_run.id)
        queued_export_run = await uow.node_runs.get(export_node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(ocr_node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert completed_ocr_run.status == NodeRunStatus.SUCCEEDED
    assert queued_export_run.status == NodeRunStatus.QUEUED
    assert queued_export_run.input_artifact_refs == {
        "records": [artifacts[0].ref()],
    }
    execute_messages = [
        message
        for message in pending_messages
        if message.subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    ]
    node_queued_events = [
        message
        for message in pending_messages
        if message.subject == NODE_RUN_QUEUED_EVENT_SUBJECT
    ]
    assert [message.subject for message in pending_messages] == [
        WORKFLOW_RUN_RUNNING_EVENT_SUBJECT,
        NODE_RUN_RUNNING_EVENT_SUBJECT,
        ARTIFACT_CREATED_EVENT_SUBJECT,
        NODE_RUN_SUCCEEDED_EVENT_SUBJECT,
        NODE_RUN_QUEUED_EVENT_SUBJECT,
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    ]
    assert len(execute_messages) == 1
    assert len(node_queued_events) == 1
    assert execute_messages[0].payload["workflow_run_id"] == str(workflow_run.id)
    assert execute_messages[0].payload["node_run_id"] == str(export_node_run.id)
    assert node_queued_events[0].payload["node_run_id"] == str(export_node_run.id)


@pytest.mark.asyncio
async def test_node_run_executor_collects_concrete_map_outputs_for_reduce_node() -> None:
    store = InMemoryDataStore()
    definition = WorkflowDefinition(
        name="Concrete OCR export",
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.provider",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="export",
                operator_id="export.dataset",
                operator_version="1.0.0",
            ),
        ],
        edges=[
            WorkflowEdge("ocr", "ocr_pages", "export", "records"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    workflow_run = WorkflowRun(workflow_version_id=version.id)
    first_ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
            "map_item_index": 1,
        },
    )
    second_ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
            "map_item_index": 2,
        },
    )
    third_ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
            "map_item_index": 3,
        },
    )
    export_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="export",
        operator_id="export.dataset",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [
                str(first_ocr_node_run.id),
                str(second_ocr_node_run.id),
                str(third_ocr_node_run.id),
            ],
        },
    )
    export_node_run.mark_blocked()
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(first_ocr_node_run)
        await uow.node_runs.add(second_ocr_node_run)
        await uow.node_runs.add(third_ocr_node_run)
        await uow.node_runs.add(export_node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {
            ("ocr.provider", "1.0.0"): SuccessfulOcrHandler(),
            ("export.dataset", "1.0.0"): ExportSequenceReadingHandler(),
        },
        node_specs={
            ("ocr.provider", "1.0.0"): NodeSpec(
                id="ocr.provider",
                version="1.0.0",
                execution_mode=ExecutionMode.MAP,
                inputs=(),
                outputs=(
                    PortSpec(
                        name="ocr_pages",
                        artifact_type="ocr.page_result",
                        schema_version=1,
                        sequence=True,
                    ),
                ),
            ),
            ("export.dataset", "1.0.0"): NodeSpec(
                id="export.dataset",
                version="1.0.0",
                execution_mode=ExecutionMode.REDUCE,
                inputs=(
                    PortSpec(
                        name="records",
                        artifact_type="ocr.page_result",
                        schema_version=1,
                        sequence=True,
                    ),
                ),
                outputs=(
                    PortSpec(
                        name="dataset",
                        artifact_type="export.dataset",
                        schema_version=1,
                    ),
                ),
            ),
        },
    )

    await executor.execute_node_run(second_ocr_node_run.id)
    await executor.execute_node_run(first_ocr_node_run.id)
    await executor.execute_node_run(third_ocr_node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        queued_export_run = await uow.node_runs.get(export_node_run.id)
        sequence_ref = queued_export_run.input_artifact_refs["records"]
        assert isinstance(sequence_ref, ArtifactSequenceRef)
        collected_sequence = await uow.artifact_sequences.get(sequence_ref.sequence_id)
        completed_first_ocr_run = await uow.node_runs.get(first_ocr_node_run.id)
        completed_second_ocr_run = await uow.node_runs.get(second_ocr_node_run.id)
        completed_third_ocr_run = await uow.node_runs.get(third_ocr_node_run.id)

    assert queued_export_run.status == NodeRunStatus.QUEUED
    assert collected_sequence is not None
    assert completed_first_ocr_run is not None
    assert completed_second_ocr_run is not None
    assert completed_third_ocr_run is not None
    first_output_refs = completed_first_ocr_run.output_artifact_refs["ocr_pages"]
    second_output_refs = completed_second_ocr_run.output_artifact_refs["ocr_pages"]
    third_output_refs = completed_third_ocr_run.output_artifact_refs["ocr_pages"]
    assert isinstance(first_output_refs, list)
    assert isinstance(second_output_refs, list)
    assert isinstance(third_output_refs, list)
    assert [str(ref.artifact_id) for ref in collected_sequence.item_refs] == [
        str(first_output_refs[0].artifact_id),
        str(second_output_refs[0].artifact_id),
        str(third_output_refs[0].artifact_id),
    ]

    await executor.execute_node_run(export_node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        completed_export_run = await uow.node_runs.get(export_node_run.id)
        export_artifacts = await uow.artifacts.list_for_node_run(export_node_run.id)

    assert completed_export_run.status == NodeRunStatus.SUCCEEDED
    assert export_artifacts[0].metadata["ordered_ocr_artifact_ids"] == [
        str(ref.artifact_id) for ref in collected_sequence.item_refs
    ]


@pytest.mark.asyncio
async def test_node_run_executor_binds_two_same_type_sequence_edges_by_target_port() -> None:
    store = InMemoryDataStore()
    definition = WorkflowDefinition(
        name="OCR comparison",
        nodes=[
            WorkflowNode(
                id="ocr_a",
                operator_id="ocr.sequence.output",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="ocr_b",
                operator_id="ocr.sequence.output",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="compare",
                operator_id="ocr.compare_pages",
                operator_version="1.0.0",
            ),
        ],
        edges=[
            WorkflowEdge("ocr_a", "ocr_pages", "compare", "candidate_a_pages"),
            WorkflowEdge("ocr_b", "ocr_pages", "compare", "candidate_b_pages"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    workflow_run = WorkflowRun(workflow_version_id=version.id)
    first_ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr_a",
        operator_id="ocr.sequence.output",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
        },
    )
    second_ocr_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr_b",
        operator_id="ocr.sequence.output",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [],
        },
    )
    compare_node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="compare",
        operator_id="ocr.compare_pages",
        operator_version="1.0.0",
        metadata={
            "workflow_version_id": str(version.id),
            "upstream_node_run_ids": [
                str(first_ocr_node_run.id),
                str(second_ocr_node_run.id),
            ],
        },
    )
    compare_node_run.mark_blocked()
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(first_ocr_node_run)
        await uow.node_runs.add(second_ocr_node_run)
        await uow.node_runs.add(compare_node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("ocr.sequence.output", "1.0.0"): OcrSequenceOutputHandler()},
    )

    await executor.execute_node_run(first_ocr_node_run.id)
    await executor.execute_node_run(second_ocr_node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        queued_compare_run = await uow.node_runs.get(compare_node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    candidate_a_ref = queued_compare_run.input_artifact_refs["candidate_a_pages"]
    candidate_b_ref = queued_compare_run.input_artifact_refs["candidate_b_pages"]
    assert queued_compare_run.status == NodeRunStatus.QUEUED
    assert isinstance(candidate_a_ref, ArtifactSequenceRef)
    assert isinstance(candidate_b_ref, ArtifactSequenceRef)
    assert candidate_a_ref.artifact_type == "ocr.page_result"
    assert candidate_b_ref.artifact_type == "ocr.page_result"
    assert candidate_a_ref.sequence_id != candidate_b_ref.sequence_id
    execute_messages = [
        message
        for message in pending_messages
        if message.subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    ]
    node_queued_events = [
        message
        for message in pending_messages
        if message.subject == NODE_RUN_QUEUED_EVENT_SUBJECT
    ]
    assert [message.payload["node_run_id"] for message in execute_messages] == [
        str(compare_node_run.id)
    ]
    assert [message.payload["node_run_id"] for message in node_queued_events] == [
        str(compare_node_run.id)
    ]


@pytest.mark.asyncio
async def test_node_run_executor_loads_sequence_input_in_declared_order() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    first_page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=None,
        payload_ref="memory://source/page-1.png",
    )
    second_page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=None,
        payload_ref="memory://source/page-2.png",
    )
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[second_page.ref(), first_page.ref()],
        index_key="page_number",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.sequence",
        operator_version="1.0.0",
        input_artifact_refs={"pages": sequence.ref()},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.artifacts.add(first_page)
        await uow.artifacts.add(second_page)
        await uow.artifact_sequences.add(sequence)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.sequence", "1.0.0"): SequenceReadingHandler()},
    )

    await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)
        processed_node_run = await uow.node_runs.get(node_run.id)

    assert processed_node_run.status == NodeRunStatus.SUCCEEDED
    assert artifacts[0].metadata == {
        "ordered_payload_refs": [
            "memory://source/page-2.png",
            "memory://source/page-1.png",
        ],
        "sequence_id": str(sequence.id),
        "index_key": "page_number",
    }


@pytest.mark.asyncio
async def test_node_run_executor_marks_missing_input_sequence_permanent_failure() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    missing_ref = ArtifactSequenceRef(
        sequence_id=uuid4(),
        artifact_type="source.page_image",
        schema_version=1,
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.sequence",
        operator_version="1.0.0",
        input_artifact_refs={"pages": missing_ref},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.sequence", "1.0.0"): SequenceReadingHandler()},
    )

    with pytest.raises(NodeRunExecutionError, match="pages"):
        await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        failed_node_run = await uow.node_runs.get(node_run.id)

    assert failed_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert "Input artifact sequence not found" in failed_workflow_run.error
    assert failed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert str(missing_ref.sequence_id) in failed_node_run.error
    assert "pages" in failed_node_run.error


@pytest.mark.asyncio
async def test_node_run_executor_persists_output_artifact_sequences() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.sequence.output",
        operator_version="1.0.0",
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.sequence.output", "1.0.0"): SequenceOutputHandler()},
    )

    await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        processed_node_run = await uow.node_runs.get(node_run.id)
        sequences = await uow.artifact_sequences.list_by_artifact_type("text.markdown")

    assert processed_node_run.status == NodeRunStatus.SUCCEEDED
    assert sequences[0].metadata == {"handler": "sequence-output"}
    assert processed_node_run.output_artifact_refs == {
        "markdown_pages": sequences[0].ref()
    }


@pytest.mark.asyncio
async def test_node_run_executor_marks_missing_input_artifact_permanent_failure() -> (
    None
):
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
        input_artifact_refs={
            "page": ArtifactRef(
                artifact_id=uuid4(),
                artifact_type="source.page_image",
                schema_version=1,
            )
        },
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): SuccessfulHandler()},
    )

    with pytest.raises(NodeRunExecutionError, match="Input artifact not found"):
        await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        failed_node_run = await uow.node_runs.get(node_run.id)

    assert failed_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert "Input artifact not found" in failed_workflow_run.error
    assert failed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert failed_node_run.attempt_count == 1
    assert "Input artifact not found" in failed_node_run.error


@pytest.mark.asyncio
async def test_node_run_executor_preserves_retryable_handler_failure() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("ocr.provider", "1.0.0"): RetryableFailureHandler()},
    )

    with pytest.raises(NodeRunExecutionError, match="provider unavailable"):
        await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_node_run = await uow.node_runs.get(node_run.id)

    assert failed_node_run.status == NodeRunStatus.FAILED_RETRYABLE
    assert failed_node_run.attempt_count == 1
    assert failed_node_run.error == "provider unavailable"


@pytest.mark.asyncio
async def test_node_run_executor_marks_missing_workflow_run_permanent_failure() -> None:
    store = InMemoryDataStore()
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("ocr.provider", "1.0.0"): UnexpectedHandler()},
    )

    with pytest.raises(NodeRunExecutionError, match="WorkflowRun not found"):
        await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_node_run = await uow.node_runs.get(node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert failed_node_run is not None
    assert failed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert failed_node_run.error == f"WorkflowRun not found: {node_run.workflow_run_id}"
    assert [message.subject for message in pending_messages] == [
        NODE_RUN_FAILED_PERMANENT_EVENT_SUBJECT,
    ]
    assert pending_messages[0].payload["node_run_id"] == str(node_run.id)
    assert pending_messages[0].payload["error"]["error_code"] == (
        "node_run_execution_failed"
    )


@pytest.mark.asyncio
async def test_node_run_executor_stops_retrying_after_attempt_limit() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
        max_attempts=2,
    )
    for _ in range(node_run.max_attempts):
        node_run.mark_running()
        node_run.mark_failed("provider unavailable", retryable=True)
    workflow_run.mark_running()
    workflow_run.mark_failed("provider unavailable", retryable=True)
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("ocr.provider", "1.0.0"): UnexpectedHandler()},
    )

    await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        failed_node_run = await uow.node_runs.get(node_run.id)

    assert failed_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert "Retry attempts exhausted" in failed_workflow_run.error
    assert failed_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert failed_node_run.attempt_count == 2
    assert failed_node_run.error == (
        f"Retry attempts exhausted for node run {node_run.id}: 2/2"
    )


@pytest.mark.asyncio
async def test_node_run_executor_marks_retryable_ocr_provider_failure(
    tmp_path,
) -> None:
    store = InMemoryDataStore()
    storage = LocalArtifactPayloadStorage(tmp_path)
    saved = storage.save(
        SaveArtifactPayloadCommand(
            bucket="source-page-images",
            key="pages/page-1.png",
            payload=b"image-bytes",
        )
    )
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(bucket=saved.bucket, key=saved.key),
    )
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[page.ref()],
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
        operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
        input_artifact_refs={"pages": sequence.ref()},
        metadata={
            "workflow_node_config": {"engine": RetryableOcrPageEngine.engine_id}
        },
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.artifacts.add(page)
        await uow.artifact_sequences.add(sequence)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {
            (OCR_EXTRACT_PAGES_OPERATOR_ID, OCR_EXTRACT_PAGES_OPERATOR_VERSION): (
                OcrExtractPagesHandler(
                    storage,
                    engines={
                        RetryableOcrPageEngine.engine_id: RetryableOcrPageEngine()
                    },
                )
            )
        },
    )

    with pytest.raises(NodeRunExecutionError, match="provider unavailable"):
        await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        failed_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        failed_node_run = await uow.node_runs.get(node_run.id)

    assert failed_workflow_run.status == WorkflowRunStatus.FAILED_RETRYABLE
    assert failed_node_run.status == NodeRunStatus.FAILED_RETRYABLE
    assert failed_node_run.attempt_count == 1
    assert "OCR engine failed for page 1" in failed_node_run.error


@pytest.mark.asyncio
async def test_node_run_executor_does_not_overwrite_midflight_node_cancel() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): CancellingSuccessHandler(store)},
    )

    await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        processed_node_run = await uow.node_runs.get(node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)

    assert processed_node_run.status == NodeRunStatus.CANCELLED
    assert processed_node_run.attempt_count == 1
    assert processed_node_run.output_artifact_refs == {}
    assert artifacts == []


@pytest.mark.asyncio
async def test_node_run_executor_does_not_persist_outputs_after_workflow_cancel() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="markdown",
        operator_id="text.markdown",
        operator_version="1.0.0",
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.commit()
    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("text.markdown", "1.0.0"): WorkflowCancellingSuccessHandler(store)},
    )

    await executor.execute_node_run(node_run.id)

    async with InMemoryUnitOfWork(store) as uow:
        cancelled_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        processed_node_run = await uow.node_runs.get(node_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)

    assert cancelled_workflow_run.status == WorkflowRunStatus.CANCELLED
    assert processed_node_run.status == NodeRunStatus.CANCELLED
    assert processed_node_run.output_artifact_refs == {}
    assert artifacts == []


@pytest.mark.asyncio
async def test_node_run_executor_returns_none_when_no_work_is_queued() -> None:
    store = InMemoryDataStore()
    executor = NodeRunExecutor(lambda: InMemoryUnitOfWork(store), {})

    assert await executor.execute_next_node_run() is None

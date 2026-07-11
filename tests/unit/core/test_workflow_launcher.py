from uuid import uuid4

import pytest

from notarius_core.application.workflows import WorkflowRunLauncher
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    ArtifactRef,
    ArtifactSequence,
    ExecutionMode,
    NodeRun,
    NodeRunStatus,
    NodeSpec,
    OutboxMessage,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_persistence.adapters.in_memory import InMemoryUnitOfWork

OCR_SPEC = NodeSpec(
    id="ocr.run",
    version="1.0.0",
    execution_mode=ExecutionMode.MAP,
    inputs=(
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
            sequence=True,
        ),
    ),
    outputs=(
        PortSpec(
            name="ocr_pages",
            artifact_type="ocr.page_result",
            schema_version=1,
            sequence=True,
        ),
    ),
)

EXPORT_SPEC = NodeSpec(
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
)

REGISTRY = {
    (OCR_SPEC.id, OCR_SPEC.version): OCR_SPEC,
    (EXPORT_SPEC.id, EXPORT_SPEC.version): EXPORT_SPEC,
}


@pytest.mark.asyncio
async def test_workflow_run_launcher_persists_run_and_schedules_root_nodes() -> None:
    page_refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
    ]
    definition = WorkflowDefinition(
        name="OCR export",
        declared_inputs=[
            PortSpec(
                name="pages",
                artifact_type="source.page_image",
                schema_version=1,
                sequence=True,
            )
        ],
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
                config={"batch_size": 2},
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

    async with InMemoryUnitOfWork() as uow:
        result = await WorkflowRunLauncher(REGISTRY).launch(
            uow,
            version,
            page_refs,
            metadata={"experiment": "ocr-a"},
            node_run_outbox_message_builder=_test_outbox_message,
        )

        stored_run = await uow.workflow_runs.get(result.workflow_run.id)
        node_runs = await uow.node_runs.list_for_workflow_run(result.workflow_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    node_runs_by_node_id = {
        node_run.workflow_node_id: node_run for node_run in node_runs
    }
    ocr_run = node_runs_by_node_id["ocr"]
    export_run = node_runs_by_node_id["export"]

    assert stored_run == result.workflow_run
    assert stored_run is not None
    assert stored_run.input_artifact_refs == page_refs
    assert stored_run.metadata == {"experiment": "ocr-a"}
    assert result.queued_node_run_ids == (ocr_run.id,)
    assert ocr_run.status == NodeRunStatus.QUEUED
    assert ocr_run.input_artifact_refs == {"pages": page_refs}
    assert ocr_run.metadata["workflow_node_config"] == {"batch_size": 2}
    assert ocr_run.metadata["upstream_node_run_ids"] == []
    assert [message.subject for message in pending_messages] == ["test.node_run"]
    assert pending_messages[0].payload == {
        "workflow_run_id": str(result.workflow_run.id),
        "node_run_id": str(ocr_run.id),
    }
    assert export_run.status == NodeRunStatus.BLOCKED
    assert export_run.input_artifact_refs == {}
    assert export_run.metadata["upstream_node_run_ids"] == [str(ocr_run.id)]


@pytest.mark.asyncio
async def test_workflow_run_launcher_binds_sequence_input_as_sequence_ref() -> None:
    page_refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
    ]
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=page_refs,
        metadata={"source": "unit-test"},
    )
    definition = WorkflowDefinition(
        name="OCR export",
        declared_inputs=[
            PortSpec(
                name="pages",
                artifact_type="source.page_image",
                schema_version=1,
                sequence=True,
            )
        ],
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            )
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    async with InMemoryUnitOfWork() as uow:
        await uow.artifact_sequences.add(sequence)
        result = await WorkflowRunLauncher(REGISTRY).launch(
            uow,
            version,
            [],
            input_artifact_sequences=[sequence],
        )
        stored_run = await uow.workflow_runs.get(result.workflow_run.id)
        node_runs = await uow.node_runs.list_for_workflow_run(result.workflow_run.id)

    assert stored_run is not None
    assert stored_run.input_artifact_sequence_refs == [sequence.ref()]
    assert node_runs[0].input_artifact_refs == {"pages": sequence.ref()}


@pytest.mark.asyncio
async def test_workflow_run_launcher_expands_concrete_map_nodes_over_sequence() -> None:
    page_refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
    ]
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=page_refs,
        metadata={"source": "unit-test"},
    )
    definition = WorkflowDefinition(
        name="Concrete OCR export",
        declared_inputs=[
            PortSpec(
                name="pages",
                artifact_type="source.page_image",
                schema_version=1,
                sequence=True,
            )
        ],
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
                config={"batch_size": 1},
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
        metadata={"execution_planning": "concrete_map"},
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    async with InMemoryUnitOfWork() as uow:
        await uow.artifact_sequences.add(sequence)
        result = await WorkflowRunLauncher(REGISTRY).launch(
            uow,
            version,
            [],
            input_artifact_sequences=[sequence],
            node_run_outbox_message_builder=_test_outbox_message,
        )
        node_runs = await uow.node_runs.list_for_workflow_run(result.workflow_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    ocr_node_runs = [
        node_run for node_run in node_runs if node_run.workflow_node_id == "ocr"
    ]
    export_node_runs = [
        node_run for node_run in node_runs if node_run.workflow_node_id == "export"
    ]
    assert len(ocr_node_runs) == 3
    assert len(export_node_runs) == 1
    assert result.queued_node_run_ids == tuple(node_run.id for node_run in ocr_node_runs)
    assert [node_run.input_artifact_refs for node_run in ocr_node_runs] == [
        {"pages": [page_refs[0]]},
        {"pages": [page_refs[1]]},
        {"pages": [page_refs[2]]},
    ]
    assert [node_run.metadata["map_item_index"] for node_run in ocr_node_runs] == [
        1,
        2,
        3,
    ]
    assert {node_run.metadata["map_item_count"] for node_run in ocr_node_runs} == {3}
    assert {
        node_run.metadata["map_source_sequence_id"] for node_run in ocr_node_runs
    } == {str(sequence.id)}
    export_node_run = export_node_runs[0]
    assert export_node_run.status == NodeRunStatus.BLOCKED
    assert export_node_run.metadata["upstream_node_run_ids"] == [
        str(node_run.id) for node_run in ocr_node_runs
    ]
    assert [message.payload["node_run_id"] for message in pending_messages] == [
        str(node_run.id) for node_run in ocr_node_runs
    ]


@pytest.mark.asyncio
async def test_workflow_run_launcher_rejects_missing_required_root_input() -> None:
    definition = WorkflowDefinition(
        name="OCR export",
        declared_inputs=[
            PortSpec(
                name="pages",
                artifact_type="source.page_image",
                schema_version=1,
                sequence=True,
            )
        ],
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            )
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    async with InMemoryUnitOfWork() as uow:
        with pytest.raises(ValidationError, match="missing required input artifacts"):
            await WorkflowRunLauncher(REGISTRY).launch(uow, version, [])


def _test_outbox_message(workflow_run: WorkflowRun, node_run: NodeRun) -> OutboxMessage:
    return OutboxMessage(
        subject="test.node_run",
        message_type="TestNodeRunRequested",
        payload={
            "workflow_run_id": str(workflow_run.id),
            "node_run_id": str(node_run.id),
        },
    )

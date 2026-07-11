from uuid import uuid4

import pytest

from notarius_core.application.experiments import (
    apply_experiment_parameters,
    expand_parameter_grid,
)
from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    ExecutionMode,
    ExperimentParameter,
    NodeRun,
    NodeRunStatus,
    OutboxMessage,
    OutboxMessageStatus,
    NodeSpec,
    PortSpec,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
)


def test_artifact_ref_preserves_contract_identity_and_hash() -> None:
    workflow_run_id = uuid4()
    node_run_id = uuid4()
    artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=node_run_id,
        producer_operator_id="ocr.mistral",
        producer_operator_version="1.0.0",
        payload_ref="s3://notarius/runs/one/ocr/page-1.json",
        content_hash="abc123",
    )

    ref = artifact.ref()

    assert ref == ArtifactRef(
        artifact_id=artifact.id,
        artifact_type="ocr.page_result",
        schema_version=1,
        content_hash="abc123",
    )


def test_artifact_sequence_rejects_mixed_artifact_types() -> None:
    refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="ocr.page_result",
            schema_version=1,
        ),
    ]

    with pytest.raises(ValueError, match="item type mismatch"):
        ArtifactSequence(
            artifact_type="source.page_image",
            schema_version=1,
            item_refs=refs,
        )


def test_artifact_sequence_rejects_mixed_schema_versions() -> None:
    refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="ocr.page_result",
            schema_version=1,
        ),
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="ocr.page_result",
            schema_version=2,
        ),
    ]

    with pytest.raises(ValueError, match="schema version mismatch"):
        ArtifactSequence(
            artifact_type="ocr.page_result",
            schema_version=1,
            item_refs=refs,
        )


def test_artifact_sequence_ref_preserves_sequence_contract() -> None:
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[
            ArtifactRef(
                artifact_id=uuid4(),
                artifact_type="source.page_image",
                schema_version=1,
            )
        ],
    )

    assert sequence.ref() == ArtifactSequenceRef(
        sequence_id=sequence.id,
        artifact_type="source.page_image",
        schema_version=1,
    )


def test_node_spec_declares_generic_artifact_contracts_and_execution_mode() -> None:
    spec = NodeSpec(
        id="ocr.mistral",
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
        config_schema={"type": "object"},
    )

    assert spec.execution_mode == ExecutionMode.MAP
    assert spec.inputs[0].artifact_type == "source.page_image"
    assert spec.outputs[0].artifact_type == "ocr.page_result"


def test_workflow_version_keeps_immutable_snapshot_reference() -> None:
    definition = WorkflowDefinition(
        name="Compare OCR",
        nodes=[
            WorkflowNode(
                id="ocr_a",
                operator_id="ocr.mistral",
                operator_version="1.0.0",
            )
        ],
    )

    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    assert version.workflow_definition_id == definition.id
    assert version.definition_snapshot.nodes[0].operator_id == "ocr.mistral"


def test_workflow_run_status_transitions_distinguish_retryability() -> None:
    run = WorkflowRun(workflow_version_id=uuid4())

    run.mark_running()
    assert run.status == WorkflowRunStatus.RUNNING
    assert run.started_at is not None

    run.mark_failed("provider timed out", retryable=True)
    assert run.status == WorkflowRunStatus.FAILED_RETRYABLE
    assert run.finished_at is not None
    assert not run.is_terminal


def test_node_run_records_attempt_and_outputs() -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="ocr_a",
        operator_id="ocr.mistral",
        operator_version="1.0.0",
    )
    output_ref = ArtifactRef(
        artifact_id=uuid4(),
        artifact_type="ocr.page_result",
        schema_version=1,
    )

    node_run.mark_running()
    node_run.mark_succeeded({"ocr_pages": [output_ref]})

    assert node_run.status == NodeRunStatus.SUCCEEDED
    assert node_run.attempt_count == 1
    assert node_run.output_artifact_refs == {"ocr_pages": [output_ref]}
    assert node_run.is_terminal


def test_outbox_message_can_be_marked_permanently_failed() -> None:
    message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )

    message.mark_failed("nats unavailable")
    message.mark_permanently_failed("nats unavailable")

    assert message.status == OutboxMessageStatus.FAILED
    assert message.attempts == 1
    assert message.error == "nats unavailable"


def test_outbox_message_requeue_resets_terminal_failure_state() -> None:
    message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )

    message.mark_failed("nats unavailable")
    message.mark_permanently_failed("nats unavailable")
    message.requeue()

    assert message.status == OutboxMessageStatus.PENDING
    assert message.attempts == 0
    assert message.error is None
    assert message.published_at is None


def test_experiment_parameter_requires_values() -> None:
    with pytest.raises(ValueError, match="requires values"):
        ExperimentParameter(
            name="ocr_engine",
            node_id="ocr",
            config_path=("engine",),
            values=(),
        )


def test_experiment_grid_rejects_duplicate_names_across_nodes() -> None:
    parameters = [
        ExperimentParameter(
            name="engine",
            node_id="ocr_a",
            config_path=("engine",),
            values=("local.text",),
        ),
        ExperimentParameter(
            name="engine",
            node_id="ocr_b",
            config_path=("engine",),
            values=("mistral.ocr",),
        ),
    ]

    with pytest.raises(ValueError, match="unique names"):
        expand_parameter_grid(parameters)


def test_experiment_grid_applies_parameter_values_to_workflow_snapshot() -> None:
    definition = WorkflowDefinition(
        name="OCR variants",
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.extract_pages",
                operator_version="1.0.0",
                config={"engine": "local.text", "engine_config": {"language": "eng"}},
            )
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    parameters = [
        ExperimentParameter(
            name="engine",
            node_id="ocr",
            config_path=("engine",),
            values=("local.text", "local.tesseract"),
        ),
        ExperimentParameter(
            name="language",
            node_id="ocr",
            config_path=("engine_config", "language"),
            values=("eng", "pol"),
        ),
    ]

    grid = expand_parameter_grid(parameters)
    variant_version = apply_experiment_parameters(version, parameters, grid[3])

    assert grid == [
        {"engine": "local.text", "language": "eng"},
        {"engine": "local.text", "language": "pol"},
        {"engine": "local.tesseract", "language": "eng"},
        {"engine": "local.tesseract", "language": "pol"},
    ]
    assert variant_version.id == version.id
    assert variant_version.definition_snapshot.nodes[0].config == {
        "engine": "local.tesseract",
        "engine_config": {"language": "pol"},
    }
    assert definition.nodes[0].config == {
        "engine": "local.text",
        "engine_config": {"language": "eng"},
    }

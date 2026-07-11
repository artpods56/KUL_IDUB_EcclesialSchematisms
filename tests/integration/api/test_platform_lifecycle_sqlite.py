from pathlib import Path
from uuid import uuid4

import pytest

from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    ArtifactSequence,
    Experiment,
    ExperimentParameter,
    ExperimentVariant,
    InputAssemblyTrace,
    InvocationTrace,
    NodeRun,
    NodeRunStatus,
    OutboxMessage,
    OutboxMessageStatus,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
)
from notarius_persistence.unit_of_work import create_sqlite_uow_factory


@pytest.mark.asyncio
async def test_sqlite_uow_persists_platform_lifecycle(tmp_path: Path) -> None:
    factory = create_sqlite_uow_factory(f"sqlite:///{tmp_path / 'studio.db'}")
    source_id = uuid4()
    input_ref = ArtifactRef(
        artifact_id=uuid4(),
        artifact_type="source.page_image",
        schema_version=1,
        content_hash="source-hash",
    )
    definition = WorkflowDefinition(
        name="OCR comparison",
        description="Compare page OCR providers",
        nodes=[
            WorkflowNode(
                id="ocr_a",
                operator_id="ocr.mistral",
                operator_version="1.0.0",
                config={"language": "pl"},
                label="Mistral OCR",
                ui_position={"x": 120, "y": 40},
            )
        ],
        edges=[
            WorkflowEdge(
                from_node_id="source",
                from_port="page",
                to_node_id="ocr_a",
                to_port="page",
            )
        ],
        declared_inputs=[
            PortSpec(
                name="page",
                artifact_type="source.page_image",
                schema_version=1,
            )
        ],
        metadata={"workspace": "integration"},
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
        created_by="integration-test",
        change_note="initial version",
    )
    run = WorkflowRun(
        workflow_version_id=version.id,
        input_artifact_refs=[input_ref],
        metadata={"priority": "normal"},
    )
    node_run = NodeRun(
        workflow_run_id=run.id,
        workflow_node_id="ocr_a",
        operator_id="ocr.mistral",
        operator_version="1.0.0",
        input_artifact_refs={"page": input_ref},
        metadata={"page_number": 1},
    )
    source_artifact = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/source/page-1.png",
        content_hash="source-hash",
        metadata={"source_id": str(source_id), "page_number": 1},
    )
    artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=run.id,
        producer_node_run_id=node_run.id,
        payload_ref="object://runs/one/ocr/page-1.json",
        producer_operator_id="ocr.mistral",
        producer_operator_version="1.0.0",
        input_artifact_ids=[input_ref.artifact_id],
        content_hash="ocr-hash",
        metadata={"confidence": 0.98},
    )
    output_ref = artifact.ref()
    sequence = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[output_ref],
        metadata={"source_id": str(source_id), "kind": "pages"},
    )
    run.input_artifact_sequence_refs = [sequence.ref()]
    node_run.input_artifact_refs["ocr_sequence"] = sequence.ref()
    input_trace = InputAssemblyTrace(
        node_run_id=node_run.id,
        selected_inputs={"page": input_ref, "pages": [input_ref]},
        omitted_inputs={"context": "not requested"},
        policies={"selection": "latest"},
        metadata={"source": "test"},
    )
    invocation_trace = InvocationTrace(
        node_run_id=node_run.id,
        invocation_type="ocr",
        input_artifact_refs=[input_ref],
        output_artifact_refs=[output_ref],
        provider="mistral",
        model="mistral-ocr",
        request_ref="object://requests/one.json",
        response_ref="object://responses/one.json",
        runtime={"duration_ms": 42},
        metadata={"tokens": 0},
    )
    outbox_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={
            "workflow_run_id": str(run.id),
            "node_run_id": str(node_run.id),
        },
    )
    run.mark_succeeded([output_ref])
    node_run.mark_succeeded(
        {
            "ocr_pages": [output_ref],
            "ocr_sequence": sequence.ref(),
        }
    )
    experiment = Experiment(
        name="OCR engine comparison",
        workflow_version_id=version.id,
        parameters=[
            ExperimentParameter(
                name="engine",
                node_id="ocr_a",
                config_path=("operator",),
                values=("mistral", "tesseract"),
            )
        ],
        input_artifact_refs=[input_ref],
        input_artifact_sequence_refs=[sequence.ref()],
        variants=[
            ExperimentVariant(
                key="variant-0001",
                ordinal=1,
                parameter_values={"engine": "mistral"},
                workflow_run_id=run.id,
            )
        ],
        metadata={"metric": "mean_confidence"},
    )

    async with factory() as uow:
        await uow.workflow_definitions.add(definition)
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(run)
        await uow.node_runs.add(node_run)
        await uow.artifacts.add(source_artifact)
        await uow.artifacts.add(artifact)
        await uow.artifact_sequences.add(sequence)
        await uow.experiments.add(experiment)
        await uow.input_assembly_traces.add(input_trace)
        await uow.invocation_traces.add(invocation_trace)
        await uow.outbox_messages.add(outbox_message)
        await uow.workflow_runs.update(run)
        await uow.node_runs.update(node_run)
        await uow.commit()

    async with factory() as uow:
        stored_definition = await uow.workflow_definitions.get(definition.id)
        definitions = await uow.workflow_definitions.list()
        stored_version = await uow.workflow_versions.get(version.id)
        versions = await uow.workflow_versions.list_for_definition(definition.id)
        latest_version = await uow.workflow_versions.latest_for_definition(
            definition.id
        )
        stored_run = await uow.workflow_runs.get(run.id)
        runs = await uow.workflow_runs.list_for_version(version.id)
        succeeded_runs = await uow.workflow_runs.list_by_status(
            WorkflowRunStatus.SUCCEEDED
        )
        stored_node_run = await uow.node_runs.get(node_run.id)
        node_runs = await uow.node_runs.list_for_workflow_run(run.id)
        succeeded_node_runs = await uow.node_runs.list_by_status(
            NodeRunStatus.SUCCEEDED
        )
        stored_source_artifact = await uow.artifacts.get(source_artifact.id)
        artifacts_for_source = await uow.artifacts.list_for_source(source_id)
        source_artifacts = await uow.artifacts.list_by_type("source.page_image")
        stored_artifact = await uow.artifacts.get(artifact.id)
        artifacts_for_run = await uow.artifacts.list_for_workflow_run(run.id)
        artifacts_for_node = await uow.artifacts.list_for_node_run(node_run.id)
        artifacts_by_type = await uow.artifacts.list_by_type("ocr.page_result")
        stored_sequence = await uow.artifact_sequences.get(sequence.id)
        sequences_for_source = await uow.artifact_sequences.list_for_source(source_id)
        sequences = await uow.artifact_sequences.list_by_artifact_type(
            "ocr.page_result"
        )
        stored_experiment = await uow.experiments.get(experiment.id)
        experiments = await uow.experiments.list()
        version_experiments = await uow.experiments.list_for_workflow_version(
            version.id
        )
        input_traces = await uow.input_assembly_traces.list_for_node_run(node_run.id)
        invocation_traces = await uow.invocation_traces.list_for_node_run(node_run.id)
        stored_outbox_message = await uow.outbox_messages.get(outbox_message.id)
        pending_outbox_messages = await uow.outbox_messages.list_pending()
        workflow_run_outbox_messages = await uow.outbox_messages.list_for_workflow_run(
            run.id
        )
        other_workflow_run_outbox_messages = (
            await uow.outbox_messages.list_for_workflow_run(uuid4())
        )

        outbox_message.mark_published()
        await uow.outbox_messages.update(outbox_message)
        await uow.commit()

    async with factory() as uow:
        published_outbox_messages = await uow.outbox_messages.list_by_status(
            OutboxMessageStatus.PUBLISHED
        )

    async with factory() as uow:
        deleted_outbox_messages_count = await uow.outbox_messages.delete_many(
            [outbox_message.id]
        )
        await uow.commit()

    async with factory() as uow:
        deleted_outbox_message = await uow.outbox_messages.get(outbox_message.id)

    assert stored_definition is not None
    assert stored_definition.id == definition.id
    assert stored_definition.nodes == definition.nodes
    assert stored_definition.edges == definition.edges
    assert stored_definition.declared_inputs == definition.declared_inputs
    assert stored_definition.metadata == {"workspace": "integration"}
    assert [item.id for item in definitions] == [definition.id]

    assert stored_version is not None
    assert stored_version.definition_snapshot.nodes == definition.nodes
    assert [item.id for item in versions] == [version.id]
    assert latest_version is not None
    assert latest_version.id == version.id

    assert stored_run is not None
    assert stored_run.status == WorkflowRunStatus.SUCCEEDED
    assert stored_run.input_artifact_refs == [input_ref]
    assert stored_run.input_artifact_sequence_refs == [sequence.ref()]
    assert stored_run.output_artifact_refs == [output_ref]
    assert [item.id for item in runs] == [run.id]
    assert [item.id for item in succeeded_runs] == [run.id]

    assert stored_node_run is not None
    assert stored_node_run.status == NodeRunStatus.SUCCEEDED
    assert stored_node_run.input_artifact_refs == {
        "page": input_ref,
        "ocr_sequence": sequence.ref(),
    }
    assert stored_node_run.output_artifact_refs == {
        "ocr_pages": [output_ref],
        "ocr_sequence": sequence.ref(),
    }
    assert [item.id for item in node_runs] == [node_run.id]
    assert [item.id for item in succeeded_node_runs] == [node_run.id]

    assert stored_source_artifact is not None
    assert stored_source_artifact.workflow_run_id is None
    assert [item.id for item in artifacts_for_source] == [source_artifact.id]
    assert [item.id for item in source_artifacts] == [source_artifact.id]

    assert stored_artifact is not None
    assert stored_artifact.input_artifact_ids == [input_ref.artifact_id]
    assert stored_artifact.metadata == {"confidence": 0.98}
    assert [item.id for item in artifacts_for_run] == [artifact.id]
    assert [item.id for item in artifacts_for_node] == [artifact.id]
    assert [item.id for item in artifacts_by_type] == [artifact.id]

    assert stored_sequence is not None
    assert stored_sequence.item_refs == [output_ref]
    assert [item.id for item in sequences_for_source] == [sequence.id]
    assert [item.id for item in sequences] == [sequence.id]

    assert stored_experiment is not None
    assert stored_experiment.workflow_version_id == version.id
    assert stored_experiment.parameters == experiment.parameters
    assert stored_experiment.workflow_run_ids == [run.id]
    assert stored_experiment.metadata == {"metric": "mean_confidence"}
    assert [item.id for item in experiments] == [experiment.id]
    assert [item.id for item in version_experiments] == [experiment.id]

    assert input_traces == [input_trace]
    assert invocation_traces == [invocation_trace]
    assert stored_outbox_message is not None
    assert stored_outbox_message.payload == {
        "workflow_run_id": str(run.id),
        "node_run_id": str(node_run.id),
    }
    assert [item.id for item in pending_outbox_messages] == [outbox_message.id]
    assert [item.id for item in workflow_run_outbox_messages] == [outbox_message.id]
    assert other_workflow_run_outbox_messages == []
    assert [item.id for item in published_outbox_messages] == [outbox_message.id]
    assert deleted_outbox_messages_count == 1
    assert deleted_outbox_message is None

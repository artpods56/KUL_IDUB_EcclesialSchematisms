import pytest

from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    Experiment,
    ExperimentParameter,
    ExperimentVariant,
    InputAssemblyTrace,
    InvocationTrace,
    NodeRun,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)


@pytest.mark.asyncio
async def test_in_memory_uow_persists_workflow_version_and_run() -> None:
    store = InMemoryDataStore()
    definition = WorkflowDefinition(
        name="OCR comparison",
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
    run = WorkflowRun(workflow_version_id=version.id)

    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_definitions.add(definition)
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(run)
        await uow.commit()

    async with InMemoryUnitOfWork(store) as uow:
        stored_definition = await uow.workflow_definitions.get(definition.id)
        latest_version = await uow.workflow_versions.latest_for_definition(
            definition.id
        )
        queued_run = await uow.workflow_runs.next_queued()

    assert stored_definition == definition
    assert latest_version == version
    assert queued_run == run


@pytest.mark.asyncio
async def test_in_memory_uow_persists_node_artifacts_and_traces() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=WorkflowDefinition(name="base").id)
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr_a",
        operator_id="ocr.mistral",
        operator_version="1.0.0",
    )
    artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="s3://notarius/runs/one/ocr/page-1.json",
    )
    input_trace = InputAssemblyTrace(
        node_run_id=node_run.id,
        selected_inputs={"page": ArtifactRef(artifact.id, artifact.artifact_type, 1)},
    )
    invocation_trace = InvocationTrace(
        node_run_id=node_run.id,
        invocation_type="ocr",
        provider="mistral",
        model="mistral-ocr",
        output_artifact_refs=[artifact.ref()],
    )

    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.artifacts.add(artifact)
        await uow.input_assembly_traces.add(input_trace)
        await uow.invocation_traces.add(invocation_trace)
        await uow.commit()

    async with InMemoryUnitOfWork(store) as uow:
        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)
        input_traces = await uow.input_assembly_traces.list_for_node_run(node_run.id)
        invocation_traces = await uow.invocation_traces.list_for_node_run(node_run.id)

    assert node_runs == [node_run]
    assert artifacts == [artifact]
    assert input_traces == [input_trace]
    assert invocation_traces == [invocation_trace]


@pytest.mark.asyncio
async def test_in_memory_uow_persists_experiments() -> None:
    store = InMemoryDataStore()
    definition = WorkflowDefinition(name="Prompt variants")
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    run = WorkflowRun(workflow_version_id=version.id)
    experiment = Experiment(
        name="Prompt comparison",
        workflow_version_id=version.id,
        parameters=[
            ExperimentParameter(
                name="prompt",
                node_id="prompt",
                config_path=("template",),
                values=("A", "B"),
            )
        ],
        variants=[
            ExperimentVariant(
                key="variant-0001",
                ordinal=1,
                parameter_values={"prompt": "A"},
                workflow_run_id=run.id,
            )
        ],
    )

    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_definitions.add(definition)
        await uow.workflow_versions.add(version)
        await uow.workflow_runs.add(run)
        await uow.experiments.add(experiment)
        await uow.commit()

    async with InMemoryUnitOfWork(store) as uow:
        stored = await uow.experiments.get(experiment.id)
        experiments = await uow.experiments.list()
        version_experiments = await uow.experiments.list_for_workflow_version(
            version.id
        )

    assert stored == experiment
    assert experiments == [experiment]
    assert version_experiments == [experiment]

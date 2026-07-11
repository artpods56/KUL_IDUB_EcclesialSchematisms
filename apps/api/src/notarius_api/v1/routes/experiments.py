from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    ExperimentComparisonResponse,
    ExperimentCreate,
    ExperimentEvaluationMetricResponse,
    ExperimentExecutionCreate,
    ExperimentExecutionResponse,
    ExperimentExecutionVariantResponse,
    ExperimentMetricValueResponse,
    ExperimentOutputBundleResponse,
    ExperimentRerunFailedResponse,
    ExperimentRerunVariantResponse,
    ExperimentResponse,
    ExperimentVariantComparisonResponse,
    ExperimentVariantOutputBundleResponse,
    WorkflowRunExecutionNodeError,
    WorkflowRunResponse,
)
from notarius_api.services.output_bundles import WorkflowRunOutputBundleService
from notarius_api.services.workflow_execution import WorkflowRunExecutionService
from notarius_core.application.experiments import (
    apply_experiment_parameters,
    expand_parameter_grid,
)
from notarius_core.application.workflows import NodeSpecRegistry, WorkflowRunLauncher
from notarius_core.domain.errors import ConflictError, NotFoundError, ValidationError
from notarius_core.domain.models import (
    Artifact,
    ArtifactSequence,
    ArtifactSequenceRef,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
    InvocationTrace,
    NodeRun,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowVersion,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import RunEventType
from notarius_messaging.outbox import (
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)

router = APIRouter(tags=["experiments"])

WorkflowRunExecutionServiceDependency = Annotated[
    WorkflowRunExecutionService,
    Depends(deps.create_workflow_run_execution_service),
]
WorkflowRunOutputBundleServiceDependency = Annotated[
    WorkflowRunOutputBundleService,
    Depends(deps.create_workflow_run_output_bundle_service),
]
UowFactoryDependency = Annotated[
    Callable[[], StudioUnitOfWorkPort],
    Depends(deps.create_uow_factory),
]

_DURATION_RUNTIME_KEYS = {
    "duration_ms",
    "elapsed_ms",
    "latency_ms",
    "runtime_ms",
}
_COST_KEYS = {
    "cost",
    "cost_usd",
    "estimated_cost",
    "estimated_cost_usd",
    "total_cost",
    "total_cost_usd",
}
_RERUNNABLE_WORKFLOW_RUN_STATUSES = {
    WorkflowRunStatus.FAILED_RETRYABLE,
    WorkflowRunStatus.FAILED_PERMANENT,
    WorkflowRunStatus.CANCELLED,
}
_FAILED_WORKFLOW_RUN_STATUSES = {
    WorkflowRunStatus.FAILED_RETRYABLE,
    WorkflowRunStatus.FAILED_PERMANENT,
}
_FINISHED_EXPERIMENT_RUN_STATUSES = {
    WorkflowRunStatus.SUCCEEDED,
    WorkflowRunStatus.FAILED_RETRYABLE,
    WorkflowRunStatus.FAILED_PERMANENT,
    WorkflowRunStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class _ExperimentVariantRerunResult:
    variant_id: UUID
    variant_key: str
    previous_workflow_run_id: UUID
    workflow_run_id: UUID


@router.post(
    "/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_experiment(
    body: ExperimentCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> ExperimentResponse:
    async with uow:
        version = await deps.get_workflow_version_or_404(uow, body.workflow_version_id)
        input_artifact_refs = [ref.to_domain() for ref in body.input_artifact_refs]
        input_sequences = [
            await _get_matching_artifact_sequence(uow, sequence_ref.to_domain())
            for sequence_ref in body.input_artifact_sequence_refs
        ]
        try:
            parameters = [
                parameter.to_domain()
                for parameter in [*body.parameters, *body.parameter_presets]
            ]
            parameter_grid = expand_parameter_grid(parameters)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        experiment = Experiment(
            name=body.name,
            description=body.description,
            workflow_version_id=version.id,
            parameters=parameters,
            input_artifact_refs=input_artifact_refs,
            input_artifact_sequence_refs=[
                sequence.ref() for sequence in input_sequences
            ],
            metadata=body.metadata,
            status=ExperimentStatus.QUEUED,
        )
        variants: list[ExperimentVariant] = []
        for ordinal, parameter_values in enumerate(parameter_grid, start=1):
            variant_key = f"variant-{ordinal:04d}"
            try:
                variant_version = apply_experiment_parameters(
                    version,
                    parameters,
                    parameter_values,
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc

            launch_result = await WorkflowRunLauncher(node_specs).launch(
                uow,
                variant_version,
                input_artifact_refs,
                metadata={
                    "experiment_id": str(experiment.id),
                    "experiment_variant_key": variant_key,
                    "experiment_parameter_values": parameter_values,
                },
                node_run_outbox_message_builder=node_run_execute_requested_outbox_message,
                workflow_run_queued_event_builder=partial(
                    workflow_run_event_outbox_message,
                    event_type=RunEventType.QUEUED,
                ),
                node_run_queued_event_builder=partial(
                    node_run_event_outbox_message,
                    event_type=RunEventType.QUEUED,
                ),
                input_artifact_sequences=input_sequences,
            )
            variants.append(
                ExperimentVariant(
                    key=variant_key,
                    ordinal=ordinal,
                    parameter_values=parameter_values,
                    workflow_run_id=launch_result.workflow_run.id,
                )
            )

        experiment.variants = variants
        await uow.experiments.add(experiment)
        await uow.commit()
        return ExperimentResponse.from_domain(experiment)


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_experiments(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ExperimentResponse]:
    async with uow:
        return [
            ExperimentResponse.from_domain(experiment)
            for experiment in await uow.experiments.list()
        ]


@router.get(
    "/workflow-versions/{workflow_version_id}/experiments",
    response_model=list[ExperimentResponse],
)
async def list_workflow_version_experiments(
    workflow_version_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ExperimentResponse]:
    async with uow:
        await deps.get_workflow_version_or_404(uow, workflow_version_id)
        return [
            ExperimentResponse.from_domain(experiment)
            for experiment in await uow.experiments.list_for_workflow_version(
                workflow_version_id
            )
        ]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ExperimentResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        return ExperimentResponse.from_domain(experiment)


@router.get(
    "/experiments/{experiment_id}/comparison",
    response_model=ExperimentComparisonResponse,
)
async def compare_experiment(
    experiment_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ExperimentComparisonResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        variants: list[ExperimentVariantComparisonResponse] = []
        for variant in experiment.variants:
            run = await deps.get_workflow_run_or_404(uow, variant.workflow_run_id)
            node_runs = await uow.node_runs.list_for_workflow_run(run.id)
            artifacts = await uow.artifacts.list_for_workflow_run(run.id)
            invocation_traces: list[InvocationTrace] = []
            for node_run in node_runs:
                invocation_traces.extend(
                    await uow.invocation_traces.list_for_node_run(node_run.id)
                )

            node_run_status_counts = _node_run_status_counts(node_runs)
            artifact_counts = _artifact_counts(artifacts)
            invocation_count = len(invocation_traces)
            validation_error_count = _validation_error_count(
                artifacts,
                invocation_traces,
            )
            total_duration_ms = _total_duration_ms(
                run,
                node_runs,
                invocation_traces,
            )
            total_cost = _total_cost(invocation_traces)
            evaluation_metrics = [
                ExperimentEvaluationMetricResponse(
                    artifact_id=artifact.id,
                    producer_node_run_id=artifact.producer_node_run_id,
                    metadata=artifact.metadata,
                )
                for artifact in artifacts
                if artifact.artifact_type == "evaluation.metrics"
            ]
            metric_values = _experiment_metric_values(
                artifacts=artifacts,
                invocation_count=invocation_count,
                validation_error_count=validation_error_count,
                total_duration_ms=total_duration_ms,
                total_cost=total_cost,
            )

            variants.append(
                ExperimentVariantComparisonResponse(
                    variant_id=variant.id,
                    variant_key=variant.key,
                    ordinal=variant.ordinal,
                    parameter_values=variant.parameter_values,
                    workflow_run_id=run.id,
                    workflow_run_status=run.status,
                    node_run_status_counts=node_run_status_counts,
                    artifact_counts=artifact_counts,
                    invocation_count=invocation_count,
                    validation_error_count=validation_error_count,
                    total_duration_ms=total_duration_ms,
                    total_cost=total_cost,
                    evaluation_metrics=evaluation_metrics,
                    metric_values=metric_values,
                    errors=_run_errors(run, node_runs, invocation_traces),
                )
            )

        return ExperimentComparisonResponse(
            experiment_id=experiment.id,
            workflow_version_id=experiment.workflow_version_id,
            variant_count=len(experiment.variants),
            metric_names=sorted(
                {
                    metric.name
                    for variant in variants
                    for metric in variant.metric_values
                }
            ),
            variants=variants,
        )


@router.get(
    "/experiments/{experiment_id}/outputs",
    response_model=ExperimentOutputBundleResponse,
)
async def get_experiment_outputs(
    experiment_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    output_bundle_service: WorkflowRunOutputBundleServiceDependency,
    artifact_type: str | None = None,
    include_payloads: bool = False,
    include_text_payloads: bool = False,
    include_traces: bool = False,
) -> ExperimentOutputBundleResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        variants = list(experiment.variants)

    variant_bundles: list[ExperimentVariantOutputBundleResponse] = []
    for variant in variants:
        output_bundle = await output_bundle_service.build_workflow_run_output_bundle(
            variant.workflow_run_id,
            artifact_type=artifact_type,
            include_payloads=include_payloads,
            include_text_payloads=include_text_payloads,
            include_traces=include_traces,
        )
        variant_bundles.append(
            ExperimentVariantOutputBundleResponse(
                variant_id=variant.id,
                variant_key=variant.key,
                ordinal=variant.ordinal,
                parameter_values=variant.parameter_values,
                workflow_run_id=variant.workflow_run_id,
                output_bundle=output_bundle,
            )
        )

    return ExperimentOutputBundleResponse(
        experiment=ExperimentResponse.from_domain(experiment),
        variants=variant_bundles,
    )


@router.post(
    "/experiments/{experiment_id}/execute",
    response_model=ExperimentExecutionResponse,
)
async def execute_experiment(
    experiment_id: UUID,
    body: ExperimentExecutionCreate,
    execution_service: WorkflowRunExecutionServiceDependency,
    uow_factory: UowFactoryDependency,
) -> ExperimentExecutionResponse:
    async with uow_factory() as uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        variants = list(experiment.variants)

    variant_responses: list[ExperimentExecutionVariantResponse] = []
    for variant in variants:
        result = await execution_service.execute_workflow_run(
            variant.workflow_run_id,
            max_node_runs=body.max_node_runs_per_variant,
        )
        errors = [
            WorkflowRunExecutionNodeError(
                node_run_id=error.node_run_id,
                error=error.error,
            )
            for error in result.errors
        ]
        variant_responses.append(
            ExperimentExecutionVariantResponse(
                variant_id=variant.id,
                variant_key=variant.key,
                workflow_run_id=variant.workflow_run_id,
                workflow_run=WorkflowRunResponse.from_domain(result.workflow_run),
                processed_node_run_ids=result.processed_node_run_ids,
                errors=errors,
            )
        )
        if body.stop_on_error and errors:
            break

    async with uow_factory() as uow:
        updated_experiment = await deps.get_experiment_or_404(uow, experiment_id)
        updated_experiment.status = await _experiment_status_from_variant_runs(
            uow,
            updated_experiment,
        )
        updated_experiment.updated_at = datetime.now(UTC)
        await uow.experiments.update(updated_experiment)
        await uow.commit()

    return ExperimentExecutionResponse(
        experiment=ExperimentResponse.from_domain(updated_experiment),
        variants=variant_responses,
    )


@router.post(
    "/experiments/{experiment_id}/cancel",
    response_model=ExperimentResponse,
)
async def cancel_experiment(
    experiment_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ExperimentResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        cancelled_count = 0
        for variant in experiment.variants:
            run = await deps.get_workflow_run_or_404(uow, variant.workflow_run_id)
            if run.is_terminal:
                continue

            await _cancel_open_workflow_run(uow, run)
            cancelled_count += 1

        if cancelled_count == 0 and experiment.status != ExperimentStatus.CANCELLED:
            raise ConflictError(
                f"Cannot cancel experiment {experiment.id}: no open workflow runs"
            )

        experiment.status = ExperimentStatus.CANCELLED
        experiment.updated_at = datetime.now(UTC)
        await uow.experiments.update(experiment)
        await uow.commit()
        return ExperimentResponse.from_domain(experiment)


@router.post(
    "/experiments/{experiment_id}/variants/{variant_id}/cancel",
    response_model=ExperimentResponse,
)
async def cancel_experiment_variant(
    experiment_id: UUID,
    variant_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ExperimentResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        variant = _experiment_variant_or_404(experiment, variant_id)
        run = await deps.get_workflow_run_or_404(uow, variant.workflow_run_id)
        if run.is_terminal and run.status != WorkflowRunStatus.CANCELLED:
            raise ConflictError(
                "Cannot cancel experiment variant "
                f"{variant.key}: workflow run {run.id} is {run.status.value}"
            )

        await _cancel_open_workflow_run(uow, run)

        experiment.status = await _experiment_status_from_variant_runs(uow, experiment)
        experiment.updated_at = datetime.now(UTC)
        await uow.experiments.update(experiment)
        await uow.commit()
        return ExperimentResponse.from_domain(experiment)


@router.post(
    "/experiments/{experiment_id}/variants/{variant_id}/rerun",
    response_model=ExperimentResponse,
)
async def rerun_experiment_variant(
    experiment_id: UUID,
    variant_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> ExperimentResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        variant = _experiment_variant_or_404(experiment, variant_id)
        previous_run = await deps.get_workflow_run_or_404(
            uow,
            variant.workflow_run_id,
        )
        if previous_run.status not in _RERUNNABLE_WORKFLOW_RUN_STATUSES:
            raise ConflictError(
                "Cannot rerun experiment variant "
                f"{variant.key}: workflow run {previous_run.id} is "
                f"{previous_run.status.value}"
            )

        version = await deps.get_workflow_version_or_404(
            uow,
            experiment.workflow_version_id,
        )
        input_sequences = [
            await _get_matching_artifact_sequence(uow, sequence_ref)
            for sequence_ref in experiment.input_artifact_sequence_refs
        ]

        await _rerun_experiment_variant(
            uow,
            experiment,
            variant,
            previous_run,
            version,
            input_sequences,
            node_specs,
        )
        experiment.status = ExperimentStatus.QUEUED
        experiment.updated_at = datetime.now(UTC)
        await uow.experiments.update(experiment)
        await uow.commit()
        return ExperimentResponse.from_domain(experiment)


@router.post(
    "/experiments/{experiment_id}/rerun-failed",
    response_model=ExperimentRerunFailedResponse,
)
async def rerun_failed_experiment_variants(
    experiment_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> ExperimentRerunFailedResponse:
    async with uow:
        experiment = await deps.get_experiment_or_404(uow, experiment_id)
        version = await deps.get_workflow_version_or_404(
            uow,
            experiment.workflow_version_id,
        )
        input_sequences = [
            await _get_matching_artifact_sequence(uow, sequence_ref)
            for sequence_ref in experiment.input_artifact_sequence_refs
        ]

        rerun_results: list[_ExperimentVariantRerunResult] = []
        for variant in experiment.variants:
            previous_run = await deps.get_workflow_run_or_404(
                uow,
                variant.workflow_run_id,
            )
            if previous_run.status not in _FAILED_WORKFLOW_RUN_STATUSES:
                continue

            rerun_results.append(
                await _rerun_experiment_variant(
                    uow,
                    experiment,
                    variant,
                    previous_run,
                    version,
                    input_sequences,
                    node_specs,
                )
            )

        if not rerun_results:
            raise ConflictError(
                f"Cannot rerun failed variants for experiment {experiment.id}: "
                "no failed workflow runs"
            )

        experiment.status = ExperimentStatus.QUEUED
        experiment.updated_at = datetime.now(UTC)
        await uow.experiments.update(experiment)
        await uow.commit()
        return ExperimentRerunFailedResponse(
            experiment=ExperimentResponse.from_domain(experiment),
            variants=[
                ExperimentRerunVariantResponse(
                    variant_id=result.variant_id,
                    variant_key=result.variant_key,
                    previous_workflow_run_id=result.previous_workflow_run_id,
                    workflow_run_id=result.workflow_run_id,
                )
                for result in rerun_results
            ],
        )


async def _get_matching_artifact_sequence(
    uow: StudioUnitOfWorkPort,
    sequence_ref: ArtifactSequenceRef,
) -> ArtifactSequence:
    sequence = await deps.get_artifact_sequence_or_404(uow, sequence_ref.sequence_id)
    if sequence.artifact_type != sequence_ref.artifact_type:
        raise ValidationError(
            "Artifact sequence type mismatch for "
            f"{sequence.id}: expected {sequence_ref.artifact_type}, "
            f"got {sequence.artifact_type}"
        )
    if sequence.schema_version != sequence_ref.schema_version:
        raise ValidationError(
            "Artifact sequence schema version mismatch for "
            f"{sequence.id}: expected {sequence_ref.schema_version}, "
            f"got {sequence.schema_version}"
        )
    return sequence


def _experiment_variant_or_404(
    experiment: Experiment,
    variant_id: UUID,
) -> ExperimentVariant:
    for variant in experiment.variants:
        if variant.id == variant_id:
            return variant
    raise NotFoundError(
        "ExperimentVariant",
        f"{experiment.id}/{variant_id}",
    )


async def _cancel_open_workflow_run(
    uow: StudioUnitOfWorkPort,
    run: WorkflowRun,
) -> None:
    node_runs = await uow.node_runs.list_for_workflow_run(run.id)
    for node_run in node_runs:
        if node_run.is_terminal:
            continue
        node_run.mark_cancelled()
        await uow.node_runs.update(node_run)
        await uow.outbox_messages.add(
            node_run_event_outbox_message(
                node_run,
                RunEventType.CANCELLED,
            )
        )

    if run.status != WorkflowRunStatus.CANCELLED:
        run.mark_cancelled()
        await uow.workflow_runs.update(run)
        await uow.outbox_messages.add(
            workflow_run_event_outbox_message(
                run,
                RunEventType.CANCELLED,
            )
        )


async def _rerun_experiment_variant(
    uow: StudioUnitOfWorkPort,
    experiment: Experiment,
    variant: ExperimentVariant,
    previous_run: WorkflowRun,
    version: WorkflowVersion,
    input_sequences: list[ArtifactSequence],
    node_specs: NodeSpecRegistry,
) -> _ExperimentVariantRerunResult:
    try:
        variant_version = apply_experiment_parameters(
            version,
            experiment.parameters,
            variant.parameter_values,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    launch_result = await WorkflowRunLauncher(node_specs).launch(
        uow,
        variant_version,
        experiment.input_artifact_refs,
        metadata={
            "experiment_id": str(experiment.id),
            "experiment_variant_id": str(variant.id),
            "experiment_variant_key": variant.key,
            "experiment_parameter_values": variant.parameter_values,
            "experiment_rerun_of_workflow_run_id": str(previous_run.id),
        },
        node_run_outbox_message_builder=node_run_execute_requested_outbox_message,
        workflow_run_queued_event_builder=partial(
            workflow_run_event_outbox_message,
            event_type=RunEventType.QUEUED,
        ),
        node_run_queued_event_builder=partial(
            node_run_event_outbox_message,
            event_type=RunEventType.QUEUED,
        ),
        input_artifact_sequences=input_sequences,
        commit=False,
    )

    previous_run_ids = _previous_workflow_run_ids(variant)
    previous_run_ids.append(str(previous_run.id))
    variant.workflow_run_id = launch_result.workflow_run.id
    variant.metadata = {
        **variant.metadata,
        "previous_workflow_run_ids": previous_run_ids,
        "rerun_count": len(previous_run_ids),
        "rerun_of_workflow_run_id": str(previous_run.id),
    }
    return _ExperimentVariantRerunResult(
        variant_id=variant.id,
        variant_key=variant.key,
        previous_workflow_run_id=previous_run.id,
        workflow_run_id=launch_result.workflow_run.id,
    )


async def _experiment_status_from_variant_runs(
    uow: StudioUnitOfWorkPort,
    experiment: Experiment,
) -> ExperimentStatus:
    statuses = [
        (await deps.get_workflow_run_or_404(uow, variant.workflow_run_id)).status
        for variant in experiment.variants
    ]
    if not statuses:
        return ExperimentStatus.QUEUED
    if all(status == WorkflowRunStatus.CANCELLED for status in statuses):
        return ExperimentStatus.CANCELLED
    if all(status == WorkflowRunStatus.SUCCEEDED for status in statuses):
        return ExperimentStatus.SUCCEEDED
    if (
        any(status in _FAILED_WORKFLOW_RUN_STATUSES for status in statuses)
        and all(status in _FINISHED_EXPERIMENT_RUN_STATUSES for status in statuses)
    ):
        return ExperimentStatus.FAILED
    if any(status == WorkflowRunStatus.RUNNING for status in statuses):
        return ExperimentStatus.RUNNING
    return ExperimentStatus.QUEUED


def _previous_workflow_run_ids(variant: ExperimentVariant) -> list[str]:
    value = variant.metadata.get("previous_workflow_run_ids")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _node_run_status_counts(node_runs: list[NodeRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_run in node_runs:
        status = node_run.status.value
        counts[status] = counts.get(status, 0) + 1
    return counts


def _artifact_counts(artifacts: list[Artifact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact.artifact_type] = counts.get(artifact.artifact_type, 0) + 1
    return counts


def _experiment_metric_values(
    artifacts: list[Artifact],
    invocation_count: int,
    validation_error_count: int,
    total_duration_ms: float | None,
    total_cost: float | None,
) -> list[ExperimentMetricValueResponse]:
    metric_values = [
        ExperimentMetricValueResponse(
            name="summary.invocation_count",
            value=invocation_count,
            source="summary",
        ),
        ExperimentMetricValueResponse(
            name="summary.validation_error_count",
            value=validation_error_count,
            source="summary",
        ),
    ]
    if total_duration_ms is not None:
        metric_values.append(
            ExperimentMetricValueResponse(
                name="summary.total_duration_ms",
                value=total_duration_ms,
                source="summary",
            )
        )
    if total_cost is not None:
        metric_values.append(
            ExperimentMetricValueResponse(
                name="summary.total_cost",
                value=total_cost,
                source="summary",
            )
        )

    for artifact in artifacts:
        if artifact.artifact_type != "evaluation.metrics":
            continue

        metric_family_value = artifact.metadata.get("metric_family")
        metric_family = (
            metric_family_value
            if isinstance(metric_family_value, str) and metric_family_value
            else "evaluation.metrics"
        )
        for key in sorted(artifact.metadata):
            if key == "metric_family":
                continue
            value = artifact.metadata[key]
            if not _is_metric_primitive(value):
                continue
            metric_values.append(
                ExperimentMetricValueResponse(
                    name=f"{metric_family}.{key}",
                    value=value,
                    source="evaluation.metrics",
                    artifact_id=artifact.id,
                    producer_node_run_id=artifact.producer_node_run_id,
                )
            )

    return metric_values


def _is_metric_primitive(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _validation_error_count(
    artifacts: list[Artifact],
    invocation_traces: list[InvocationTrace],
) -> int:
    document_counts = [
        _int_metadata_value(artifact.metadata, "validation_error_count")
        for artifact in artifacts
        if artifact.artifact_type == "extraction.document_result"
        and _int_metadata_value(artifact.metadata, "validation_error_count") is not None
    ]
    if document_counts:
        return sum(document_counts)

    record_counts = [
        _int_metadata_value(artifact.metadata, "validation_error_count")
        for artifact in artifacts
        if artifact.artifact_type == "extraction.record_result"
        and _int_metadata_value(artifact.metadata, "validation_error_count") is not None
    ]
    if record_counts:
        return sum(record_counts)

    return sum(
        count
        for count in (
            _int_metadata_value(trace.runtime, "validation_error_count")
            for trace in invocation_traces
        )
        if count is not None
    )


def _total_duration_ms(
    run: WorkflowRun,
    node_runs: list[NodeRun],
    invocation_traces: list[InvocationTrace],
) -> float | None:
    trace_total = _sum_numeric_keys(
        [trace.runtime for trace in invocation_traces],
        _DURATION_RUNTIME_KEYS,
    )
    if trace_total is not None:
        return trace_total

    node_durations = [
        _elapsed_ms(node_run.started_at, node_run.finished_at)
        for node_run in node_runs
        if node_run.started_at is not None and node_run.finished_at is not None
    ]
    if node_durations:
        return sum(node_durations)

    return _elapsed_ms(run.started_at, run.finished_at)


def _total_cost(invocation_traces: list[InvocationTrace]) -> float | None:
    values = [trace.runtime for trace in invocation_traces]
    values.extend(trace.metadata for trace in invocation_traces)
    return _sum_numeric_keys(values, _COST_KEYS)


def _run_errors(
    run: WorkflowRun,
    node_runs: list[NodeRun],
    invocation_traces: list[InvocationTrace],
) -> list[str]:
    errors: list[str] = []
    if run.error:
        errors.append(run.error)
    errors.extend(node_run.error for node_run in node_runs if node_run.error)
    errors.extend(trace.error for trace in invocation_traces if trace.error)
    return errors


def _sum_numeric_keys(
    values: list[dict[str, object]],
    keys: set[str],
) -> float | None:
    total = 0.0
    found = False
    for value in values:
        for key in keys:
            number = _float_metadata_value(value, key)
            if number is None:
                continue
            total += number
            found = True
    return total if found else None


def _int_metadata_value(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_metadata_value(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _elapsed_ms(started_at: object, finished_at: object) -> float | None:
    if started_at is None or finished_at is None:
        return None
    return (finished_at - started_at).total_seconds() * 1000

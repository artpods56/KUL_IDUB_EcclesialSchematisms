from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    WorkflowDefinitionResponse,
    WorkflowTemplateLaunchCreate,
    WorkflowTemplateLaunchResponse,
    WorkflowTemplateMaterializeCreate,
    WorkflowTemplateMaterializeResponse,
    WorkflowTemplateResponse,
    WorkflowRunResponse,
    WorkflowVersionResponse,
)
from notarius_core.application.workflows import (
    NodeSpecRegistry,
    WorkflowRunLauncher,
)
from notarius_core.application.workflows.templates import (
    WorkflowTemplate,
    build_workflow_definition_from_template,
    list_workflow_templates,
    workflow_template,
    workflow_template_id,
)
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    ArtifactSequence,
    ArtifactSequenceRef,
    WorkflowVersion,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import RunEventType
from notarius_messaging.outbox import (
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)

router = APIRouter(tags=["workflow-templates"])


@router.get("/workflow-templates", response_model=list[WorkflowTemplateResponse])
async def list_templates() -> list[WorkflowTemplateResponse]:
    return [
        _workflow_template_response(template)
        for template in list_workflow_templates()
    ]


@router.get(
    "/workflow-templates/{template_id}",
    response_model=WorkflowTemplateResponse,
)
async def get_template(template_id: str) -> WorkflowTemplateResponse:
    resolved_template_id = workflow_template_id(template_id)
    return _workflow_template_response(workflow_template(resolved_template_id))


@router.post(
    "/workflow-templates/{template_id}/materialize",
    response_model=WorkflowTemplateMaterializeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def materialize_template(
    template_id: str,
    body: WorkflowTemplateMaterializeCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowTemplateMaterializeResponse:
    resolved_template_id = workflow_template_id(template_id)
    definition = build_workflow_definition_from_template(
        resolved_template_id,
        body.config,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
    )
    template = workflow_template(resolved_template_id)

    async with uow:
        await uow.workflow_definitions.add(definition)
        version = WorkflowVersion(
            workflow_definition_id=definition.id,
            version_number=1,
            definition_snapshot=definition,
            created_by=body.created_by,
            change_note=body.change_note
            or f"Materialized from template {template.id.value}",
        )
        await uow.workflow_versions.add(version)
        await uow.commit()
        return WorkflowTemplateMaterializeResponse(
            template=_workflow_template_response(template),
            workflow_definition=WorkflowDefinitionResponse.from_domain(definition),
            workflow_version=WorkflowVersionResponse.from_domain(version),
        )


@router.post(
    "/workflow-templates/{template_id}/launch",
    response_model=WorkflowTemplateLaunchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def launch_template(
    template_id: str,
    body: WorkflowTemplateLaunchCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> WorkflowTemplateLaunchResponse:
    resolved_template_id = workflow_template_id(template_id)
    definition = build_workflow_definition_from_template(
        resolved_template_id,
        body.config,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
    )
    template = workflow_template(resolved_template_id)

    async with uow:
        input_sequences = [
            await _get_matching_artifact_sequence(uow, sequence_ref.to_domain())
            for sequence_ref in body.input_artifact_sequence_refs
        ]
        await uow.workflow_definitions.add(definition)
        version = WorkflowVersion(
            workflow_definition_id=definition.id,
            version_number=1,
            definition_snapshot=definition,
            created_by=body.created_by,
            change_note=body.change_note
            or f"Launched from template {template.id.value}",
        )
        await uow.workflow_versions.add(version)
        result = await WorkflowRunLauncher(node_specs).launch(
            uow,
            version,
            [ref.to_domain() for ref in body.input_artifact_refs],
            metadata=body.metadata,
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
        return WorkflowTemplateLaunchResponse(
            template=_workflow_template_response(template),
            workflow_definition=WorkflowDefinitionResponse.from_domain(definition),
            workflow_version=WorkflowVersionResponse.from_domain(version),
            workflow_run=WorkflowRunResponse.from_domain(result.workflow_run),
            queued_node_run_ids=list(result.queued_node_run_ids),
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


def _workflow_template_response(
    template: WorkflowTemplate,
) -> WorkflowTemplateResponse:
    return WorkflowTemplateResponse(
        id=template.id.value,
        version=template.version,
        display_name=template.display_name,
        description=template.description,
        config_schema=template.config_schema,
    )

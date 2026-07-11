from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from notarius_core.application.workflows import (
    NodeRunDependency,
    WorkflowExecutionPlan,
)
from notarius_core.domain.models import (
    Artifact,
    ArtifactPortRef,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    ExecutionMode,
    Experiment,
    ExperimentParameter,
    ExperimentStatus,
    ExperimentVariant,
    InputAssemblyTrace,
    InvocationTrace,
    NodeSpec,
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


class PlatformResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ArtifactRefSchema(PlatformResponse):
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_hash: str | None = None

    def to_domain(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            schema_version=self.schema_version,
            content_hash=self.content_hash,
        )

    @classmethod
    def from_domain(cls, ref: ArtifactRef) -> "ArtifactRefSchema":
        return cls.model_validate(ref)


class ArtifactSequenceRefSchema(PlatformResponse):
    sequence_id: UUID
    artifact_type: str
    schema_version: int

    def to_domain(self) -> ArtifactSequenceRef:
        return ArtifactSequenceRef(
            sequence_id=self.sequence_id,
            artifact_type=self.artifact_type,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_domain(cls, ref: ArtifactSequenceRef) -> "ArtifactSequenceRefSchema":
        return cls.model_validate(ref)


class PortSpecSchema(PlatformResponse):
    name: str
    artifact_type: str
    schema_version: int
    sequence: bool = False
    required: bool = True
    description: str | None = None

    def to_domain(self) -> PortSpec:
        return PortSpec(
            name=self.name,
            artifact_type=self.artifact_type,
            schema_version=self.schema_version,
            sequence=self.sequence,
            required=self.required,
            description=self.description,
        )

    @classmethod
    def from_domain(cls, port: PortSpec) -> "PortSpecSchema":
        return cls.model_validate(port)


class ArtifactTypePortUseResponse(PlatformResponse):
    operator_id: str
    operator_version: str
    port_name: str
    sequence: bool
    required: bool


class ArtifactTypeResponse(PlatformResponse):
    artifact_type: str
    schema_version: int
    sequence: bool
    consumed_by: list[ArtifactTypePortUseResponse] = Field(default_factory=list)
    produced_by: list[ArtifactTypePortUseResponse] = Field(default_factory=list)


class ArtifactPayloadSchemaResponse(PlatformResponse):
    artifact_type: str
    schema_version: int
    content_type: str = "application/json"
    json_schema: dict[str, Any]


class WorkflowNodeSchema(PlatformResponse):
    id: str
    operator_id: str
    operator_version: str
    config: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None
    ui_position: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> WorkflowNode:
        return WorkflowNode(
            id=self.id,
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            config=self.config,
            label=self.label,
            ui_position=self.ui_position,
        )

    @classmethod
    def from_domain(cls, node: WorkflowNode) -> "WorkflowNodeSchema":
        return cls.model_validate(node)


class WorkflowEdgeSchema(PlatformResponse):
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str

    def to_domain(self) -> WorkflowEdge:
        return WorkflowEdge(
            from_node_id=self.from_node_id,
            from_port=self.from_port,
            to_node_id=self.to_node_id,
            to_port=self.to_port,
        )

    @classmethod
    def from_domain(cls, edge: WorkflowEdge) -> "WorkflowEdgeSchema":
        return cls.model_validate(edge)


class WorkflowDefinitionCreate(BaseModel):
    name: str
    description: str | None = None
    nodes: list[WorkflowNodeSchema] = Field(default_factory=list)
    edges: list[WorkflowEdgeSchema] = Field(default_factory=list)
    declared_inputs: list[PortSpecSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            name=self.name,
            description=self.description,
            nodes=[node.to_domain() for node in self.nodes],
            edges=[edge.to_domain() for edge in self.edges],
            declared_inputs=[port.to_domain() for port in self.declared_inputs],
            metadata=self.metadata,
        )


class WorkflowDefinitionResponse(PlatformResponse):
    id: UUID
    name: str
    description: str | None
    nodes: list[WorkflowNodeSchema]
    edges: list[WorkflowEdgeSchema]
    declared_inputs: list[PortSpecSchema]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls, definition: WorkflowDefinition
    ) -> "WorkflowDefinitionResponse":
        return cls(
            id=definition.id,
            name=definition.name,
            description=definition.description,
            nodes=[WorkflowNodeSchema.from_domain(node) for node in definition.nodes],
            edges=[WorkflowEdgeSchema.from_domain(edge) for edge in definition.edges],
            declared_inputs=[
                PortSpecSchema.from_domain(port) for port in definition.declared_inputs
            ],
            metadata=definition.metadata,
            created_at=definition.created_at,
            updated_at=definition.updated_at,
        )


class WorkflowExecutionPlanNodeResponse(PlatformResponse):
    node_run_id: UUID
    workflow_node_id: str
    operator_id: str
    operator_version: str
    execution_index: int
    status: NodeRunStatus
    execution_mode: ExecutionMode | None = None
    input_ports: list[PortSpecSchema] = Field(default_factory=list)
    output_ports: list[PortSpecSchema] = Field(default_factory=list)
    upstream_node_run_ids: list[UUID] = Field(default_factory=list)
    upstream_workflow_node_ids: list[str] = Field(default_factory=list)
    downstream_node_run_ids: list[UUID] = Field(default_factory=list)
    downstream_workflow_node_ids: list[str] = Field(default_factory=list)
    root: bool
    leaf: bool
    input_artifact_refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ] = Field(default_factory=dict)
    output_artifact_refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ] = Field(default_factory=dict)


class WorkflowExecutionPlanResponse(PlatformResponse):
    workflow_version_id: UUID
    workflow_run_id: UUID
    execution_order: list[str]
    root_node_run_ids: list[UUID]
    leaf_node_run_ids: list[UUID]
    nodes: list[WorkflowExecutionPlanNodeResponse]

    @classmethod
    def from_compiled_plan(
        cls,
        plan: WorkflowExecutionPlan,
        node_specs: Mapping[tuple[str, str], NodeSpec],
    ) -> "WorkflowExecutionPlanResponse":
        dependencies_by_node_run_id = {
            dependency.node_run_id: dependency
            for dependency in plan.dependencies
        }
        node_runs_by_id = {node_run.id: node_run for node_run in plan.node_runs}
        downstream_ids_by_node_run_id = _downstream_node_run_ids(
            plan.dependencies,
        )
        nodes: list[WorkflowExecutionPlanNodeResponse] = []
        for execution_index, node_run in enumerate(plan.node_runs):
            spec = node_specs[(node_run.operator_id, node_run.operator_version)]
            dependency = dependencies_by_node_run_id[node_run.id]
            nodes.append(
                _execution_plan_node_response(
                    node_run=node_run,
                    execution_index=execution_index,
                    execution_mode=spec.execution_mode,
                    input_ports=[
                        PortSpecSchema.from_domain(port) for port in spec.inputs
                    ],
                    output_ports=[
                        PortSpecSchema.from_domain(port) for port in spec.outputs
                    ],
                    upstream_node_run_ids=list(dependency.upstream_node_run_ids),
                    downstream_node_run_ids=downstream_ids_by_node_run_id[node_run.id],
                    node_runs_by_id=node_runs_by_id,
                )
            )

        return _execution_plan_response(
            workflow_version_id=plan.workflow_version_id,
            workflow_run_id=plan.workflow_run_id,
            nodes=nodes,
        )

    @classmethod
    def from_node_runs(
        cls,
        workflow_version_id: UUID,
        workflow_run_id: UUID,
        node_runs: Sequence[NodeRun],
        node_specs: Mapping[tuple[str, str], NodeSpec],
    ) -> "WorkflowExecutionPlanResponse":
        ordered_node_runs = sorted(node_runs, key=_node_run_execution_index)
        node_runs_by_id = {node_run.id: node_run for node_run in ordered_node_runs}
        upstream_ids_by_node_run_id = {
            node_run.id: _node_run_upstream_ids(node_run)
            for node_run in ordered_node_runs
        }
        dependencies = [
            NodeRunDependency(
                node_run_id=node_run_id,
                upstream_node_run_ids=tuple(upstream_node_run_ids),
            )
            for node_run_id, upstream_node_run_ids in upstream_ids_by_node_run_id.items()
        ]
        downstream_ids_by_node_run_id = _downstream_node_run_ids(dependencies)
        nodes: list[WorkflowExecutionPlanNodeResponse] = []
        for fallback_index, node_run in enumerate(ordered_node_runs):
            spec = node_specs.get((node_run.operator_id, node_run.operator_version))
            execution_mode = _node_run_execution_mode(node_run, spec)
            input_ports = _node_run_expected_ports(
                node_run,
                "expected_input_ports",
                spec.inputs if spec is not None else (),
            )
            output_ports = _node_run_expected_ports(
                node_run,
                "expected_output_ports",
                spec.outputs if spec is not None else (),
            )
            nodes.append(
                _execution_plan_node_response(
                    node_run=node_run,
                    execution_index=_node_run_execution_index_with_fallback(
                        node_run,
                        fallback_index,
                    ),
                    execution_mode=execution_mode,
                    input_ports=input_ports,
                    output_ports=output_ports,
                    upstream_node_run_ids=upstream_ids_by_node_run_id[node_run.id],
                    downstream_node_run_ids=downstream_ids_by_node_run_id[node_run.id],
                    node_runs_by_id=node_runs_by_id,
                )
            )

        return _execution_plan_response(
            workflow_version_id=workflow_version_id,
            workflow_run_id=workflow_run_id,
            nodes=nodes,
        )


class WorkflowValidationResponse(PlatformResponse):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    node_count: int
    edge_count: int
    execution_order: list[str] = Field(default_factory=list)
    execution_plan: WorkflowExecutionPlanResponse | None = None


class WorkflowTemplateResponse(PlatformResponse):
    id: str
    version: str
    display_name: str
    description: str
    config_schema: dict[str, Any]


class WorkflowVersionCreate(BaseModel):
    change_note: str | None = None
    created_by: str | None = None


class WorkflowVersionResponse(PlatformResponse):
    id: UUID
    workflow_definition_id: UUID
    version_number: int
    definition_snapshot: WorkflowDefinitionResponse
    created_at: datetime
    created_by: str | None
    change_note: str | None

    @classmethod
    def from_domain(cls, version: WorkflowVersion) -> "WorkflowVersionResponse":
        return cls(
            id=version.id,
            workflow_definition_id=version.workflow_definition_id,
            version_number=version.version_number,
            definition_snapshot=WorkflowDefinitionResponse.from_domain(
                version.definition_snapshot
            ),
            created_at=version.created_at,
            created_by=version.created_by,
            change_note=version.change_note,
        )


class WorkflowRunCreate(BaseModel):
    workflow_version_id: UUID
    input_artifact_refs: list[ArtifactRefSchema] = Field(default_factory=list)
    input_artifact_sequence_refs: list[ArtifactSequenceRefSchema] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowTemplateLaunchCreate(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    input_artifact_refs: list[ArtifactRefSchema] = Field(default_factory=list)
    input_artifact_sequence_refs: list[ArtifactSequenceRefSchema] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    change_note: str | None = None


class WorkflowTemplateMaterializeCreate(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    change_note: str | None = None


class WorkflowRunResponse(PlatformResponse):
    id: UUID
    workflow_version_id: UUID
    status: WorkflowRunStatus
    input_artifact_refs: list[ArtifactRefSchema]
    input_artifact_sequence_refs: list[ArtifactSequenceRefSchema]
    output_artifact_refs: list[ArtifactRefSchema]
    metadata: dict[str, Any]
    error: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_domain(cls, run: WorkflowRun) -> "WorkflowRunResponse":
        return cls(
            id=run.id,
            workflow_version_id=run.workflow_version_id,
            status=run.status,
            input_artifact_refs=[
                ArtifactRefSchema.from_domain(ref) for ref in run.input_artifact_refs
            ],
            input_artifact_sequence_refs=[
                ArtifactSequenceRefSchema.from_domain(ref)
                for ref in run.input_artifact_sequence_refs
            ],
            output_artifact_refs=[
                ArtifactRefSchema.from_domain(ref) for ref in run.output_artifact_refs
            ],
            metadata=run.metadata,
            error=run.error,
            queued_at=run.queued_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


class WorkflowTemplateLaunchResponse(PlatformResponse):
    template: WorkflowTemplateResponse
    workflow_definition: WorkflowDefinitionResponse
    workflow_version: WorkflowVersionResponse
    workflow_run: WorkflowRunResponse
    queued_node_run_ids: list[UUID]


class WorkflowTemplateMaterializeResponse(PlatformResponse):
    template: WorkflowTemplateResponse
    workflow_definition: WorkflowDefinitionResponse
    workflow_version: WorkflowVersionResponse


class WorkflowRunExecutionCreate(BaseModel):
    max_node_runs: int = Field(default=100, ge=1)


class WorkflowRunExecutionNodeError(PlatformResponse):
    node_run_id: UUID
    error: str


class WorkflowRunExecutionResponse(PlatformResponse):
    workflow_run_id: UUID
    workflow_run: WorkflowRunResponse
    processed_node_run_ids: list[UUID]
    errors: list[WorkflowRunExecutionNodeError]


class ExperimentParameterSchema(PlatformResponse):
    name: str
    node_id: str
    config_path: list[str]
    values: list[Any]
    description: str | None = None

    def to_domain(self) -> ExperimentParameter:
        return ExperimentParameter(
            name=self.name,
            node_id=self.node_id,
            config_path=tuple(self.config_path),
            values=tuple(self.values),
            description=self.description,
        )

    @classmethod
    def from_domain(
        cls,
        parameter: ExperimentParameter,
    ) -> "ExperimentParameterSchema":
        return cls(
            name=parameter.name,
            node_id=parameter.node_id,
            config_path=list(parameter.config_path),
            values=list(parameter.values),
            description=parameter.description,
        )


ExperimentParameterPresetKind = Literal[
    "ocr_engine",
    "ocr_engine_config",
    "ocr_language_hints",
    "ocr_candidate_a_label",
    "ocr_candidate_b_label",
    "ocr_selected_candidate",
    "ocr_selection_note",
    "model_provider",
    "model_name",
    "model_parameters",
    "prompt_template",
    "extraction_schema",
    "static_context",
    "input_policy_type",
    "input_policy_settings",
    "export_format",
    "export_filename",
]

_EXPERIMENT_PARAMETER_PRESET_PATHS: Mapping[
    ExperimentParameterPresetKind,
    tuple[str, tuple[str, ...]],
] = {
    "ocr_engine": ("engine", ("engine",)),
    "ocr_engine_config": ("engine_config", ("engine_config",)),
    "ocr_language_hints": ("language_hints", ("language_hints",)),
    "ocr_candidate_a_label": ("candidate_a_label", ("candidate_a_label",)),
    "ocr_candidate_b_label": ("candidate_b_label", ("candidate_b_label",)),
    "ocr_selected_candidate": ("selected_candidate", ("selected_candidate",)),
    "ocr_selection_note": ("decision_note", ("decision_note",)),
    "model_provider": ("provider", ("provider",)),
    "model_name": ("model", ("model",)),
    "model_parameters": ("parameters", ("parameters",)),
    "prompt_template": ("template", ("template",)),
    "extraction_schema": ("json_schema", ("json_schema",)),
    "static_context": ("context", ("context",)),
    "input_policy_type": ("policy_type", ("policy_type",)),
    "input_policy_settings": ("settings", ("settings",)),
    "export_format": ("format", ("format",)),
    "export_filename": ("filename", ("filename",)),
}


class ExperimentParameterPresetSchema(PlatformResponse):
    kind: ExperimentParameterPresetKind
    node_id: str
    values: list[Any]
    name: str | None = None
    description: str | None = None

    def to_domain(self) -> ExperimentParameter:
        preset = _EXPERIMENT_PARAMETER_PRESET_PATHS.get(self.kind)
        if preset is None:
            raise ValueError(f"Unsupported experiment parameter preset: {self.kind}")

        default_name, config_path = preset
        parameter_name = self.name if self.name is not None else default_name
        return ExperimentParameter(
            name=parameter_name,
            node_id=self.node_id,
            config_path=config_path,
            values=tuple(self.values),
            description=self.description,
        )


class ExperimentVariantSchema(PlatformResponse):
    id: UUID
    key: str
    ordinal: int
    parameter_values: dict[str, Any]
    workflow_run_id: UUID
    metadata: dict[str, Any]

    @classmethod
    def from_domain(cls, variant: ExperimentVariant) -> "ExperimentVariantSchema":
        return cls.model_validate(variant)


class ExperimentCreate(BaseModel):
    name: str
    workflow_version_id: UUID
    description: str | None = None
    parameters: list[ExperimentParameterSchema] = Field(default_factory=list)
    parameter_presets: list[ExperimentParameterPresetSchema] = Field(
        default_factory=list
    )
    input_artifact_refs: list[ArtifactRefSchema] = Field(default_factory=list)
    input_artifact_sequence_refs: list[ArtifactSequenceRefSchema] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(PlatformResponse):
    id: UUID
    name: str
    description: str | None
    workflow_version_id: UUID
    status: ExperimentStatus
    parameters: list[ExperimentParameterSchema]
    input_artifact_refs: list[ArtifactRefSchema]
    input_artifact_sequence_refs: list[ArtifactSequenceRefSchema]
    variants: list[ExperimentVariantSchema]
    workflow_run_ids: list[UUID]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, experiment: Experiment) -> "ExperimentResponse":
        return cls(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            workflow_version_id=experiment.workflow_version_id,
            status=experiment.status,
            parameters=[
                ExperimentParameterSchema.from_domain(parameter)
                for parameter in experiment.parameters
            ],
            input_artifact_refs=[
                ArtifactRefSchema.from_domain(ref)
                for ref in experiment.input_artifact_refs
            ],
            input_artifact_sequence_refs=[
                ArtifactSequenceRefSchema.from_domain(ref)
                for ref in experiment.input_artifact_sequence_refs
            ],
            variants=[
                ExperimentVariantSchema.from_domain(variant)
                for variant in experiment.variants
            ],
            workflow_run_ids=experiment.workflow_run_ids,
            metadata=experiment.metadata,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at,
        )


class ExperimentEvaluationMetricResponse(PlatformResponse):
    artifact_id: UUID
    producer_node_run_id: UUID | None
    metadata: dict[str, Any]


class ExperimentMetricValueResponse(PlatformResponse):
    name: str
    value: str | int | float | bool | None
    source: Literal["summary", "evaluation.metrics"]
    artifact_id: UUID | None = None
    producer_node_run_id: UUID | None = None


class ExperimentVariantComparisonResponse(PlatformResponse):
    variant_id: UUID
    variant_key: str
    ordinal: int
    parameter_values: dict[str, Any]
    workflow_run_id: UUID
    workflow_run_status: WorkflowRunStatus
    node_run_status_counts: dict[str, int]
    artifact_counts: dict[str, int]
    invocation_count: int
    validation_error_count: int
    total_duration_ms: float | None
    total_cost: float | None
    evaluation_metrics: list[ExperimentEvaluationMetricResponse]
    metric_values: list[ExperimentMetricValueResponse]
    errors: list[str]


class ExperimentComparisonResponse(PlatformResponse):
    experiment_id: UUID
    workflow_version_id: UUID
    variant_count: int
    metric_names: list[str]
    variants: list[ExperimentVariantComparisonResponse]


class ExperimentExecutionCreate(BaseModel):
    max_node_runs_per_variant: int = Field(default=100, ge=1)
    stop_on_error: bool = False


class ExperimentExecutionVariantResponse(PlatformResponse):
    variant_id: UUID
    variant_key: str
    workflow_run_id: UUID
    workflow_run: WorkflowRunResponse
    processed_node_run_ids: list[UUID]
    errors: list[WorkflowRunExecutionNodeError]


class ExperimentExecutionResponse(PlatformResponse):
    experiment: ExperimentResponse
    variants: list[ExperimentExecutionVariantResponse]


class ExperimentRerunVariantResponse(PlatformResponse):
    variant_id: UUID
    variant_key: str
    previous_workflow_run_id: UUID
    workflow_run_id: UUID


class ExperimentRerunFailedResponse(PlatformResponse):
    experiment: ExperimentResponse
    variants: list[ExperimentRerunVariantResponse]


class NodeRunCreate(BaseModel):
    workflow_node_id: str
    operator_id: str
    operator_version: str
    input_artifact_refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeRunResponse(PlatformResponse):
    id: UUID
    workflow_run_id: UUID
    workflow_node_id: str
    operator_id: str
    operator_version: str
    status: NodeRunStatus
    input_artifact_refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ]
    output_artifact_refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ]
    attempt_count: int
    max_attempts: int
    metadata: dict[str, Any]
    error: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_domain(cls, node_run: NodeRun) -> "NodeRunResponse":
        return cls(
            id=node_run.id,
            workflow_run_id=node_run.workflow_run_id,
            workflow_node_id=node_run.workflow_node_id,
            operator_id=node_run.operator_id,
            operator_version=node_run.operator_version,
            status=node_run.status,
            input_artifact_refs=_artifact_ref_map_from_domain(
                node_run.input_artifact_refs
            ),
            output_artifact_refs=_artifact_ref_map_from_domain(
                node_run.output_artifact_refs
            ),
            attempt_count=node_run.attempt_count,
            max_attempts=node_run.max_attempts,
            metadata=node_run.metadata,
            error=node_run.error,
            queued_at=node_run.queued_at,
            started_at=node_run.started_at,
            finished_at=node_run.finished_at,
        )


class NodeRunExecutionResponse(PlatformResponse):
    requested_node_run_id: UUID | None
    processed_node_run_id: UUID | None
    node_run: NodeRunResponse | None
    error: str | None = None


class NodeRunRetryResponse(PlatformResponse):
    workflow_run: WorkflowRunResponse
    node_run: NodeRunResponse
    outbox_message_id: UUID


class OutboxMessageResponse(PlatformResponse):
    id: UUID
    subject: str
    message_type: str
    payload: dict[str, Any]
    status: OutboxMessageStatus
    attempts: int
    error: str | None
    created_at: datetime
    published_at: datetime | None

    @classmethod
    def from_domain(cls, message: OutboxMessage) -> "OutboxMessageResponse":
        return cls(
            id=message.id,
            subject=message.subject,
            message_type=message.message_type,
            payload=message.payload,
            status=message.status,
            attempts=message.attempts,
            error=message.error,
            created_at=message.created_at,
            published_at=message.published_at,
        )


class OutboxCleanupRequest(BaseModel):
    statuses: list[OutboxMessageStatus] = Field(min_length=1)
    older_than: datetime
    subject_prefix: str | None = None
    message_type: str | None = None
    dry_run: bool = True


class OutboxCleanupResponse(PlatformResponse):
    dry_run: bool
    matched_count: int
    deleted_count: int
    messages: list[OutboxMessageResponse]


class DlqSummaryResponse(PlatformResponse):
    consumer_name: str
    error_code: str
    original_subject: str
    count: int
    latest_failed_at: datetime
    latest_outbox_message_id: UUID


class InputAssemblyTraceResponse(PlatformResponse):
    id: UUID
    node_run_id: UUID
    selected_inputs: dict[str, ArtifactRefSchema | list[ArtifactRefSchema]]
    omitted_inputs: dict[str, str]
    policies: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_domain(cls, trace: InputAssemblyTrace) -> "InputAssemblyTraceResponse":
        selected_inputs: dict[str, ArtifactRefSchema | list[ArtifactRefSchema]] = {}
        for key, value in trace.selected_inputs.items():
            if isinstance(value, list):
                selected_inputs[key] = [
                    ArtifactRefSchema.from_domain(item) for item in value
                ]
            else:
                selected_inputs[key] = ArtifactRefSchema.from_domain(value)
        return cls(
            id=trace.id,
            node_run_id=trace.node_run_id,
            selected_inputs=selected_inputs,
            omitted_inputs=trace.omitted_inputs,
            policies=trace.policies,
            metadata=trace.metadata,
            created_at=trace.created_at,
        )


class InvocationTraceResponse(PlatformResponse):
    id: UUID
    node_run_id: UUID
    invocation_type: str
    input_artifact_refs: list[ArtifactRefSchema]
    output_artifact_refs: list[ArtifactRefSchema]
    provider: str | None
    model: str | None
    request_ref: str | None
    response_ref: str | None
    runtime: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None
    created_at: datetime

    @classmethod
    def from_domain(cls, trace: InvocationTrace) -> "InvocationTraceResponse":
        return cls(
            id=trace.id,
            node_run_id=trace.node_run_id,
            invocation_type=trace.invocation_type,
            input_artifact_refs=[
                ArtifactRefSchema.from_domain(ref) for ref in trace.input_artifact_refs
            ],
            output_artifact_refs=[
                ArtifactRefSchema.from_domain(ref) for ref in trace.output_artifact_refs
            ],
            provider=trace.provider,
            model=trace.model,
            request_ref=trace.request_ref,
            response_ref=trace.response_ref,
            runtime=trace.runtime,
            metadata=trace.metadata,
            error=trace.error,
            created_at=trace.created_at,
        )


class ArtifactCreate(BaseModel):
    artifact_type: str
    schema_version: int
    workflow_run_id: UUID | None = None
    producer_node_run_id: UUID | None = None
    payload_ref: str
    producer_operator_id: str | None = None
    producer_operator_version: str | None = None
    input_artifact_ids: list[UUID] = Field(default_factory=list)
    content_hash: str | None = None
    preview_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactJsonPayloadCreate(BaseModel):
    artifact_type: str
    schema_version: int
    payload: Any
    workflow_run_id: UUID | None = None
    producer_node_run_id: UUID | None = None
    producer_operator_id: str | None = None
    producer_operator_version: str | None = None
    input_artifact_ids: list[UUID] = Field(default_factory=list)
    preview_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    bucket: str = "script-artifacts"
    key: str | None = None
    content_type: str = "application/json"


class ArtifactResponse(PlatformResponse):
    id: UUID
    artifact_type: str
    schema_version: int
    workflow_run_id: UUID | None
    producer_node_run_id: UUID | None
    payload_ref: str
    producer_operator_id: str | None
    producer_operator_version: str | None
    input_artifact_ids: list[UUID]
    content_hash: str | None
    preview_ref: str | None
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_domain(cls, artifact: Artifact) -> "ArtifactResponse":
        return cls.model_validate(artifact)


class ArtifactSequenceCreate(BaseModel):
    artifact_type: str
    schema_version: int
    item_refs: list[ArtifactRefSchema]
    ordered: bool = True
    index_key: str = "sequence_index"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactSequenceResponse(PlatformResponse):
    id: UUID
    artifact_type: str
    schema_version: int
    item_refs: list[ArtifactRefSchema]
    ordered: bool
    index_key: str
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_domain(cls, sequence: ArtifactSequence) -> "ArtifactSequenceResponse":
        return cls(
            id=sequence.id,
            artifact_type=sequence.artifact_type,
            schema_version=sequence.schema_version,
            item_refs=[ArtifactRefSchema.from_domain(ref) for ref in sequence.item_refs],
            ordered=sequence.ordered,
            index_key=sequence.index_key,
            metadata=sequence.metadata,
            created_at=sequence.created_at,
        )


ArtifactGraphNodeKind = Literal["artifact", "artifact_sequence", "node_run"]
ArtifactGraphEdgeKind = Literal[
    "artifact_input",
    "artifact_sequence_item",
    "node_input",
    "node_output",
]


class ArtifactGraphEdgeResponse(PlatformResponse):
    edge_type: ArtifactGraphEdgeKind
    from_kind: ArtifactGraphNodeKind
    from_id: UUID
    to_kind: ArtifactGraphNodeKind
    to_id: UUID
    port_name: str | None = None


class ArtifactGraphResponse(PlatformResponse):
    workflow_run: WorkflowRunResponse | None = None
    root_artifact: ArtifactResponse | None = None
    node_runs: list[NodeRunResponse] = Field(default_factory=list)
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    artifact_sequences: list[ArtifactSequenceResponse] = Field(default_factory=list)
    edges: list[ArtifactGraphEdgeResponse] = Field(default_factory=list)


class WorkflowRunSummaryError(PlatformResponse):
    node_run_id: UUID | None
    status: str
    error: str


class WorkflowRunSummaryResponse(PlatformResponse):
    workflow_run: WorkflowRunResponse
    node_runs: list[NodeRunResponse]
    artifacts: list[ArtifactResponse]
    node_run_status_counts: dict[str, int]
    artifact_counts: dict[str, int]
    errors: list[WorkflowRunSummaryError]


WorkflowRunTimelineEventKind = Literal[
    "workflow_run",
    "node_run",
    "artifact",
    "dead_letter",
    "malformed_outbox",
]


class WorkflowRunTimelineEventResponse(PlatformResponse):
    outbox_message_id: UUID
    subject: str
    message_type: str
    outbox_status: OutboxMessageStatus
    outbox_created_at: datetime
    outbox_published_at: datetime | None
    event_kind: WorkflowRunTimelineEventKind
    event_type: str
    occurred_at: datetime
    workflow_run_id: UUID
    node_run_id: UUID | None = None
    artifact_id: UUID | None = None
    artifact_type: str | None = None
    error: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunTimelineResponse(PlatformResponse):
    workflow_run: WorkflowRunResponse
    events: list[WorkflowRunTimelineEventResponse]


class OutputArtifactPayloadResponse(PlatformResponse):
    content_type: str
    byte_size: int
    json_payload: Any | None = None
    text: str | None = None
    error: str | None = None


class ArtifactInspectionResponse(PlatformResponse):
    artifact: ArtifactResponse
    payload: OutputArtifactPayloadResponse | None = None
    lineage: ArtifactGraphResponse | None = None


class OutputArtifactResponse(PlatformResponse):
    artifact: ArtifactResponse
    payload: OutputArtifactPayloadResponse | None = None


class WorkflowRunTraceBundleResponse(PlatformResponse):
    node_run: NodeRunResponse
    input_assembly_traces: list[InputAssemblyTraceResponse]
    invocation_traces: list[InvocationTraceResponse]


class WorkflowRunOutputBundleResponse(PlatformResponse):
    workflow_run: WorkflowRunResponse
    artifacts: list[OutputArtifactResponse]
    artifact_sequences: list[ArtifactSequenceResponse] = Field(default_factory=list)
    traces: list[WorkflowRunTraceBundleResponse] = Field(default_factory=list)


class ExperimentVariantOutputBundleResponse(PlatformResponse):
    variant_id: UUID
    variant_key: str
    ordinal: int
    parameter_values: dict[str, Any]
    workflow_run_id: UUID
    output_bundle: WorkflowRunOutputBundleResponse


class ExperimentOutputBundleResponse(PlatformResponse):
    experiment: ExperimentResponse
    variants: list[ExperimentVariantOutputBundleResponse]


class NodeSpecResponse(PlatformResponse):
    id: str
    version: str
    inputs: list[PortSpecSchema]
    outputs: list[PortSpecSchema]
    execution_mode: ExecutionMode
    config_schema: dict[str, Any]
    display_name: str | None
    description: str | None

    @classmethod
    def from_domain(cls, spec: NodeSpec) -> "NodeSpecResponse":
        return cls(
            id=spec.id,
            version=spec.version,
            inputs=[PortSpecSchema.from_domain(port) for port in spec.inputs],
            outputs=[PortSpecSchema.from_domain(port) for port in spec.outputs],
            execution_mode=spec.execution_mode,
            config_schema=spec.config_schema,
            display_name=spec.display_name,
            description=spec.description,
        )


def _execution_plan_node_response(
    *,
    node_run: NodeRun,
    execution_index: int,
    execution_mode: ExecutionMode | None,
    input_ports: list[PortSpecSchema],
    output_ports: list[PortSpecSchema],
    upstream_node_run_ids: list[UUID],
    downstream_node_run_ids: list[UUID],
    node_runs_by_id: Mapping[UUID, NodeRun],
) -> WorkflowExecutionPlanNodeResponse:
    upstream_workflow_node_ids = [
        node_runs_by_id[node_run_id].workflow_node_id
        for node_run_id in upstream_node_run_ids
        if node_run_id in node_runs_by_id
    ]
    downstream_workflow_node_ids = [
        node_runs_by_id[node_run_id].workflow_node_id
        for node_run_id in downstream_node_run_ids
        if node_run_id in node_runs_by_id
    ]
    return WorkflowExecutionPlanNodeResponse(
        node_run_id=node_run.id,
        workflow_node_id=node_run.workflow_node_id,
        operator_id=node_run.operator_id,
        operator_version=node_run.operator_version,
        execution_index=execution_index,
        status=node_run.status,
        execution_mode=execution_mode,
        input_ports=input_ports,
        output_ports=output_ports,
        upstream_node_run_ids=upstream_node_run_ids,
        upstream_workflow_node_ids=upstream_workflow_node_ids,
        downstream_node_run_ids=downstream_node_run_ids,
        downstream_workflow_node_ids=downstream_workflow_node_ids,
        root=upstream_node_run_ids == [],
        leaf=downstream_node_run_ids == [],
        input_artifact_refs=_artifact_ref_map_from_domain(node_run.input_artifact_refs),
        output_artifact_refs=_artifact_ref_map_from_domain(
            node_run.output_artifact_refs
        ),
    )


def _execution_plan_response(
    *,
    workflow_version_id: UUID,
    workflow_run_id: UUID,
    nodes: list[WorkflowExecutionPlanNodeResponse],
) -> WorkflowExecutionPlanResponse:
    ordered_nodes = sorted(nodes, key=lambda node: node.execution_index)
    return WorkflowExecutionPlanResponse(
        workflow_version_id=workflow_version_id,
        workflow_run_id=workflow_run_id,
        execution_order=[node.workflow_node_id for node in ordered_nodes],
        root_node_run_ids=[
            node.node_run_id for node in ordered_nodes if node.root
        ],
        leaf_node_run_ids=[
            node.node_run_id for node in ordered_nodes if node.leaf
        ],
        nodes=ordered_nodes,
    )


def _downstream_node_run_ids(
    dependencies: Sequence[NodeRunDependency],
) -> dict[UUID, list[UUID]]:
    downstream_ids_by_node_run_id = {
        dependency.node_run_id: [] for dependency in dependencies
    }
    for dependency in dependencies:
        for upstream_node_run_id in dependency.upstream_node_run_ids:
            downstream_ids = downstream_ids_by_node_run_id.setdefault(
                upstream_node_run_id,
                [],
            )
            downstream_ids.append(dependency.node_run_id)
    return downstream_ids_by_node_run_id


def _node_run_execution_index(node_run: NodeRun) -> int:
    return _node_run_execution_index_with_fallback(node_run, 0)


def _node_run_execution_index_with_fallback(
    node_run: NodeRun,
    fallback_index: int,
) -> int:
    execution_index = node_run.metadata.get("execution_index")
    if isinstance(execution_index, int):
        return execution_index
    return fallback_index


def _node_run_upstream_ids(node_run: NodeRun) -> list[UUID]:
    raw_ids = node_run.metadata.get("upstream_node_run_ids")
    if not isinstance(raw_ids, list):
        return []

    upstream_node_run_ids: list[UUID] = []
    for raw_id in raw_ids:
        if isinstance(raw_id, str):
            upstream_node_run_ids.append(UUID(raw_id))
    return upstream_node_run_ids


def _node_run_execution_mode(
    node_run: NodeRun,
    spec: NodeSpec | None,
) -> ExecutionMode | None:
    raw_execution_mode = node_run.metadata.get("execution_mode")
    if isinstance(raw_execution_mode, str):
        try:
            return ExecutionMode(raw_execution_mode)
        except ValueError:
            pass
    if spec is not None:
        return spec.execution_mode
    return None


def _node_run_expected_ports(
    node_run: NodeRun,
    metadata_key: str,
    fallback_ports: Sequence[PortSpec],
) -> list[PortSpecSchema]:
    raw_ports = node_run.metadata.get(metadata_key)
    if isinstance(raw_ports, list):
        try:
            return [PortSpecSchema.model_validate(port) for port in raw_ports]
        except PydanticValidationError:
            pass

    return [PortSpecSchema.from_domain(port) for port in fallback_ports]


def artifact_ref_map_to_domain(
    refs: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ],
) -> dict[str, ArtifactPortRef]:
    converted: dict[str, ArtifactPortRef] = {}
    for key, value in refs.items():
        if isinstance(value, list):
            converted[key] = [item.to_domain() for item in value]
        elif isinstance(value, ArtifactSequenceRefSchema):
            converted[key] = value.to_domain()
        else:
            converted[key] = value.to_domain()
    return converted


def _artifact_ref_map_from_domain(
    refs: dict[str, ArtifactPortRef],
) -> dict[
    str,
    ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
]:
    converted: dict[
        str,
        ArtifactRefSchema | ArtifactSequenceRefSchema | list[ArtifactRefSchema],
    ] = {}
    for key, value in refs.items():
        if isinstance(value, list):
            converted[key] = [ArtifactRefSchema.from_domain(item) for item in value]
        elif isinstance(value, ArtifactSequenceRef):
            converted[key] = ArtifactSequenceRefSchema.from_domain(value)
        else:
            converted[key] = ArtifactRefSchema.from_domain(value)
    return converted

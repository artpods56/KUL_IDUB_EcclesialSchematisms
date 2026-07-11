from uuid import uuid4

import pytest

from notarius_core.application.workflows import WorkflowCompiler
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    ExecutionMode,
    NodeSpec,
    PortSpec,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    WorkflowVersion,
)

SOURCE_SPEC = NodeSpec(
    id="source.load",
    version="1.0.0",
    execution_mode=ExecutionMode.SINGLE,
    inputs=(),
    outputs=(
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
            sequence=True,
        ),
    ),
)

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
    (SOURCE_SPEC.id, SOURCE_SPEC.version): SOURCE_SPEC,
    (OCR_SPEC.id, OCR_SPEC.version): OCR_SPEC,
    (EXPORT_SPEC.id, EXPORT_SPEC.version): EXPORT_SPEC,
}


def test_workflow_compiler_returns_node_runs_in_topological_order() -> None:
    definition = WorkflowDefinition(
        name="OCR export",
        nodes=[
            WorkflowNode(
                id="export",
                operator_id="export.dataset",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
                config={"batch_size": 8},
            ),
        ],
        edges=[
            WorkflowEdge("source", "pages", "ocr", "pages"),
            WorkflowEdge("ocr", "ocr_pages", "export", "records"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    workflow_run_id = uuid4()

    plan = WorkflowCompiler(REGISTRY).compile(version, workflow_run_id)

    assert [node_run.workflow_node_id for node_run in plan.node_runs] == [
        "source",
        "ocr",
        "export",
    ]
    assert [node_run.workflow_run_id for node_run in plan.node_runs] == [
        workflow_run_id,
        workflow_run_id,
        workflow_run_id,
    ]
    assert plan.node_runs[1].metadata["workflow_node_config"] == {"batch_size": 8}

    node_runs_by_node_id = {
        node_run.workflow_node_id: node_run for node_run in plan.node_runs
    }
    dependencies_by_node_run_id = {
        dependency.node_run_id: dependency.upstream_node_run_ids
        for dependency in plan.dependencies
    }
    assert dependencies_by_node_run_id[node_runs_by_node_id["source"].id] == ()
    assert dependencies_by_node_run_id[node_runs_by_node_id["ocr"].id] == (
        node_runs_by_node_id["source"].id,
    )
    assert dependencies_by_node_run_id[node_runs_by_node_id["export"].id] == (
        node_runs_by_node_id["ocr"].id,
    )


def test_workflow_compiler_rejects_unknown_operator() -> None:
    definition = WorkflowDefinition(
        name="Unknown operator",
        nodes=[
            WorkflowNode(
                id="unknown",
                operator_id="missing.operator",
                operator_version="1.0.0",
            )
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="Unknown operator"):
        WorkflowCompiler(REGISTRY).compile(version, uuid4())


def test_workflow_compiler_rejects_invalid_node_config() -> None:
    configured_spec = NodeSpec(
        id="source.load",
        version="2.0.0",
        execution_mode=ExecutionMode.SINGLE,
        inputs=(),
        outputs=SOURCE_SPEC.outputs,
        config_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    registry = {
        (configured_spec.id, configured_spec.version): configured_spec,
    }
    definition = WorkflowDefinition(
        name="Invalid node config",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="2.0.0",
                config={},
            )
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="node config"):
        WorkflowCompiler(registry).compile(version, uuid4())


def test_workflow_compiler_rejects_duplicate_node_ids() -> None:
    definition = WorkflowDefinition(
        name="Duplicate nodes",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="source",
                operator_id="ocr.run",
                operator_version="1.0.0",
            ),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="Duplicate workflow node ids: source"):
        WorkflowCompiler(REGISTRY).compile(version, uuid4())


def test_workflow_compiler_rejects_missing_edge_endpoint() -> None:
    definition = WorkflowDefinition(
        name="Missing endpoint",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="1.0.0",
            )
        ],
        edges=[
            WorkflowEdge("source", "pages", "ocr", "pages"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="missing target node"):
        WorkflowCompiler(REGISTRY).compile(version, uuid4())


@pytest.mark.parametrize(
    ("edge", "message"),
    [
        (WorkflowEdge("source", "missing", "ocr", "pages"), "source output port"),
        (WorkflowEdge("source", "pages", "ocr", "missing"), "target input port"),
    ],
)
def test_workflow_compiler_rejects_missing_edge_ports(
    edge: WorkflowEdge,
    message: str,
) -> None:
    definition = WorkflowDefinition(
        name="Missing ports",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            ),
        ],
        edges=[edge],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match=message):
        WorkflowCompiler(REGISTRY).compile(version, uuid4())


def test_workflow_compiler_rejects_multiple_edges_into_same_target_port() -> None:
    definition = WorkflowDefinition(
        name="Duplicate target port",
        nodes=[
            WorkflowNode(
                id="source_a",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="source_b",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            ),
        ],
        edges=[
            WorkflowEdge("source_a", "pages", "ocr", "pages"),
            WorkflowEdge("source_b", "pages", "ocr", "pages"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="multiple edges into target input port"):
        WorkflowCompiler(REGISTRY).compile(version, uuid4())


@pytest.mark.parametrize(
    "target_port",
    [
        PortSpec(
            name="pages",
            artifact_type="text.markdown",
            schema_version=1,
            sequence=True,
        ),
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=2,
            sequence=True,
        ),
        PortSpec(
            name="pages",
            artifact_type="source.page_image",
            schema_version=1,
        ),
    ],
)
def test_workflow_compiler_rejects_incompatible_artifact_contracts(
    target_port: PortSpec,
) -> None:
    incompatible_ocr_spec = NodeSpec(
        id="ocr.run",
        version="2.0.0",
        execution_mode=ExecutionMode.MAP,
        inputs=(target_port,),
        outputs=OCR_SPEC.outputs,
    )
    registry = {
        (SOURCE_SPEC.id, SOURCE_SPEC.version): SOURCE_SPEC,
        (
            incompatible_ocr_spec.id,
            incompatible_ocr_spec.version,
        ): incompatible_ocr_spec,
    }
    definition = WorkflowDefinition(
        name="Incompatible contracts",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="2.0.0",
            ),
        ],
        edges=[WorkflowEdge("source", "pages", "ocr", "pages")],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="incompatible artifact contracts"):
        WorkflowCompiler(registry).compile(version, uuid4())


def test_workflow_compiler_rejects_cycles() -> None:
    source_from_ocr_spec = NodeSpec(
        id="source.load",
        version="2.0.0",
        execution_mode=ExecutionMode.SINGLE,
        inputs=(
            PortSpec(
                name="seed",
                artifact_type="ocr.page_result",
                schema_version=1,
                sequence=True,
            ),
        ),
        outputs=SOURCE_SPEC.outputs,
    )
    registry = {
        (source_from_ocr_spec.id, source_from_ocr_spec.version): source_from_ocr_spec,
        (OCR_SPEC.id, OCR_SPEC.version): OCR_SPEC,
    }
    definition = WorkflowDefinition(
        name="Cycle",
        nodes=[
            WorkflowNode(
                id="source",
                operator_id="source.load",
                operator_version="2.0.0",
            ),
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            ),
        ],
        edges=[
            WorkflowEdge("source", "pages", "ocr", "pages"),
            WorkflowEdge("ocr", "ocr_pages", "source", "seed"),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    with pytest.raises(ValidationError, match="contains a cycle"):
        WorkflowCompiler(registry).compile(version, uuid4())

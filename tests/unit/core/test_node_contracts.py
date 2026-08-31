from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from grafy_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    NoConfig,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.nodes import (
    MAX_NODE_ERROR_MESSAGE_LENGTH,
    ArtifactTypeVariable,
    InPort,
    Node,
    NodeContractError,
    NodeContractResolutionError,
    NodeExecutionContext,
    OutPort,
    PortShape,
    UserFacingNodeError,
    derive_input_contract,
    resolve_node_contracts,
)
from grafy_core.artifact_contracts import RASTER_IMAGE


class ExampleImage:
    pass


class ExampleConfig(NodeConfig):
    language: str = "eng"


class ExampleInput(NodeInput):
    pages: Annotated[
        list[ExampleImage],
        InPort(RASTER_IMAGE),
        Field(title="Source pages", description="Images to process in order."),
    ]


class ExampleOutput(NodeOutput):
    pages: Annotated[ArtifactRefSequence, OutPort(RASTER_IMAGE)]


class ExampleNode(Node[ExampleConfig, ExampleInput, ExampleOutput]):
    operator_id = "example.node"
    operator_version = 1

    async def run(
        self,
        _context: NodeExecutionContext,
        _config: ExampleConfig,
        inputs: ExampleInput,
        /,
    ) -> ExampleOutput:
        del inputs
        raise NotImplementedError


class AnnotationShapesInput(NodeInput):
    optional_page: Annotated[
        ExampleImage | None,
        InPort(RASTER_IMAGE),
    ] = None
    page_batches: Annotated[
        list[list[ExampleImage]],
        InPort(RASTER_IMAGE, variadic=True),
    ]
    page_refs: Annotated[ArtifactRefSequence, InPort(RASTER_IMAGE)]


class InstancePlugInput(NodeInput):
    items: Annotated[
        list[ArtifactRef | ArtifactRefSequence],
        InPort(RASTER_IMAGE, variadic=True, instance_plugs=True),
    ]


class NonVariadicInstancePlugInput(NodeInput):
    items: Annotated[
        list[ArtifactRef | ArtifactRefSequence],
        InPort(RASTER_IMAGE, instance_plugs=True),
    ]


class BroadInstancePlugInput(NodeInput):
    items: Annotated[
        list[ArtifactRef | object],
        InPort(RASTER_IMAGE, variadic=True, instance_plugs=True),
    ]


GENERIC_ARTIFACT = ArtifactTypeVariable("T")


class GenericInput(NodeInput):
    items: Annotated[
        list[ArtifactRef | ArtifactRefSequence],
        InPort(GENERIC_ARTIFACT, variadic=True, instance_plugs=True),
    ]


class GenericOutput(NodeOutput):
    items: Annotated[ArtifactRefSequence, OutPort(GENERIC_ARTIFACT)]


class GenericNode(Node[NoConfig, GenericInput, GenericOutput]):
    operator_id = "example.generic"
    operator_version = 1

    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: GenericInput,
        /,
    ) -> GenericOutput:
        del inputs
        raise NotImplementedError


def test_node_contracts_derive_shape_and_resolver_target_from_annotations() -> None:
    assert ExampleNode.config_contract.model is ExampleConfig
    assert ExampleNode.input_contract.model is ExampleInput
    assert ExampleNode.input_contract.ports["pages"].shape == "many"
    assert ExampleNode.input_contract.ports["pages"].accepted_shapes == (
        PortShape.MANY,
    )
    assert ExampleNode.input_contract.ports["pages"].target_type is ExampleImage
    assert ExampleNode.input_contract.ports["pages"].title == "Source pages"
    assert (
        ExampleNode.input_contract.ports["pages"].description
        == "Images to process in order."
    )
    assert ExampleNode.output_contract.model is ExampleOutput
    assert ExampleNode.output_contract.ports["pages"].shape == "many"


def test_node_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NoConfig.model_validate({"unexpected": True})


def test_user_facing_node_error_requires_a_bounded_nonblank_message() -> None:
    assert str(UserFacingNodeError("  Safe provider failure.  ")) == (
        "Safe provider failure."
    )

    with pytest.raises(ValueError, match="must not be blank"):
        UserFacingNodeError("  ")
    with pytest.raises(ValueError, match="must be at most"):
        UserFacingNodeError("x" * (MAX_NODE_ERROR_MESSAGE_LENGTH + 1))


def test_input_contract_supports_optional_variadic_and_structural_types() -> None:
    contract = derive_input_contract(AnnotationShapesInput)

    optional_page = contract.ports["optional_page"]
    assert optional_page.shape is PortShape.ONE
    assert optional_page.target_type is ExampleImage
    assert optional_page.allows_none is True

    page_batches = contract.ports["page_batches"]
    assert page_batches.shape is PortShape.MANY
    assert page_batches.variadic is True
    assert page_batches.target_type is ExampleImage

    page_refs = contract.ports["page_refs"]
    assert page_refs.shape is PortShape.MANY
    assert page_refs.target_type is None
    assert page_refs.preserves_ref_container is True


def test_input_contract_derives_mixed_instance_plug_shapes() -> None:
    items = derive_input_contract(InstancePlugInput).ports["items"]

    assert items.shape is PortShape.ONE
    assert items.accepted_shapes == (PortShape.ONE, PortShape.MANY)
    assert items.variadic is True
    assert items.instance_plugs is True
    assert items.target_type is None
    assert items.preserves_ref_container is True


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            NonVariadicInstancePlugInput,
            "NonVariadicInstancePlugInput.items.*require variadic=True",
        ),
        (
            BroadInstancePlugInput,
            "BroadInstancePlugInput.items.*must be a concrete Python type",
        ),
    ],
)
def test_input_contract_rejects_invalid_instance_plug_declarations(
    model: type[NodeInput],
    message: str,
) -> None:
    with pytest.raises(NodeContractError, match=message):
        derive_input_contract(model)


def test_generic_node_contract_resolves_shared_variable_without_mutation() -> None:
    node = GenericNode()
    artifact_type = ArtifactTypeKey("example.value", 2)

    resolved = resolve_node_contracts(node, {"T": artifact_type})

    assert node.input_contract.ports["items"].accepts is GENERIC_ARTIFACT
    assert node.output_contract.ports["items"].produces is GENERIC_ARTIFACT
    assert resolved.input_contract.ports["items"].accepts == artifact_type
    assert resolved.output_contract.ports["items"].produces == artifact_type


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ({}, "missing artifact type bindings: T"),
        (
            {
                "T": ArtifactTypeKey("example.value", 1),
                "Unknown": ArtifactTypeKey("example.value", 1),
            },
            "unknown artifact type bindings: Unknown",
        ),
        (
            {"T": ArtifactTypeKey("", 1)},
            "binding 'T'.*non-empty artifact type id",
        ),
        (
            {"T": ArtifactTypeKey("example.value", 0)},
            "binding 'T'.*positive schema version",
        ),
    ],
)
def test_generic_node_contract_rejects_contextual_binding_errors(
    bindings: dict[str, ArtifactTypeKey],
    message: str,
) -> None:
    with pytest.raises(
        NodeContractResolutionError,
        match=f"Node 'example.generic'.*{message}",
    ):
        resolve_node_contracts(GenericNode(), bindings)


@pytest.mark.parametrize("name", ["", " T", "T ", "x" * 256])
def test_artifact_type_variable_rejects_noncanonical_names(name: str) -> None:
    with pytest.raises(ValueError, match="Artifact type variable name"):
        ArtifactTypeVariable(name)

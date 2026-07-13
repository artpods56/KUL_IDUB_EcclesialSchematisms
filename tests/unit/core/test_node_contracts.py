from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from notarius_core.artifacts import (
    SOURCE_PAGE_IMAGE,
    ArtifactRefSequence,
    NoConfig,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
    PortShape,
    derive_input_contract,
)


class ExampleImage:
    pass


class ExampleConfig(NodeConfig):
    language: str = "eng"


class ExampleInput(NodeInput):
    pages: Annotated[
        list[ExampleImage],
        InPort(SOURCE_PAGE_IMAGE),
        Field(title="Source pages", description="Images to process in order."),
    ]


class ExampleOutput(NodeOutput):
    pages: Annotated[ArtifactRefSequence, OutPort(SOURCE_PAGE_IMAGE)]


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
        InPort(SOURCE_PAGE_IMAGE),
    ] = None
    page_batches: Annotated[
        list[list[ExampleImage]],
        InPort(SOURCE_PAGE_IMAGE, variadic=True),
    ]
    page_refs: Annotated[ArtifactRefSequence, InPort(SOURCE_PAGE_IMAGE)]


def test_node_contracts_derive_shape_and_resolver_target_from_annotations() -> None:
    assert ExampleNode.config_contract.model is ExampleConfig
    assert ExampleNode.input_contract.model is ExampleInput
    assert ExampleNode.input_contract.ports["pages"].shape == "many"
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

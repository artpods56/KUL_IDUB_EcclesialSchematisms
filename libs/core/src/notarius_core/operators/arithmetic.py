import json
from hashlib import sha256
from typing import Annotated, cast, final, override

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from notarius_core.domain.errors import NotFoundError
from notarius_core.artifacts import (
    ArtifactFieldProjection,
    ArtifactObject,
    ArtifactRef,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NoConfig,
    NodeConfig,
    NodeInput,
    NodeOutput,
    UnitOfWorkPort,
)
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.plugins import Plugin
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
)
from notarius_core.runtime.resolvers import (
    ArtifactContractError,
    ResolutionError,
    Resolver,
)


class IntegerValuePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictInt


class ArithmeticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    addition: StrictInt
    subtraction: StrictInt


INTEGER_VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("scalar.integer", 1),
    title="Integer value",
    payload_schema=cast(JsonObject, IntegerValuePayload.model_json_schema()),
)

ARITHMETIC_RESULT = ArtifactTypeSpec(
    key=ArtifactTypeKey("arithmetic.result", 1),
    title="Arithmetic result",
    payload_schema=cast(JsonObject, ArithmeticResult.model_json_schema()),
    field_projections=(
        ArtifactFieldProjection(
            path=("addition",),
            target=INTEGER_VALUE.key,
            title="Addition",
        ),
        ArtifactFieldProjection(
            path=("subtraction",),
            target=INTEGER_VALUE.key,
            title="Subtraction",
        ),
    ),
)

ARITHMETIC = Plugin(
    slug="builtin.arithmetic",
    title="Arithmetic",
)
ARITHMETIC.register_artifact_type(INTEGER_VALUE)
ARITHMETIC.register_artifact_type(ARITHMETIC_RESULT)


class NumberConfig(NodeConfig):
    value: StrictInt = Field(description="Integer emitted by the node.")


class NumberInput(NodeInput):
    pass


class NumberOutput(NodeOutput):
    value: Annotated[
        StrictInt,
        OutPort(INTEGER_VALUE),
        Field(title="Value", description="The configured integer value."),
    ]


@ARITHMETIC.node(
    operator_id="arithmetic.number",
    version=1,
    title="Number",
)
@final
class NumberNode(Node[NumberConfig, NumberInput, NumberOutput]):
    """Produces a configured integer value."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: NumberConfig,
        _inputs: NumberInput,
        /,
    ) -> NumberOutput:
        return NumberOutput(value=config.value)


class IntegerSequenceConfig(NodeConfig):
    start: StrictInt = Field(default=0, description="First integer in the sequence.")
    count: StrictInt = Field(
        default=3,
        ge=1,
        le=10_000,
        description="Number of integers to produce.",
    )
    step: StrictInt = Field(default=1, description="Difference between values.")


class IntegerSequenceInput(NodeInput):
    pass


class IntegerSequenceOutput(NodeOutput):
    values: Annotated[
        list[StrictInt],
        OutPort(INTEGER_VALUE),
        Field(title="Values", description="Ordered generated integer sequence."),
    ]


@ARITHMETIC.node(
    operator_id="arithmetic.integer_sequence",
    version=1,
    title="Integer sequence",
)
@final
class IntegerSequenceNode(
    Node[IntegerSequenceConfig, IntegerSequenceInput, IntegerSequenceOutput]
):
    """Produces an ordered sequence of configured integer values."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: IntegerSequenceConfig,
        _inputs: IntegerSequenceInput,
        /,
    ) -> IntegerSequenceOutput:
        return IntegerSequenceOutput(
            values=[config.start + index * config.step for index in range(config.count)]
        )


class AddSubtractInput(NodeInput):
    left: Annotated[
        StrictInt,
        InPort(INTEGER_VALUE),
        Field(title="Left", description="Left-hand integer operand."),
    ]
    right: Annotated[
        StrictInt,
        InPort(INTEGER_VALUE),
        Field(title="Right", description="Right-hand integer operand."),
    ]


class AddSubtractOutput(NodeOutput):
    result: Annotated[
        ArithmeticResult,
        OutPort(ARITHMETIC_RESULT),
        Field(description="Compound addition and subtraction result."),
    ]


@ARITHMETIC.node(
    operator_id="arithmetic.add_subtract",
    version=1,
    title="Add & subtract",
)
@final
class AddSubtractNode(Node[NoConfig, AddSubtractInput, AddSubtractOutput]):
    """Produces addition and subtraction fields from two integer inputs."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: AddSubtractInput,
        /,
    ) -> AddSubtractOutput:
        return AddSubtractOutput(
            result=ArithmeticResult(
                addition=inputs.left + inputs.right,
                subtraction=inputs.left - inputs.right,
            )
        )


class MultiplyInput(NodeInput):
    left: Annotated[
        StrictInt,
        InPort(INTEGER_VALUE),
        Field(title="Left", description="Left-hand integer factor."),
    ]
    right: Annotated[
        StrictInt,
        InPort(INTEGER_VALUE),
        Field(title="Right", description="Right-hand integer factor."),
    ]


class MultiplyOutput(NodeOutput):
    result: Annotated[
        StrictInt,
        OutPort(INTEGER_VALUE),
        Field(description="Product of the two input integers."),
    ]


@ARITHMETIC.node(
    operator_id="arithmetic.multiply",
    version=1,
    title="Multiply",
)
@final
class MultiplyNode(Node[NoConfig, MultiplyInput, MultiplyOutput]):
    """Multiplies two integer inputs."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: MultiplyInput,
        /,
    ) -> MultiplyOutput:
        return MultiplyOutput(result=inputs.left * inputs.right)


class SumIntegersInput(NodeInput):
    values: Annotated[
        list[StrictInt],
        InPort(INTEGER_VALUE),
        Field(
            min_length=1,
            title="Values",
            description="Ordered integer sequence to sum.",
        ),
    ]


class SumIntegersOutput(NodeOutput):
    result: Annotated[
        StrictInt,
        OutPort(INTEGER_VALUE),
        Field(description="Sum of all input integers."),
    ]


@ARITHMETIC.node(
    operator_id="arithmetic.sum",
    version=1,
    title="Sum integers",
)
@final
class SumIntegersNode(Node[NoConfig, SumIntegersInput, SumIntegersOutput]):
    """Sums an ordered integer sequence into one integer value."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: SumIntegersInput,
        /,
    ) -> SumIntegersOutput:
        return SumIntegersOutput(result=sum(inputs.values))


@final
class IntegerValueOutputWriter(ArtifactOutputWriter):
    artifact_type = INTEGER_VALUE.key

    def __init__(self, *, uow: UnitOfWorkPort) -> None:
        self._uow = uow

    @override
    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        try:
            payload = IntegerValuePayload.model_validate({"value": value})
        except ValidationError as exc:
            message = (
                f"Failed to serialize {self.artifact_type.id}@"
                f"{self.artifact_type.schema_version} value produced by node "
                f"{context.node_context.node_id!r}"
            )
            raise RuntimeError(message) from exc

        payload_json = cast(JsonObject, payload.model_dump(mode="json"))
        payload_bytes = json.dumps(
            payload_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        provenance: dict[str, object] = {
            input_name: [
                {
                    "artifact_id": str(ref.artifact_id),
                    "artifact_type": ref.artifact_type,
                    "schema_version": ref.schema_version,
                }
                for ref in refs
            ]
            for input_name, refs in context.provenance.refs_by_input.items()
        }
        metadata: JsonObject = {
            "producer_node_id": context.node_context.node_id,
        }
        if provenance:
            metadata["provenance"] = provenance
        metadata.update(context.metadata)
        artifact = ArtifactObject(
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type="application/json",
            storage_backend="inline",
            inline_payload=payload_json,
            byte_size=len(payload_bytes),
            sha256=sha256(payload_bytes).hexdigest(),
            metadata=metadata,
        )
        try:
            async with self._uow as uow:
                await uow.artifacts.add(artifact)
                await uow.commit()
        except Exception as exc:
            message = (
                f"Failed to persist {self.artifact_type.id}@"
                f"{self.artifact_type.schema_version} produced by node "
                f"{context.node_context.node_id!r}"
            )
            raise RuntimeError(message) from exc
        return artifact.ref()


@final
class IntegerValueResolver(Resolver[int]):
    source = INTEGER_VALUE.key
    target: type[object] = int

    def __init__(self, *, uow: UnitOfWorkPort) -> None:
        self._uow = uow

    @override
    async def resolve(self, ref: ArtifactRef) -> int:
        if ref.key() != self.source:
            message = (
                f"Integer resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for artifact {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(ref.artifact_id)
        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))
        if artifact.ref() != ref:
            message = (
                f"Artifact repository returned a different artifact ref for "
                f"integer artifact {ref.artifact_id}"
            )
            raise ArtifactContractError(message)
        if artifact.inline_payload is None:
            message = (
                f"Integer artifact {ref.artifact_id} does not have an inline "
                f"JSON payload"
            )
            raise ArtifactContractError(message)

        try:
            return IntegerValuePayload.model_validate(artifact.inline_payload).value
        except ValidationError as exc:
            message = (
                f"Failed to resolve artifact {ref.artifact_id} as "
                f"{self.source.id}@{self.source.schema_version} integer value"
            )
            raise ResolutionError(message) from exc

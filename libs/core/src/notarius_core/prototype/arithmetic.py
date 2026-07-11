import json
from hashlib import sha256
from typing import Annotated, ClassVar, cast, final, override

from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

from notarius_core.domain.errors import NotFoundError
from notarius_core.prototype.artifacts import (
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
from notarius_core.prototype.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.prototype.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
)
from notarius_core.prototype.resolvers import (
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


class NumberConfig(NodeConfig):
    value: StrictInt


class NumberInput(NodeInput):
    pass


class NumberOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@final
class NumberNode(Node[NumberConfig, NumberInput, NumberOutput]):
    operator_id: ClassVar[str] = "arithmetic.number"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: NumberConfig,
        _inputs: NumberInput,
        /,
    ) -> NumberOutput:
        return NumberOutput(value=config.value)


class AddSubtractInput(NodeInput):
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class AddSubtractOutput(NodeOutput):
    result: Annotated[ArithmeticResult, OutPort(ARITHMETIC_RESULT)]


@final
class AddSubtractNode(Node[NoConfig, AddSubtractInput, AddSubtractOutput]):
    operator_id: ClassVar[str] = "arithmetic.add_subtract"
    operator_version: ClassVar[int] = 1

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
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class MultiplyOutput(NodeOutput):
    result: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@final
class MultiplyNode(Node[NoConfig, MultiplyInput, MultiplyOutput]):
    operator_id: ClassVar[str] = "arithmetic.multiply"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: MultiplyInput,
        /,
    ) -> MultiplyOutput:
        return MultiplyOutput(result=inputs.left * inputs.right)


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

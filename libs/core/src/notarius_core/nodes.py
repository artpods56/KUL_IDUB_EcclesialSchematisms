from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import UnionType
from typing import Annotated, Any, ClassVar, Union, cast, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel

from notarius_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    ArtifactRef,
    ArtifactRefSequence,
    NodeConfig,
    NodeInput,
    NodeOutput,
)


class PortShape(StrEnum):
    ONE = "one"
    MANY = "many"


class NodeContractError(TypeError):
    pass


class NodeContractResolutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactTypeVariable:
    """A named artifact type chosen when a node instance is bound."""

    name: str

    def __post_init__(self) -> None:
        if self.name.strip() == "":
            raise ValueError("Artifact type variable name must not be empty")
        if self.name != self.name.strip():
            raise ValueError(
                "Artifact type variable name must not have surrounding whitespace"
            )
        if len(self.name) > 255:
            raise ValueError(
                "Artifact type variable name must be at most 255 characters"
            )


ArtifactTypeContract = ArtifactTypeKey | ArtifactTypeVariable


@dataclass(frozen=True, slots=True)
class InPort:
    """Marks an input model field as an artifact port via Annotated metadata."""

    accepts: ArtifactTypeSpec | ArtifactTypeContract
    variadic: bool = False
    instance_plugs: bool = False


@dataclass(frozen=True, slots=True)
class OutPort:
    """Marks an output model field as an artifact port via Annotated metadata."""

    produces: ArtifactTypeSpec | ArtifactTypeContract


@dataclass(frozen=True, slots=True)
class InputPortSpec:
    name: str
    accepts: ArtifactTypeContract
    title: str | None = None
    description: str | None = None
    shape: PortShape = PortShape.ONE
    variadic: bool = False
    instance_plugs: bool = False
    accepted_shapes: tuple[PortShape, ...] = (PortShape.ONE,)
    target_type: type[object] | None = None
    preserves_ref_container: bool = False
    allows_none: bool = False
    required: bool = True


@dataclass(frozen=True, slots=True)
class OutputPortSpec:
    name: str
    produces: ArtifactTypeContract
    title: str | None = None
    description: str | None = None
    shape: PortShape = PortShape.ONE
    required: bool = True


@dataclass(frozen=True, slots=True)
class InputContract[T: BaseModel]:
    model: type[T]
    ports: dict[str, InputPortSpec]


@dataclass(frozen=True, slots=True)
class OutputContract[T: BaseModel]:
    model: type[T]
    ports: dict[str, OutputPortSpec]


@dataclass(frozen=True, slots=True)
class ConfigContract[T: BaseModel]:
    model: type[T]


@dataclass(frozen=True, slots=True)
class ResolvedNodeContracts:
    input_contract: InputContract[Any]
    output_contract: OutputContract[Any]


@dataclass(frozen=True, slots=True)
class NodeExecutionContext:
    workflow_run_id: UUID | None = None
    node_run_id: UUID | None = None
    node_id: str | None = None
    invocation_index: int | None = None


def _artifact_type_contract(
    value: ArtifactTypeSpec | ArtifactTypeContract,
) -> ArtifactTypeContract:
    if isinstance(value, ArtifactTypeSpec):
        return value.key
    return value


def derive_input_contract[T: BaseModel](model: type[T]) -> InputContract[T]:
    ports: dict[str, InputPortSpec] = {}
    for name, field in model.model_fields.items():
        port = _single_port_meta(model, name, field.metadata, InPort, OutPort)
        if port is None:
            continue
        (
            shape,
            accepted_shapes,
            target_type,
            preserves_ref_container,
            allows_none,
        ) = _input_annotation_contract(
            model=model,
            field_name=name,
            annotation=field.annotation,
            variadic=port.variadic,
            instance_plugs=port.instance_plugs,
        )
        ports[name] = InputPortSpec(
            name=name,
            accepts=_artifact_type_contract(port.accepts),
            title=field.title,
            description=field.description,
            shape=shape,
            variadic=port.variadic,
            instance_plugs=port.instance_plugs,
            accepted_shapes=accepted_shapes,
            target_type=target_type,
            preserves_ref_container=preserves_ref_container,
            allows_none=allows_none,
            required=field.is_required(),
        )
    return InputContract(model=model, ports=ports)


def derive_output_contract[T: BaseModel](model: type[T]) -> OutputContract[T]:
    ports: dict[str, OutputPortSpec] = {}
    for name, field in model.model_fields.items():
        port = _single_port_meta(model, name, field.metadata, OutPort, InPort)
        if port is None:
            continue
        shape = _output_annotation_shape(
            model=model,
            field_name=name,
            annotation=field.annotation,
        )
        ports[name] = OutputPortSpec(
            name=name,
            produces=_artifact_type_contract(port.produces),
            title=field.title,
            description=field.description,
            shape=shape,
            required=field.is_required(),
        )
    return OutputContract(model=model, ports=ports)


def _single_port_meta[PortT](
    model: type[BaseModel],
    field_name: str,
    metadata: list[Any],
    expected: type[PortT],
    forbidden: type[object],
) -> PortT | None:
    if any(isinstance(meta, forbidden) for meta in metadata):
        message = (
            f"{model.__name__}.{field_name} declares a {forbidden.__name__}; "
            f"only {expected.__name__} is valid on this side of the contract"
        )
        raise NodeContractError(message)
    ports = [meta for meta in metadata if isinstance(meta, expected)]
    if len(ports) > 1:
        message = (
            f"{model.__name__}.{field_name} declares {len(ports)} "
            f"{expected.__name__} annotations; at most one is allowed"
        )
        raise NodeContractError(message)
    return ports[0] if ports else None


def _input_annotation_contract(
    *,
    model: type[BaseModel],
    field_name: str,
    annotation: object,
    variadic: bool,
    instance_plugs: bool,
) -> tuple[
    PortShape,
    tuple[PortShape, ...],
    type[object] | None,
    bool,
    bool,
]:
    if instance_plugs:
        if not variadic:
            message = (
                f"{model.__name__}.{field_name} declares instance_plugs=True; "
                "instance plugs require variadic=True"
            )
            raise NodeContractError(message)
        if get_origin(annotation) is not list:
            message = (
                f"{model.__name__}.{field_name} must use exactly "
                "list[ArtifactRef | ArtifactRefSequence] for instance plugs"
            )
            raise NodeContractError(message)
        item_type = _sequence_item_type(
            model=model,
            field_name=field_name,
            annotation=annotation,
            purpose="instance plugs",
        )
        item_origin = get_origin(item_type)
        item_types = set(get_args(item_type))
        if item_origin not in (UnionType, Union) or item_types != {
            ArtifactRef,
            ArtifactRefSequence,
        }:
            message = (
                f"{model.__name__}.{field_name} must use exactly "
                "list[ArtifactRef | ArtifactRefSequence] for instance plugs"
            )
            raise NodeContractError(message)
        return (
            PortShape.ONE,
            (PortShape.ONE, PortShape.MANY),
            None,
            True,
            False,
        )

    value_type, allows_none = _unwrap_optional(annotation)
    if variadic:
        value_type = _sequence_item_type(
            model=model,
            field_name=field_name,
            annotation=value_type,
            purpose="a variadic input port",
        )
        value_type, item_allows_none = _unwrap_optional(value_type)
        allows_none = allows_none or item_allows_none

    if value_type is ArtifactRef:
        return PortShape.ONE, (PortShape.ONE,), None, True, allows_none
    if value_type is ArtifactRefSequence:
        return PortShape.MANY, (PortShape.MANY,), None, True, allows_none

    origin = get_origin(value_type)
    if _is_sequence_origin(origin):
        item_type = _sequence_item_type(
            model=model,
            field_name=field_name,
            annotation=value_type,
            purpose="an artifact sequence",
        )
        item_type, _ = _unwrap_optional(item_type)
        target_type = _concrete_type(model, field_name, item_type)
        return (
            PortShape.MANY,
            (PortShape.MANY,),
            target_type,
            False,
            allows_none,
        )

    return (
        PortShape.ONE,
        (PortShape.ONE,),
        _concrete_type(model, field_name, value_type),
        False,
        allows_none,
    )


def _output_annotation_shape(
    *,
    model: type[BaseModel],
    field_name: str,
    annotation: object,
) -> PortShape:
    value_type, _ = _unwrap_optional(annotation)
    if value_type is ArtifactRef:
        return PortShape.ONE
    if value_type is ArtifactRefSequence:
        return PortShape.MANY
    if _is_sequence_origin(get_origin(value_type)):
        _sequence_item_type(
            model=model,
            field_name=field_name,
            annotation=value_type,
            purpose="an artifact sequence",
        )
        return PortShape.MANY
    return PortShape.ONE


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _unwrap_optional(get_args(annotation)[0])
    if origin not in (UnionType, Union):
        return annotation, False

    args = get_args(annotation)
    non_none = tuple(arg for arg in args if arg is not type(None))
    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0], True
    return annotation, False


def _is_sequence_origin(origin: object) -> bool:
    return origin is list or origin is tuple or origin is Sequence


def _sequence_item_type(
    *,
    model: type[BaseModel],
    field_name: str,
    annotation: object,
    purpose: str,
) -> object:
    origin = get_origin(annotation)
    if not _is_sequence_origin(origin):
        message = (
            f"{model.__name__}.{field_name} must use a list, tuple, or Sequence "
            f"annotation for {purpose}"
        )
        raise NodeContractError(message)
    args = get_args(annotation)
    if len(args) == 0:
        message = (
            f"{model.__name__}.{field_name} must declare an item type for {purpose}"
        )
        raise NodeContractError(message)
    return args[0]


def _concrete_type(
    model: type[BaseModel],
    field_name: str,
    annotation: object,
) -> type[object]:
    if isinstance(annotation, type):
        return annotation
    message = (
        f"{model.__name__}.{field_name} artifact value type must be a concrete "
        f"Python type, got {annotation!r}"
    )
    raise NodeContractError(message)


class Node[
    ConfigT: NodeConfig,
    InputT: NodeInput,
    OutputT: NodeOutput,
](ABC):
    operator_id: ClassVar[str]
    operator_version: ClassVar[int]
    plugin_slug: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    config_contract: ClassVar[ConfigContract[Any]]
    input_contract: ClassVar[InputContract[Any]]
    output_contract: ClassVar[OutputContract[Any]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is not Node:
                continue
            type_args = get_args(base)
            if len(type_args) != 3:
                continue
            config_model, input_model, output_model = type_args
            if not (
                isinstance(config_model, type)
                and issubclass(config_model, NodeConfig)
                and isinstance(input_model, type)
                and issubclass(input_model, NodeInput)
                and isinstance(output_model, type)
                and issubclass(output_model, NodeOutput)
            ):
                continue
            cls.config_contract = ConfigContract(model=config_model)
            cls.input_contract = derive_input_contract(input_model)
            cls.output_contract = derive_output_contract(output_model)
            return

    @abstractmethod
    async def run(
        self,
        context: NodeExecutionContext,
        config: ConfigT,
        inputs: InputT,
        /,
    ) -> OutputT: ...


def resolve_node_contracts(
    node: Node[Any, Any, Any],
    bindings: Mapping[str, ArtifactTypeKey],
) -> ResolvedNodeContracts:
    """Resolve every artifact type variable declared by one node instance."""

    variables: set[str] = set()
    for port in node.input_contract.ports.values():
        if isinstance(port.accepts, ArtifactTypeVariable):
            variables.add(port.accepts.name)
    for port in node.output_contract.ports.values():
        if isinstance(port.produces, ArtifactTypeVariable):
            variables.add(port.produces.name)
    raw_bindings = cast(Mapping[object, object], bindings)
    binding_names: set[str] = set()
    for raw_name in raw_bindings:
        if not isinstance(raw_name, str):
            raise NodeContractResolutionError(
                f"Node {node.operator_id!r} artifact type binding names must be "
                f"strings, got {type(raw_name).__name__}"
            )
        binding_names.add(raw_name)

    unknown = sorted(binding_names - variables)
    if unknown:
        rendered = ", ".join(unknown)
        raise NodeContractResolutionError(
            f"Node {node.operator_id!r} received unknown artifact type bindings: "
            f"{rendered}"
        )

    missing = sorted(variables - binding_names)
    if missing:
        rendered = ", ".join(missing)
        raise NodeContractResolutionError(
            f"Node {node.operator_id!r} is missing artifact type bindings: {rendered}"
        )

    for raw_name, raw_key in raw_bindings.items():
        if not isinstance(raw_name, str):
            continue
        if not isinstance(raw_key, ArtifactTypeKey):
            raise NodeContractResolutionError(
                f"Node {node.operator_id!r} artifact type binding {raw_name!r} "
                f"must be an ArtifactTypeKey, got {type(raw_key).__name__}"
            )
        if raw_key.id.strip() == "":
            raise NodeContractResolutionError(
                f"Node {node.operator_id!r} artifact type binding {raw_name!r} must "
                "reference a non-empty artifact type id"
            )
        if raw_key.id != raw_key.id.strip():
            raise NodeContractResolutionError(
                f"Node {node.operator_id!r} artifact type binding {raw_name!r} must "
                "reference an artifact type id without surrounding whitespace"
            )
        if isinstance(raw_key.schema_version, bool) or raw_key.schema_version < 1:
            raise NodeContractResolutionError(
                f"Node {node.operator_id!r} artifact type binding {raw_name!r} must "
                "reference a positive schema version"
            )

    input_ports = {
        name: replace(port, accepts=bindings[port.accepts.name])
        if isinstance(port.accepts, ArtifactTypeVariable)
        else port
        for name, port in node.input_contract.ports.items()
    }
    output_ports = {
        name: replace(port, produces=bindings[port.produces.name])
        if isinstance(port.produces, ArtifactTypeVariable)
        else port
        for name, port in node.output_contract.ports.items()
    }
    return ResolvedNodeContracts(
        input_contract=replace(node.input_contract, ports=input_ports),
        output_contract=replace(node.output_contract, ports=output_ports),
    )

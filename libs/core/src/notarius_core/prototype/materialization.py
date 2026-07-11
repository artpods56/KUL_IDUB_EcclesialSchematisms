from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from pydantic import BaseModel

from notarius_core.prototype.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
)
from notarius_core.prototype.nodes import InputContract, PortShape
from notarius_core.prototype.resolvers import ResolverRegistry


class MaterializationError(RuntimeError):
    def __init__(self, port_name: str, message: str) -> None:
        super().__init__(f"Input {port_name!r} {message}")
        self.port_name = port_name


@dataclass(frozen=True, slots=True)
class MaterializationProvenance:
    refs_by_input: dict[str, tuple[ArtifactRef, ...]]

    def refs_for(self, input_name: str) -> tuple[ArtifactRef, ...]:
        return self.refs_by_input.get(input_name, ())


_EXPECTED_PORT_VALUE: Final = {
    (False, PortShape.ONE): "an ArtifactRef",
    (False, PortShape.MANY): (
        "an ArtifactRefSequence (wrap refs in an ArtifactRefSequence)"
    ),
    (True, PortShape.ONE): "a list with one ArtifactRef per incoming edge",
    (True, PortShape.MANY): "a list with one ArtifactRefSequence per incoming edge",
}


class InputMaterializer:
    def __init__(self, resolver_registry: ResolverRegistry) -> None:
        self._resolver_registry = resolver_registry

    async def materialize[T: BaseModel](
        self,
        contract: InputContract[T],
        inputs: Mapping[str, object],
    ) -> tuple[T, MaterializationProvenance]:
        values: dict[str, object] = {}
        refs_by_input: dict[str, tuple[ArtifactRef, ...]] = {}

        for name in contract.model.model_fields:
            if name in inputs:
                values[name] = inputs[name]

        for name, spec in contract.ports.items():
            if name not in inputs:
                if spec.required:
                    raise MaterializationError(name, "is required")
                continue

            value = inputs[name]
            if value is None and spec.allows_none:
                values[name] = None
                continue

            # Normalize every legal input into one canonical form: a list of
            # refs per incoming edge. Everything below reads from `edges` and
            # never re-inspects the raw value.
            edges: list[list[ArtifactRef]]
            match spec.variadic, spec.shape, value:
                case False, PortShape.ONE, ArtifactRef() as ref:
                    edges = [[ref]]
                case False, PortShape.MANY, ArtifactRefSequence() as sequence:
                    edges = [list(sequence.item_refs)]
                case True, PortShape.ONE, list() | tuple():
                    edges = []
                    for item in cast(Sequence[object], value):
                        if not isinstance(item, ArtifactRef):
                            raise MaterializationError(
                                name,
                                f"expected one ArtifactRef per incoming edge, "
                                f"got {type(item).__name__}",
                            )
                        edges.append([item])
                case True, PortShape.MANY, list() | tuple():
                    edges = []
                    for item in cast(Sequence[object], value):
                        if not isinstance(item, ArtifactRefSequence):
                            raise MaterializationError(
                                name,
                                f"expected one ArtifactRefSequence per incoming "
                                f"edge, got {type(item).__name__}",
                            )
                        edges.append(list(item.item_refs))
                case _:
                    expected = _EXPECTED_PORT_VALUE[(spec.variadic, spec.shape)]
                    raise MaterializationError(
                        name, f"expected {expected}, got {type(value).__name__}"
                    )

            if spec.variadic and len(edges) == 0 and spec.required:
                raise MaterializationError(name, "expected at least one incoming edge")

            refs = tuple(ref for edge in edges for ref in edge)
            for ref in refs:
                if ref.key() != spec.accepts:
                    raise MaterializationError(
                        name,
                        f"expected {spec.accepts.id}@"
                        f"{spec.accepts.schema_version}, got "
                        f"{ref.artifact_type}@{ref.schema_version}",
                    )
            if len(refs) > 0:
                refs_by_input[name] = refs

            if spec.preserves_ref_container:
                values[name] = value
                continue

            target = spec.target_type
            if target is None:
                raise MaterializationError(
                    name, "does not declare a resolvable Python value type"
                )
            resolved = [
                [
                    await self._resolver_registry.resolve(ref=ref, target=target)
                    for ref in edge
                ]
                for edge in edges
            ]
            match spec.variadic, spec.shape:
                case False, PortShape.ONE:
                    values[name] = resolved[0][0]
                case False, PortShape.MANY:
                    values[name] = resolved[0]
                case True, PortShape.ONE:
                    values[name] = [edge[0] for edge in resolved]
                case True, PortShape.MANY:
                    values[name] = resolved

        return contract.model.model_validate(values), MaterializationProvenance(
            refs_by_input=refs_by_input,
        )

from collections.abc import Callable
from dataclasses import dataclass

from grafy_core.artifacts import ArtifactTypeKey


MAX_ARTIFACT_CONVERSION_HOPS = 8


def conversion_runtime_types_are_compatible(
    produced: type[object],
    accepted: type[object],
) -> bool:
    """Conservatively prove that every declared output fits the next input."""
    if produced is accepted or accepted is object:
        return True
    if produced is int and accepted is float:
        return True
    return False


@dataclass(frozen=True, slots=True)
class ArtifactConversionKey:
    id: str
    version: int

    def __post_init__(self) -> None:
        if self.id.strip() == "":
            raise ValueError("Artifact conversion id must not be blank")
        if self.version < 1:
            raise ValueError("Artifact conversion version must be positive")


@dataclass(frozen=True, slots=True)
class ArtifactConversion[SourceT, TargetT]:
    key: ArtifactConversionKey
    source: ArtifactTypeKey
    target: ArtifactTypeKey
    source_type: type[SourceT]
    target_type: type[TargetT]
    title: str
    convert: Callable[[SourceT], TargetT]

    def __post_init__(self) -> None:
        if self.title.strip() == "":
            raise ValueError("Artifact conversion title must not be blank")

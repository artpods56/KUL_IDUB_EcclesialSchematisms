from collections.abc import Callable
from dataclasses import dataclass

from notarius_core.artifacts import ArtifactTypeKey


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
    title: str
    convert: Callable[[SourceT], TargetT]

    def __post_init__(self) -> None:
        if self.title.strip() == "":
            raise ValueError("Artifact conversion title must not be blank")

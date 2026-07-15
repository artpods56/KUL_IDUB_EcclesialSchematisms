from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence


MaterializedOutputValue = ArtifactRef | ArtifactRefSequence


class MaterializedOutputEnvelope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    port: str = Field(min_length=1, max_length=255)
    kind: Literal["single", "sequence"]
    value: MaterializedOutputValue

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        expected_kind = (
            "sequence" if isinstance(self.value, ArtifactRefSequence) else "single"
        )
        if self.kind != expected_kind:
            raise ValueError(
                f"Materialized output {self.port!r} declares kind {self.kind!r}, "
                f"but its value is {expected_kind!r}"
            )
        return self


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class MaterializedNodeOutputs:
    graph_id: UUID
    graph_revision: int
    node_id: str
    workflow_run_id: UUID
    outputs: dict[str, MaterializedOutputValue]
    materialized_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.graph_revision < 1:
            raise ValueError("Materialized output graph revision must be at least 1")
        self.node_id = self.node_id.strip()
        if self.node_id == "":
            raise ValueError("Materialized output node id must not be blank")
        if len(self.node_id) > 255:
            raise ValueError("Materialized output node id must be at most 255 characters")
        if self.materialized_at.tzinfo is None:
            raise ValueError("Materialized output timestamp must be timezone-aware")

        normalized: dict[str, MaterializedOutputValue] = {}
        for raw_port, value in self.outputs.items():
            port = raw_port.strip()
            if port == "":
                raise ValueError("Materialized output port must not be blank")
            if len(port) > 255:
                raise ValueError(
                    "Materialized output port must be at most 255 characters"
                )
            if port in normalized:
                raise ValueError(f"Duplicate materialized output port {port!r}")
            normalized[port] = value.model_copy(deep=True)
        self.outputs = normalized

    def storage_envelopes(self) -> list[dict[str, object]]:
        return self.outputs_to_storage(self.outputs)

    @staticmethod
    def outputs_to_storage(
        outputs: dict[str, MaterializedOutputValue],
    ) -> list[dict[str, object]]:
        return [
            MaterializedOutputEnvelope(
                port=port,
                kind=(
                    "sequence"
                    if isinstance(value, ArtifactRefSequence)
                    else "single"
                ),
                value=value,
            ).model_dump(mode="json")
            for port, value in sorted(outputs.items())
        ]

    @staticmethod
    def outputs_from_storage(
        value: object,
    ) -> dict[str, MaterializedOutputValue]:
        if not isinstance(value, list):
            raise ValueError("Materialized outputs storage payload must be a list")
        outputs: dict[str, MaterializedOutputValue] = {}
        for raw_envelope in cast(list[object], value):
            envelope = MaterializedOutputEnvelope.model_validate(raw_envelope)
            if envelope.port in outputs:
                raise ValueError(
                    f"Duplicate materialized output port {envelope.port!r}"
                )
            outputs[envelope.port] = envelope.value
        return outputs

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from notarius_core.domain.artifact_outputs import (
    ArtifactOutputEnvelope,
    ArtifactOutputValue,
    artifact_outputs_from_storage,
    artifact_outputs_to_storage,
    normalize_artifact_outputs,
)


# Compatibility aliases for existing callers while the shared artifact-output
# vocabulary becomes the persistence boundary used by both materializations and
# invocation-cache entries.
MaterializedOutputValue = ArtifactOutputValue
MaterializedOutputEnvelope = ArtifactOutputEnvelope


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
            raise ValueError(
                "Materialized output node id must be at most 255 characters"
            )
        if self.materialized_at.tzinfo is None:
            raise ValueError("Materialized output timestamp must be timezone-aware")

        self.outputs = normalize_artifact_outputs(self.outputs)

    def storage_envelopes(self) -> list[dict[str, object]]:
        return self.outputs_to_storage(self.outputs)

    @staticmethod
    def outputs_to_storage(
        outputs: dict[str, MaterializedOutputValue],
    ) -> list[dict[str, object]]:
        return artifact_outputs_to_storage(outputs)

    @staticmethod
    def outputs_from_storage(
        value: object,
    ) -> dict[str, MaterializedOutputValue]:
        return artifact_outputs_from_storage(value)

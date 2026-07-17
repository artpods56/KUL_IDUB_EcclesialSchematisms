import json
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs

from notarius_api.schemas.workbench import (
    ArtifactSummaryResponse,
    GraphMaterializationsResponse,
    RunNodeResponse,
    RunPortOutputResponse,
    RunResponse,
)
from notarius_api.services.artifacts import ArtifactService
from notarius_api.services.execution.models import GraphExecutionResult


class RunResultPresenter:
    """Maps graph execution and materialization results to HTTP response models."""

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    async def run_response(self, execution: GraphExecutionResult) -> RunResponse:
        return RunResponse(
            status=execution.status,
            node_runs=[
                RunNodeResponse(
                    node_id=node_result.node_id,
                    status=node_result.status,
                    error=node_result.error,
                    outputs=[
                        await self.port_output_response(port_name, value)
                        for port_name, value in node_result.outputs.items()
                    ],
                )
                for node_result in execution.node_results
            ],
        )

    async def materializations_response(
        self,
        graph_id: UUID,
        graph_revision: int,
        materializations: Sequence[MaterializedNodeOutputs],
    ) -> GraphMaterializationsResponse:
        node_runs: list[RunNodeResponse] = []
        for materialization in materializations:
            accessible_outputs: list[RunPortOutputResponse] = []
            for port_name, value in materialization.outputs.items():
                if await self._artifacts.is_accessible(value):
                    accessible_outputs.append(
                        await self.port_output_response(port_name, value)
                    )
            if materialization.outputs and not accessible_outputs:
                continue
            node_runs.append(
                RunNodeResponse(
                    node_id=materialization.node_id,
                    status="succeeded",
                    error=None,
                    outputs=accessible_outputs,
                )
            )
        return GraphMaterializationsResponse(
            graph_id=graph_id,
            graph_revision=graph_revision,
            node_runs=node_runs,
        )

    async def port_output_response(
        self,
        port_name: str,
        value: ArtifactOutputValue,
    ) -> RunPortOutputResponse:
        if isinstance(value, ArtifactRefSequence):
            refs = list(value.item_refs)
            kind: Literal["single", "sequence"] = "sequence"
        else:
            refs = [value]
            kind = "single"
        return RunPortOutputResponse(
            port=port_name,
            kind=kind,
            value=value,
            artifacts=[await self.artifact_summary(ref) for ref in refs],
        )

    async def artifact_summary(
        self,
        ref: ArtifactRef,
    ) -> ArtifactSummaryResponse:
        artifact = await self._artifacts.get(ref.artifact_id)
        if artifact is None:
            return ArtifactSummaryResponse(
                artifact_id=ref.artifact_id,
                artifact_type=ref.artifact_type,
                schema_version=ref.schema_version,
                content_type="application/octet-stream",
            )
        text: str | None = None
        if artifact.inline_payload is not None:
            payload_text = artifact.inline_payload.get("text")
            if isinstance(payload_text, str):
                text = payload_text
            payload_markdown = artifact.inline_payload.get("markdown")
            if text is None and isinstance(payload_markdown, str):
                text = payload_markdown
            if text is None:
                if set(artifact.inline_payload) == {"value"}:
                    text = json.dumps(
                        artifact.inline_payload["value"],
                        ensure_ascii=False,
                    )
                else:
                    text = json.dumps(
                        artifact.inline_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
        return ArtifactSummaryResponse(
            artifact_id=artifact.id,
            artifact_type=artifact.artifact_type,
            schema_version=artifact.schema_version,
            content_type=artifact.content_type,
            byte_size=artifact.byte_size,
            sha256=artifact.sha256,
            text=text,
            content_url=f"./artifacts/{artifact.id}/content",
            metadata=artifact.metadata,
        )


__all__ = ["RunResultPresenter"]

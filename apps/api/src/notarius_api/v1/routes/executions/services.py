import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionDetail,
    GraphExecutionNodeResult,
    GraphExecutionPage,
    GraphExecutionScope,
    GraphExecutionStatus,
)
from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.saved_graphs import SavedGraphRevision
from notarius_core.operators.tables import TABLE_DATA
from notarius_core.ports.execution_history import ExecutionHistoryUnitOfWorkPort
from notarius_core.ports.materialized_outputs import WorkbenchUnitOfWorkPort

from notarius_api.v1.routes.artifacts.models import ArtifactSummaryResponse
from notarius_api.v1.routes.artifacts.services import ArtifactService

from .models import (
    GraphMaterializationsResponse,
    RunExecutionResponse,
    RunNodeResponse,
    RunPortOutputResponse,
    RunResponse,
)
from .runtime.errors import GraphExecutionError

if TYPE_CHECKING:
    from .runtime.manager import RunExecutionSnapshot
    from .runtime.models import GraphExecutionResult


INLINE_SUMMARY_TEXT_BYTE_LIMIT = 64 * 1_024


class ExecutionHistoryService:
    """Own the durable lifecycle and browsing of saved-graph executions."""

    def __init__(
        self,
        unit_of_work: ExecutionHistoryUnitOfWorkPort,
        saved_graphs: SavedGraphService | None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._saved_graphs = saved_graphs

    async def create_queued(
        self,
        *,
        workspace_id: UUID,
        execution_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        scope: GraphExecutionScope,
        requested_node_ids: tuple[str, ...],
    ) -> GraphExecution:
        if self._saved_graphs is None:
            raise RuntimeError(
                "Saved graph context is not configured for execution history"
            )
        await self._saved_graphs.get_revision(
            workspace_id,
            graph_id,
            graph_revision,
        )
        execution = GraphExecution(
            workspace_id=workspace_id,
            execution_id=execution_id,
            graph_id=graph_id,
            graph_revision=graph_revision,
            scope=scope,
            status="queued",
            requested_node_ids=requested_node_ids,
        )
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.add(execution)
            await unit_of_work.commit()
        return execution

    async def mark_running(
        self,
        workspace_id: UUID,
        execution: GraphExecution,
    ) -> None:
        if execution.workspace_id != workspace_id:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        execution.status = "running"
        execution.started_at = datetime.now(UTC)
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            await unit_of_work.commit()

    async def mark_cancelling(
        self,
        workspace_id: UUID,
        execution: GraphExecution,
    ) -> None:
        if execution.workspace_id != workspace_id:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        execution.status = "cancelling"
        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            await unit_of_work.commit()

    async def complete(
        self,
        workspace_id: UUID,
        execution: GraphExecution,
        *,
        status: GraphExecutionStatus,
        result: "GraphExecutionResult | None",
        error: str | None,
    ) -> None:
        if execution.workspace_id != workspace_id:
            raise NotFoundError("Graph execution", str(execution.execution_id))
        if status not in {"cancelled", "succeeded", "failed"}:
            raise ValueError(f"Execution completion status {status!r} is not terminal")
        completed_at = datetime.now(UTC)
        execution.status = status
        execution.finished_at = completed_at
        execution.error = error
        if result is not None:
            execution.workflow_run_id = result.workflow_run_id

        async with self._unit_of_work as unit_of_work:
            await unit_of_work.execution_history.update(execution)
            if result is not None:
                for position, node_result in enumerate(result.node_results):
                    await unit_of_work.execution_history.add_node_result(
                        GraphExecutionNodeResult(
                            workspace_id=workspace_id,
                            execution_id=execution.execution_id,
                            node_id=node_result.node_id,
                            position=position,
                            status=node_result.status,
                            outputs=dict(node_result.outputs),
                            error=node_result.error,
                            completed_at=completed_at,
                        )
                    )
            await unit_of_work.commit()

    async def get_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        execution_id: UUID,
    ) -> GraphExecutionDetail | None:
        async with self._unit_of_work as unit_of_work:
            detail = await unit_of_work.execution_history.get(
                workspace_id,
                execution_id,
            )
        if detail is None or detail.execution.graph_id != graph_id:
            return None
        return detail

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        *,
        limit: int,
        cursor: GraphExecutionCursor | None = None,
        graph_revision: int | None = None,
        status: GraphExecutionStatus | None = None,
        node_id: str | None = None,
    ) -> GraphExecutionPage:
        async with self._unit_of_work as unit_of_work:
            return await unit_of_work.execution_history.list_for_graph(
                workspace_id,
                graph_id,
                limit=limit,
                cursor=cursor,
                graph_revision=graph_revision,
                status=status,
                node_id=node_id,
            )

    async def interrupt_active(self, workspace_id: UUID) -> int:
        async with self._unit_of_work as unit_of_work:
            interrupted = await unit_of_work.execution_history.interrupt_active(
                workspace_id=workspace_id,
                finished_at=datetime.now(UTC),
                error=(
                    "Execution was interrupted because the API process stopped "
                    "before reporting a terminal result"
                ),
            )
            await unit_of_work.commit()
        return interrupted


class MaterializationService:
    """Owns persisted graph-output snapshots and submitted pinned outputs."""

    def __init__(
        self,
        unit_of_work: WorkbenchUnitOfWorkPort,
        artifacts: ArtifactService,
        saved_graphs: SavedGraphService | None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._artifacts = artifacts
        self._saved_graphs = saved_graphs

    async def saved_graph_revision(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
    ) -> SavedGraphRevision:
        if graph_revision < 1:
            raise GraphExecutionError("Graph revision must be positive")
        if self._saved_graphs is None:
            raise GraphExecutionError(
                "Saved graph context is not configured for this workbench"
            )
        return await self._saved_graphs.get_revision(
            workspace_id,
            graph_id,
            graph_revision,
        )

    async def validate_latest_pins(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        submitted_pins: Mapping[tuple[str, str], ArtifactOutputValue],
    ) -> None:
        if not submitted_pins:
            return
        async with self._unit_of_work as unit_of_work:
            materializations = await unit_of_work.materialized_outputs.list_for_graph(
                workspace_id,
                graph_id,
                graph_revision,
            )
        by_node = {
            materialization.node_id: materialization
            for materialization in materializations
        }

        for (from_node, from_port), submitted_value in submitted_pins.items():
            materialization = by_node.get(from_node)
            materialized_value = (
                materialization.outputs.get(from_port)
                if materialization is not None
                else None
            )
            if materialized_value is None or not await self._artifacts.is_accessible(
                workspace_id,
                materialized_value,
            ):
                raise GraphExecutionError(
                    f"Cannot reuse upstream output {from_node!r}.{from_port!r}: "
                    "there is no accessible materialized artifact for this graph "
                    "revision. Run the upstream node too or choose "
                    "'Run with dependencies'."
                )
            if submitted_value != materialized_value:
                raise GraphExecutionError(
                    f"Pinned output {from_node!r}.{from_port!r} is not the latest "
                    f"materialized output for graph {graph_id} revision "
                    f"{graph_revision}. Refresh the graph materializations and "
                    "try again, or choose 'Run with dependencies'."
                )

    async def resolve_pinned_outputs(
        self,
        workspace_id: UUID,
        pinned_outputs: Mapping[tuple[str, str], ArtifactOutputValue],
    ) -> dict[str, dict[str, ArtifactOutputValue]]:
        outputs: dict[str, dict[str, ArtifactOutputValue]] = {}
        for (from_node, from_port), value in pinned_outputs.items():
            context = f"Pinned output {from_node!r}.{from_port!r}"
            await self._artifacts.validate_refs(
                workspace_id,
                value,
                context=context,
            )
            if not await self._artifacts.is_accessible(workspace_id, value):
                raise GraphExecutionError(f"{context} is not accessible")
            outputs.setdefault(from_node, {})[from_port] = value
        return outputs

    async def persist_execution(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        execution: "GraphExecutionResult",
    ) -> None:
        successful_results = [
            node_result
            for node_result in execution.node_results
            if node_result.status == "succeeded"
        ]
        if not successful_results:
            return
        async with self._unit_of_work as unit_of_work:
            for node_result in successful_results:
                await unit_of_work.materialized_outputs.upsert(
                    MaterializedNodeOutputs(
                        workspace_id=workspace_id,
                        graph_id=graph_id,
                        graph_revision=graph_revision,
                        node_id=node_result.node_id,
                        workflow_run_id=execution.workflow_run_id,
                        outputs=dict(node_result.outputs),
                        materialized_at=datetime.now(UTC),
                    )
                )
            await unit_of_work.commit()

    async def list_for_graph(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
    ) -> list[MaterializedNodeOutputs]:
        await self.saved_graph_revision(workspace_id, graph_id, graph_revision)
        async with self._unit_of_work as unit_of_work:
            return await unit_of_work.materialized_outputs.list_for_graph(
                workspace_id,
                graph_id,
                graph_revision,
            )


class RunResultPresenter:
    """Maps graph execution and materialization results to HTTP response models."""

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    async def run_response(
        self,
        workspace_id: UUID,
        execution: "GraphExecutionResult",
    ) -> RunResponse:
        return RunResponse(
            status=execution.status,
            node_runs=[
                RunNodeResponse(
                    node_id=node_result.node_id,
                    status=node_result.status,
                    error=node_result.error,
                    outputs=[
                        await self.port_output_response(
                            workspace_id,
                            port_name,
                            value,
                        )
                        for port_name, value in node_result.outputs.items()
                    ],
                )
                for node_result in execution.node_results
            ],
        )

    async def execution_response(
        self,
        execution: "RunExecutionSnapshot",
    ) -> RunExecutionResponse:
        result = None
        if execution.result is not None:
            result = await self.run_response(
                execution.workspace_id,
                execution.result,
            )
        return RunExecutionResponse(
            execution_id=execution.execution_id,
            status=execution.status,
            active_node_id=execution.active_node_id,
            result=result,
            error=execution.error,
        )

    async def materializations_response(
        self,
        workspace_id: UUID,
        graph_id: UUID,
        graph_revision: int,
        materializations: Sequence[MaterializedNodeOutputs],
    ) -> GraphMaterializationsResponse:
        node_runs: list[RunNodeResponse] = []
        for materialization in materializations:
            accessible_outputs: list[RunPortOutputResponse] = []
            for port_name, value in materialization.outputs.items():
                if await self._artifacts.is_accessible(workspace_id, value):
                    accessible_outputs.append(
                        await self.port_output_response(
                            workspace_id,
                            port_name,
                            value,
                        )
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
        workspace_id: UUID,
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
            artifacts=[
                await self.artifact_summary(workspace_id, ref) for ref in refs
            ],
        )

    async def artifact_summary(
        self,
        workspace_id: UUID,
        ref: ArtifactRef,
    ) -> ArtifactSummaryResponse:
        artifact = await self._artifacts.get(workspace_id, ref.artifact_id)
        if artifact is None:
            return ArtifactSummaryResponse(
                artifact_id=ref.artifact_id,
                artifact_type=ref.artifact_type,
                schema_version=ref.schema_version,
                content_type="application/octet-stream",
            )
        text: str | None = None
        include_inline_text = (
            artifact.inline_payload is not None
            and artifact.artifact_type != TABLE_DATA.key.id
            and artifact.byte_size is not None
            and artifact.byte_size <= INLINE_SUMMARY_TEXT_BYTE_LIMIT
        )
        if include_inline_text and artifact.inline_payload is not None:
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


__all__ = [
    "ExecutionHistoryService",
    "MaterializationService",
    "RunResultPresenter",
]

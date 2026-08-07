"""Concrete MCP operations backed by application services on the API owner."""

from typing import cast
from uuid import UUID, uuid4

from fastapi import FastAPI
from pydantic import JsonValue, ValidationError

from notarius_core.application.collaboration import CollaborationService
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.collaboration import (
    GRAPH_COMMAND_ADAPTER,
    CollaborativeGraphHead,
    CommandReceiptOutcome,
    GraphCommand,
    GraphCommandReceipt,
    ReplaceDocumentCommand,
)
from notarius_core.domain.errors import (
    CapabilityDeniedError,
    CollaborationCommandRejectedError,
    CollaborationHeadConflictError,
    CollaborationIdempotencyMismatchError,
    CollaborationUncheckpointedError,
    MissingCollaborativeHeadError,
    NotFoundError,
    SavedGraphRevisionConflictError,
    UserDisabledError,
)
from notarius_core.domain.identity import (
    ActorContext,
    WorkspaceCapability,
)
from notarius_mcp.models import (
    CollaborativeHeadResponse,
    CreateSavedGraphRequest,
    NodeRegistryResponse,
    SavedGraphListResponse,
    SavedGraphResponse,
    SubmitGraphCommandResponse,
    UpdateSavedGraphRequest,
)
from notarius_mcp.operations import McpCallerContext, McpOperationError

from notarius_api.mcp.document import document_from_mcp_request
from notarius_api.v1.routes.catalog.models import (
    NodeRegistryResponse as ApiNodeRegistryResponse,
)
from notarius_api.v1.routes.collaboration.hub import GraphRoomHub
from notarius_api.v1.routes.collaboration.models import (
    GraphCommandAcceptedMessage,
    RoomRehydrateMessage,
)
from notarius_api.v1.routes.collaboration.publish import actor_presentation_for
from notarius_api.v1.routes.saved_graphs.models import (
    CollaborativeHeadResponse as ApiCollaborativeHeadResponse,
    GraphCommandReceiptResponse as ApiGraphCommandReceiptResponse,
    SavedGraphListResponse as ApiSavedGraphListResponse,
    SavedGraphResponse as ApiSavedGraphResponse,
)


class ApiGraphWorkspaceOperations:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    async def get_registry(self, caller: McpCallerContext) -> NodeRegistryResponse:
        self._require_scope(caller, WorkspaceCapability.VIEW_GRAPH)
        registry = self._app.state.workbench_plugin_registry
        modules = self._app.state.graph_modules
        module_executor = self._app.state.run_graph
        module_listing = await modules.list(caller.workspace_id)
        api_response = ApiNodeRegistryResponse.from_registry(
            registry,
            module_listing,
            module_executor,
        )
        return NodeRegistryResponse.model_validate(
            api_response.model_dump(mode="json")
        )

    async def list_graphs(self, caller: McpCallerContext) -> SavedGraphListResponse:
        self._require_scope(caller, WorkspaceCapability.VIEW_GRAPH)
        service = cast(SavedGraphService, self._app.state.saved_graphs)
        graphs = await service.list(caller.workspace_id)
        api_response = ApiSavedGraphListResponse.from_graphs(graphs)
        return SavedGraphListResponse.model_validate(
            api_response.model_dump(mode="json")
        )

    async def get_live_head(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
    ) -> CollaborativeHeadResponse:
        self._require_scope(caller, WorkspaceCapability.VIEW_GRAPH)
        collaboration = cast(CollaborationService, self._app.state.collaboration)
        try:
            head = await collaboration.get_head(
                actor=self._actor(caller),
                workspace_id=caller.workspace_id,
                graph_id=graph_id,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc
        api_response = ApiCollaborativeHeadResponse.from_head(head)
        return CollaborativeHeadResponse.model_validate(
            api_response.model_dump(mode="json")
        )

    async def create_graph(
        self,
        caller: McpCallerContext,
        request: CreateSavedGraphRequest,
    ) -> SavedGraphResponse:
        self._require_scope(caller, WorkspaceCapability.CREATE_GRAPH)
        collaboration = cast(CollaborationService, self._app.state.collaboration)
        try:
            graph, _, _ = await collaboration.bootstrap_graph(
                actor=self._actor(caller),
                workspace_id=caller.workspace_id,
                command_id=uuid4(),
                command=ReplaceDocumentCommand(
                    name=request.name,
                    document=document_from_mcp_request(request),
                ),
            )
        except Exception as exc:
            raise self._map_error(exc) from exc
        api_response = ApiSavedGraphResponse.from_graph(graph)
        return SavedGraphResponse.model_validate(api_response.model_dump(mode="json"))

    async def replace_graph(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
        request: UpdateSavedGraphRequest,
    ) -> SavedGraphResponse:
        self._require_scope(caller, WorkspaceCapability.EDIT_GRAPH)
        collaboration = cast(CollaborationService, self._app.state.collaboration)
        try:
            graph, head = await collaboration.replace_complete_document(
                actor=self._actor(caller),
                workspace_id=caller.workspace_id,
                graph_id=graph_id,
                name=request.name,
                document=document_from_mcp_request(request),
                expected_revision=request.expected_revision,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc
        await self._publish_epoch_reset(
            workspace_id=caller.workspace_id,
            graph_id=graph_id,
            head=head,
        )
        api_response = ApiSavedGraphResponse.from_graph(graph)
        return SavedGraphResponse.model_validate(api_response.model_dump(mode="json"))

    async def submit_command(
        self,
        caller: McpCallerContext,
        *,
        graph_id: UUID,
        command_id: UUID,
        room_epoch: UUID,
        observed_sequence: int,
        command: JsonValue,
    ) -> SubmitGraphCommandResponse:
        self._require_scope(caller, WorkspaceCapability.EDIT_GRAPH)
        try:
            parsed_command = GRAPH_COMMAND_ADAPTER.validate_python(command)
        except ValidationError as exc:
            raise McpOperationError(
                status_code=422,
                message="Graph command payload is invalid.",
            ) from exc
        collaboration = cast(CollaborationService, self._app.state.collaboration)
        actor = self._actor(caller)
        try:
            head, receipt = await collaboration.accept_command(
                actor=actor,
                workspace_id=caller.workspace_id,
                graph_id=graph_id,
                command_id=command_id,
                observed_sequence=observed_sequence,
                observed_room_epoch=room_epoch,
                command=parsed_command,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc
        if receipt.outcome is not CommandReceiptOutcome.IDEMPOTENT_REPLAY:
            await self._publish_accepted_command(
                actor=actor,
                workspace_id=caller.workspace_id,
                graph_id=graph_id,
                command=parsed_command,
                receipt=receipt,
            )
        api_head = ApiCollaborativeHeadResponse.from_head(head)
        api_receipt = ApiGraphCommandReceiptResponse.from_receipt(receipt)
        return SubmitGraphCommandResponse.model_validate(
            {
                "head": api_head.model_dump(mode="json"),
                "receipt": api_receipt.model_dump(mode="json"),
            }
        )

    def _require_scope(
        self,
        caller: McpCallerContext,
        capability: WorkspaceCapability,
    ) -> None:
        if capability.value not in caller.scopes:
            raise McpOperationError(
                status_code=403,
                message="Personal access token scope is insufficient.",
            )

    @staticmethod
    def _actor(caller: McpCallerContext) -> ActorContext:
        return ActorContext(
            user_id=caller.user_id,
            credential_reference=caller.credential_reference,
        )

    def _hub(self) -> GraphRoomHub:
        hub = getattr(self._app.state, "graph_room_hub", None)
        if not isinstance(hub, GraphRoomHub):
            raise RuntimeError("Graph room hub is not configured")
        return hub

    async def _publish_accepted_command(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
        command: GraphCommand,
        receipt: GraphCommandReceipt,
    ) -> None:
        presentation = await actor_presentation_for(self._app, actor)
        await self._hub().publish_accepted(
            workspace_id=workspace_id,
            graph_id=graph_id,
            accepted=GraphCommandAcceptedMessage(
                command_id=receipt.command_id,
                room_epoch=receipt.room_epoch,
                sequence=receipt.accepted_sequence,
                actor=presentation,
                graph_room_session_id=None,
                command=command,
            ),
        )

    async def _publish_epoch_reset(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        head: CollaborativeGraphHead,
    ) -> None:
        await self._hub().publish_rehydrate(
            workspace_id=workspace_id,
            graph_id=graph_id,
            message=RoomRehydrateMessage(
                head=ApiCollaborativeHeadResponse.from_head(head),
            ),
        )

    @staticmethod
    def _map_error(exc: Exception) -> McpOperationError:
        if isinstance(exc, UserDisabledError):
            return McpOperationError(status_code=401, message="Authentication required.")
        if isinstance(exc, CapabilityDeniedError):
            return McpOperationError(status_code=403, message="Forbidden.")
        if isinstance(
            exc,
            (
                NotFoundError,
                MissingCollaborativeHeadError,
            ),
        ):
            return McpOperationError(status_code=404, message="Not found.")
        if isinstance(
            exc,
            (
                CollaborationHeadConflictError,
                CollaborationIdempotencyMismatchError,
                CollaborationUncheckpointedError,
                SavedGraphRevisionConflictError,
            ),
        ):
            return McpOperationError(status_code=409, message="Conflict.")
        if isinstance(exc, CollaborationCommandRejectedError):
            return McpOperationError(status_code=422, message="Command rejected.")
        if isinstance(exc, ValidationError):
            return McpOperationError(status_code=422, message="Invalid request.")
        return McpOperationError(status_code=500, message="Operation failed.")


__all__ = ["ApiGraphWorkspaceOperations"]

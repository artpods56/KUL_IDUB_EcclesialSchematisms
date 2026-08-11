from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from notarius_core.domain.identity import (
    ActorContext,
    PAT_ALLOWED_CAPABILITIES,
    PersonalAccessToken,
    WorkspaceCapability,
    WorkspaceMembership,
    WorkspaceRole,
)

from notarius_api.app_state import get_identity
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.auth.models import (
    PersonalAccessTokenCreatedResponse,
    PersonalAccessTokenCreateRequest,
    PersonalAccessTokenResponse,
    UserResponse,
    WorkspaceCreateRequest,
    WorkspaceMemberRequest,
    WorkspaceMemberResponse,
    WorkspaceMemberRoleRequest,
    WorkspaceResponse,
)
from notarius_api.v1.routes.auth.services import AuthService
from notarius_api.v1.routes.collaboration.publish import (
    close_user_rooms_for_permission_change,
)


router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@dataclass(frozen=True)
class WorkspaceFailureRoute:
    operation: str
    resource_type: str | None = None
    resource_path_param: str | None = None


_WORKSPACE_FAILURE_ROUTES: dict[str, WorkspaceFailureRoute] = {
    "list_workspaces": WorkspaceFailureRoute(
        operation="workspace.list",
        resource_type="workspace",
    ),
    "create_workspace": WorkspaceFailureRoute(
        operation="workspace.create",
        resource_type="workspace",
    ),
    "list_members": WorkspaceFailureRoute(
        operation="workspace.membership.list",
        resource_type="workspace_membership",
    ),
    "add_member": WorkspaceFailureRoute(
        operation="workspace.membership.upsert",
        resource_type="user",
    ),
    "change_member_role": WorkspaceFailureRoute(
        operation="workspace.membership.role_change",
        resource_type="user",
        resource_path_param="user_id",
    ),
    "remove_member": WorkspaceFailureRoute(
        operation="workspace.membership.remove",
        resource_type="user",
        resource_path_param="user_id",
    ),
    "list_personal_access_tokens": WorkspaceFailureRoute(
        operation="credential.pat.list",
        resource_type="personal_access_token",
    ),
    "create_personal_access_token": WorkspaceFailureRoute(
        operation="credential.pat.create",
        resource_type="personal_access_token",
    ),
    "revoke_personal_access_token": WorkspaceFailureRoute(
        operation="credential.pat.revoke",
        resource_type="personal_access_token",
        resource_path_param="token_id",
    ),
    "list_graph_folders": WorkspaceFailureRoute(
        operation="graph.folder.list",
        resource_type="graph_folder",
    ),
    "create_graph_folder": WorkspaceFailureRoute(
        operation="graph.folder.create",
        resource_type="graph_folder",
    ),
    "rename_graph_folder": WorkspaceFailureRoute(
        operation="graph.folder.rename",
        resource_type="graph_folder",
        resource_path_param="folder_id",
    ),
    "delete_graph_folder": WorkspaceFailureRoute(
        operation="graph.folder.delete",
        resource_type="graph_folder",
        resource_path_param="folder_id",
    ),
    "assign_graph_folder": WorkspaceFailureRoute(
        operation="graph.folder.assign",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
    "archive_graph": WorkspaceFailureRoute(
        operation="graph.archive",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
    "restore_graph": WorkspaceFailureRoute(
        operation="graph.restore",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
    "star_graph": WorkspaceFailureRoute(
        operation="graph.star",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
    "unstar_graph": WorkspaceFailureRoute(
        operation="graph.unstar",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
    "record_graph_open": WorkspaceFailureRoute(
        operation="graph.open.record",
        resource_type="saved_graph",
        resource_path_param="graph_id",
    ),
}


def workspace_failure_metadata(
    request: Request,
) -> tuple[str, UUID | None, str | None, str | None] | None:
    route = request.scope.get("route")
    route_name = getattr(route, "name", None)
    if not isinstance(route_name, str):
        return None
    failure_route = _WORKSPACE_FAILURE_ROUTES.get(route_name)
    if failure_route is None:
        return None

    workspace_id: UUID | None = None
    raw_workspace_id = request.path_params.get("workspace_id")
    if isinstance(raw_workspace_id, str):
        try:
            workspace_id = UUID(raw_workspace_id)
        except ValueError:
            workspace_id = None

    resource_id: str | None = None
    if failure_route.resource_path_param is not None:
        raw_resource_id = request.path_params.get(failure_route.resource_path_param)
        if isinstance(raw_resource_id, str):
            try:
                resource_id = str(UUID(raw_resource_id))
            except ValueError:
                resource_id = None
    return (
        failure_route.operation,
        workspace_id,
        failure_route.resource_type,
        resource_id,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> list[WorkspaceResponse]:
    rows = await get_identity(request.app).identity_service.list_workspaces(actor=actor)
    return [
        WorkspaceResponse(
            id=workspace.id,
            slug=workspace.slug,
            name=workspace.name,
            kind=workspace.kind,
            role=membership.role,
            capabilities=tuple(
                sorted(membership.capabilities, key=lambda item: item.value)
            ),
        )
        for workspace, membership in rows
    ]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> WorkspaceResponse:
    workspace = await get_identity(request.app).identity_service.create_shared_workspace(
        actor=actor,
        slug=payload.slug,
        name=payload.name,
    )
    return WorkspaceResponse(
        id=workspace.id,
        slug=workspace.slug,
        name=workspace.name,
        kind=workspace.kind,
        role=WorkspaceRole.OWNER,
        capabilities=tuple(WorkspaceCapability),
    )


@router.get("/{workspace_id}/members", response_model=list[WorkspaceMemberResponse])
async def list_members(
    workspace_id: UUID,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> list[WorkspaceMemberResponse]:
    rows = await get_identity(request.app).identity_service.list_members(
        actor=actor,
        workspace_id=workspace_id,
    )
    return [
        WorkspaceMemberResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                display_name=user.display_name,
                active=user.active,
            ),
            role=membership.role,
            authorization_version=membership.authorization_version,
            revoked_at=membership.revoked_at,
        )
        for user, membership in rows
    ]


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberResponse)
async def add_member(
    workspace_id: UUID,
    payload: WorkspaceMemberRequest,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> WorkspaceMemberResponse:
    membership = await get_identity(request.app).identity_service.add_or_reactivate_member(
        actor=actor,
        workspace_id=workspace_id,
        user_id=payload.user_id,
        role=payload.role,
    )
    await close_user_rooms_for_permission_change(
        request,
        workspace_id=workspace_id,
        user_id=payload.user_id,
        access_revoked=False,
    )
    return await _member_response(request, membership.user_id, membership)


@router.patch(
    "/{workspace_id}/members/{user_id}", response_model=WorkspaceMemberResponse
)
async def change_member_role(
    workspace_id: UUID,
    user_id: UUID,
    payload: WorkspaceMemberRoleRequest,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> WorkspaceMemberResponse:
    membership = await get_identity(request.app).identity_service.change_member_role(
        actor=actor,
        workspace_id=workspace_id,
        user_id=user_id,
        role=payload.role,
    )
    await close_user_rooms_for_permission_change(
        request,
        workspace_id=workspace_id,
        user_id=user_id,
        access_revoked=False,
    )
    return await _member_response(request, user_id, membership)


@router.delete("/{workspace_id}/members/{user_id}", status_code=204)
async def remove_member(
    workspace_id: UUID,
    user_id: UUID,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> Response:
    await get_identity(request.app).identity_service.remove_member(
        actor=actor,
        workspace_id=workspace_id,
        user_id=user_id,
    )
    await close_user_rooms_for_permission_change(
        request,
        workspace_id=workspace_id,
        user_id=user_id,
        access_revoked=True,
    )
    return Response(status_code=204)


@router.get(
    "/{workspace_id}/personal-access-tokens",
    response_model=list[PersonalAccessTokenResponse],
)
async def list_personal_access_tokens(
    workspace_id: UUID,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> list[PersonalAccessTokenResponse]:
    tokens = await get_identity(request.app).identity_service.list_personal_access_tokens(
        actor=actor,
        workspace_id=workspace_id,
    )
    return [_pat_response(token) for token in tokens]


@router.post(
    "/{workspace_id}/personal-access-tokens",
    response_model=PersonalAccessTokenCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_personal_access_token(
    workspace_id: UUID,
    payload: PersonalAccessTokenCreateRequest,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> JSONResponse:
    now = datetime.now(UTC)
    if payload.expires_at.tzinfo is None or payload.expires_at <= now:
        raise HTTPException(status_code=422, detail="PAT expiry must be in the future")
    maximum_expiry = now + timedelta(
        seconds=request.app.state.settings.personal_access_token_max_lifetime_seconds
    )
    if payload.expires_at > maximum_expiry:
        raise HTTPException(
            status_code=422, detail="PAT expiry exceeds configured lifetime"
        )
    scopes = tuple(WorkspaceCapability(scope.value) for scope in payload.scopes)
    if not set(scopes).issubset(PAT_ALLOWED_CAPABILITIES):
        raise HTTPException(
            status_code=422,
            detail="Personal access token scope is not available",
        )
    auth: AuthService = get_identity(request.app).auth_service
    if not await auth.allow_pat_creation(str(actor.user_id)):
        raise HTTPException(status_code=429, detail="Too many token creation attempts")
    token, raw_token = auth.issue_personal_access_token(
        user_id=actor.user_id,
        workspace_id=workspace_id,
        label=payload.label,
        scopes=scopes,
        expires_at=payload.expires_at,
    )
    created = await get_identity(request.app).identity_service.create_personal_access_token(
        actor=actor,
        token=token,
    )
    response = PersonalAccessTokenCreatedResponse(
        **_pat_response(created).model_dump(),
        token=SecretStr(raw_token),
    )
    # FastAPI's response-model serialization is intentionally bypassed here:
    # this is the sole opt-in boundary that may deliver the raw PAT once.
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            **response.model_dump(mode="json"),
            "token": response.token.get_secret_value(),
        },
    )


@router.delete(
    "/{workspace_id}/personal-access-tokens/{token_id}",
    status_code=204,
)
async def revoke_personal_access_token(
    workspace_id: UUID,
    token_id: UUID,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> Response:
    await get_identity(request.app).identity_service.revoke_personal_access_token(
        actor=actor,
        workspace_id=workspace_id,
        token_id=token_id,
    )
    return Response(status_code=204)


async def _member_response(
    request: Request,
    user_id: UUID,
    membership: WorkspaceMembership,
) -> WorkspaceMemberResponse:
    async with get_identity(request.app).identity_uow_factory() as unit_of_work:
        user = await unit_of_work.identity.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return WorkspaceMemberResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            active=user.active,
        ),
        role=membership.role,
        authorization_version=membership.authorization_version,
        revoked_at=membership.revoked_at,
    )


def _pat_response(token: PersonalAccessToken) -> PersonalAccessTokenResponse:
    return PersonalAccessTokenResponse(
        id=token.id,
        public_prefix=token.public_prefix,
        workspace_id=token.workspace_id,
        label=token.label,
        scopes=token.scopes,
        created_at=token.created_at,
        last_used_at=token.last_used_at,
        expires_at=token.expires_at,
        revoked_at=token.revoked_at,
    )


__all__ = ["router"]

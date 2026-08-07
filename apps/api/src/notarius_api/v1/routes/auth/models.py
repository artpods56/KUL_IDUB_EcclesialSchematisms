from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from notarius_core.domain.identity import WorkspaceCapability, WorkspaceKind, WorkspaceRole


class SessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime
    revoked_at: datetime | None
    current: bool


class WorkspaceResponse(BaseModel):
    id: UUID
    slug: str
    name: str
    kind: WorkspaceKind
    role: WorkspaceRole
    capabilities: tuple[WorkspaceCapability, ...]


class UserResponse(BaseModel):
    id: UUID
    email: str | None
    display_name: str | None
    active: bool


class WorkspaceMemberResponse(BaseModel):
    user: UserResponse
    role: WorkspaceRole
    authorization_version: int
    revoked_at: datetime | None


class WorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)


class WorkspaceMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    role: WorkspaceRole


class WorkspaceMemberRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: WorkspaceRole


class PersonalAccessTokenCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=160)
    scopes: tuple[WorkspaceCapability, ...] = Field(min_length=1, max_length=8)
    expires_at: datetime


class PersonalAccessTokenResponse(BaseModel):
    id: UUID
    public_prefix: str
    workspace_id: UUID
    label: str
    scopes: tuple[WorkspaceCapability, ...]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime
    revoked_at: datetime | None


class PersonalAccessTokenCreatedResponse(PersonalAccessTokenResponse):
    token: str

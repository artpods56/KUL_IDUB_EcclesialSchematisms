from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from notarius_core.domain.identity import (
    WorkspaceCapability,
    WorkspaceKind,
    WorkspaceRole,
    normalize_workspace_slug,
)


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

    @field_validator("slug")
    @classmethod
    def normalize_slug(cls, value: str) -> str:
        return normalize_workspace_slug(value)

    @field_validator("name")
    @classmethod
    def reject_whitespace_only_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must contain a non-whitespace character")
        return value


class WorkspaceMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    role: WorkspaceRole


class WorkspaceMemberRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: WorkspaceRole


class PersonalAccessTokenScope(StrEnum):
    VIEW_GRAPH = "view_graph"
    VIEW_ARTIFACTS = "view_artifacts"
    VIEW_MATERIALIZATIONS = "view_materializations"
    VIEW_HISTORY = "view_history"
    VIEW_EXECUTION = "view_execution"
    CREATE_GRAPH = "create_graph"
    EDIT_GRAPH = "edit_graph"
    CHECKPOINT_GRAPH = "checkpoint_graph"
    EXECUTE_GRAPH = "execute_graph"
    CANCEL_EXECUTION = "cancel_execution"


class PersonalAccessTokenCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=160)
    scopes: tuple[PersonalAccessTokenScope, ...] = Field(min_length=1, max_length=8)
    expires_at: datetime

    @field_validator("label")
    @classmethod
    def reject_whitespace_only_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label must contain a non-whitespace character")
        return value

    @field_validator("scopes")
    @classmethod
    def reject_duplicate_scopes(
        cls,
        value: tuple[PersonalAccessTokenScope, ...],
    ) -> tuple[PersonalAccessTokenScope, ...]:
        if len(value) != len(set(value)):
            raise ValueError("PAT scopes must be unique")
        return value


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
    token: SecretStr = Field(
        description="Raw personal access token; returned once and never retrievable again.",
    )

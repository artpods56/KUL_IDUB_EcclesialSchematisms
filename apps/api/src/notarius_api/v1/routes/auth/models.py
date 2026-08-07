from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.main import IncEx

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


class PersonalAccessTokenCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=160)
    scopes: tuple[WorkspaceCapability, ...] = Field(min_length=1, max_length=8)
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
        value: tuple[WorkspaceCapability, ...],
    ) -> tuple[WorkspaceCapability, ...]:
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
    token: str = Field(
        repr=False,
        description="Raw personal access token; returned once and never retrievable again.",
    )

    def model_dump(
        self, *args: object, include_sensitive: bool = False, **kwargs: object
    ) -> dict[str, object]:
        dumped = super().model_dump(*args, **kwargs)
        if not include_sensitive:
            dumped.pop("token", None)
        return dumped

    def model_dump_json(
        self,
        *args: object,
        include_sensitive: bool = False,
        **kwargs: object,
    ) -> str:
        if include_sensitive:
            return super().model_dump_json(*args, **kwargs)
        caller_exclude = kwargs.pop("exclude", None)
        if isinstance(caller_exclude, set):
            merged_exclude_set = set(cast(set[str], caller_exclude))
            merged_exclude_set.add("token")
            return super().model_dump_json(*args, exclude=merged_exclude_set, **kwargs)
        if isinstance(caller_exclude, dict):
            merged_exclude: IncEx = dict(cast(dict[str, bool | IncEx], caller_exclude))
            merged_exclude["token"] = True
        else:
            merged_exclude = {"token"}
        return super().model_dump_json(*args, exclude=merged_exclude, **kwargs)

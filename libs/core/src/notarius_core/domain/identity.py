from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import re
from uuid import UUID, uuid4

from notarius_core.domain.errors import (
    BootstrapOwnerMismatchError,
    CapabilityDeniedError,
    IdentityInvariantError,
    LastWorkspaceOwnerError,
)


LOCAL_WORKSPACE_SLUG = "local"
PERSONAL_WORKSPACE_SLUG_PREFIX = "personal-"


class WorkspaceKind(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"


class WorkspaceRole(StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    OWNER = "owner"


class WorkspaceCapability(StrEnum):
    VIEW_GRAPH = "view_graph"
    VIEW_ARTIFACTS = "view_artifacts"
    VIEW_MATERIALIZATIONS = "view_materializations"
    VIEW_HISTORY = "view_history"
    VIEW_EXECUTION = "view_execution"
    JOIN_GRAPH_ROOM = "join_graph_room"
    PUBLISH_PRESENCE = "publish_presence"
    CREATE_GRAPH = "create_graph"
    EDIT_GRAPH = "edit_graph"
    CHECKPOINT_GRAPH = "checkpoint_graph"
    EXECUTE_GRAPH = "execute_graph"
    CANCEL_EXECUTION = "cancel_execution"
    MANAGE_SECRETS = "manage_secrets"
    DELETE_GRAPH = "delete_graph"
    MANAGE_MEMBERS = "manage_members"
    RENAME_WORKSPACE = "rename_workspace"


_SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?$")
_ROLE_CAPABILITIES: dict[WorkspaceRole, frozenset[WorkspaceCapability]] = {
    WorkspaceRole.VIEWER: frozenset(
        {
            WorkspaceCapability.VIEW_GRAPH,
            WorkspaceCapability.VIEW_ARTIFACTS,
            WorkspaceCapability.VIEW_MATERIALIZATIONS,
            WorkspaceCapability.VIEW_HISTORY,
            WorkspaceCapability.VIEW_EXECUTION,
            WorkspaceCapability.JOIN_GRAPH_ROOM,
            WorkspaceCapability.PUBLISH_PRESENCE,
        }
    ),
    WorkspaceRole.EDITOR: frozenset(
        {
            WorkspaceCapability.VIEW_GRAPH,
            WorkspaceCapability.VIEW_ARTIFACTS,
            WorkspaceCapability.VIEW_MATERIALIZATIONS,
            WorkspaceCapability.VIEW_HISTORY,
            WorkspaceCapability.VIEW_EXECUTION,
            WorkspaceCapability.JOIN_GRAPH_ROOM,
            WorkspaceCapability.PUBLISH_PRESENCE,
            WorkspaceCapability.CREATE_GRAPH,
            WorkspaceCapability.EDIT_GRAPH,
            WorkspaceCapability.CHECKPOINT_GRAPH,
            WorkspaceCapability.EXECUTE_GRAPH,
            WorkspaceCapability.CANCEL_EXECUTION,
        }
    ),
    WorkspaceRole.OWNER: frozenset(WorkspaceCapability),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")


def _require_nonempty(value: str, label: str, maximum: int) -> str:
    if value == "" or value.strip() == "":
        raise ValueError(f"{label} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{label} must be at most {maximum} characters")
    return value


def normalize_workspace_slug(value: str) -> str:
    slug = value.strip().lower()
    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            "Workspace slug must contain 1-80 lowercase letters, numbers, or "
            "internal hyphens"
        )
    return slug


def personal_workspace_slug(user_id: UUID) -> str:
    return f"{PERSONAL_WORKSPACE_SLUG_PREFIX}{user_id.hex}"


def capabilities_for_role(role: WorkspaceRole) -> frozenset[WorkspaceCapability]:
    return _ROLE_CAPABILITIES[role]


@dataclass
class User:
    id: UUID = field(default_factory=uuid4)
    email: str | None = None
    display_name: str | None = None
    active: bool = True
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.email is not None:
            self.email = _require_nonempty(self.email.strip(), "User email", 320)
        if self.display_name is not None:
            self.display_name = _require_nonempty(
                self.display_name.strip(), "User display name", 160
            )
        _require_aware(self.created_at, "User creation timestamp")
        _require_aware(self.updated_at, "User update timestamp")

    @property
    def is_active(self) -> bool:
        return self.active

    def update_profile(
        self,
        *,
        email: str | None,
        display_name: str | None,
        updated_at: datetime | None = None,
    ) -> None:
        self.email = (
            None
            if email is None
            else _require_nonempty(email.strip(), "User email", 320)
        )
        self.display_name = (
            None
            if display_name is None
            else _require_nonempty(display_name.strip(), "User display name", 160)
        )
        self.updated_at = updated_at or _utc_now()
        _require_aware(self.updated_at, "User update timestamp")


@dataclass
class OidcIdentity:
    user_id: UUID
    issuer: str
    subject: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        _require_nonempty(self.issuer, "OIDC issuer", 2048)
        _require_nonempty(self.subject, "OIDC subject", 512)
        _require_aware(self.created_at, "OIDC identity creation timestamp")
        _require_aware(self.updated_at, "OIDC identity update timestamp")


@dataclass
class Workspace:
    slug: str
    name: str
    kind: WorkspaceKind
    id: UUID = field(default_factory=uuid4)
    personal_owner_user_id: UUID | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.kind = WorkspaceKind(self.kind)
        self.slug = normalize_workspace_slug(self.slug)
        self.name = _require_nonempty(self.name.strip(), "Workspace name", 160)
        if self.kind is WorkspaceKind.PERSONAL and self.personal_owner_user_id is None:
            raise IdentityInvariantError(
                "Personal workspace must have a personal owner"
            )
        if self.kind is WorkspaceKind.SHARED and self.personal_owner_user_id is not None:
            raise IdentityInvariantError(
                "Shared workspace cannot have a personal owner"
            )
        _require_aware(self.created_at, "Workspace creation timestamp")
        _require_aware(self.updated_at, "Workspace update timestamp")

    @classmethod
    def personal(cls, *, owner_user_id: UUID, name: str = "Personal workspace") -> "Workspace":
        return cls(
            slug=personal_workspace_slug(owner_user_id),
            name=name,
            kind=WorkspaceKind.PERSONAL,
            personal_owner_user_id=owner_user_id,
        )

    @classmethod
    def shared(cls, *, slug: str, name: str) -> "Workspace":
        return cls(slug=slug, name=name, kind=WorkspaceKind.SHARED)

    @property
    def is_sealed_bootstrap_workspace(self) -> bool:
        return (
            self.kind is WorkspaceKind.SHARED
            and self.slug == LOCAL_WORKSPACE_SLUG
        )


@dataclass
class WorkspaceMembership:
    workspace_id: UUID
    user_id: UUID
    role: WorkspaceRole
    authorization_version: int = 1
    revoked_at: datetime | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.role = WorkspaceRole(self.role)
        if self.authorization_version < 1:
            raise ValueError("Membership authorization version must be positive")
        _require_aware(self.created_at, "Membership creation timestamp")
        _require_aware(self.updated_at, "Membership update timestamp")
        if self.revoked_at is not None:
            _require_aware(self.revoked_at, "Membership revocation timestamp")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    @property
    def capabilities(self) -> frozenset[WorkspaceCapability]:
        if not self.is_active:
            return frozenset()
        return capabilities_for_role(self.role)

    def grants(self, capability: WorkspaceCapability) -> bool:
        return capability in self.capabilities

    def change_role(
        self,
        role: WorkspaceRole,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        if self.role is not role:
            self.role = role
            self.authorization_version += 1
        self.updated_at = updated_at or _utc_now()

    def revoke(self, *, revoked_at: datetime | None = None) -> None:
        if self.revoked_at is None:
            self.revoked_at = revoked_at or _utc_now()
            self.authorization_version += 1
        self.updated_at = revoked_at or _utc_now()

    def reactivate(
        self,
        *,
        role: WorkspaceRole,
        updated_at: datetime | None = None,
    ) -> None:
        self.role = role
        self.revoked_at = None
        self.authorization_version += 1
        self.updated_at = updated_at or _utc_now()


@dataclass
class OidcLoginTransaction:
    state_digest: bytes = field(repr=False)
    nonce_digest: bytes = field(repr=False)
    encrypted_pkce_verifier: bytes = field(repr=False)
    pkce_key_version: int
    return_path: str
    expires_at: datetime
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    consumed_at: datetime | None = None

    def __post_init__(self) -> None:
        if len(self.state_digest) == 0 or len(self.nonce_digest) == 0:
            raise ValueError("OIDC transaction digests must not be empty")
        if len(self.encrypted_pkce_verifier) == 0:
            raise ValueError("OIDC transaction verifier must not be empty")
        if self.pkce_key_version < 1:
            raise ValueError("OIDC transaction PKCE key version must be positive")
        _require_nonempty(self.return_path, "OIDC transaction return path", 2048)
        _require_aware(self.created_at, "OIDC transaction creation timestamp")
        _require_aware(self.expires_at, "OIDC transaction expiry timestamp")
        if self.consumed_at is not None:
            _require_aware(self.consumed_at, "OIDC transaction consumption timestamp")

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at is not None

    def consume(self, *, consumed_at: datetime | None = None) -> None:
        if self.is_consumed:
            raise IdentityInvariantError("OIDC login transaction was already consumed")
        self.consumed_at = consumed_at or _utc_now()


@dataclass
class OidcBootstrapOwnerMapping:
    workspace_id: UUID
    issuer: str
    subject: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    consumed_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.issuer, "Bootstrap OIDC issuer", 2048)
        _require_nonempty(self.subject, "Bootstrap OIDC subject", 512)
        _require_aware(self.created_at, "Bootstrap mapping creation timestamp")
        if self.consumed_at is not None:
            _require_aware(self.consumed_at, "Bootstrap mapping consumption timestamp")

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at is not None

    def consume(self, *, consumed_at: datetime | None = None) -> None:
        if self.is_consumed:
            raise IdentityInvariantError("Bootstrap owner mapping was already consumed")
        self.consumed_at = consumed_at or _utc_now()


@dataclass
class AuthSession:
    user_id: UUID
    secret_digest: bytes = field(repr=False)
    csrf_digest: bytes = field(repr=False)
    expires_at: datetime
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if len(self.secret_digest) == 0 or len(self.csrf_digest) == 0:
            raise ValueError("Auth session digests must not be empty")
        _require_aware(self.created_at, "Auth session creation timestamp")
        _require_aware(self.expires_at, "Auth session expiry timestamp")
        if self.last_used_at is not None:
            _require_aware(self.last_used_at, "Auth session last-used timestamp")
        if self.revoked_at is not None:
            _require_aware(self.revoked_at, "Auth session revocation timestamp")

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def revoke(self, *, revoked_at: datetime | None = None) -> None:
        self.revoked_at = revoked_at or _utc_now()


@dataclass
class PersonalAccessToken:
    user_id: UUID
    workspace_id: UUID
    public_prefix: str
    secret_digest: bytes = field(repr=False)
    label: str
    scopes: tuple[WorkspaceCapability, ...]
    expires_at: datetime
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        self.scopes = tuple(WorkspaceCapability(scope) for scope in self.scopes)
        _require_nonempty(self.public_prefix, "Personal access token prefix", 32)
        if len(self.secret_digest) == 0:
            raise ValueError("Personal access token digest must not be empty")
        _require_nonempty(self.label.strip(), "Personal access token label", 160)
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("Personal access token scopes must be unique")
        for scope in self.scopes:
            _require_nonempty(scope, "Personal access token scope", 64)
        _require_aware(self.created_at, "Personal access token creation timestamp")
        _require_aware(self.expires_at, "Personal access token expiry timestamp")
        if self.last_used_at is not None:
            _require_aware(self.last_used_at, "Personal access token last-used timestamp")
        if self.revoked_at is not None:
            _require_aware(self.revoked_at, "Personal access token revocation timestamp")

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def revoke(self, *, revoked_at: datetime | None = None) -> None:
        self.revoked_at = revoked_at or _utc_now()


@dataclass(frozen=True, slots=True)
class ActorContext:
    user_id: UUID
    credential_reference: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceAccess:
    actor: ActorContext
    workspace_id: UUID
    membership: WorkspaceMembership

    @property
    def capabilities(self) -> frozenset[WorkspaceCapability]:
        return self.membership.capabilities

    def require(self, capability: WorkspaceCapability) -> None:
        if not self.membership.grants(capability):
            raise CapabilityDeniedError(
                capability=capability.value,
                workspace_id=self.workspace_id,
                user_id=self.actor.user_id,
            )


@dataclass(frozen=True, slots=True)
class IdentityProvisioningResult:
    user: User
    oidc_identity: OidcIdentity
    personal_workspace: Workspace
    local_workspace_membership: WorkspaceMembership | None


def ensure_last_owner_can_change(
    *,
    workspace: Workspace,
    memberships: tuple[WorkspaceMembership, ...] | list[WorkspaceMembership],
    target: WorkspaceMembership,
    replacement_role: WorkspaceRole | None = None,
    removing: bool = False,
) -> None:
    if workspace.kind is WorkspaceKind.PERSONAL:
        if target.user_id != workspace.personal_owner_user_id:
            raise IdentityInvariantError(
                "Personal workspace cannot accept another membership"
            )
        if removing or replacement_role is not WorkspaceRole.OWNER:
            raise IdentityInvariantError(
                "Personal workspace owner membership cannot be removed or changed"
            )
        return

    if not target.is_active or target.role is not WorkspaceRole.OWNER:
        return
    becomes_owner = replacement_role is WorkspaceRole.OWNER and not removing
    active_owner_count = sum(
        membership.is_active and membership.role is WorkspaceRole.OWNER
        for membership in memberships
    )
    if active_owner_count <= 1 and not becomes_owner:
        raise LastWorkspaceOwnerError(
            f"Workspace {workspace.id} must retain an active owner"
        )


def validate_bootstrap_match(
    mapping: OidcBootstrapOwnerMapping,
    *,
    issuer: str,
    subject: str,
) -> None:
    if mapping.issuer != issuer or mapping.subject != subject:
        raise BootstrapOwnerMismatchError(
            "OIDC identity does not match the configured bootstrap owner"
        )


__all__ = [
    "ActorContext",
    "AuthSession",
    "IdentityProvisioningResult",
    "LOCAL_WORKSPACE_SLUG",
    "OidcBootstrapOwnerMapping",
    "OidcIdentity",
    "OidcLoginTransaction",
    "PersonalAccessToken",
    "User",
    "Workspace",
    "WorkspaceAccess",
    "WorkspaceCapability",
    "WorkspaceKind",
    "WorkspaceMembership",
    "WorkspaceRole",
    "capabilities_for_role",
    "ensure_last_owner_can_change",
    "normalize_workspace_slug",
    "personal_workspace_slug",
    "validate_bootstrap_match",
]

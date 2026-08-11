"""Domain types for Notarius core."""

from notarius_core.domain.modules import (
    GraphModuleDefinition,
    GraphModuleDefinitionError,
    GraphModulePort,
    GraphModuleReference,
    GraphModuleReferenceError,
    ModuleBoundaryConfig,
)
from notarius_core.domain.identity import (
    ActorContext,
    AuthSession,
    OidcBootstrapOwnerMapping,
    OidcIdentity,
    OidcLoginTransaction,
    PersonalAccessToken,
    User,
    Workspace,
    WorkspaceAccess,
    WorkspaceCapability,
    WorkspaceKind,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_core.domain.security_audit import (
    SecurityAuditActorKind,
    SecurityAuditEvent,
    SecurityAuditOutcome,
)
from notarius_core.domain.templates import Template, TemplateLibraryError, TemplateState

__all__ = [
    "GraphModuleDefinition",
    "GraphModuleDefinitionError",
    "GraphModulePort",
    "GraphModuleReference",
    "GraphModuleReferenceError",
    "ModuleBoundaryConfig",
    "ActorContext",
    "AuthSession",
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
    "SecurityAuditActorKind",
    "SecurityAuditEvent",
    "SecurityAuditOutcome",
    "Template",
    "TemplateLibraryError",
    "TemplateState",
]

"""Plugin release scope and deployment-owned execution metadata."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class PluginReleaseScope(StrEnum):
    SYSTEM = "system"
    WORKSPACE = "workspace"


class PluginExecutionPolicy(StrEnum):
    HOST_ELIGIBLE = "host-eligible"
    ISOLATED_ONLY = "isolated-only"


class PluginDistribution(StrEnum):
    BUNDLED = "bundled"
    OPTIONAL = "optional"
    PUBLISHED = "published"


@dataclass(frozen=True, slots=True)
class PlatformPluginActor:
    """Non-secret deployment identity authorized to manage System Plugins."""

    reference: str

    def __post_init__(self) -> None:
        normalized = self.reference.strip()
        if normalized == "" or len(normalized) > 255:
            raise ValueError(
                "Platform Plugin actor reference must contain 1 to 255 characters"
            )
        object.__setattr__(self, "reference", normalized)


@dataclass(frozen=True, slots=True)
class PluginReleaseNamespace:
    """Visibility and ownership boundary for one Plugin release family."""

    scope: PluginReleaseScope
    workspace_id: UUID | None

    def __post_init__(self) -> None:
        if self.scope is PluginReleaseScope.SYSTEM and self.workspace_id is not None:
            raise ValueError(
                "System Plugin release namespaces cannot have a Workspace owner"
            )
        if self.scope is PluginReleaseScope.WORKSPACE and self.workspace_id is None:
            raise ValueError(
                "Workspace Plugin release namespaces require a Workspace owner"
            )

    @property
    def storage_path(self) -> str:
        if self.scope is PluginReleaseScope.SYSTEM:
            return "system"
        if self.workspace_id is None:
            raise ValueError(
                "Workspace Plugin release namespace has no Workspace owner"
            )
        return f"workspaces/{self.workspace_id}"


__all__ = [
    "PlatformPluginActor",
    "PluginDistribution",
    "PluginExecutionPolicy",
    "PluginReleaseNamespace",
    "PluginReleaseScope",
]

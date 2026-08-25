"""Exact Plugin release revocation facts kept beside immutable releases."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from grafy_core.domain.plugin_identity import (
    PlatformPluginActor,
    PluginReleaseNamespace,
    PluginReleaseScope,
)
from grafy_core.domain.plugin_releases import PluginRelease


class PluginReleaseRevocationReason(StrEnum):
    """Non-sensitive categories safe to retain in diagnostics and audit data."""

    SECURITY = "security"
    INTEGRITY = "integrity"
    POLICY = "policy"
    OPERATIONAL = "operational"


class PluginReleaseRevocationError(ValueError):
    """An exact release revocation would violate its immutable identity."""


@dataclass
class PluginReleaseRevocation:
    """Permanent execution denial for one exact, still-retained release."""

    release_id: UUID
    scope: PluginReleaseScope
    workspace_id: UUID | None
    slug: str
    revision: int
    reason: PluginReleaseRevocationReason
    revoked_by_user_id: UUID | None = None
    revoked_by_platform_actor: str | None = None
    revoked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.scope = PluginReleaseScope(self.scope)
        self.reason = PluginReleaseRevocationReason(self.reason)
        try:
            PluginReleaseNamespace(
                scope=self.scope,
                workspace_id=self.workspace_id,
            )
        except ValueError as exc:
            raise PluginReleaseRevocationError(str(exc)) from exc
        normalized_slug = self.slug.strip()
        if normalized_slug == "" or len(normalized_slug) > 100:
            raise PluginReleaseRevocationError(
                "Plugin release revocation slug must contain 1 to 100 characters"
            )
        self.slug = normalized_slug
        if isinstance(self.revision, bool) or self.revision < 1:
            raise PluginReleaseRevocationError(
                "Plugin release revocation revision must be a positive integer"
            )
        if self.revoked_at.tzinfo is None:
            raise PluginReleaseRevocationError(
                "Plugin release revocation timestamp must be timezone-aware"
            )
        if self.scope is PluginReleaseScope.SYSTEM:
            if self.revoked_by_user_id is not None:
                raise PluginReleaseRevocationError(
                    "System Plugin releases cannot be revoked by a Workspace user"
                )
            if self.revoked_by_platform_actor is None:
                raise PluginReleaseRevocationError(
                    "System Plugin release revocation requires a platform actor"
                )
            try:
                actor = PlatformPluginActor(self.revoked_by_platform_actor)
            except ValueError as exc:
                raise PluginReleaseRevocationError(str(exc)) from exc
            self.revoked_by_platform_actor = actor.reference
        elif self.revoked_by_user_id is None:
            raise PluginReleaseRevocationError(
                "Workspace Plugin release revocation requires a Workspace user"
            )
        elif self.revoked_by_platform_actor is not None:
            raise PluginReleaseRevocationError(
                "Workspace Plugin releases cannot be revoked by a platform actor"
            )

    @classmethod
    def from_release(
        cls,
        release: PluginRelease,
        *,
        reason: PluginReleaseRevocationReason,
        revoked_by_user_id: UUID | None = None,
        revoked_by_platform_actor: str | None = None,
        revoked_at: datetime | None = None,
    ) -> Self:
        return cls(
            release_id=release.id,
            scope=release.scope,
            workspace_id=release.workspace_id,
            slug=release.slug,
            revision=release.revision,
            reason=reason,
            revoked_by_user_id=revoked_by_user_id,
            revoked_by_platform_actor=revoked_by_platform_actor,
            revoked_at=revoked_at or datetime.now(UTC),
        )

    @property
    def namespace(self) -> PluginReleaseNamespace:
        return PluginReleaseNamespace(
            scope=self.scope,
            workspace_id=self.workspace_id,
        )

    def has_same_intent(self, other: Self) -> bool:
        """Compare immutable request intent while ignoring creation time."""

        return (
            self.release_id == other.release_id
            and self.scope is other.scope
            and self.workspace_id == other.workspace_id
            and self.slug == other.slug
            and self.revision == other.revision
            and self.reason is other.reason
            and self.revoked_by_user_id == other.revoked_by_user_id
            and self.revoked_by_platform_actor == other.revoked_by_platform_actor
        )


__all__ = [
    "PluginReleaseRevocation",
    "PluginReleaseRevocationError",
    "PluginReleaseRevocationReason",
]

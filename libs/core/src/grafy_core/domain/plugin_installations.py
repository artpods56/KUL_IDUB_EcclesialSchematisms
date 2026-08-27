"""Scoped Plugin installations and their resolved releases."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.domain.plugin_identity import (
    PlatformPluginActor,
    PluginDistribution,
    PluginExecutionPolicy,
    PluginReleaseNamespace,
    PluginReleaseScope,
)
from grafy_core.domain.plugin_releases import (
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginRelease,
    PluginReleaseDescriptor,
    PluginReleaseError,
    PluginRuntimeArtifact,
)


@dataclass
class PluginInstallation:
    """Append-only assignment of one release to one visibility namespace."""

    release_id: UUID
    scope: PluginReleaseScope
    workspace_id: UUID | None
    slug: str
    release_revision: int
    execution_policy: PluginExecutionPolicy
    distribution: PluginDistribution | None
    installed_by_user_id: UUID | None = None
    installed_by_platform_actor: str | None = None
    installed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        self.scope = PluginReleaseScope(self.scope)
        self.execution_policy = PluginExecutionPolicy(self.execution_policy)
        if self.distribution is not None:
            self.distribution = PluginDistribution(self.distribution)
        try:
            PluginReleaseNamespace(
                scope=self.scope,
                workspace_id=self.workspace_id,
            )
        except ValueError as exc:
            raise PluginReleaseError(str(exc)) from exc
        if self.scope is PluginReleaseScope.SYSTEM:
            if self.installed_by_user_id is not None:
                raise PluginReleaseError(
                    "System Plugin installations cannot use a Workspace user actor"
                )
            if self.installed_by_platform_actor is None:
                raise PluginReleaseError(
                    "System Plugin installations require a platform actor"
                )
            try:
                actor = PlatformPluginActor(self.installed_by_platform_actor)
            except ValueError as exc:
                raise PluginReleaseError(str(exc)) from exc
            self.installed_by_platform_actor = actor.reference
        else:
            if self.installed_by_user_id is None:
                raise PluginReleaseError(
                    "Workspace Plugin installations require a Workspace user actor"
                )
            if self.installed_by_platform_actor is not None:
                raise PluginReleaseError(
                    "Workspace Plugin installations cannot use a platform actor"
                )
            if self.execution_policy is not PluginExecutionPolicy.ISOLATED_ONLY:
                raise PluginReleaseError(
                    "Workspace Plugin installations must use isolated-only execution"
                )
        if self.scope is PluginReleaseScope.SYSTEM and self.distribution is None:
            raise PluginReleaseError(
                "System Plugin installations require distribution metadata"
            )
        if self.scope is PluginReleaseScope.WORKSPACE and self.distribution is not None:
            raise PluginReleaseError(
                "Workspace Plugin installations cannot declare System distribution metadata"
            )
        if self.slug.strip() == "" or len(self.slug) > 100:
            raise PluginReleaseError(
                "Plugin installation slug must contain 1 to 100 characters"
            )
        if isinstance(self.release_revision, bool) or self.release_revision < 1:
            raise PluginReleaseError(
                "Plugin installation release revision must be positive"
            )
        if self.installed_at.tzinfo is None:
            raise PluginReleaseError("Plugin installed_at must be timezone-aware")

    @classmethod
    def from_release(
        cls,
        release: PluginRelease,
        *,
        namespace: PluginReleaseNamespace,
        execution_policy: PluginExecutionPolicy,
        distribution: PluginDistribution | None,
        installed_by_user_id: UUID | None,
        installed_by_platform_actor: str | None,
    ) -> "PluginInstallation":
        return cls(
            release_id=release.id,
            scope=namespace.scope,
            workspace_id=namespace.workspace_id,
            slug=release.slug,
            release_revision=release.revision,
            execution_policy=execution_policy,
            distribution=distribution,
            installed_by_user_id=installed_by_user_id,
            installed_by_platform_actor=installed_by_platform_actor,
        )

    @property
    def namespace(self) -> PluginReleaseNamespace:
        return PluginReleaseNamespace(
            scope=self.scope,
            workspace_id=self.workspace_id,
        )


@dataclass(frozen=True, slots=True)
class InstalledPluginRelease:
    """One immutable release resolved through an exact scoped installation."""

    release: PluginRelease
    installation: PluginInstallation

    def __post_init__(self) -> None:
        if (
            self.installation.release_id != self.release.id
            or self.installation.slug != self.release.slug
            or self.installation.release_revision != self.release.revision
        ):
            raise PluginReleaseError(
                "Plugin installation does not match its immutable release"
            )

    @property
    def id(self) -> UUID:
        return self.release.id

    @property
    def installation_id(self) -> UUID:
        return self.installation.id

    @property
    def scope(self) -> PluginReleaseScope:
        return self.installation.scope

    @property
    def workspace_id(self) -> UUID | None:
        return self.installation.workspace_id

    @property
    def namespace(self) -> PluginReleaseNamespace:
        return self.installation.namespace

    @property
    def slug(self) -> str:
        return self.release.slug

    @property
    def revision(self) -> int:
        return self.release.revision

    @property
    def catalog(self) -> PluginCatalogManifest:
        return self.release.catalog

    @property
    def capabilities(self) -> PluginCapabilityManifest:
        return self.release.capabilities

    @property
    def capability_digest(self) -> str:
        return self.release.capability_digest

    @property
    def execution_policy(self) -> PluginExecutionPolicy:
        return self.installation.execution_policy

    @property
    def distribution(self) -> PluginDistribution | None:
        return self.installation.distribution

    @property
    def runtime_artifact(self) -> PluginRuntimeArtifact | None:
        return self.release.runtime_artifact

    @property
    def runtime_image_digest(self) -> str | None:
        return self.release.runtime_image_digest

    @property
    def runtime_profile(self) -> str:
        return self.release.runtime_profile

    @property
    def loader_target(self) -> str:
        return self.release.loader_target

    @property
    def published_by_user_id(self) -> UUID | None:
        return self.release.published_by_user_id

    @property
    def published_by_platform_actor(self) -> str | None:
        return self.release.published_by_platform_actor

    @property
    def published_at(self) -> datetime:
        return self.release.published_at

    @property
    def source_digest(self) -> str:
        return self.release.source_digest

    @property
    def contract_digest(self) -> str:
        return self.release.contract_digest

    @property
    def protocol_digest(self) -> str:
        return self.release.protocol_digest

    @property
    def profile_digest(self) -> str:
        return self.release.profile_digest

    @property
    def lock_digest(self) -> str:
        return self.release.lock_digest

    @property
    def descriptor_digest(self) -> str:
        if self.release.descriptor_digest is None:
            raise PluginReleaseError("Plugin release has no descriptor digest")
        return self.release.descriptor_digest

    @property
    def descriptor(self) -> PluginReleaseDescriptor:
        return self.release.descriptor

    @property
    def executable(self) -> bool:
        return self.release.executable

    @property
    def required_capabilities(self) -> tuple[PluginRuntimeCapability, ...]:
        return self.release.capabilities.capabilities


__all__ = ["InstalledPluginRelease", "PluginInstallation"]

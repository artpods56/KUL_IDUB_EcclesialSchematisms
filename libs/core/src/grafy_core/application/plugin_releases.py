"""Publish, select, and resolve immutable scoped Plugin releases."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from uuid import UUID

from grafy_core.canonical_conversions import CANONICAL_ARTIFACT_CONVERSIONS
from grafy_core.domain.errors import (
    NotFoundError,
    ObjectAlreadyExistsError,
    UserDisabledError,
)
from grafy_core.domain.identity import (
    ActorContext,
    WorkspaceAccess,
    WorkspaceCapability,
)
from grafy_core.domain.plugin_releases import (
    PluginArtifactConversionContract,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginDistribution,
    PluginExecutionPolicy,
    PlatformPluginActor,
    PluginRelease,
    PluginReleaseDescriptor,
    PluginReleaseError,
    PluginReleaseNamespace,
    PluginReleaseScope,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.domain.plugin_selection import PluginReleaseSelection
from grafy_core.domain.plugin_revocations import (
    PluginReleaseRevocation,
    PluginReleaseRevocationError,
    PluginReleaseRevocationReason,
)
from grafy_core.ports.plugin_releases import PluginReleaseUnitOfWorkPort
from grafy_core.ports.storage import FileStoragePort, SaveFileCommand


_CANONICAL_CONVERSION_CONTRACTS = {
    (contract.key.id, contract.key.version): contract
    for contract in (
        PluginArtifactConversionContract.from_conversion(conversion)
        for conversion in CANONICAL_ARTIFACT_CONVERSIONS
    )
}


@dataclass(frozen=True, slots=True)
class SystemPluginPromotionCandidate:
    """Exact release and prospective selection checked before promotion."""

    release: PluginRelease
    selection: PluginReleaseSelection
    expected_generation: int


def require_workspace_catalog_authority(catalog: PluginCatalogManifest) -> None:
    """Require every Workspace-owned identity to use its exact slug namespace."""

    prefix = f"{catalog.slug}."
    identities = [("node", node.operator_id) for node in catalog.nodes]
    identities.extend(
        ("artifact type", artifact.key.id) for artifact in catalog.artifact_types
    )
    for identity_kind, identity in identities:
        if identity.startswith(prefix):
            continue
        raise PluginReleaseError(
            f"Workspace Plugin {catalog.slug!r} owned {identity_kind} "
            f"{identity!r} must use the exact {prefix!r} namespace"
        )


def require_canonical_conversion_references(
    catalog: PluginCatalogManifest,
) -> None:
    """Reject release conversion contracts not owned exactly by deployment code."""

    for conversion in catalog.artifact_conversions:
        key = (conversion.key.id, conversion.key.version)
        if _CANONICAL_CONVERSION_CONTRACTS.get(key) == conversion:
            continue
        raise PluginReleaseError(
            f"Plugin {catalog.slug!r} artifact conversion "
            f"{conversion.key.id}@{conversion.key.version} is not an exact "
            "deployment-owned canonical conversion"
        )


class PluginReleaseService:
    """Shared release boundary for human- and agent-authored Plugin freezes.

    Source objects are content-addressed under the ``plugin-releases/``
    namespace, so a failed publication can only leave an orphaned object in
    that known garbage-collectable namespace; rows stay append-only.
    """

    def __init__(
        self,
        unit_of_work_factory: Callable[[], PluginReleaseUnitOfWorkPort],
        storage: FileStoragePort,
        *,
        bucket: str,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._storage = storage
        self._bucket = bucket

    async def authorize_publisher(
        self,
        workspace_id: UUID,
        published_by_user_id: UUID,
    ) -> None:
        """Require an active Plugin-publishing actor before expensive work."""

        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_publisher(
                unit_of_work,
                workspace_id,
                published_by_user_id,
            )

    async def publish(
        self,
        *,
        workspace_id: UUID,
        catalog: PluginCatalogManifest,
        capabilities: PluginCapabilityManifest,
        source_archive: bytes,
        lock_digest: str,
        runtime_profile: str,
        runtime_artifact: PluginRuntimeArtifact | None,
        published_by_user_id: UUID,
    ) -> PluginRelease:
        require_workspace_catalog_authority(catalog)
        return await self._append_release(
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.WORKSPACE,
                workspace_id=workspace_id,
            ),
            catalog=catalog,
            capabilities=capabilities,
            source_archive=source_archive,
            lock_digest=lock_digest,
            runtime_profile=runtime_profile,
            runtime_artifact=runtime_artifact,
            execution_policy=PluginExecutionPolicy.ISOLATED_ONLY,
            distribution=None,
            published_by_user_id=published_by_user_id,
            published_by_platform_actor=None,
            select_release=True,
        )

    async def stage_system(
        self,
        *,
        catalog: PluginCatalogManifest,
        capabilities: PluginCapabilityManifest,
        source_archive: bytes,
        lock_digest: str,
        runtime_profile: str,
        runtime_artifact: PluginRuntimeArtifact,
        execution_policy: PluginExecutionPolicy,
        distribution: PluginDistribution,
        platform_actor: PlatformPluginActor,
    ) -> PluginRelease:
        """Append an immutable global release without implicitly promoting it."""

        return await self._append_release(
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.SYSTEM,
                workspace_id=None,
            ),
            catalog=catalog,
            capabilities=capabilities,
            source_archive=source_archive,
            lock_digest=lock_digest,
            runtime_profile=runtime_profile,
            runtime_artifact=runtime_artifact,
            execution_policy=execution_policy,
            distribution=distribution,
            published_by_user_id=None,
            published_by_platform_actor=platform_actor.reference,
            select_release=False,
        )

    async def prepare_system_promotion(
        self,
        *,
        slug: str,
        revision: int,
        platform_actor: PlatformPluginActor,
        expected_generation: int | None = None,
    ) -> SystemPluginPromotionCandidate:
        """Build the exact post-promotion selection without mutating persistence."""

        namespace = PluginReleaseNamespace(
            scope=PluginReleaseScope.SYSTEM,
            workspace_id=None,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            release = await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )
            if release is None:
                raise NotFoundError("System Plugin release", f"{slug}@{revision}")
            current = await unit_of_work.plugin_releases.get_selection(
                namespace,
                slug,
            )
            actor_reference = f"platform:{platform_actor.reference}"
            if current is None:
                if expected_generation not in (None, 0):
                    raise PluginReleaseError(
                        f"System Plugin selection {slug!r} changed concurrently"
                    )
                prospective = PluginReleaseSelection.from_release(
                    release,
                    actor_reference=actor_reference,
                )
                observed_generation = 0
            else:
                if (
                    expected_generation is not None
                    and current.generation != expected_generation
                ):
                    raise PluginReleaseError(
                        f"System Plugin selection {slug!r} changed concurrently"
                    )
                observed_generation = current.generation
                prospective = replace(current)
                prospective.select(
                    release,
                    publish=True,
                    actor_reference=actor_reference,
                )
            return SystemPluginPromotionCandidate(
                release=release,
                selection=prospective,
                expected_generation=observed_generation,
            )

    async def promote_system(
        self,
        *,
        slug: str,
        revision: int,
        platform_actor: PlatformPluginActor,
        expected_generation: int,
    ) -> PluginReleaseSelection:
        namespace = PluginReleaseNamespace(
            scope=PluginReleaseScope.SYSTEM,
            workspace_id=None,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            release = await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )
            if release is None:
                raise NotFoundError("System Plugin release", f"{slug}@{revision}")
            selection = await unit_of_work.plugin_releases.get_selection(
                namespace,
                slug,
            )
            actor_reference = f"platform:{platform_actor.reference}"
            if selection is None:
                if expected_generation != 0:
                    raise PluginReleaseError(
                        f"System Plugin selection {slug!r} changed concurrently"
                    )
                selection = PluginReleaseSelection.from_release(
                    release,
                    actor_reference=actor_reference,
                )
                await unit_of_work.plugin_releases.add_selection(selection)
            else:
                if selection.generation != expected_generation:
                    raise PluginReleaseError(
                        f"System Plugin selection {slug!r} changed concurrently"
                    )
                previous_generation = selection.generation
                selection.select(
                    release,
                    publish=True,
                    actor_reference=actor_reference,
                )
                if selection.generation != previous_generation:
                    await unit_of_work.plugin_releases.update_selection(
                        selection,
                        expected_generation=previous_generation,
                    )
            await unit_of_work.commit()
            return selection

    async def _append_release(
        self,
        *,
        namespace: PluginReleaseNamespace,
        catalog: PluginCatalogManifest,
        capabilities: PluginCapabilityManifest,
        source_archive: bytes,
        lock_digest: str,
        runtime_profile: str,
        runtime_artifact: PluginRuntimeArtifact | None,
        execution_policy: PluginExecutionPolicy,
        distribution: PluginDistribution | None,
        published_by_user_id: UUID | None,
        published_by_platform_actor: str | None,
        select_release: bool,
    ) -> PluginRelease:
        require_canonical_conversion_references(catalog)
        if not source_archive:
            raise PluginReleaseError("Plugin source archive must not be empty")
        if namespace.scope is PluginReleaseScope.SYSTEM and runtime_artifact is None:
            raise PluginReleaseError(
                "System Plugin releases require a retained runtime artifact"
            )
        runtime_profile = runtime_profile.strip()
        source_digest = sha256(source_archive).hexdigest()
        contract_digest = plugin_contract_digest(catalog)
        profile_digest = plugin_profile_digest(runtime_profile)
        protocol_digest = plugin_protocol_digest()
        source_object_key = (
            f"plugin-releases/{namespace.storage_path}/{catalog.slug}/"
            f"{source_digest}.tar.gz"
        )
        descriptor = PluginReleaseDescriptor(
            source_digest=source_digest,
            contract_digest=contract_digest,
            capability_digest=capabilities.digest,
            protocol_digest=protocol_digest,
            profile_digest=profile_digest,
            lock_digest=lock_digest,
            runtime_profile=runtime_profile,
            runtime_artifact=runtime_artifact,
            scope=namespace.scope,
            execution_policy=execution_policy,
            distribution=distribution,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            if namespace.scope is PluginReleaseScope.WORKSPACE:
                if namespace.workspace_id is None or published_by_user_id is None:
                    raise PluginReleaseError(
                        "Workspace Plugin publication requires a Workspace publisher"
                    )
                await self._require_publisher(
                    unit_of_work,
                    namespace.workspace_id,
                    published_by_user_id,
                )
                system_namespace = PluginReleaseNamespace(
                    scope=PluginReleaseScope.SYSTEM,
                    workspace_id=None,
                )
                if await unit_of_work.plugin_releases.family_exists(
                    system_namespace,
                    catalog.slug,
                ):
                    raise PluginReleaseError(
                        f"Workspace Plugin {catalog.slug!r} conflicts with a "
                        "System Plugin family"
                    )
                self._require_no_cross_scope_identity_collisions(
                    catalog,
                    await unit_of_work.plugin_releases.list_catalogs(
                        system_namespace
                    ),
                    scope=PluginReleaseScope.WORKSPACE,
                )
            else:
                if await unit_of_work.plugin_releases.workspace_family_exists(
                    catalog.slug
                ):
                    raise PluginReleaseError(
                        f"System Plugin {catalog.slug!r} conflicts with a Workspace "
                        "Plugin family"
                    )
                self._require_no_cross_scope_identity_collisions(
                    catalog,
                    await unit_of_work.plugin_releases.list_workspace_catalogs(),
                    scope=PluginReleaseScope.SYSTEM,
                )
            existing = await unit_of_work.plugin_releases.get_by_descriptor_digest(
                namespace,
                catalog.slug,
                descriptor.digest,
            )
            if existing is not None:
                if existing.catalog != catalog or existing.capabilities != capabilities:
                    raise PluginReleaseError(
                        "Plugin release descriptor digest matched different metadata"
                    )
                return existing
            revision = await unit_of_work.plugin_releases.next_revision(
                namespace,
                catalog.slug,
            )
            await self._save_source_archive(
                source_object_key,
                source_archive,
                source_digest,
            )
            release = PluginRelease(
                workspace_id=namespace.workspace_id,
                slug=catalog.slug,
                revision=revision,
                catalog=catalog,
                contract_digest=contract_digest,
                capabilities=capabilities,
                capability_digest=capabilities.digest,
                protocol_digest=protocol_digest,
                profile_digest=profile_digest,
                source_object_key=source_object_key,
                source_digest=source_digest,
                lock_digest=lock_digest,
                runtime_profile=runtime_profile,
                runtime_image_digest=(
                    None
                    if runtime_artifact is None
                    else runtime_artifact.manifest_digest
                ),
                runtime_artifact=runtime_artifact,
                descriptor_digest=descriptor.digest,
                published_by_user_id=published_by_user_id,
                published_by_platform_actor=published_by_platform_actor,
                scope=namespace.scope,
                execution_policy=execution_policy,
                distribution=distribution,
            )
            await unit_of_work.plugin_releases.add(release)
            if select_release:
                selection = await unit_of_work.plugin_releases.get_selection(
                    namespace,
                    catalog.slug,
                )
                actor_reference = f"user:{published_by_user_id}"
                if selection is None:
                    await unit_of_work.plugin_releases.add_selection(
                        PluginReleaseSelection.from_release(
                            release,
                            actor_reference=actor_reference,
                        )
                    )
                else:
                    expected_generation = selection.generation
                    selection.select(
                        release,
                        publish=True,
                        actor_reference=actor_reference,
                    )
                    await unit_of_work.plugin_releases.update_selection(
                        selection,
                        expected_generation=expected_generation,
                    )
            await unit_of_work.commit()
            return release

    @staticmethod
    def _require_no_cross_scope_identity_collisions(
        catalog: PluginCatalogManifest,
        retained_catalogs: list[PluginCatalogManifest],
        *,
        scope: PluginReleaseScope,
    ) -> None:
        retained_scope = (
            PluginReleaseScope.SYSTEM
            if scope is PluginReleaseScope.WORKSPACE
            else PluginReleaseScope.WORKSPACE
        )
        retained_nodes = {
            (node.operator_id, node.operator_version)
            for retained_catalog in retained_catalogs
            for node in retained_catalog.nodes
        }
        for node in catalog.nodes:
            key = (node.operator_id, node.operator_version)
            if key in retained_nodes:
                raise PluginReleaseError(
                    f"{scope.value.title()} Plugin {catalog.slug!r} node "
                    f"{node.operator_id}@{node.operator_version} conflicts with a "
                    f"retained {retained_scope.value.title()} Plugin identity"
                )
        retained_artifacts = {
            (artifact.key.id, artifact.key.schema_version)
            for retained_catalog in retained_catalogs
            for artifact in retained_catalog.artifact_types
        }
        for artifact in catalog.artifact_types:
            key = (artifact.key.id, artifact.key.schema_version)
            if key in retained_artifacts:
                raise PluginReleaseError(
                    f"{scope.value.title()} Plugin {catalog.slug!r} artifact type "
                    f"{artifact.key.id}@{artifact.key.schema_version} conflicts "
                    f"with a retained {retained_scope.value.title()} Plugin identity"
                )

    async def list_current(self, workspace_id: UUID) -> list[PluginRelease]:
        namespace = PluginReleaseNamespace(
            scope=PluginReleaseScope.WORKSPACE,
            workspace_id=workspace_id,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.list_current(namespace)

    async def list_current_system(self) -> list[PluginRelease]:
        namespace = PluginReleaseNamespace(
            scope=PluginReleaseScope.SYSTEM,
            workspace_id=None,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.list_current(namespace)

    async def get_system_by_revision(
        self,
        slug: str,
        revision: int,
    ) -> PluginRelease | None:
        namespace = PluginReleaseNamespace(
            scope=PluginReleaseScope.SYSTEM,
            workspace_id=None,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )

    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
        *,
        scope: PluginReleaseScope = PluginReleaseScope.WORKSPACE,
    ) -> PluginRelease | None:
        namespace = PluginReleaseNamespace(
            scope=scope,
            workspace_id=(
                workspace_id if scope is PluginReleaseScope.WORKSPACE else None
            ),
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )

    async def get_selection(
        self,
        workspace_id: UUID,
        slug: str,
        *,
        scope: PluginReleaseScope = PluginReleaseScope.WORKSPACE,
    ) -> PluginReleaseSelection | None:
        """Read the exact mutable selection for one scoped Plugin family."""

        namespace = PluginReleaseNamespace(
            scope=scope,
            workspace_id=(
                workspace_id if scope is PluginReleaseScope.WORKSPACE else None
            ),
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.get_selection(namespace, slug)

    async def revoke(
        self,
        *,
        workspace_id: UUID,
        slug: str,
        revision: int,
        reason: PluginReleaseRevocationReason,
        revoked_by_user_id: UUID,
    ) -> PluginReleaseRevocation:
        """Permanently revoke one exact release owned by a Workspace."""

        return await self._revoke_release(
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.WORKSPACE,
                workspace_id=workspace_id,
            ),
            slug=slug,
            revision=revision,
            reason=reason,
            revoked_by_user_id=revoked_by_user_id,
            revoked_by_platform_actor=None,
        )

    async def revoke_system(
        self,
        *,
        slug: str,
        revision: int,
        reason: PluginReleaseRevocationReason,
        platform_actor: PlatformPluginActor,
    ) -> PluginReleaseRevocation:
        """Permanently revoke one exact System release as platform authority."""

        return await self._revoke_release(
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.SYSTEM,
                workspace_id=None,
            ),
            slug=slug,
            revision=revision,
            reason=reason,
            revoked_by_user_id=None,
            revoked_by_platform_actor=platform_actor.reference,
        )

    async def get_revocation(
        self,
        *,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginReleaseRevocation | None:
        return await self._get_revocation(
            PluginReleaseNamespace(
                scope=PluginReleaseScope.WORKSPACE,
                workspace_id=workspace_id,
            ),
            slug,
            revision,
        )

    async def get_system_revocation(
        self,
        *,
        slug: str,
        revision: int,
    ) -> PluginReleaseRevocation | None:
        return await self._get_revocation(
            PluginReleaseNamespace(
                scope=PluginReleaseScope.SYSTEM,
                workspace_id=None,
            ),
            slug,
            revision,
        )

    async def _revoke_release(
        self,
        *,
        namespace: PluginReleaseNamespace,
        slug: str,
        revision: int,
        reason: PluginReleaseRevocationReason,
        revoked_by_user_id: UUID | None,
        revoked_by_platform_actor: str | None,
    ) -> PluginReleaseRevocation:
        async with self._unit_of_work_factory() as unit_of_work:
            if namespace.scope is PluginReleaseScope.WORKSPACE:
                if namespace.workspace_id is None or revoked_by_user_id is None:
                    raise PluginReleaseRevocationError(
                        "Workspace Plugin revocation requires a Workspace actor"
                    )
                await self._require_publisher(
                    unit_of_work,
                    namespace.workspace_id,
                    revoked_by_user_id,
                )
            elif revoked_by_platform_actor is None:
                raise PluginReleaseRevocationError(
                    "System Plugin revocation requires a platform actor"
                )

            release = await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )
            if release is None:
                raise NotFoundError(
                    f"{namespace.scope.value.title()} Plugin release",
                    f"{slug}@{revision}",
                )
            proposed = PluginReleaseRevocation.from_release(
                release,
                reason=reason,
                revoked_by_user_id=revoked_by_user_id,
                revoked_by_platform_actor=revoked_by_platform_actor,
            )
            existing = (
                await unit_of_work.plugin_releases.get_revocation_by_release_id(
                    release.id
                )
            )
            if existing is not None:
                if existing.has_same_intent(proposed):
                    return existing
                raise PluginReleaseRevocationError(
                    "Plugin release revocation already exists with different "
                    f"immutable intent for {namespace.scope.value}:"
                    f"{namespace.workspace_id}:{slug}@{revision} ({release.id})"
                )
            revocation = await unit_of_work.plugin_releases.add_revocation(proposed)
            await unit_of_work.commit()
            return revocation

    async def _get_revocation(
        self,
        namespace: PluginReleaseNamespace,
        slug: str,
        revision: int,
    ) -> PluginReleaseRevocation | None:
        async with self._unit_of_work_factory() as unit_of_work:
            release = await unit_of_work.plugin_releases.get_by_revision(
                namespace,
                slug,
                revision,
            )
            if release is None:
                return None
            return await unit_of_work.plugin_releases.get_revocation_by_release_id(
                release.id
            )

    async def list_runtime_artifacts(self) -> list[PluginRuntimeArtifact]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.list_runtime_artifacts()

    async def _save_source_archive(
        self,
        object_key: str,
        source_archive: bytes,
        source_digest: str,
    ) -> None:
        try:
            stored = await self._storage.save(
                SaveFileCommand(
                    bucket=self._bucket,
                    path=object_key,
                    stream=BytesIO(source_archive),
                    content_type="application/gzip",
                    metadata={
                        "source": "plugin-release",
                        "sha256": source_digest,
                    },
                )
            )
        except ObjectAlreadyExistsError:
            existing = await self._storage.load(self._bucket, object_key)
            try:
                existing_digest = sha256(existing.read()).hexdigest()
            finally:
                existing.close()
            if existing_digest != source_digest:
                raise PluginReleaseError(
                    f"Stored Plugin source {object_key!r} does not match its digest"
                )
            return
        if stored.sha256 != source_digest:
            raise PluginReleaseError(
                f"Stored Plugin source {object_key!r} changed while being written"
            )

    @staticmethod
    async def _require_publisher(
        unit_of_work: PluginReleaseUnitOfWorkPort,
        workspace_id: UUID,
        published_by_user_id: UUID,
    ) -> None:
        user = await unit_of_work.identity.get_user(published_by_user_id)
        if user is None or not user.active:
            raise UserDisabledError(f"User {published_by_user_id} is disabled")
        membership = await unit_of_work.identity.get_membership(
            workspace_id=workspace_id,
            user_id=published_by_user_id,
        )
        if membership is None or not membership.is_active:
            raise NotFoundError("Workspace", str(workspace_id))
        WorkspaceAccess(
            actor=ActorContext(
                user_id=published_by_user_id,
                credential_reference="plugin-publication",
            ),
            workspace_id=workspace_id,
            membership=membership,
        ).require(WorkspaceCapability.PUBLISH_PLUGIN)


__all__ = [
    "PluginReleaseService",
    "SystemPluginPromotionCandidate",
    "require_canonical_conversion_references",
    "require_workspace_catalog_authority",
]

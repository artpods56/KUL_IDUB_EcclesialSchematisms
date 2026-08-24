"""Publish and list immutable Workspace Plugin releases."""

from collections.abc import Callable
from hashlib import sha256
from io import BytesIO
from uuid import UUID

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
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginRelease,
    PluginReleaseDescriptor,
    PluginReleaseError,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.ports.plugin_releases import PluginReleaseUnitOfWorkPort
from grafy_core.ports.storage import FileStoragePort, SaveFileCommand


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
        published_by_user_id: UUID | None,
    ) -> PluginRelease:
        if not source_archive:
            raise PluginReleaseError("Plugin source archive must not be empty")
        runtime_profile = runtime_profile.strip()
        source_digest = sha256(source_archive).hexdigest()
        contract_digest = plugin_contract_digest(catalog)
        profile_digest = plugin_profile_digest(runtime_profile)
        protocol_digest = plugin_protocol_digest()
        source_object_key = (
            f"plugin-releases/{workspace_id}/{catalog.slug}/{source_digest}.tar.gz"
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
        )
        async with self._unit_of_work_factory() as unit_of_work:
            if published_by_user_id is not None:
                await self._require_publisher(
                    unit_of_work,
                    workspace_id,
                    published_by_user_id,
                )
            existing = await unit_of_work.plugin_releases.get_by_descriptor_digest(
                workspace_id,
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
                workspace_id,
                catalog.slug,
            )
            await self._save_source_archive(
                source_object_key,
                source_archive,
                source_digest,
            )
            release = PluginRelease(
                workspace_id=workspace_id,
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
            )
            await unit_of_work.plugin_releases.add(release)
            await unit_of_work.commit()
            return release

    async def list_current(self, workspace_id: UUID) -> list[PluginRelease]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.list_current(workspace_id)

    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginRelease | None:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.plugin_releases.get_by_revision(
                workspace_id,
                slug,
                revision,
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


__all__ = ["PluginReleaseService"]

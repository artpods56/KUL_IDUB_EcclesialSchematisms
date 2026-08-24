"""Shared verified publication workflow for Workspace Plugin releases."""

import asyncio
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_core.domain.plugin_releases import (
    PluginRelease,
    plugin_contract_digest,
    plugin_profile_digest,
)

from grafy_api.plugin_oci import PluginOciImageBuilder
from grafy_api.plugin_publishing import (
    PluginDirectoryPublisher,
    PluginPublishingError,
    VerifiedPluginDirectory,
)


class PluginPublicationConflictError(RuntimeError):
    """The reviewed working copy or release head changed before publication."""


class PluginPublicationWorkflow:
    """Verify, build, and append one immutable Plugin release."""

    def __init__(
        self,
        publisher: PluginDirectoryPublisher,
        image_builder: PluginOciImageBuilder,
        releases: PluginReleaseService,
    ) -> None:
        self._publisher = publisher
        self._image_builder = image_builder
        self._releases = releases

    async def verify(
        self,
        directory: Path,
        *,
        expected_slug: str,
    ) -> VerifiedPluginDirectory:
        verified = await asyncio.to_thread(
            self._publisher.verify,
            directory,
            expected_slug=expected_slug,
        )
        if verified.capabilities.capabilities:
            rendered = ", ".join(verified.capabilities.capabilities)
            raise PluginPublishingError(
                "The first executable Workspace Plugin flow supports no requested "
                f"capabilities; remove: {rendered}"
            )
        return verified

    async def publish(
        self,
        *,
        workspace_id: UUID,
        directory: Path,
        expected_slug: str,
        published_by_user_id: UUID | None,
        reviewed_source_digest: str | None = None,
        reviewed_base_revision: int | None = None,
    ) -> PluginRelease:
        if published_by_user_id is not None:
            await self._releases.authorize_publisher(
                workspace_id,
                published_by_user_id,
            )
        verified = await self.verify(directory, expected_slug=expected_slug)
        source_digest = sha256(verified.source_archive).hexdigest()
        if (
            reviewed_source_digest is not None
            and source_digest != reviewed_source_digest
        ):
            raise PluginPublicationConflictError(
                "Plugin working copy changed after review"
            )
        if reviewed_base_revision is not None:
            current = next(
                (
                    release
                    for release in await self._releases.list_current(workspace_id)
                    if release.slug == expected_slug
                ),
                None,
            )
            current_revision = 0 if current is None else current.revision
            if current_revision != reviewed_base_revision:
                raise PluginPublicationConflictError(
                    "Plugin release head changed after review"
                )
        runtime_artifact = await self._image_builder.build_and_store(
            workspace_id=workspace_id,
            catalog=verified.catalog,
            source_archive=verified.source_archive,
            source_digest=source_digest,
            contract_digest=plugin_contract_digest(verified.catalog),
            profile_digest=plugin_profile_digest(verified.runtime_profile),
        )
        return await self._releases.publish(
            workspace_id=workspace_id,
            catalog=verified.catalog,
            capabilities=verified.capabilities,
            source_archive=verified.source_archive,
            lock_digest=verified.lock_digest,
            runtime_profile=verified.runtime_profile,
            runtime_artifact=runtime_artifact,
            published_by_user_id=published_by_user_id,
        )


__all__ = [
    "PluginPublicationConflictError",
    "PluginPublicationWorkflow",
]

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from grafy_core.domain.identity import Workspace, WorkspaceKind
from grafy_core.domain.plugin_releases import (
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginNodeContract,
    PluginRelease,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_persistence.database import Database, create_database
from grafy_persistence.orm import metadata
from grafy_persistence.adapters.repositories import SqlPluginReleaseRepository
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000851")
OTHER_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000852")


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = create_database(
        f"sqlite+aiosqlite:///{tmp_path / 'plugin-releases.sqlite3'}"
    )
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_workspace(
            Workspace(
                id=WORKSPACE_ID,
                slug="plugins",
                name="Plugins",
                kind=WorkspaceKind.SHARED,
            )
        )
        await unit_of_work.identity.add_workspace(
            Workspace(
                id=OTHER_WORKSPACE_ID,
                slug="other-plugins",
                name="Other Plugins",
                kind=WorkspaceKind.SHARED,
            )
        )
        await unit_of_work.commit()
    try:
        yield database
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_plugin_release_repository_is_workspace_scoped_and_lists_latest(
    database: Database,
) -> None:
    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        nodes=(
            PluginNodeContract(
                operator_id="notes.echo",
                operator_version=1,
                title="Echo",
                description="Echo text",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(),
                outputs=(),
            ),
        ),
    )
    capabilities = PluginCapabilityManifest()
    first = PluginRelease(
        workspace_id=WORKSPACE_ID,
        slug="notes",
        revision=1,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/notes/first.tar.gz",
        source_digest="1" * 64,
        lock_digest="2" * 64,
        runtime_profile="python-uv",
    )
    second = PluginRelease(
        workspace_id=WORKSPACE_ID,
        slug="notes",
        revision=2,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/notes/second.tar.gz",
        source_digest="3" * 64,
        lock_digest="4" * 64,
        runtime_profile="python-uv",
        runtime_image_digest="6" * 64,
        runtime_artifact=PluginRuntimeArtifact(
            object_key="plugin-releases/notes/runtime/5.oci.tar",
            archive_digest="5" * 64,
            manifest_digest="6" * 64,
            config_digest="7" * 64,
        ),
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.plugin_releases.next_revision(WORKSPACE_ID, "notes") == 1
        )
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.plugin_releases.get_by_source_digest(
                WORKSPACE_ID,
                "notes",
                first.source_digest,
            )
            == first
        )
        assert await unit_of_work.plugin_releases.list_current(WORKSPACE_ID) == [second]
        persisted_second = await unit_of_work.plugin_releases.get_by_descriptor_digest(
            WORKSPACE_ID,
            "notes",
            second.descriptor.digest,
        )
        assert persisted_second is not None
        assert persisted_second.runtime_artifact == second.runtime_artifact
        assert persisted_second.executable is True
        assert await unit_of_work.plugin_releases.list_runtime_artifacts() == [
            second.runtime_artifact
        ]
        assert await unit_of_work.plugin_releases.list_current(OTHER_WORKSPACE_ID) == []
        assert (
            await unit_of_work.plugin_releases.next_revision(WORKSPACE_ID, "notes") == 3
        )


@pytest.mark.asyncio
async def test_get_by_revision_still_resolves_older_releases_after_new_publish(
    database: Database,
) -> None:
    """Releases are append-only: a graph pinned to revision 1 keeps resolving
    revision 1 after revision 2 is published."""

    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        nodes=(
            PluginNodeContract(
                operator_id="notes.echo",
                operator_version=1,
                title="Echo",
                description="Echo text",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(),
                outputs=(),
            ),
        ),
    )
    capabilities = PluginCapabilityManifest()

    def _release(revision: int, source_digest: str) -> PluginRelease:
        return PluginRelease(
            workspace_id=WORKSPACE_ID,
            slug="notes",
            revision=revision,
            catalog=catalog,
            contract_digest=plugin_contract_digest(catalog),
            capabilities=capabilities,
            capability_digest=capabilities.digest,
            protocol_digest=plugin_protocol_digest(),
            profile_digest=plugin_profile_digest("python-uv"),
            source_object_key=f"plugin-releases/notes/r{revision}.tar.gz",
            source_digest=source_digest,
            lock_digest="9" * 64,
            runtime_profile="python-uv",
        )

    first = _release(1, "1" * 64)
    second = _release(2, "3" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.plugin_releases.get_by_revision(WORKSPACE_ID, "notes", 1)
            == first
        )
        # Publishing revision 2 never moves the existing pin.
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        resolved = await unit_of_work.plugin_releases.get_by_revision(
            WORKSPACE_ID,
            "notes",
            1,
        )
        assert resolved is not None
        assert resolved.revision == 1
        assert resolved.source_digest == first.source_digest
        assert (
            await unit_of_work.plugin_releases.get_by_revision(WORKSPACE_ID, "notes", 9)
            is None
        )


def test_plugin_release_repository_port_is_append_only() -> None:
    delete_members = {
        name
        for name in dir(SqlPluginReleaseRepository)
        if name.lower().startswith(("delete", "remove", "purge", "drop"))
    }
    assert delete_members == set()

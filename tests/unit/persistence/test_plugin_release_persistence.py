from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from grafy_core.domain.errors import ConcurrentWriteError
from grafy_core.domain.identity import User, Workspace, WorkspaceKind
from grafy_core.domain.plugin_releases import (
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginDistribution,
    PluginExecutionPolicy,
    PluginNodeContract,
    PluginRelease,
    PluginReleaseNamespace,
    PluginReleaseScope,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.domain.plugin_selection import (
    PluginFamilyLifecycle,
    PluginReleaseSelection,
    PluginReleaseSelectionError,
)
from grafy_core.domain.plugin_revocations import (
    PluginReleaseRevocation,
    PluginReleaseRevocationError,
    PluginReleaseRevocationReason,
)
from grafy_persistence.database import Database, create_database
from grafy_persistence.orm import metadata
from grafy_persistence.adapters.repositories import SqlPluginReleaseRepository
from grafy_persistence import schema
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000851")
OTHER_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000852")
WORKSPACE_NAMESPACE = PluginReleaseNamespace(
    scope=PluginReleaseScope.WORKSPACE,
    workspace_id=WORKSPACE_ID,
)
OTHER_WORKSPACE_NAMESPACE = PluginReleaseNamespace(
    scope=PluginReleaseScope.WORKSPACE,
    workspace_id=OTHER_WORKSPACE_ID,
)
SYSTEM_NAMESPACE = PluginReleaseNamespace(
    scope=PluginReleaseScope.SYSTEM,
    workspace_id=None,
)
REVOCATION_ACTOR_ID = UUID("00000000-0000-0000-0000-000000000853")


def _release(
    namespace: PluginReleaseNamespace,
    revision: int,
    source_digest: str,
    lock_digest: str,
    *,
    runtime_artifact: PluginRuntimeArtifact | None = None,
    slug: str = "notes",
) -> PluginRelease:
    catalog = PluginCatalogManifest(
        slug=slug,
        title="Notes",
        nodes=(
            PluginNodeContract(
                operator_id=f"{slug}.echo",
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
    is_system = namespace.scope is PluginReleaseScope.SYSTEM
    return PluginRelease(
        workspace_id=namespace.workspace_id,
        slug=slug,
        revision=revision,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key=(
            f"plugin-releases/{namespace.storage_path}/{slug}/r{revision}.tar.gz"
        ),
        source_digest=source_digest,
        lock_digest=lock_digest,
        runtime_profile="python-uv",
        runtime_image_digest=(
            None if runtime_artifact is None else runtime_artifact.manifest_digest
        ),
        runtime_artifact=runtime_artifact,
        scope=namespace.scope,
        execution_policy=(
            PluginExecutionPolicy.HOST_ELIGIBLE
            if is_system
            else PluginExecutionPolicy.ISOLATED_ONLY
        ),
        distribution=PluginDistribution.BUNDLED if is_system else None,
        published_by_platform_actor="test:system" if is_system else None,
    )


def _runtime_artifact(marker: str) -> PluginRuntimeArtifact:
    return PluginRuntimeArtifact(
        object_key=f"plugin-releases/system/{marker}/runtime.oci.tar",
        archive_digest=marker * 64,
        manifest_digest=marker * 64,
        config_digest=marker * 64,
    )


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
async def test_plugin_release_repository_scopes_identity_and_revision_sequences(
    database: Database,
) -> None:
    runtime_artifact = PluginRuntimeArtifact(
        object_key="plugin-releases/workspaces/runtime/5.oci.tar",
        archive_digest="5" * 64,
        manifest_digest="6" * 64,
        config_digest="7" * 64,
    )
    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    second = _release(
        WORKSPACE_NAMESPACE,
        2,
        "3" * 64,
        "4" * 64,
        runtime_artifact=runtime_artifact,
    )
    other_workspace_first = _release(
        OTHER_WORKSPACE_NAMESPACE,
        1,
        "1" * 64,
        "2" * 64,
    )
    system_first = _release(SYSTEM_NAMESPACE, 1, "1" * 64, "2" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert not await unit_of_work.plugin_releases.family_exists(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert not await unit_of_work.plugin_releases.workspace_family_exists("notes")
        assert (
            await unit_of_work.plugin_releases.next_revision(
                WORKSPACE_NAMESPACE,
                "notes",
            )
            == 1
        )
        assert (
            await unit_of_work.plugin_releases.next_revision(
                SYSTEM_NAMESPACE,
                "notes",
            )
            == 1
        )
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.plugin_releases.add(other_workspace_first)
        await unit_of_work.plugin_releases.add(system_first)
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(
                second,
                actor_reference=f"user:{WORKSPACE_ID}",
            )
        )
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(
                other_workspace_first,
                actor_reference=f"user:{OTHER_WORKSPACE_ID}",
            )
        )
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(
                system_first,
                actor_reference="platform:test:system",
            )
        )
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.plugin_releases.family_exists(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert await unit_of_work.plugin_releases.family_exists(
            OTHER_WORKSPACE_NAMESPACE,
            "notes",
        )
        assert await unit_of_work.plugin_releases.family_exists(
            SYSTEM_NAMESPACE,
            "notes",
        )
        assert await unit_of_work.plugin_releases.workspace_family_exists("notes")
        assert (
            await unit_of_work.plugin_releases.get_by_source_digest(
                WORKSPACE_NAMESPACE,
                "notes",
                first.source_digest,
            )
            == first
        )
        assert (
            await unit_of_work.plugin_releases.get_by_revision(
                SYSTEM_NAMESPACE,
                "notes",
                1,
            )
            == system_first
        )
        assert await unit_of_work.plugin_releases.list_current(WORKSPACE_NAMESPACE) == [
            second
        ]
        assert await unit_of_work.plugin_releases.list_current(SYSTEM_NAMESPACE) == [
            system_first
        ]
        persisted_system = await unit_of_work.plugin_releases.get_by_revision(
            SYSTEM_NAMESPACE,
            "notes",
            1,
        )
        assert persisted_system is not None
        assert persisted_system.published_by_platform_actor == "test:system"
        persisted_second = await unit_of_work.plugin_releases.get_by_descriptor_digest(
            WORKSPACE_NAMESPACE,
            "notes",
            second.descriptor.digest,
        )
        assert persisted_second is not None
        assert persisted_second.runtime_artifact == second.runtime_artifact
        assert persisted_second.executable is True
        assert await unit_of_work.plugin_releases.list_runtime_artifacts() == [
            second.runtime_artifact
        ]
        assert await unit_of_work.plugin_releases.list_current(
            OTHER_WORKSPACE_NAMESPACE
        ) == [other_workspace_first]
        assert (
            await unit_of_work.plugin_releases.next_revision(
                WORKSPACE_NAMESPACE,
                "notes",
            )
            == 3
        )
        assert (
            await unit_of_work.plugin_releases.next_revision(
                OTHER_WORKSPACE_NAMESPACE,
                "notes",
            )
            == 2
        )
        assert (
            await unit_of_work.plugin_releases.next_revision(
                SYSTEM_NAMESPACE,
                "notes",
            )
            == 2
        )


@pytest.mark.asyncio
async def test_exact_revocation_retains_release_selection_and_actor_provenance(
    database: Database,
) -> None:
    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    second = _release(WORKSPACE_NAMESPACE, 2, "3" * 64, "4" * 64)
    revocation = PluginReleaseRevocation.from_release(
        first,
        reason=PluginReleaseRevocationReason.SECURITY,
        revoked_by_user_id=REVOCATION_ACTOR_ID,
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(
            User(
                id=REVOCATION_ACTOR_ID,
                email="revoker@example.test",
                display_name="Revoker",
            )
        )
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(
                second,
                actor_reference=f"user:{REVOCATION_ACTOR_ID}",
            )
        )
        assert (
            await unit_of_work.plugin_releases.add_revocation(revocation)
            == revocation
        )
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        persisted = (
            await unit_of_work.plugin_releases.get_revocation_by_release_id(first.id)
        )
        assert persisted is not None
        assert persisted.release_id == first.id
        assert persisted.reason is PluginReleaseRevocationReason.SECURITY
        assert persisted.revoked_by_user_id == REVOCATION_ACTOR_ID
        assert persisted.revoked_by_platform_actor is None
        assert persisted.revoked_at.tzinfo is not None
        assert await unit_of_work.plugin_releases.list_current(WORKSPACE_NAMESPACE) == [
            second
        ]
        assert (
            await unit_of_work.plugin_releases.get_by_revision(
                WORKSPACE_NAMESPACE,
                first.slug,
                first.revision,
            )
            is not None
        )
        repeated = PluginReleaseRevocation.from_release(
            first,
            reason=PluginReleaseRevocationReason.SECURITY,
            revoked_by_user_id=REVOCATION_ACTOR_ID,
        )
        assert (
            await unit_of_work.plugin_releases.add_revocation(repeated)
            == persisted
        )

        conflicting = PluginReleaseRevocation.from_release(
            first,
            reason=PluginReleaseRevocationReason.POLICY,
            revoked_by_user_id=REVOCATION_ACTOR_ID,
        )
        with pytest.raises(
            PluginReleaseRevocationError,
            match="different immutable intent",
        ):
            await unit_of_work.plugin_releases.add_revocation(conflicting)

        wrong_scope = PluginReleaseRevocation(
            release_id=first.id,
            scope=PluginReleaseScope.WORKSPACE,
            workspace_id=OTHER_WORKSPACE_ID,
            slug=first.slug,
            revision=first.revision,
            reason=PluginReleaseRevocationReason.SECURITY,
            revoked_by_user_id=REVOCATION_ACTOR_ID,
        )
        with pytest.raises(
            PluginReleaseRevocationError,
            match="identity does not match exact release",
        ):
            await unit_of_work.plugin_releases.add_revocation(wrong_scope)

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.delete().where(
                    schema.plugin_releases.c.id == first.id
                )
            )


@pytest.mark.asyncio
async def test_runtime_artifact_reachability_includes_every_retained_lifecycle(
    database: Database,
) -> None:
    historical = _release(
        SYSTEM_NAMESPACE,
        1,
        "1" * 64,
        "2" * 64,
        slug="current",
        runtime_artifact=_runtime_artifact("a"),
    )
    current = _release(
        SYSTEM_NAMESPACE,
        2,
        "3" * 64,
        "4" * 64,
        slug="current",
        runtime_artifact=_runtime_artifact("b"),
    )
    deprecated = _release(
        SYSTEM_NAMESPACE,
        1,
        "5" * 64,
        "6" * 64,
        slug="deprecated",
        runtime_artifact=_runtime_artifact("c"),
    )
    withdrawn = _release(
        SYSTEM_NAMESPACE,
        1,
        "7" * 64,
        "8" * 64,
        slug="withdrawn",
        runtime_artifact=_runtime_artifact("d"),
    )
    revoked = _release(
        SYSTEM_NAMESPACE,
        1,
        "9" * 64,
        "0" * 64,
        slug="revoked",
        runtime_artifact=_runtime_artifact("e"),
    )
    current_selection = PluginReleaseSelection.from_release(current)
    deprecated_selection = PluginReleaseSelection.from_release(deprecated)
    deprecated_selection.deprecate()
    withdrawn_selection = PluginReleaseSelection.from_release(withdrawn)
    withdrawn_selection.withdraw()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        for release in (historical, current, deprecated, withdrawn, revoked):
            await unit_of_work.plugin_releases.add(release)
        for selection in (
            current_selection,
            deprecated_selection,
            withdrawn_selection,
        ):
            await unit_of_work.plugin_releases.add_selection(selection)
        await unit_of_work.plugin_releases.add_revocation(
            PluginReleaseRevocation.from_release(
                revoked,
                reason=PluginReleaseRevocationReason.SECURITY,
                revoked_by_platform_actor="test:retention",
            )
        )
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        artifacts = await unit_of_work.plugin_releases.list_runtime_artifacts()
        assert {artifact.object_key for artifact in artifacts} == {
            release.runtime_artifact.object_key
            for release in (historical, current, deprecated, withdrawn, revoked)
            if release.runtime_artifact is not None
        }
        for release in (historical, current, deprecated, withdrawn, revoked):
            retained = await unit_of_work.plugin_releases.get_by_revision(
                SYSTEM_NAMESPACE,
                release.slug,
                release.revision,
            )
            assert retained is not None
            assert retained.source_object_key == release.source_object_key


@pytest.mark.asyncio
async def test_plugin_release_family_existence_preserves_scope_boundaries(
    database: Database,
) -> None:
    system_release = _release(
        SYSTEM_NAMESPACE,
        1,
        "1" * 64,
        "2" * 64,
        slug="system-only",
    )
    workspace_release = _release(
        OTHER_WORKSPACE_NAMESPACE,
        1,
        "3" * 64,
        "4" * 64,
        slug="workspace-only",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(system_release)
        await unit_of_work.plugin_releases.add(workspace_release)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.plugin_releases.family_exists(
            SYSTEM_NAMESPACE,
            "system-only",
        )
        assert not await unit_of_work.plugin_releases.family_exists(
            WORKSPACE_NAMESPACE,
            "system-only",
        )
        assert await unit_of_work.plugin_releases.family_exists(
            OTHER_WORKSPACE_NAMESPACE,
            "workspace-only",
        )
        assert not await unit_of_work.plugin_releases.family_exists(
            WORKSPACE_NAMESPACE,
            "workspace-only",
        )
        assert not await unit_of_work.plugin_releases.workspace_family_exists(
            "system-only"
        )
        assert await unit_of_work.plugin_releases.workspace_family_exists(
            "workspace-only"
        )


@pytest.mark.asyncio
async def test_plugin_release_selection_can_roll_back_to_an_older_revision(
    database: Database,
) -> None:
    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    second = _release(WORKSPACE_NAMESPACE, 2, "3" * 64, "4" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(
                second,
                actor_reference=f"user:{WORKSPACE_ID}",
            )
        )
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        selection = await unit_of_work.plugin_releases.get_selection(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert selection is not None
        expected_generation = selection.generation
        selection.select(
            first,
            actor_reference=f"user:{OTHER_WORKSPACE_ID}",
        )
        await unit_of_work.plugin_releases.update_selection(
            selection,
            expected_generation=expected_generation,
        )
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.plugin_releases.list_current(WORKSPACE_NAMESPACE) == [
            first
        ]
        selection = await unit_of_work.plugin_releases.get_selection(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert selection is not None
        assert selection.selected_release_id == first.id
        assert selection.selected_revision == 1
        assert selection.lifecycle is PluginFamilyLifecycle.PUBLISHED
        assert selection.generation == 2
        assert selection.updated_by_actor == f"user:{OTHER_WORKSPACE_ID}"


@pytest.mark.asyncio
async def test_plugin_release_selection_rejects_stale_generation(
    database: Database,
) -> None:
    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    second = _release(WORKSPACE_NAMESPACE, 2, "3" * 64, "4" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.plugin_releases.add_selection(
            PluginReleaseSelection.from_release(first)
        )
        await unit_of_work.commit()

    stale_unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    winning_unit_of_work = SqlAlchemyUnitOfWork(database.sessions)
    async with stale_unit_of_work:
        stale = await stale_unit_of_work.plugin_releases.get_selection(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert stale is not None
        async with winning_unit_of_work:
            winner = await winning_unit_of_work.plugin_releases.get_selection(
                WORKSPACE_NAMESPACE,
                "notes",
            )
            assert winner is not None
            winner.select(second, actor_reference="user:winner")
            await winning_unit_of_work.plugin_releases.update_selection(
                winner,
                expected_generation=1,
            )
            await winning_unit_of_work.commit()

        stale.select(second, actor_reference="user:stale")
        with pytest.raises(
            ConcurrentWriteError,
            match="notes.*expected generation 1",
        ):
            await stale_unit_of_work.plugin_releases.update_selection(
                stale,
                expected_generation=1,
            )
        await stale_unit_of_work.rollback()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        selection = await unit_of_work.plugin_releases.get_selection(
            WORKSPACE_NAMESPACE,
            "notes",
        )
        assert selection is not None
        assert selection.selected_release_id == second.id
        assert selection.generation == 2
        assert selection.updated_by_actor == "user:winner"


@pytest.mark.asyncio
async def test_plugin_release_selection_write_verifies_exact_release_identity(
    database: Database,
) -> None:
    release = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(release)
        invalid = PluginReleaseSelection(
            scope=release.scope,
            workspace_id=release.workspace_id,
            slug=release.slug,
            selected_release_id=release.id,
            selected_revision=release.revision + 1,
        )
        with pytest.raises(
            PluginReleaseSelectionError, match="identity does not match"
        ):
            await unit_of_work.plugin_releases.add_selection(invalid)
        await unit_of_work.rollback()


@pytest.mark.asyncio
async def test_get_by_revision_still_resolves_older_releases_after_new_publish(
    database: Database,
) -> None:
    """Releases are append-only: a graph pinned to revision 1 keeps resolving
    revision 1 after revision 2 is published."""

    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "9" * 64)
    second = _release(WORKSPACE_NAMESPACE, 2, "3" * 64, "9" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.plugin_releases.get_by_revision(
                WORKSPACE_NAMESPACE,
                "notes",
                1,
            )
            == first
        )
        # Publishing revision 2 never moves the existing pin.
        await unit_of_work.plugin_releases.add(second)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        resolved = await unit_of_work.plugin_releases.get_by_revision(
            WORKSPACE_NAMESPACE,
            "notes",
            1,
        )
        assert resolved is not None
        assert resolved.revision == 1
        assert resolved.source_digest == first.source_digest
        assert (
            await unit_of_work.plugin_releases.get_by_revision(
                WORKSPACE_NAMESPACE,
                "notes",
                9,
            )
            is None
        )


@pytest.mark.asyncio
async def test_plugin_release_repository_enforces_revision_and_descriptor_uniqueness(
    database: Database,
) -> None:
    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(first)
        await unit_of_work.commit()

    duplicate_revision = _release(WORKSPACE_NAMESPACE, 1, "3" * 64, "4" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        with pytest.raises(IntegrityError):
            await unit_of_work.plugin_releases.add(duplicate_revision)
        await unit_of_work.rollback()

    duplicate_descriptor = _release(WORKSPACE_NAMESPACE, 2, "1" * 64, "2" * 64)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        with pytest.raises(IntegrityError):
            await unit_of_work.plugin_releases.add(duplicate_descriptor)
        await unit_of_work.rollback()


@pytest.mark.asyncio
async def test_plugin_release_table_enforces_scope_owner_and_policy(
    database: Database,
) -> None:
    release = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64)
    system_release = _release(
        SYSTEM_NAMESPACE,
        1,
        "3" * 64,
        "4" * 64,
        slug="system-notes",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.plugin_releases.add(release)
        await unit_of_work.plugin_releases.add(system_release)
        await unit_of_work.commit()

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.update()
                .where(schema.plugin_releases.c.id == release.id)
                .values(
                    scope=PluginReleaseScope.SYSTEM,
                    distribution=PluginDistribution.BUNDLED,
                )
            )

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.update()
                .where(schema.plugin_releases.c.id == release.id)
                .values(workspace_id=None)
            )

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.update()
                .where(schema.plugin_releases.c.id == release.id)
                .values(distribution=PluginDistribution.BUNDLED)
            )

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.update()
                .where(schema.plugin_releases.c.id == release.id)
                .values(published_by_platform_actor="platform:test")
            )

    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            await connection.execute(
                schema.plugin_releases.update()
                .where(schema.plugin_releases.c.id == system_release.id)
                .values(published_by_platform_actor=None)
            )


def test_plugin_release_repository_port_is_append_only() -> None:
    delete_members = {
        name
        for name in dir(SqlPluginReleaseRepository)
        if name.lower().startswith(("delete", "remove", "purge", "drop"))
    }
    assert delete_members == set()


@pytest.mark.asyncio
async def test_list_workspace_catalogs_spans_every_workspace_and_retained_revision(
    database: Database,
) -> None:
    """Cross-scope identity checks need every retained Workspace catalog,
    not only current selections or one Workspace."""

    first = _release(WORKSPACE_NAMESPACE, 1, "1" * 64, "2" * 64, slug="alpha")
    second = _release(WORKSPACE_NAMESPACE, 2, "3" * 64, "4" * 64, slug="alpha")
    other_first = _release(
        OTHER_WORKSPACE_NAMESPACE, 1, "5" * 64, "6" * 64, slug="beta"
    )
    system_first = _release(SYSTEM_NAMESPACE, 1, "7" * 64, "8" * 64, slug="gamma")
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        for release in (first, second, other_first, system_first):
            await unit_of_work.plugin_releases.add(release)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        catalogs = await unit_of_work.plugin_releases.list_workspace_catalogs()

    assert [catalog.slug for catalog in catalogs] == ["alpha", "alpha", "beta"]
    owners: dict[str, UUID] = {}
    async with database.engine.connect() as connection:
        rows = await connection.execute(
            select(
                schema.plugin_releases.c.slug,
                schema.plugin_releases.c.workspace_id,
            ).where(
                schema.plugin_releases.c.scope == "workspace",
                schema.plugin_releases.c.slug.in_(
                    {catalog.slug for catalog in catalogs}
                ),
            )
        )
        for row in rows:
            assert row.workspace_id is not None
            owners[row.slug] = row.workspace_id
    assert owners == {"alpha": WORKSPACE_ID, "beta": OTHER_WORKSPACE_ID}
    assert set(owners.values()) == {WORKSPACE_ID, OTHER_WORKSPACE_ID}

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from notarius_core.application.identity import IdentityService
from notarius_core.domain.errors import (
    BootstrapOwnerMismatchError,
    IdentityInvariantError,
    LastWorkspaceOwnerError,
)
from notarius_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    created = create_database(
        f"sqlite+aiosqlite:///{tmp_path / 'identity-app.sqlite3'}"
    )
    async with created.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    async with SqlAlchemyUnitOfWork(created.sessions) as unit_of_work:
        await unit_of_work.identity.add_workspace(
            Workspace.shared(slug="local", name="Local workspace")
        )
        await unit_of_work.commit()
    try:
        yield created
    finally:
        await created.dispose()


def _service(database: Database) -> IdentityService:
    return IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))


@pytest.mark.asyncio
async def test_bootstrap_mapping_gates_first_login_and_is_consumed_once(
    database: Database,
) -> None:
    service = _service(database)
    await service.bootstrap_oidc_owner(
        issuer="https://issuer.example.test",
        subject="owner-subject",
    )
    with pytest.raises(BootstrapOwnerMismatchError):
        await service.provision_oidc_identity(
            issuer="https://issuer.example.test",
            subject="wrong-subject",
            email="wrong@example.test",
            display_name="Wrong",
        )

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.identity.get_user(UUID(int=1)) is None

    provisioned = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="owner-subject",
        email="owner@example.test",
        display_name="Owner",
    )

    assert provisioned.local_workspace_membership is not None
    assert provisioned.local_workspace_membership.role is WorkspaceRole.OWNER
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.identity.get_unconsumed_bootstrap_mapping(
                provisioned.local_workspace_membership.workspace_id
            )
            is None
        )
        provisioning_events = await unit_of_work.security_audit.list_for_workspace(
            provisioned.local_workspace_membership.workspace_id,
            limit=10,
        )
    assert any(
        event.operation == "oidc.identity.provision"
        and event.resource_type == "user"
        and event.resource_id == str(provisioned.user.id)
        for event in provisioning_events
    )

    async with database.engine.begin() as connection:
        await connection.execute(
            text("UPDATE oidc_bootstrap_owner_mappings SET consumed_at = NULL")
        )

    second = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="second-subject",
        email="second@example.test",
        display_name="Second",
    )
    assert second.local_workspace_membership is None
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.identity.get_unconsumed_bootstrap_mapping(
                provisioned.local_workspace_membership.workspace_id
            )
        ) is not None


@pytest.mark.asyncio
async def test_local_workspace_is_visible_to_editor_and_viewer_after_bootstrap(
    database: Database,
) -> None:
    service = _service(database)
    editor = User(email="editor@example.test", display_name="Editor")
    viewer = User(email="viewer@example.test", display_name="Viewer")
    async with database.engine.connect() as connection:
        local_id = UUID(
            str(
                (
                    await connection.execute(
                        text("SELECT id FROM workspaces WHERE slug = 'local'")
                    )
                ).scalar_one()
            )
        )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        local = await unit_of_work.identity.get_workspace(local_id)
        assert local is not None
        await unit_of_work.identity.add_user(editor)
        await unit_of_work.identity.add_user(viewer)
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=local.id,
                user_id=editor.id,
                role=WorkspaceRole.EDITOR,
            )
        )
        await unit_of_work.identity.add_membership(
            WorkspaceMembership(
                workspace_id=local.id,
                user_id=viewer.id,
                role=WorkspaceRole.VIEWER,
            )
        )
        await unit_of_work.commit()

    editor_workspaces = await service.list_workspaces(
        actor=ActorContext(user_id=editor.id, credential_reference="session-editor")
    )
    viewer_workspaces = await service.list_workspaces(
        actor=ActorContext(user_id=viewer.id, credential_reference="session-viewer")
    )
    assert [
        (workspace.slug, membership.role) for workspace, membership in editor_workspaces
    ] == [("local", WorkspaceRole.EDITOR)]
    assert [
        (workspace.slug, membership.role) for workspace, membership in viewer_workspaces
    ] == [("local", WorkspaceRole.VIEWER)]


@pytest.mark.asyncio
async def test_prebootstrap_user_without_membership_cannot_discover_local_workspace(
    database: Database,
) -> None:
    service = _service(database)
    user = User(email="pending@example.test", display_name="Pending")
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(user)
        await unit_of_work.commit()

    assert (
        await service.list_workspaces(
            actor=ActorContext(user_id=user.id, credential_reference="session-pending")
        )
        == []
    )


@pytest.mark.asyncio
async def test_personal_membership_stays_owner_and_membership_changes_are_audited(
    database: Database,
) -> None:
    service = _service(database)
    await service.bootstrap_oidc_owner(
        issuer="https://issuer.example.test",
        subject="owner-subject",
    )
    owner = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="owner-subject",
        email="owner@example.test",
        display_name="Owner",
    )
    member = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="member-subject",
        email="member@example.test",
        display_name="Member",
    )
    shared = await service.create_shared_workspace(
        actor=ActorContext(
            user_id=owner.user.id,
            credential_reference="session-owner",
        ),
        slug="team",
        name="Team",
    )

    with pytest.raises(LastWorkspaceOwnerError):
        await service.add_or_reactivate_member(
            actor=ActorContext(
                user_id=owner.user.id,
                credential_reference="session-owner",
            ),
            workspace_id=shared.id,
            user_id=owner.user.id,
            role=WorkspaceRole.VIEWER,
        )

    with pytest.raises(IdentityInvariantError):
        await service.add_or_reactivate_member(
            actor=ActorContext(
                user_id=owner.user.id,
                credential_reference="session-owner",
            ),
            workspace_id=owner.personal_workspace.id,
            user_id=owner.user.id,
            role=WorkspaceRole.VIEWER,
        )

    await service.add_or_reactivate_member(
        actor=ActorContext(
            user_id=owner.user.id,
            credential_reference="session-owner",
        ),
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.VIEWER,
    )
    await service.change_member_role(
        actor=ActorContext(
            user_id=owner.user.id,
            credential_reference="session-owner",
        ),
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.EDITOR,
    )
    await service.remove_member(
        actor=ActorContext(
            user_id=owner.user.id,
            credential_reference="session-owner",
        ),
        workspace_id=shared.id,
        user_id=member.user.id,
    )

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        events = await unit_of_work.security_audit.list_for_workspace(
            shared.id,
            limit=10,
        )
    operations = {event.operation for event in events}
    assert "workspace.membership.role_change" in operations
    assert "workspace.membership.remove" in operations
    assert all(event.credential_reference == "session-owner" for event in events)

    await service.disable_user(user_id=member.user.id)
    async with database.engine.connect() as connection:
        disabled_event = (
            (
                await connection.execute(
                    text(
                        "SELECT resource_type, resource_id "
                        "FROM security_audit_events "
                        "WHERE operation = 'user.disable'"
                    )
                )
            )
            .mappings()
            .one()
        )
    assert disabled_event["resource_type"] == "user"
    assert disabled_event["resource_id"] == str(member.user.id)


@pytest.mark.asyncio
async def test_concurrent_owner_removals_preserve_one_shared_owner(
    database: Database,
) -> None:
    service = _service(database)
    await service.bootstrap_oidc_owner(
        issuer="https://issuer.example.test",
        subject="owner-one-subject",
    )
    owner_one = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="owner-one-subject",
        email="owner-one@example.test",
        display_name="Owner One",
    )
    owner_two = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="owner-two-subject",
        email="owner-two@example.test",
        display_name="Owner Two",
    )
    shared = await service.create_shared_workspace(
        actor=ActorContext(user_id=owner_one.user.id),
        slug="concurrent-team",
        name="Concurrent Team",
    )
    await service.add_or_reactivate_member(
        actor=ActorContext(user_id=owner_one.user.id),
        workspace_id=shared.id,
        user_id=owner_two.user.id,
        role=WorkspaceRole.OWNER,
    )

    results = await asyncio.gather(
        service.remove_member(
            actor=ActorContext(user_id=owner_one.user.id),
            workspace_id=shared.id,
            user_id=owner_one.user.id,
        ),
        service.remove_member(
            actor=ActorContext(user_id=owner_two.user.id),
            workspace_id=shared.id,
            user_id=owner_two.user.id,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, WorkspaceMembership) for result in results) == 1
    assert sum(isinstance(result, LastWorkspaceOwnerError) for result in results) == 1
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        owner_count = await unit_of_work.identity.count_active_owners(shared.id)
    assert owner_count == 1

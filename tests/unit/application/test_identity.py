from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from notarius_core.application.identity import IdentityService
from notarius_core.domain.errors import (
    BootstrapOwnerMismatchError,
    IdentityInvariantError,
)
from notarius_core.domain.identity import Workspace, WorkspaceRole
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    created = create_database(f"sqlite+aiosqlite:///{tmp_path / 'identity-app.sqlite3'}")
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

    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE oidc_bootstrap_owner_mappings SET consumed_at = NULL"
            )
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
        actor_user_id=owner.user.id,
        slug="team",
        name="Team",
    )

    with pytest.raises(IdentityInvariantError):
        await service.add_or_reactivate_member(
            actor_user_id=owner.user.id,
            workspace_id=owner.personal_workspace.id,
            user_id=owner.user.id,
            role=WorkspaceRole.VIEWER,
        )

    await service.add_or_reactivate_member(
        actor_user_id=owner.user.id,
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.VIEWER,
    )
    await service.change_member_role(
        actor_user_id=owner.user.id,
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.EDITOR,
    )
    await service.remove_member(
        actor_user_id=owner.user.id,
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

    await service.disable_user(user_id=member.user.id)
    async with database.engine.connect() as connection:
        disabled_event = (
            await connection.execute(
                text(
                    "SELECT resource_type, resource_id "
                    "FROM security_audit_events "
                    "WHERE operation = 'user.disable'"
                )
            )
        ).mappings().one()
    assert disabled_event["resource_type"] == "user"
    assert disabled_event["resource_id"] == str(member.user.id)

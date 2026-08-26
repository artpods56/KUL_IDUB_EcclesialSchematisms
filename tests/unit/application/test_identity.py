import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from grafy_core.application.identity import IdentityService
from grafy_core.domain.errors import (
    IdentityInvariantError,
    LastWorkspaceOwnerError,
)
from grafy_core.domain.identity import (
    ActorContext,
    AuthSession,
    PersonalAccessToken,
    WorkspaceCapability,
    WorkspaceMembership,
    WorkspaceRole,
)
from grafy_persistence.database import Database, create_database
from grafy_persistence.orm import metadata
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    created = create_database(
        f"sqlite+aiosqlite:///{tmp_path / 'identity-app.sqlite3'}"
    )
    async with created.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    try:
        yield created
    finally:
        await created.dispose()


def _service(database: Database) -> IdentityService:
    return IdentityService(lambda: SqlAlchemyUnitOfWork(database.sessions))


@pytest.mark.asyncio
async def test_oidc_provisioning_creates_only_a_personal_workspace(
    database: Database,
) -> None:
    service = _service(database)
    provisioned = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="new-user-subject",
        email="owner@example.test",
        display_name="Owner",
    )

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        workspaces = await unit_of_work.identity.list_workspaces_for_user(
            provisioned.user.id
        )
        provisioning_events = await unit_of_work.security_audit.list_for_workspace(
            provisioned.personal_workspace.id,
            limit=10,
        )
    assert workspaces == [provisioned.personal_workspace]
    assert any(
        event.operation == "oidc.identity.provision"
        and event.resource_type == "user"
        and event.resource_id == str(provisioned.user.id)
        for event in provisioning_events
    )
    with pytest.raises(ValueError, match="slug 'local' is reserved"):
        await service.create_shared_workspace(
            actor=ActorContext(user_id=provisioned.user.id),
            slug="local",
            name="Local",
        )

@pytest.mark.asyncio
async def test_personal_membership_stays_owner_and_membership_changes_are_audited(
    database: Database,
) -> None:
    service = _service(database)
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
    with pytest.raises(IdentityInvariantError):
        await service.create_shared_workspace(
            actor=ActorContext(
                user_id=owner.user.id,
                credential_reference="session-owner",
            ),
            slug=" team ",
            name="Duplicate team",
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

    now = datetime.now(UTC)
    active_session = AuthSession(
        user_id=member.user.id,
        secret_digest=b"session-secret-digest",
        csrf_digest=b"csrf-secret-digest",
        expires_at=now + timedelta(hours=1),
    )
    active_token = PersonalAccessToken(
        user_id=member.user.id,
        workspace_id=shared.id,
        public_prefix="nrt_disable_user_test",
        secret_digest=b"pat-secret-digest",
        label="disable-user-test",
        scopes=(WorkspaceCapability.VIEW_GRAPH,),
        expires_at=now + timedelta(hours=1),
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_auth_session(active_session)
        await unit_of_work.identity.add_personal_access_token(active_token)
        await unit_of_work.commit()

    await service.disable_user(user_id=member.user.id)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        stored_session = await unit_of_work.identity.get_auth_session(active_session.id)
        stored_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=active_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
    assert stored_session is not None and stored_session.is_revoked
    assert stored_token is not None and stored_token.is_revoked
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


def _workspace_pat(
    *,
    user_id: UUID,
    workspace_id: UUID,
    label: str,
    scopes: tuple[WorkspaceCapability, ...],
    secret_digest: bytes,
) -> PersonalAccessToken:
    return PersonalAccessToken(
        user_id=user_id,
        workspace_id=workspace_id,
        public_prefix=f"nrt_{label}"[:32],
        secret_digest=secret_digest,
        label=label,
        scopes=scopes,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_membership_and_role_changes_revoke_affected_workspace_pats(
    database: Database,
) -> None:
    service = _service(database)
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
        slug="pat-revocation-team",
        name="PAT revocation team",
    )
    owner_actor = ActorContext(
        user_id=owner.user.id,
        credential_reference="session-owner",
    )
    await service.add_or_reactivate_member(
        actor=owner_actor,
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.EDITOR,
    )

    editor_only_token = _workspace_pat(
        user_id=member.user.id,
        workspace_id=shared.id,
        label="editor-only",
        scopes=(WorkspaceCapability.EDIT_GRAPH,),
        secret_digest=b"pat-editor-only-digest",
    )
    viewer_compatible_token = _workspace_pat(
        user_id=member.user.id,
        workspace_id=shared.id,
        label="viewer-compatible",
        scopes=(WorkspaceCapability.VIEW_GRAPH,),
        secret_digest=b"pat-viewer-compatible-digest",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_personal_access_token(editor_only_token)
        await unit_of_work.identity.add_personal_access_token(viewer_compatible_token)
        await unit_of_work.commit()

    await service.change_member_role(
        actor=owner_actor,
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.VIEWER,
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        demoted_editor_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=editor_only_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        demoted_viewer_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=viewer_compatible_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        demotion_events = await unit_of_work.security_audit.list_for_workspace(
            shared.id,
            limit=20,
        )
    assert demoted_editor_token is not None and demoted_editor_token.is_revoked
    assert demoted_viewer_token is not None and not demoted_viewer_token.is_revoked
    assert any(
        event.operation == "credential.pat.revoke"
        and event.resource_id == str(editor_only_token.id)
        and event.user_id == owner.user.id
        for event in demotion_events
    )

    removal_token = _workspace_pat(
        user_id=member.user.id,
        workspace_id=shared.id,
        label="remove-member",
        scopes=(WorkspaceCapability.VIEW_GRAPH,),
        secret_digest=b"pat-remove-member-digest",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_personal_access_token(removal_token)
        await unit_of_work.commit()

    await service.remove_member(
        actor=owner_actor,
        workspace_id=shared.id,
        user_id=member.user.id,
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        removed_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=removal_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        still_revoked_editor_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=editor_only_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        removal_events = await unit_of_work.security_audit.list_for_workspace(
            shared.id,
            limit=30,
        )
    assert removed_token is not None and removed_token.is_revoked
    assert (
        still_revoked_editor_token is not None and still_revoked_editor_token.is_revoked
    )
    assert any(
        event.operation == "credential.pat.revoke"
        and event.resource_id == str(removal_token.id)
        for event in removal_events
    )

    editor_revoked_at = still_revoked_editor_token.revoked_at
    viewer_revoked_at = removed_token.revoked_at
    assert editor_revoked_at is not None
    assert viewer_revoked_at is not None

    await service.add_or_reactivate_member(
        actor=owner_actor,
        workspace_id=shared.id,
        user_id=member.user.id,
        role=WorkspaceRole.EDITOR,
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        restored_editor_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=editor_only_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        restored_removal_token = (
            await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=removal_token.id,
                user_id=member.user.id,
                workspace_id=shared.id,
            )
        )
        membership = await unit_of_work.identity.get_membership(
            workspace_id=shared.id,
            user_id=member.user.id,
        )
    assert membership is not None and membership.is_active
    assert restored_editor_token is not None and restored_editor_token.is_revoked
    assert restored_editor_token.revoked_at == editor_revoked_at
    assert restored_removal_token is not None and restored_removal_token.is_revoked
    assert restored_removal_token.revoked_at == viewer_revoked_at


@pytest.mark.asyncio
async def test_concurrent_owner_removals_preserve_one_shared_owner(
    database: Database,
) -> None:
    service = _service(database)
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

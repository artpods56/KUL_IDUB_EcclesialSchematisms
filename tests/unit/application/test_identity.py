import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from grafy_core.application.identity import IdentityService
from grafy_core.domain.errors import (
    BootstrapOwnerMismatchError,
    IdentityInvariantError,
    LastWorkspaceOwnerError,
    NotFoundError,
)
from grafy_core.domain.identity import (
    ActorContext,
    AuthSession,
    PersonalAccessToken,
    User,
    Workspace,
    WorkspaceCapability,
    WorkspaceInvitation,
    WorkspaceInvitationStatus,
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


def test_workspace_invitation_lifecycle_expires_at_the_boundary() -> None:
    now = datetime(2026, 8, 26, 8, 0, tzinfo=UTC)
    invitation = WorkspaceInvitation(
        workspace_id=UUID(int=1),
        invitee_user_id=UUID(int=2),
        invited_by_user_id=UUID(int=3),
        role=WorkspaceRole.VIEWER,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=7),
    )

    assert not invitation.expire_if_due(now=invitation.expires_at - timedelta(microseconds=1))
    assert invitation.expire_if_due(now=invitation.expires_at)
    assert invitation.status is WorkspaceInvitationStatus.EXPIRED
    assert invitation.resolved_at == invitation.expires_at


@pytest.mark.asyncio
async def test_verified_email_invitation_grants_access_only_after_acceptance(
    database: Database,
) -> None:
    service = _service(database)
    await service.bootstrap_oidc_owner(
        issuer="https://issuer.example.test",
        subject="invitation-owner",
    )
    owner = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="invitation-owner",
        email="owner@example.test",
        display_name="Owner",
        email_verified=True,
    )
    invitee = await service.provision_oidc_identity(
        issuer="https://issuer.example.test",
        subject="invitation-recipient",
        email="Invitee@Example.Test",
        display_name="Invitee",
        email_verified=True,
    )
    workspace = await service.create_shared_workspace(
        actor=ActorContext(user_id=owner.user.id, credential_reference="owner-session"),
        slug="invited-team",
        name="Invited team",
    )

    candidate = await service.resolve_workspace_invitation_candidate(
        actor=ActorContext(user_id=owner.user.id),
        workspace_id=workspace.id,
        email="invitee@example.test",
    )
    assert candidate.id == invitee.user.id
    invitation, _ = await service.create_workspace_invitation(
        actor=ActorContext(user_id=owner.user.id, credential_reference="owner-session"),
        workspace_id=workspace.id,
        email="INVITEE@example.test",
        role=WorkspaceRole.EDITOR,
    )
    assert all(
        listed_workspace.id != workspace.id
        for listed_workspace, _ in await service.list_workspaces(
            actor=ActorContext(user_id=invitee.user.id)
        )
    )

    pending = await service.list_my_workspace_invitations(
        actor=ActorContext(user_id=invitee.user.id)
    )
    assert [(item.id, listed_workspace.id) for item, listed_workspace, _ in pending] == [
        (invitation.id, workspace.id)
    ]
    accepted, membership = await service.accept_workspace_invitation(
        actor=ActorContext(
            user_id=invitee.user.id,
            credential_reference="invitee-session",
        ),
        invitation_id=invitation.id,
    )
    assert accepted.status is WorkspaceInvitationStatus.ACCEPTED
    assert membership.role is WorkspaceRole.EDITOR
    assert [(listed_workspace.id, listed_membership.role) for listed_workspace, listed_membership in await service.list_workspaces(actor=ActorContext(user_id=invitee.user.id)) if listed_workspace.id == workspace.id] == [
        (workspace.id, WorkspaceRole.EDITOR)
    ]

    unverified_user = User(
        email="unverified@example.test",
        display_name="Unverified user",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(unverified_user)
        await unit_of_work.commit()
    with pytest.raises(NotFoundError):
        await service.resolve_workspace_invitation_candidate(
            actor=ActorContext(user_id=owner.user.id),
            workspace_id=workspace.id,
            email="unverified@example.test",
        )

    with pytest.raises(IdentityInvariantError):
        await service.create_workspace_invitation(
            actor=ActorContext(user_id=owner.user.id),
            workspace_id=workspace.id,
            email="invitee@example.test",
            role=WorkspaceRole.VIEWER,
        )

    declining_user = User(
        email="decline@example.test",
        email_verified=True,
        display_name="Declining user",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(declining_user)
        await unit_of_work.commit()
    declining_invitation, _ = await service.create_workspace_invitation(
        actor=ActorContext(user_id=owner.user.id),
        workspace_id=workspace.id,
        email="decline@example.test",
        role=WorkspaceRole.VIEWER,
    )
    with pytest.raises(IdentityInvariantError):
        await service.create_workspace_invitation(
            actor=ActorContext(user_id=owner.user.id),
            workspace_id=workspace.id,
            email="decline@example.test",
            role=WorkspaceRole.OWNER,
        )
    declined = await service.decline_workspace_invitation(
        actor=ActorContext(user_id=declining_user.id),
        invitation_id=declining_invitation.id,
    )
    assert declined.status is WorkspaceInvitationStatus.DECLINED
    assert all(
        listed_workspace.id != workspace.id
        for listed_workspace, _ in await service.list_workspaces(
            actor=ActorContext(user_id=declining_user.id)
        )
    )


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

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from notarius_core.domain.identity import (
    AuthSession,
    OidcBootstrapOwnerMapping,
    OidcIdentity,
    OidcLoginTransaction,
    PersonalAccessToken,
    User,
    Workspace,
    WorkspaceCapability,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_core.domain.security_audit import (
    SecurityAuditActorKind,
    SecurityAuditEvent,
    SecurityAuditOutcome,
)
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    created = create_database(f"sqlite+aiosqlite:///{tmp_path / 'identity.sqlite3'}")
    async with created.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    try:
        yield created
    finally:
        await created.dispose()


@pytest.mark.asyncio
async def test_identity_records_round_trip_without_plain_credential_material(
    database: Database,
) -> None:
    now = datetime(2026, 8, 7, 8, 0, tzinfo=UTC)
    user = User(
        id=UUID(int=101),
        email="owner@example.test",
        display_name="Owner",
        created_at=now,
        updated_at=now,
    )
    identity = OidcIdentity(
        id=UUID(int=102),
        user_id=user.id,
        issuer="https://issuer.example.test",
        subject="subject-101",
        created_at=now,
        updated_at=now,
    )
    workspace = Workspace(
        id=UUID(int=103),
        slug="local",
        name="Local workspace",
        kind="shared",
        created_at=now,
        updated_at=now,
    )
    membership = WorkspaceMembership(
        workspace_id=workspace.id,
        user_id=user.id,
        role="owner",
        created_at=now,
        updated_at=now,
    )
    transaction = OidcLoginTransaction(
        id=UUID(int=104),
        state_digest=b"state-digest",
        nonce_digest=b"nonce-digest",
        encrypted_pkce_verifier=b"encrypted-verifier",
        pkce_key_version=2,
        return_path="/workspaces/local",
        expires_at=now + timedelta(minutes=5),
        created_at=now,
    )
    mapping = OidcBootstrapOwnerMapping(
        id=UUID(int=105),
        workspace_id=workspace.id,
        issuer=identity.issuer,
        subject=identity.subject,
        created_at=now,
    )
    session = AuthSession(
        id=UUID(int=106),
        user_id=user.id,
        secret_digest=b"session-digest",
        csrf_digest=b"csrf-digest",
        expires_at=now + timedelta(hours=1),
        created_at=now,
    )
    token = PersonalAccessToken(
        id=UUID(int=107),
        user_id=user.id,
        workspace_id=workspace.id,
        public_prefix="nrt_test_107",
        secret_digest=b"pat-digest",
        label="automation",
        scopes=(WorkspaceCapability.VIEW_GRAPH,),
        expires_at=now + timedelta(days=1),
        created_at=now,
    )
    audit = SecurityAuditEvent(
        id=UUID(int=108),
        occurred_at=now,
        actor_kind=SecurityAuditActorKind.AUTHENTICATED,
        user_id=user.id,
        credential_reference=str(session.id),
        workspace_id=workspace.id,
        resource_type="workspace",
        resource_id=str(workspace.id),
        operation="workspace.read",
        outcome=SecurityAuditOutcome.SUCCESS,
    )

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(user)
        await unit_of_work.identity.add_oidc_identity(identity)
        await unit_of_work.identity.add_workspace(workspace)
        await unit_of_work.identity.add_membership(membership)
        await unit_of_work.identity.add_login_transaction(transaction)
        await unit_of_work.identity.add_bootstrap_mapping(mapping)
        await unit_of_work.identity.add_auth_session(session)
        await unit_of_work.identity.add_personal_access_token(token)
        await unit_of_work.security_audit.add(audit)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        loaded_identity = await unit_of_work.identity.get_oidc_identity(
            issuer=identity.issuer,
            subject=identity.subject,
        )
        loaded_membership = await unit_of_work.identity.get_membership(
            workspace_id=workspace.id,
            user_id=user.id,
        )
        loaded_session = await unit_of_work.identity.get_auth_session_by_digest(
            session.secret_digest
        )
        loaded_token = await unit_of_work.identity.get_personal_access_token_by_digest(
            token.secret_digest
        )
        loaded_audit = (
            await unit_of_work.security_audit.list_for_workspace(
                workspace.id,
                limit=10,
            )
        )[0]

    assert loaded_identity == identity
    assert loaded_membership is not None
    assert loaded_membership.role is WorkspaceRole.OWNER
    assert loaded_session is not None
    assert loaded_session.secret_digest == session.secret_digest
    assert loaded_token is not None
    assert loaded_token.scopes == (WorkspaceCapability.VIEW_GRAPH,)
    assert loaded_audit.actor_kind is SecurityAuditActorKind.AUTHENTICATED
    assert loaded_audit.user_id == user.id

    async with database.engine.connect() as connection:
        stored = (
            await connection.execute(
            text(
                "SELECT state_digest, nonce_digest, encrypted_pkce_verifier "
                "FROM oidc_login_transactions WHERE id = :id"
            ),
            {"id": transaction.id.hex},
            )
        ).mappings().one()
        assert stored["state_digest"] == transaction.state_digest
        assert stored["nonce_digest"] == transaction.nonce_digest
        assert stored["encrypted_pkce_verifier"] == transaction.encrypted_pkce_verifier

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        stored_transaction = await unit_of_work.identity.get_login_transaction(
            transaction.id
        )
        stored_session = await unit_of_work.identity.get_auth_session(session.id)
        stored_token = await unit_of_work.identity.get_personal_access_token_by_digest(
            token.secret_digest
        )
        assert stored_transaction is not None
        assert stored_session is not None
        assert stored_token is not None
        stored_transaction.consume(consumed_at=now)
        stored_session.expires_at = now - timedelta(seconds=1)
        stored_token.expires_at = now - timedelta(seconds=1)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.identity.delete_expired_login_transactions(now) == 1
        assert await unit_of_work.identity.delete_expired_sessions(now) == 1
        assert await unit_of_work.identity.delete_expired_personal_access_tokens(now) == 1
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.identity.get_login_transaction(transaction.id) is None
        assert await unit_of_work.identity.get_auth_session(session.id) is None
        assert (
            await unit_of_work.identity.get_personal_access_token_by_digest(
                token.secret_digest
            )
            is None
        )

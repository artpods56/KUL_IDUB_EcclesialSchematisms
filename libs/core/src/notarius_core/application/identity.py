from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from notarius_core.domain.errors import (
    BootstrapOwnerRequiredError,
    CapabilityDeniedError,
    IdentityInvariantError,
    NotFoundError,
    UserDisabledError,
)
from notarius_core.domain.identity import (
    ActorContext,
    IdentityProvisioningResult,
    OidcBootstrapOwnerMapping,
    OidcIdentity,
    PAT_ALLOWED_CAPABILITIES,
    PersonalAccessToken,
    User,
    Workspace,
    WorkspaceAccess,
    WorkspaceCapability,
    WorkspaceKind,
    WorkspaceMembership,
    WorkspaceRole,
    ensure_last_owner_can_change,
    normalize_workspace_slug,
    validate_bootstrap_match,
)
from notarius_core.domain.security_audit import (
    SecurityAuditActorKind,
    SecurityAuditEvent,
    SecurityAuditOutcome,
)
from notarius_core.ports.identity import IdentityUnitOfWorkPort


def _utc_now() -> datetime:
    return datetime.now(UTC)


class IdentityService:
    """Application-owned identity, workspace, and membership workflows."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], IdentityUnitOfWorkPort],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def list_workspaces(
        self, *, actor: ActorContext
    ) -> list[tuple[Workspace, WorkspaceMembership]]:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_active_user(unit_of_work, actor.user_id)
            memberships = await unit_of_work.identity.list_memberships_for_user(
                actor.user_id
            )
            workspaces: list[tuple[Workspace, WorkspaceMembership]] = []
            for membership in memberships:
                if not membership.is_active:
                    continue
                workspace = await unit_of_work.identity.get_workspace(
                    membership.workspace_id
                )
                if workspace is not None:
                    workspaces.append((workspace, membership))
            return workspaces

    async def list_members(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
    ) -> list[tuple[User, WorkspaceMembership]]:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_workspace_owner(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
            )
            members: list[tuple[User, WorkspaceMembership]] = []
            for membership in await unit_of_work.identity.list_memberships(
                workspace_id
            ):
                user = await unit_of_work.identity.get_user(membership.user_id)
                if user is not None:
                    members.append((user, membership))
            return members

    async def create_personal_access_token(
        self,
        *,
        actor: ActorContext,
        token: PersonalAccessToken,
    ) -> PersonalAccessToken:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_active_user(unit_of_work, actor.user_id)
            if token.user_id != actor.user_id:
                raise IdentityInvariantError(
                    "PAT owner must match the authenticated user"
                )
            if not set(token.scopes).issubset(PAT_ALLOWED_CAPABILITIES):
                raise IdentityInvariantError(
                    "Personal access token scope is not available"
                )
            membership = await self._require_membership(
                unit_of_work,
                workspace_id=token.workspace_id,
                user_id=actor.user_id,
            )
            if not set(token.scopes).issubset(membership.capabilities):
                raise CapabilityDeniedError(
                    capability="personal_access_token_scope",
                    workspace_id=token.workspace_id,
                    user_id=actor.user_id,
                )
            await unit_of_work.identity.add_personal_access_token(token)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="credential.pat.create",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=token.workspace_id,
                    resource_type="personal_access_token",
                    resource_id=str(token.id),
                )
            )
            await unit_of_work.commit()
            return token

    async def list_personal_access_tokens(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
    ) -> list[PersonalAccessToken]:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_active_user(unit_of_work, actor.user_id)
            membership = await self._require_membership(
                unit_of_work,
                workspace_id=workspace_id,
                user_id=actor.user_id,
            )
            if not membership.grants(WorkspaceCapability.VIEW_GRAPH):
                raise CapabilityDeniedError(
                    capability=WorkspaceCapability.VIEW_GRAPH.value,
                    workspace_id=workspace_id,
                    user_id=actor.user_id,
                )
            return await unit_of_work.identity.list_personal_access_tokens_for_user_workspace(
                user_id=actor.user_id,
                workspace_id=workspace_id,
            )

    async def revoke_personal_access_token(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        token_id: UUID,
    ) -> PersonalAccessToken:
        async with self._unit_of_work_factory() as unit_of_work:
            await self._require_active_user(unit_of_work, actor.user_id)
            membership = await self._require_membership(
                unit_of_work,
                workspace_id=workspace_id,
                user_id=actor.user_id,
            )
            if not membership.grants(WorkspaceCapability.VIEW_GRAPH):
                raise CapabilityDeniedError(
                    capability=WorkspaceCapability.VIEW_GRAPH.value,
                    workspace_id=workspace_id,
                    user_id=actor.user_id,
                )
            token = await unit_of_work.identity.get_personal_access_token_for_user_workspace(
                token_id=token_id,
                user_id=actor.user_id,
                workspace_id=workspace_id,
            )
            if token is None:
                raise NotFoundError("Personal access token", str(token_id))
            token.revoke()
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    workspace_id=workspace_id,
                    resource_type="personal_access_token",
                    resource_id=str(token.id),
                    operation="credential.pat.revoke",
                    outcome=SecurityAuditOutcome.SUCCESS,
                )
            )
            await unit_of_work.commit()
            return token

    async def bootstrap_oidc_owner(
        self,
        *,
        issuer: str,
        subject: str,
    ) -> OidcBootstrapOwnerMapping:
        async with self._unit_of_work_factory() as unit_of_work:
            workspace = await unit_of_work.identity.get_workspace_by_slug("local")
            if workspace is None:
                raise NotFoundError("Workspace", "local")
            if workspace.kind is not WorkspaceKind.SHARED:
                raise IdentityInvariantError(
                    "The local bootstrap workspace must be shared"
                )
            if await unit_of_work.identity.count_active_owners(workspace.id) != 0:
                raise IdentityInvariantError(
                    "The local workspace already has an active owner"
                )
            existing = await unit_of_work.identity.get_unconsumed_bootstrap_mapping(
                workspace.id
            )
            if existing is not None:
                raise IdentityInvariantError(
                    "The local workspace already has a pending bootstrap mapping"
                )
            mapping = OidcBootstrapOwnerMapping(
                workspace_id=workspace.id,
                issuer=issuer,
                subject=subject,
            )
            await unit_of_work.identity.add_bootstrap_mapping(mapping)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.SYSTEM,
                    operation="bootstrap.owner.mapping.create",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace.id,
                )
            )
            await unit_of_work.commit()
        return mapping

    async def provision_oidc_identity(
        self,
        *,
        issuer: str,
        subject: str,
        email: str | None,
        display_name: str | None,
    ) -> IdentityProvisioningResult:
        """Provision or refresh one validated OIDC identity atomically."""
        async with self._unit_of_work_factory() as unit_of_work:
            local_workspace = await unit_of_work.identity.lock_workspace_by_slug_for_membership_mutation(
                "local"
            )
            identity = await unit_of_work.identity.get_oidc_identity(
                issuer=issuer,
                subject=subject,
            )
            if identity is not None:
                user = await unit_of_work.identity.get_user(identity.user_id)
                if user is None:
                    raise IdentityInvariantError(
                        f"OIDC identity {identity.id} references a missing user"
                    )
                if not user.active:
                    raise UserDisabledError(f"User {user.id} is disabled")
                user.update_profile(email=email, display_name=display_name)
                personal_workspace = await unit_of_work.identity.get_personal_workspace(
                    user.id
                )
                if personal_workspace is None:
                    personal_workspace = Workspace.personal(owner_user_id=user.id)
                    await unit_of_work.identity.add_workspace(personal_workspace)
                    await unit_of_work.identity.add_membership(
                        WorkspaceMembership(
                            workspace_id=personal_workspace.id,
                            user_id=user.id,
                            role=WorkspaceRole.OWNER,
                        )
                    )
                local_membership = await self._consume_local_bootstrap_if_needed(
                    unit_of_work,
                    user=user,
                    issuer=issuer,
                    subject=subject,
                )
                await unit_of_work.commit()
                return IdentityProvisioningResult(
                    user=user,
                    oidc_identity=identity,
                    personal_workspace=personal_workspace,
                    local_workspace_membership=local_membership,
                )

            if local_workspace is None:
                raise BootstrapOwnerRequiredError(
                    "Identity provisioning requires the migrated local workspace"
                )
            bootstrap_mapping = (
                await unit_of_work.identity.get_unconsumed_bootstrap_mapping(
                    local_workspace.id
                )
            )
            local_owner_count = await unit_of_work.identity.count_active_owners(
                local_workspace.id
            )
            local_is_sealed = local_owner_count == 0
            if local_is_sealed:
                if bootstrap_mapping is None:
                    raise BootstrapOwnerRequiredError(
                        "The local workspace is sealed until its bootstrap owner "
                        "mapping is configured"
                    )
                validate_bootstrap_match(
                    bootstrap_mapping,
                    issuer=issuer,
                    subject=subject,
                )

            user = User(email=email, display_name=display_name)
            identity = OidcIdentity(
                user_id=user.id,
                issuer=issuer,
                subject=subject,
            )
            personal_workspace = Workspace.personal(owner_user_id=user.id)
            personal_membership = WorkspaceMembership(
                workspace_id=personal_workspace.id,
                user_id=user.id,
                role=WorkspaceRole.OWNER,
            )
            await unit_of_work.identity.add_user(user)
            await unit_of_work.identity.add_oidc_identity(identity)
            await unit_of_work.identity.add_workspace(personal_workspace)
            await unit_of_work.identity.add_membership(personal_membership)

            local_membership = None
            if local_is_sealed:
                if bootstrap_mapping is None:
                    raise BootstrapOwnerRequiredError(
                        "The local workspace is sealed until its bootstrap owner "
                        "mapping is configured"
                    )
                bootstrap_mapping.consume()
                local_membership = WorkspaceMembership(
                    workspace_id=local_workspace.id,
                    user_id=user.id,
                    role=WorkspaceRole.OWNER,
                )
                await unit_of_work.identity.add_membership(local_membership)

            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.UNAUTHENTICATED,
                    operation="oidc.identity.provision",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=local_workspace.id
                    if local_membership is not None
                    else personal_workspace.id,
                    resource_type="user",
                    resource_id=str(user.id),
                )
            )
            await unit_of_work.commit()
        return IdentityProvisioningResult(
            user=user,
            oidc_identity=identity,
            personal_workspace=personal_workspace,
            local_workspace_membership=local_membership,
        )

    async def create_shared_workspace(
        self,
        *,
        actor: ActorContext,
        slug: str,
        name: str,
    ) -> Workspace:
        normalized_slug = normalize_workspace_slug(slug)
        async with self._unit_of_work_factory() as unit_of_work:
            existing = await unit_of_work.identity.lock_workspace_by_slug_for_membership_mutation(
                normalized_slug
            )
            if existing is not None:
                raise IdentityInvariantError("Workspace slug is already in use")
            await self._require_active_user(unit_of_work, actor.user_id)
            workspace = Workspace.shared(slug=normalized_slug, name=name)
            await unit_of_work.identity.add_workspace(workspace)
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=workspace.id,
                    user_id=actor.user_id,
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="workspace.create",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace.id,
                )
            )
            await unit_of_work.commit()
        return workspace

    async def authorize(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        capability: WorkspaceCapability,
    ) -> WorkspaceAccess:
        async with self._unit_of_work_factory() as unit_of_work:
            user = await self._require_active_user(unit_of_work, actor.user_id)
            del user
            membership = await unit_of_work.identity.get_membership(
                workspace_id=workspace_id,
                user_id=actor.user_id,
            )
            if membership is None or not membership.is_active:
                raise NotFoundError("Workspace", str(workspace_id))
            access = WorkspaceAccess(
                actor=actor,
                workspace_id=workspace_id,
                membership=membership,
            )
            access.require(capability)
            return access

    async def add_or_reactivate_member(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        user_id: UUID,
        role: WorkspaceRole,
    ) -> WorkspaceMembership:
        role = WorkspaceRole(role)
        async with self._unit_of_work_factory() as unit_of_work:
            workspace = (
                await unit_of_work.identity.lock_workspace_for_membership_mutation(
                    workspace_id
                )
            )
            if workspace is None:
                raise NotFoundError("Workspace", str(workspace_id))
            await self._require_workspace_owner(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
            )
            if (
                workspace.kind is WorkspaceKind.PERSONAL
                and user_id != workspace.personal_owner_user_id
            ):
                raise IdentityInvariantError(
                    "Personal workspace cannot accept another membership"
                )
            if (
                workspace.kind is WorkspaceKind.PERSONAL
                and role is not WorkspaceRole.OWNER
            ):
                raise IdentityInvariantError(
                    "Personal workspace membership must remain owner-authorized"
                )
            target_user = await self._require_active_user(unit_of_work, user_id)
            del target_user
            membership = await unit_of_work.identity.get_membership(
                workspace_id=workspace_id,
                user_id=user_id,
            )
            if membership is None:
                membership = WorkspaceMembership(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    role=role,
                )
                await unit_of_work.identity.add_membership(membership)
            elif membership.is_active:
                memberships = await unit_of_work.identity.list_memberships(workspace_id)
                ensure_last_owner_can_change(
                    workspace=workspace,
                    memberships=memberships,
                    target=membership,
                    replacement_role=role,
                )
                membership.change_role(role)
            else:
                membership.reactivate(role=role)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="workspace.membership.upsert",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="user",
                    resource_id=str(user_id),
                )
            )
            await unit_of_work.commit()
        return membership

    async def change_member_role(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        user_id: UUID,
        role: WorkspaceRole,
    ) -> WorkspaceMembership:
        role = WorkspaceRole(role)
        async with self._unit_of_work_factory() as unit_of_work:
            workspace = (
                await unit_of_work.identity.lock_workspace_for_membership_mutation(
                    workspace_id
                )
            )
            if workspace is None:
                raise NotFoundError("Workspace", str(workspace_id))
            await self._require_workspace_owner(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
            )
            membership = await self._require_membership(
                unit_of_work,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            memberships = await unit_of_work.identity.list_memberships(workspace_id)
            ensure_last_owner_can_change(
                workspace=workspace,
                memberships=memberships,
                target=membership,
                replacement_role=role,
            )
            membership.change_role(role)
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="workspace.membership.role_change",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="user",
                    resource_id=str(user_id),
                )
            )
            await unit_of_work.commit()
        return membership

    async def remove_member(
        self,
        *,
        actor: ActorContext,
        workspace_id: UUID,
        user_id: UUID,
    ) -> WorkspaceMembership:
        async with self._unit_of_work_factory() as unit_of_work:
            workspace = (
                await unit_of_work.identity.lock_workspace_for_membership_mutation(
                    workspace_id
                )
            )
            if workspace is None:
                raise NotFoundError("Workspace", str(workspace_id))
            await self._require_workspace_owner(
                unit_of_work,
                actor=actor,
                workspace_id=workspace_id,
            )
            membership = await self._require_membership(
                unit_of_work,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            memberships = await unit_of_work.identity.list_memberships(workspace_id)
            ensure_last_owner_can_change(
                workspace=workspace,
                memberships=memberships,
                target=membership,
                removing=True,
            )
            membership.revoke()
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.AUTHENTICATED,
                    user_id=actor.user_id,
                    credential_reference=actor.credential_reference,
                    operation="workspace.membership.remove",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    workspace_id=workspace_id,
                    resource_type="user",
                    resource_id=str(user_id),
                )
            )
            await unit_of_work.commit()
        return membership

    async def disable_user(self, *, user_id: UUID) -> User:
        async with self._unit_of_work_factory() as unit_of_work:
            user = await self._require_user(unit_of_work, user_id)
            user.active = False
            user.updated_at = _utc_now()
            for session in await unit_of_work.identity.list_auth_sessions_for_user(
                user_id
            ):
                session.revoke()
                await unit_of_work.security_audit.add(
                    SecurityAuditEvent(
                        actor_kind=SecurityAuditActorKind.SYSTEM,
                        operation="credential.session.revoke",
                        outcome=SecurityAuditOutcome.SUCCESS,
                        resource_type="auth_session",
                        resource_id=str(session.id),
                    )
                )
            for (
                token
            ) in await unit_of_work.identity.list_personal_access_tokens_for_user(
                user_id
            ):
                token.revoke()
                await unit_of_work.security_audit.add(
                    SecurityAuditEvent(
                        actor_kind=SecurityAuditActorKind.SYSTEM,
                        operation="credential.pat.revoke",
                        outcome=SecurityAuditOutcome.SUCCESS,
                        workspace_id=token.workspace_id,
                        resource_type="personal_access_token",
                        resource_id=str(token.id),
                    )
                )
            await unit_of_work.security_audit.add(
                SecurityAuditEvent(
                    actor_kind=SecurityAuditActorKind.SYSTEM,
                    operation="user.disable",
                    outcome=SecurityAuditOutcome.SUCCESS,
                    resource_type="user",
                    resource_id=str(user_id),
                )
            )
            await unit_of_work.commit()
        return user

    async def _consume_local_bootstrap_if_needed(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        *,
        user: User,
        issuer: str,
        subject: str,
    ) -> WorkspaceMembership | None:
        local_workspace = await unit_of_work.identity.get_workspace_by_slug("local")
        if local_workspace is None:
            return None
        if await unit_of_work.identity.count_active_owners(local_workspace.id) != 0:
            return None
        mapping = await unit_of_work.identity.get_unconsumed_bootstrap_mapping(
            local_workspace.id
        )
        if mapping is None:
            raise BootstrapOwnerRequiredError(
                "The local workspace is sealed until its bootstrap owner mapping "
                "is configured"
            )
        validate_bootstrap_match(mapping, issuer=issuer, subject=subject)
        mapping.consume()
        membership = WorkspaceMembership(
            workspace_id=local_workspace.id,
            user_id=user.id,
            role=WorkspaceRole.OWNER,
        )
        await unit_of_work.identity.add_membership(membership)
        return membership

    async def _require_active_user(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        user_id: UUID,
    ) -> User:
        user = await self._require_user(unit_of_work, user_id)
        if not user.active:
            raise UserDisabledError(f"User {user.id} is disabled")
        return user

    async def _require_user(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        user_id: UUID,
    ) -> User:
        user = await unit_of_work.identity.get_user(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return user

    async def _require_workspace(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        workspace_id: UUID,
    ) -> Workspace:
        workspace = await unit_of_work.identity.get_workspace(workspace_id)
        if workspace is None:
            raise NotFoundError("Workspace", str(workspace_id))
        return workspace

    async def _require_membership(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> WorkspaceMembership:
        membership = await unit_of_work.identity.get_membership(
            workspace_id=workspace_id,
            user_id=user_id,
        )
        if membership is None or not membership.is_active:
            raise NotFoundError("Workspace membership", f"{workspace_id}/{user_id}")
        return membership

    async def _require_workspace_owner(
        self,
        unit_of_work: IdentityUnitOfWorkPort,
        *,
        actor: ActorContext,
        workspace_id: UUID,
    ) -> WorkspaceMembership:
        await self._require_active_user(unit_of_work, actor.user_id)
        membership = await self._require_membership(
            unit_of_work,
            workspace_id=workspace_id,
            user_id=actor.user_id,
        )
        if not membership.grants(WorkspaceCapability.MANAGE_MEMBERS):
            raise CapabilityDeniedError(
                capability=WorkspaceCapability.MANAGE_MEMBERS.value,
                workspace_id=workspace_id,
                user_id=actor.user_id,
            )
        return membership


__all__ = ["IdentityService"]

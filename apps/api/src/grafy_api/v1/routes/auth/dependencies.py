"""Identity-plane FastAPI dependencies.

These thin resolvers expose the application-lifetime ``AppIdentity`` services
through the standard ``Depends()`` graph, mirroring the workbench plane's
per-service resolvers.  Registering an override for any of the resolver
functions (``identity_service``, ``auth_service``, ``identity_uow_factory``)
replaces that service across every route that consumes it, including
``require_workspace_capability``.
"""

from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyCookie

from grafy_core.application.identity import IdentityService
from grafy_core.domain.identity import (
    ActorContext,
    WorkspaceAccess,
    WorkspaceCapability,
)
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork

from grafy_api.app_state import get_identity
from grafy_api.v1.routes.auth.services import AuthService, SESSION_COOKIE


session_cookie_scheme = APIKeyCookie(
    name=SESSION_COOKIE,
    auto_error=False,
    description="Opaque host-only browser session cookie.",
)


def identity_service(request: Request) -> IdentityService:
    return get_identity(request.app).identity_service


IdentityServiceDependency = Annotated[
    IdentityService,
    Depends(identity_service),
]


def auth_service(request: Request) -> AuthService:
    return get_identity(request.app).auth_service


AuthServiceDependency = Annotated[AuthService, Depends(auth_service)]


def identity_uow_factory(
    request: Request,
) -> Callable[[], SqlAlchemyUnitOfWork]:
    return get_identity(request.app).identity_uow_factory


IdentityUnitOfWorkFactoryDependency = Annotated[
    Callable[[], SqlAlchemyUnitOfWork],
    Depends(identity_uow_factory),
]


async def browser_actor(
    request: Request,
    auth: AuthServiceDependency,
    _session_cookie: Annotated[str | None, Security(session_cookie_scheme)],
) -> ActorContext:
    if "authorization" in request.headers:
        error = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Browser routes accept cookie authentication only",
        )
        await auth.audit_unauthenticated_failure(
            operation="auth.session.verify",
            error_code="authorization_header_rejected",
        )
        request.state.auth_failure_audited = True
        raise error
    try:
        return await auth.require_browser_actor(request)
    except HTTPException as error:
        if error.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            error_code = "rate_limited"
        elif error.detail == "Origin validation failed":
            error_code = "origin_rejected"
        elif error.detail == "CSRF validation failed":
            error_code = "csrf_rejected"
        else:
            error_code = "authentication_required"
        await auth.audit_request_failure(
            request,
            operation="auth.session.verify",
            error_code=error_code,
        )
        request.state.auth_failure_audited = True
        raise


def require_workspace_capability(capability: WorkspaceCapability):
    async def dependency(
        workspace_id: UUID,
        actor: Annotated[ActorContext, Depends(browser_actor)],
        identity: IdentityServiceDependency,
    ) -> WorkspaceAccess:
        return await identity.authorize(
            actor=actor,
            workspace_id=workspace_id,
            capability=capability,
        )

    return Annotated[WorkspaceAccess, Depends(dependency)]


__all__ = [
    "AuthServiceDependency",
    "IdentityServiceDependency",
    "IdentityUnitOfWorkFactoryDependency",
    "auth_service",
    "browser_actor",
    "identity_service",
    "identity_uow_factory",
    "require_workspace_capability",
    "session_cookie_scheme",
]

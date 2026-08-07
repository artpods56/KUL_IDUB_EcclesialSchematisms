from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyCookie

from notarius_core.domain.identity import ActorContext
from notarius_api.v1.routes.auth.services import SESSION_COOKIE


session_cookie_scheme = APIKeyCookie(
    name=SESSION_COOKIE,
    auto_error=False,
    description="Opaque host-only browser session cookie.",
)


async def browser_actor(
    request: Request,
    _session_cookie: Annotated[str | None, Security(session_cookie_scheme)],
) -> ActorContext:
    auth = request.app.state.auth_service
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

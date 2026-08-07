import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from notarius_core.domain.identity import ActorContext

from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.auth.models import SessionResponse
from notarius_api.v1.routes.auth.services import (
    OIDC_TRANSACTION_COOKIE,
    SESSION_COOKIE,
    AuthService,
    OidcProtocolError,
    bounded_oidc_failure,
)
from notarius_api.v1.routes.auth.abuse import request_browser_key


router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.get("/oidc/login", include_in_schema=True)
async def oidc_login(
    request: Request,
    return_path: Annotated[str, Query(max_length=2048)] = "/",
) -> Response:
    auth: AuthService = request.app.state.auth_service
    browser_key = request_browser_key(request)
    if not await auth.allow_login_start(browser_key):
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.start",
            error_code="rate_limited",
        )
        raise HTTPException(status_code=429, detail="Too many login attempts")
    replaced = await auth.replace_login_transaction(
        request.cookies.get(OIDC_TRANSACTION_COOKIE)
    )
    if replaced:
        await auth.release_login(browser_key)
    if not await auth.reserve_login(browser_key):
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.start",
            error_code="too_many_login_transactions",
        )
        raise HTTPException(status_code=429, detail="Too many login attempts")
    try:
        authorization_url, transaction_id = await auth.start_login(
            return_path=return_path
        )
    except OidcProtocolError as error:
        await auth.release_login(browser_key)
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.start",
            error_code=error.code,
        )
        raise bounded_oidc_failure(error) from error
    except Exception:
        await auth.release_login(browser_key)
        raise
    response = RedirectResponse(authorization_url, status_code=307)
    auth.set_transaction_cookie(response, str(transaction_id))
    return response


@router.get("/oidc/callback", include_in_schema=True)
async def oidc_callback(
    request: Request,
    state: Annotated[str | None, Query(max_length=512)] = None,
    code: Annotated[str | None, Query(max_length=4096)] = None,
    error: Annotated[str | None, Query(max_length=256)] = None,
) -> Response:
    auth: AuthService = request.app.state.auth_service
    browser_key = request_browser_key(request)
    if not await auth.allow_callback(browser_key):
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.callback",
            error_code="rate_limited",
        )
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many callback attempts"},
        )
    try:
        _, issued, return_path = await auth.callback(
            transaction_id_value=request.cookies.get(OIDC_TRANSACTION_COOKIE),
            state=state,
            code=code,
            error=error,
            current_session_cookie=request.cookies.get(SESSION_COOKIE),
        )
    except OidcProtocolError as protocol_error:
        if protocol_error.transaction_consumed:
            await auth.release_login(browser_key)
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.callback",
            error_code=protocol_error.code,
        )
        failure = bounded_oidc_failure(protocol_error)
        response = JSONResponse(
            status_code=failure.status_code,
            content={"detail": failure.detail},
        )
        auth.clear_transaction_cookie(response)
        return response
    except Exception as exception:
        logger.warning(
            "oidc_callback_failed operation=oidc.login.callback error_class=%s",
            type(exception).__name__,
        )
        await auth.audit_unauthenticated_failure(
            operation="oidc.login.callback",
            error_code="internal_error",
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "OIDC callback failed"},
        )
        return response
    await auth.release_login(browser_key)
    response = RedirectResponse(return_path, status_code=303)
    auth.set_session_cookies(response, issued)
    auth.clear_transaction_cookie(response)
    return response


@router.get("/session", response_model=SessionResponse)
async def get_session(
    request: Request,
    _actor: Annotated[ActorContext, Depends(browser_actor)],
) -> SessionResponse:
    session = await request.app.state.auth_service.current_session(request)
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        created_at=session.created_at,
        last_used_at=session.last_used_at,
        expires_at=session.expires_at,
        revoked_at=session.revoked_at,
        current=True,
    )


@router.delete("/session", status_code=204)
async def delete_session(
    request: Request,
    _actor: Annotated[ActorContext, Depends(browser_actor)],
) -> Response:
    auth: AuthService = request.app.state.auth_service
    await auth.logout(request)
    response = Response(status_code=204)
    auth.clear_session_cookies(response)
    return response


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> list[SessionResponse]:
    sessions = await request.app.state.auth_service.list_sessions(actor=actor)
    current = await request.app.state.auth_service.current_session(request)
    return [
        SessionResponse(
            id=session.id,
            user_id=session.user_id,
            created_at=session.created_at,
            last_used_at=session.last_used_at,
            expires_at=session.expires_at,
            revoked_at=session.revoked_at,
            current=session.id == current.id,
        )
        for session in sessions
    ]


@router.delete("/sessions/{session_id}", status_code=204)
async def revoke_session(
    session_id: str,
    request: Request,
    actor: Annotated[ActorContext, Depends(browser_actor)],
) -> Response:
    try:
        parsed_session_id = UUID(session_id)
    except ValueError as exc:
        await request.app.state.auth_service.audit_request_failure(
            request,
            operation="auth.session.request",
            error_code="not_found",
        )
        raise HTTPException(status_code=404, detail="Session not found") from exc
    session = await request.app.state.auth_service.revoke_session(
        actor=actor,
        session_id=parsed_session_id,
    )
    response = Response(status_code=204)
    if session.id == (await request.app.state.auth_service.current_session(request)).id:
        request.app.state.auth_service.clear_session_cookies(response)
    return response

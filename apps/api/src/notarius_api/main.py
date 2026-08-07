import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
    request_validation_exception_handler as default_validation_error_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from notarius_core.application.collaboration import CollaborationService
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.application.identity import IdentityService
from notarius_core.domain.errors import (
    CapabilityDeniedError,
    IdentityInvariantError,
    NotFoundError,
    UserDisabledError,
)

from notarius_persistence.database import create_database
from notarius_persistence.unit_of_work import (
    SqlAlchemySavedGraphUnitOfWork,
    SqlAlchemyUnitOfWork,
)
from notarius_storage import create_file_storage

from notarius_api.app_state import AppIdentity, AppResources, get_identity
from notarius_api.builtins import builtin_plugins
from notarius_api.mcp import create_mounted_mcp_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.composition import build_workbench_components
from notarius_api.settings import Settings, get_settings
from notarius_api.single_owner import ApiOwnerLease
from notarius_api.v1.routes.auth.services import (
    OIDC_TRANSACTION_COOKIE,
    AuthService,
)
from notarius_api.v1.routes.auth.views import router as auth_router
from notarius_api.v1.routes.artifacts.views import router as artifacts_router
from notarius_api.v1.routes.catalog.views import router as catalog_router
from notarius_api.v1.routes.collaboration.hub import GraphRoomHub
from notarius_api.v1.routes.collaboration.publish import ActiveExecutionRoomPublisher
from notarius_api.v1.routes.collaboration.views import router as collaboration_router
from notarius_api.v1.routes.executions.views import router as executions_router
from notarius_api.v1.routes.node_secrets.services import NodeSecretService
from notarius_api.v1.routes.node_secrets.views import router as node_secrets_router
from notarius_api.v1.routes.saved_graphs.views import router as saved_graphs_router
from notarius_api.v1.routes.uploads.views import router as uploads_router
from notarius_api.v1.routes.workspaces.views import (
    router as workspaces_router,
    workspace_failure_metadata,
)


logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok"]


async def health() -> HealthResponse:
    return HealthResponse(status="ok")


async def _request_validation_error_handler(
    request: Request,
    exception: Exception,
) -> Response:
    if not isinstance(exception, RequestValidationError):
        raise exception
    is_oidc_callback = request.url.path.endswith("/auth/oidc/callback")
    is_oidc_login = request.url.path.endswith("/auth/oidc/login")
    if is_oidc_login:
        login_auth: AuthService = get_identity(request.app).auth_service
        abuse_keys = login_auth.browser_abuse_keys(request)
        allowed = await login_auth.allow_login_start(
            abuse_keys.browser_key,
            abuse_keys.network_key,
        )
        await login_auth.audit_request_failure(
            request,
            operation="oidc.login.start",
            error_code="validation_failed" if allowed else "rate_limited",
        )
        return JSONResponse(
            status_code=422 if allowed else 429,
            content=(
                {
                    "detail": [
                        {"loc": error.get("loc"), "type": error.get("type")}
                        for error in exception.errors()
                    ]
                }
                if allowed
                else {"detail": "Too many login attempts"}
            ),
        )
    if is_oidc_callback:
        auth: AuthService = get_identity(request.app).auth_service
        abuse_keys = auth.browser_abuse_keys(request)
        allowed = await auth.allow_callback(
            abuse_keys.browser_key,
            abuse_keys.network_key,
        )
        consumed_transaction_id = await auth.replace_login_transaction(
            request.cookies.get(OIDC_TRANSACTION_COOKIE)
        )
        if consumed_transaction_id is not None:
            await auth.release_login(str(consumed_transaction_id))
        error_code = "validation_failed" if allowed else "rate_limited"
        await auth.audit_request_failure(
            request,
            operation="oidc.login.callback",
            error_code=error_code,
        )
        response = JSONResponse(
            status_code=422 if allowed else 429,
            content=(
                {
                    "detail": [
                        {"loc": error.get("loc"), "type": error.get("type")}
                        for error in exception.errors()
                    ]
                }
                if allowed
                else {"detail": "Too many callback attempts"}
            ),
        )
        auth.clear_transaction_cookie(response)
        return response
    await _audit_workspace_failure(request, "validation_failed")
    if request.method != "PUT" or "/secrets/" not in request.url.path:
        return await default_validation_error_handler(request, exception)
    redacted_errors = [
        {key: value for key, value in error.items() if key not in {"input", "ctx"}}
        for error in exception.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": redacted_errors})


async def _not_found_error_handler(
    request: Request,
    _exception: Exception,
) -> JSONResponse:
    await _audit_workspace_failure(request, "not_found")
    await _audit_auth_failure(request, "not_found")
    return JSONResponse(status_code=404, content={"detail": "Not found"})


async def _capability_denied_error_handler(
    request: Request,
    _exception: Exception,
) -> JSONResponse:
    await _audit_workspace_failure(request, "capability_denied")
    return JSONResponse(status_code=403, content={"detail": "Forbidden"})


async def _disabled_user_error_handler(
    request: Request,
    _exception: Exception,
) -> JSONResponse:
    await _audit_workspace_failure(request, "disabled_user")
    return JSONResponse(status_code=401, content={"detail": "Authentication required"})


async def _identity_invariant_error_handler(
    request: Request,
    _exception: Exception,
) -> JSONResponse:
    await _audit_workspace_failure(request, "identity_invariant")
    return JSONResponse(
        status_code=409, content={"detail": "Identity operation failed"}
    )


async def _http_error_handler(request: Request, exception: Exception) -> Response:
    if isinstance(exception, HTTPException):
        error_code = "not_found" if exception.status_code == 404 else "http_error"
        await _audit_workspace_failure(request, error_code)
        await _audit_auth_failure(request, error_code)
        return await default_http_exception_handler(request, exception)
    raise exception


async def _audit_workspace_failure(
    request: Request,
    error_code: str,
) -> None:
    if getattr(request.state, "auth_failure_audited", False):
        return
    metadata = workspace_failure_metadata(request)
    if metadata is None:
        return
    operation, workspace_id, resource_type, resource_id = metadata
    await get_identity(request.app).auth_service.audit_request_failure(
        request,
        operation=operation,
        error_code=error_code,
        workspace_id=workspace_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )


async def _audit_auth_failure(request: Request, error_code: str) -> None:
    if getattr(request.state, "auth_failure_audited", False):
        return
    if request.url.path.startswith("/v1/auth/") and not request.url.path.endswith(
        ("/oidc/login", "/oidc/callback")
    ):
        await get_identity(request.app).auth_service.audit_request_failure(
            request,
            operation="auth.session.request",
            error_code=error_code,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    database = create_database(resolved_settings.resolved_database_url)

    # Placeholder replaced after the FastAPI app exists so PAT middleware can
    # close over the parent application state.
    mcp_http_app_holder: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        mcp_http_app = mcp_http_app_holder["app"]
        owner_lease: ApiOwnerLease | None = None
        if resolved_settings.require_single_api_owner:
            owner_lease = ApiOwnerLease(
                resolved_settings.workspace / ".notarius-api-owner.lock"
            )
            owner_lease.acquire()
        try:
            async with mcp_http_app.lifespan(app):  # type: ignore[attr-defined]
                registry = build_plugin_registry(builtin_plugins())
                s3_access_key_id: str | None = None
                if resolved_settings.s3_access_key_id is not None:
                    configured_access_key_id = (
                        resolved_settings.s3_access_key_id.get_secret_value().strip()
                    )
                    if configured_access_key_id != "":
                        s3_access_key_id = configured_access_key_id
                s3_secret_access_key: str | None = None
                if resolved_settings.s3_secret_access_key is not None:
                    configured_secret_access_key = (
                        resolved_settings.s3_secret_access_key.get_secret_value()
                    )
                    if configured_secret_access_key != "":
                        s3_secret_access_key = configured_secret_access_key
                s3_endpoint_url = resolved_settings.s3_endpoint_url
                if s3_endpoint_url == "":
                    s3_endpoint_url = None
                storage = create_file_storage(
                    backend=resolved_settings.storage_backend,
                    local_root=resolved_settings.workspace / "objects",
                    s3_endpoint_url=s3_endpoint_url,
                    s3_region=resolved_settings.s3_region,
                    s3_access_key_id=s3_access_key_id,
                    s3_secret_access_key=s3_secret_access_key,
                    s3_force_path_style=resolved_settings.s3_force_path_style,
                )
                saved_graphs = SavedGraphService(
                    lambda: SqlAlchemySavedGraphUnitOfWork(database.sessions),
                    registry,
                )
                collaboration = CollaborationService(
                    lambda: SqlAlchemyUnitOfWork(database.sessions),
                    registry,
                    command_hmac_key=resolved_settings.resolved_command_hmac_key(),
                    command_hmac_key_version=resolved_settings.command_hmac_key_version,
                    saved_graphs=saved_graphs,
                )
                node_secrets = NodeSecretService(
                    unit_of_work_factory=lambda: SqlAlchemyUnitOfWork(database.sessions),
                    plugin_registry=registry,
                    encryption_key=resolved_settings.credential_encryption_key,
                )
                components = build_workbench_components(
                    plugin_registry=registry,
                    workspace=resolved_settings.workspace,
                    unit_of_work=SqlAlchemyUnitOfWork(database.sessions),
                    storage=storage,
                    execution_backend=resolved_settings.execution_backend,
                    map_max_concurrency=resolved_settings.map_max_concurrency,
                    prefect_task_retries=resolved_settings.prefect_task_retries,
                    prefect_task_retry_delay_seconds=(
                        resolved_settings.prefect_task_retry_delay_seconds
                    ),
                    storage_backend=resolved_settings.storage_backend,
                    bucket=resolved_settings.storage_bucket,
                    saved_graphs=saved_graphs,
                    node_secrets=node_secrets,
                )
                await components.execution_history.interrupt_all_active()
                # Migration 0009 backfills heads; refuse to serve if any graph still lacks one.
                await collaboration.verify_every_graph_has_head()
                resources = AppResources(
                    plugin_registry=components.plugin_registry,
                    uploads=components.uploads,
                    graph_modules=components.modules,
                    run_graph=components.run_graph,
                    execution_manager=components.execution_manager,
                    execution_history=components.execution_history,
                    materializations=components.materializations,
                    presenter=components.presenter,
                    artifacts=components.artifacts,
                    saved_graphs=saved_graphs,
                    collaboration=collaboration,
                    node_secrets=node_secrets,
                    graph_room_hub=graph_room_hub,
                )
                resources.execution_manager.bind_room_publisher(
                    ActiveExecutionRoomPublisher(resources.graph_room_hub)
                )
                app.state.resources = resources

                async def cleanup_expired_auth_data() -> None:
                    while True:
                        await asyncio.sleep(
                            resolved_settings.auth_cleanup_interval_seconds
                        )
                        try:
                            await auth_service.cleanup_expired()
                        except asyncio.CancelledError:
                            raise
                        except Exception as error:
                            logger.warning(
                                "auth_cleanup_failed operation=cleanup_expired "
                                "error_class=%s",
                                type(error).__name__,
                            )
                            continue

                cleanup_task = asyncio.create_task(cleanup_expired_auth_data())
                try:
                    yield
                finally:
                    cleanup_task.cancel()
                    await asyncio.gather(cleanup_task, return_exceptions=True)
                    await resources.cleanup()
                    del app.state.resources
                    await database.dispose()
        finally:
            if owner_lease is not None:
                owner_lease.release()

    application = FastAPI(
        title="Notarius API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    mcp_http_app = create_mounted_mcp_app(application)
    mcp_http_app_holder["app"] = mcp_http_app
    application.mount("/mcp", mcp_http_app)

    async def browser_abuse_cookie_boundary(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        if "/auth/oidc/" in request.url.path:
            browser_key = getattr(request.state, "auth_browser_key", None)
            if isinstance(browser_key, str):
                get_identity(request.app).auth_service.set_browser_abuse_cookie(
                    response,
                    browser_key,
                )
        return response

    application.middleware("http")(browser_abuse_cookie_boundary)

    def identity_uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(database.sessions)

    identity_service = IdentityService(identity_uow_factory)
    auth_service = AuthService(
        settings=resolved_settings,
        unit_of_work_factory=identity_uow_factory,
        identity_service=identity_service,
    )
    graph_room_hub = GraphRoomHub(
        presence_ttl_seconds=resolved_settings.graph_room_presence_ttl_seconds,
        presence_max_updates_per_second=(
            resolved_settings.graph_room_presence_max_updates_per_second
        ),
    )
    application.state.settings = resolved_settings
    application.state.identity = AppIdentity(
        identity_uow_factory=identity_uow_factory,
        identity_service=identity_service,
        auth_service=auth_service,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "Accept-Ranges",
            "Cache-Control",
            "Content-Length",
            "Content-Range",
            "ETag",
        ],
    )
    application.add_exception_handler(
        RequestValidationError,
        _request_validation_error_handler,
    )
    application.add_exception_handler(HTTPException, _http_error_handler)
    application.add_exception_handler(NotFoundError, _not_found_error_handler)
    application.add_exception_handler(
        CapabilityDeniedError, _capability_denied_error_handler
    )
    application.add_exception_handler(UserDisabledError, _disabled_user_error_handler)
    application.add_exception_handler(
        IdentityInvariantError, _identity_invariant_error_handler
    )
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    application.include_router(auth_router, prefix="/v1")
    application.include_router(workspaces_router, prefix="/v1")
    application.include_router(saved_graphs_router, prefix="/v1")
    application.include_router(collaboration_router, prefix="/v1")
    application.include_router(node_secrets_router, prefix="/v1")
    application.include_router(catalog_router, prefix="/v1")
    application.include_router(uploads_router, prefix="/v1")
    application.include_router(executions_router, prefix="/v1")
    application.include_router(artifacts_router, prefix="/v1")
    return application


app = create_app()

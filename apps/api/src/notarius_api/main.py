import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
    request_validation_exception_handler as default_validation_error_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

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

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.composition import build_workbench_components
from notarius_api.settings import Settings, get_settings
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.auth.services import (
    OIDC_TRANSACTION_COOKIE,
    AuthService,
)
from notarius_api.v1.routes.auth.abuse import request_browser_key
from notarius_api.v1.routes.auth.views import router as auth_router
from notarius_api.v1.routes.artifacts.views import router as artifacts_router
from notarius_api.v1.routes.catalog.views import router as catalog_router
from notarius_api.v1.routes.executions.views import router as executions_router
from notarius_api.v1.routes.node_secrets.services import NodeSecretService
from notarius_api.v1.routes.node_secrets.views import router as node_secrets_router
from notarius_api.v1.routes.saved_graphs.views import router as saved_graphs_router
from notarius_api.v1.routes.uploads.views import router as uploads_router
from notarius_api.v1.routes.workspaces.views import router as workspaces_router


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
        login_auth: AuthService = request.app.state.auth_service
        browser_key = request_browser_key(request)
        allowed = await login_auth.allow_login_start(browser_key)
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
        auth: AuthService = request.app.state.auth_service
        browser_key = request_browser_key(request)
        allowed = await auth.allow_callback(browser_key)
        consumed = await auth.replace_login_transaction(
            request.cookies.get(OIDC_TRANSACTION_COOKIE)
        )
        if consumed:
            await auth.release_login(browser_key)
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
    if request.url.path.startswith("/v1/workspaces"):
        await request.app.state.auth_service.audit_request_failure(
            request,
            operation="workspace.request",
            error_code="validation_failed",
        )
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
        await _audit_workspace_failure(request, "http_error")
        await _audit_auth_failure(request, "http_error")
        return await default_http_exception_handler(request, exception)
    raise exception


async def _audit_workspace_failure(request: Request, error_code: str) -> None:
    if request.url.path.startswith("/v1/workspaces"):
        await request.app.state.auth_service.audit_request_failure(
            request,
            operation="workspace.request",
            error_code=error_code,
        )


async def _audit_auth_failure(request: Request, error_code: str) -> None:
    if request.url.path.startswith("/v1/auth/") and not request.url.path.endswith(
        ("/oidc/login", "/oidc/callback")
    ):
        await request.app.state.auth_service.audit_request_failure(
            request,
            operation="auth.session.request",
            error_code=error_code,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    database = create_database(resolved_settings.resolved_database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
        await components.execution_history.interrupt_active()
        app.state.workbench_plugin_registry = components.plugin_registry
        app.state.image_uploads = components.uploads
        app.state.graph_modules = components.modules
        app.state.run_graph = components.run_graph
        app.state.execution_manager = components.execution_manager
        app.state.execution_history = components.execution_history
        app.state.materializations = components.materializations
        app.state.run_result_presenter = components.presenter
        app.state.artifacts = components.artifacts
        app.state.saved_graphs = saved_graphs
        app.state.node_secrets = node_secrets

        async def cleanup_expired_auth_data() -> None:
            while True:
                await asyncio.sleep(resolved_settings.auth_cleanup_interval_seconds)
                try:
                    await auth_service.cleanup_expired()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning(
                        "auth_cleanup_failed operation=cleanup_expired error_class=%s",
                        type(error).__name__,
                    )
                    continue

        cleanup_task = asyncio.create_task(cleanup_expired_auth_data())
        try:
            yield
        finally:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
            await components.execution_manager.shutdown()
            await components.artifacts.close()
            del app.state.node_secrets
            del app.state.saved_graphs
            del app.state.artifacts
            del app.state.run_result_presenter
            del app.state.materializations
            del app.state.run_graph
            del app.state.execution_manager
            del app.state.execution_history
            del app.state.graph_modules
            del app.state.image_uploads
            del app.state.workbench_plugin_registry
            await database.dispose()

    application = FastAPI(
        title="Notarius API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def identity_uow_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(database.sessions)

    identity_service = IdentityService(identity_uow_factory)
    auth_service = AuthService(
        settings=resolved_settings,
        unit_of_work_factory=identity_uow_factory,
        identity_service=identity_service,
    )
    application.state.settings = resolved_settings
    application.state.identity_uow_factory = identity_uow_factory
    application.state.identity_service = identity_service
    application.state.auth_service = auth_service
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
    # Phase 1 gates legacy global resources by browser authentication only.
    # Phase 2 replaces these routes with workspace-qualified authorization.
    application.include_router(
        saved_graphs_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    application.include_router(
        node_secrets_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    application.include_router(
        catalog_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    application.include_router(
        uploads_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    application.include_router(
        executions_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    application.include_router(
        artifacts_router,
        prefix="/v1",
        dependencies=[Depends(browser_actor)],
    )
    return application


app = create_app()

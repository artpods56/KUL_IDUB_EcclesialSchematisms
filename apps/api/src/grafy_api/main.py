import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
    request_validation_exception_handler as default_validation_error_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from grafy_api.health import HealthResponse, health, readiness
from grafy_core.application.collaboration import CollaborationService
from grafy_core.application.modules import ModuleLibraryService
from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_core.application.saved_graphs import SavedGraphService
from grafy_core.application.templates import TemplateService
from grafy_core.application.identity import IdentityService
from grafy_core.domain.errors import (
    CapabilityDeniedError,
    IdentityInvariantError,
    NotFoundError,
    UserDisabledError,
)
from grafy_core.operators.modules import MODULE_BOUNDARY_REGISTRATIONS
from grafy_core.plugins import PluginRegistry

from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import (
    SqlAlchemySavedGraphUnitOfWork,
    SqlAlchemyUnitOfWork,
)
from grafy_api.app_state import AppIdentity, AppResources, get_identity
from grafy_api.plugin_oci import runtime_profile
from grafy_api.services.composition import build_workbench_components
from grafy_api.settings import Settings, get_settings
from grafy_api.single_owner import ApiOwnerLease
from grafy_api.storage import configured_file_storage
from grafy_api.system_plugin_loader import (
    LoadedSystemPluginDeployment,
    load_system_plugin_deployment_file,
)
from grafy_api.v1.routes.auth.services import (
    OIDC_TRANSACTION_COOKIE,
    AuthService,
)
from grafy_api.v1.routes.auth.views import router as auth_router
from grafy_api.v1.routes.artifacts.views import router as artifacts_router
from grafy_api.v1.routes.catalog.views import router as catalog_router
from grafy_api.v1.routes.modules.views import router as modules_router
from grafy_api.v1.routes.templates.views import router as templates_router
from grafy_api.v1.routes.collaboration.hub import GraphRoomHub
from grafy_api.v1.routes.collaboration.views import router as collaboration_router
from grafy_api.v1.routes.executions.views import router as executions_router
from grafy_api.v1.routes.executions.runtime.plugin_docker import DockerPluginRuntime
from grafy_api.v1.routes.node_secrets.services import NodeSecretService
from grafy_api.v1.routes.node_secrets.views import router as node_secrets_router
from grafy_api.v1.routes.saved_graphs.views import (
    browser_router as graph_browser_router,
    folder_router as graph_folders_router,
    router as saved_graphs_router,
)
from grafy_api.v1.routes.uploads.views import router as uploads_router
from grafy_api.v1.routes.workspaces.views import (
    router as workspaces_router,
    workspace_failure_metadata,
)


logger = logging.getLogger(__name__)


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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owner_lease: ApiOwnerLease | None = None
        try:
            if resolved_settings.require_single_api_owner:
                owner_lease = ApiOwnerLease(
                    resolved_settings.workspace / ".grafy-api-owner.lock"
                )
                owner_lease.acquire()
            loaded_deployment = LoadedSystemPluginDeployment(
                plugins=(),
                loaded_plugins=(),
                bindings=(),
            )
            deployment_manifest = (
                resolved_settings.resolved_system_plugin_deployment_manifest
            )
            if deployment_manifest is not None:
                loaded_deployment = load_system_plugin_deployment_file(
                    deployment_manifest
                )

            registry = PluginRegistry()
            registry.register_module_boundaries(MODULE_BOUNDARY_REGISTRATIONS)
            for plugin in loaded_deployment.plugins:
                registry.install(plugin)
            registry.freeze()
            storage = configured_file_storage(resolved_settings)
            saved_graphs = SavedGraphService(
                lambda: SqlAlchemySavedGraphUnitOfWork(database.sessions),
                registry,
            )
            module_library = ModuleLibraryService(
                lambda: SqlAlchemyUnitOfWork(database.sessions),
                registry,
            )
            plugin_releases = PluginReleaseService(
                lambda: SqlAlchemyUnitOfWork(database.sessions),
                storage,
                bucket=resolved_settings.storage_bucket,
            )
            plugin_runtime: DockerPluginRuntime | None = None
            network_policy = resolved_settings.resolved_network_policy
            if resolved_settings.plugin_runtime_enabled:
                seccomp_profile = resolved_settings.resolved_plugin_seccomp_profile
                if seccomp_profile is not None and not seccomp_profile.is_file():
                    raise RuntimeError(
                        "Configured Plugin seccomp profile is not a regular file"
                    )
                plugin_runtime = DockerPluginRuntime(
                    releases=plugin_releases,
                    storage=storage,
                    bucket=resolved_settings.storage_bucket,
                    profile=runtime_profile(
                        resolved_settings.plugin_runtime_profile,
                        native_base_image=(
                            resolved_settings.plugin_runtime_native_base_image
                        ),
                        native_base_image_digest=(
                            resolved_settings.plugin_runtime_native_base_image_digest
                        ),
                    ),
                    scratch_root=(
                        resolved_settings.workspace / "plugin-runtime" / "scratch"
                    ),
                    docker_binary=resolved_settings.plugin_docker_binary,
                    seccomp_profile=seccomp_profile,
                    max_live_sandboxes=(resolved_settings.max_live_plugin_sandboxes),
                    max_distinct_releases_per_scope=(
                        resolved_settings.max_distinct_plugin_releases_per_graph
                    ),
                    max_sandbox_variants_per_scope=(
                        resolved_settings.max_plugin_sandbox_variants_per_execution
                    ),
                    egress_policy=(
                        resolved_settings.resolved_plugin_egress_policy
                    ),
                    network_policy=network_policy,
                )
                await plugin_runtime.recover_orphans()
                for profile in network_policy.profiles:
                    logger.info(
                        "network_profile plane=%s name=%s mode=%s digest=%s",
                        profile.plane.value,
                        profile.name,
                        profile.mode.value,
                        profile.policy_digest,
                    )
            templates = TemplateService(lambda: SqlAlchemyUnitOfWork(database.sessions))
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
                map_max_concurrency=resolved_settings.map_max_concurrency,
                max_active_executions=resolved_settings.max_active_executions,
                max_pending_graphs=resolved_settings.max_pending_graphs,
                max_active_plugin_invocations=(
                    resolved_settings.max_active_plugin_invocations
                ),
                storage_backend=resolved_settings.storage_backend,
                bucket=resolved_settings.storage_bucket,
                staged_upload_max_bytes=resolved_settings.staged_upload_max_bytes,
                saved_graphs=saved_graphs,
                module_library=module_library,
                plugin_releases=plugin_releases,
                plugin_runtime=plugin_runtime,
                system_host_bindings=loaded_deployment.bindings,
                loaded_system_plugins=loaded_deployment.loaded_plugins,
                node_secrets=node_secrets,
                graph_room_hub=graph_room_hub,
                network_policy=network_policy,
            )
            resources = AppResources(
                database=database,
                plugin_registry=components.plugin_registry,
                uploads=components.uploads,
                graph_modules=components.modules,
                plugin_releases=components.plugin_releases,
                module_library=module_library,
                templates=templates,
                run_graph=components.run_graph,
                execution_admission=components.execution_admission,
                execution_manager=components.execution_manager,
                execution_history=components.execution_history,
                materializations=components.materializations,
                presenter=components.presenter,
                artifacts=components.artifacts,
                saved_graphs=saved_graphs,
                collaboration=collaboration,
                node_secrets=node_secrets,
                graph_room_hub=graph_room_hub,
                plugin_invoker=components.plugin_invoker,
                plugin_runtime=components.plugin_runtime,
                release_admission=components.release_admission,
            )
            try:
                await components.execution_history.interrupt_started()
                await components.execution_manager.recover_queued()
                capacity = await resources.capacity_diagnostics()
                logger.info(
                    "capacity_diagnostics active_executions=%s "
                    "max_active_executions=%s pending_graphs=%s "
                    "max_pending_graphs=%s active_plugin_invocations=%s "
                    "max_active_plugin_invocations=%s live_plugin_sandboxes=%s "
                    "max_live_plugin_sandboxes=%s",
                    capacity.execution_admission.active_executions,
                    capacity.execution_admission.max_active_executions,
                    capacity.execution_queue.pending_graphs,
                    capacity.execution_queue.max_pending_graphs,
                    (
                        None
                        if capacity.plugin_invocations is None
                        else capacity.plugin_invocations.active_invocations
                    ),
                    (
                        None
                        if capacity.plugin_invocations is None
                        else capacity.plugin_invocations.max_active_invocations
                    ),
                    (
                        None
                        if capacity.plugin_sandboxes is None
                        else capacity.plugin_sandboxes.live_sandboxes
                    ),
                    (
                        None
                        if capacity.plugin_sandboxes is None
                        else capacity.plugin_sandboxes.max_live_sandboxes
                    ),
                )
                # Migration 0009 backfills heads; refuse to serve if any graph still lacks one.
                await collaboration.verify_every_graph_has_head()
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
            finally:
                try:
                    await resources.cleanup()
                finally:
                    if getattr(app.state, "resources", None) is resources:
                        del app.state.resources
        finally:
            try:
                await database.dispose()
            finally:
                if owner_lease is not None:
                    owner_lease.release()

    application = FastAPI(
        title="Grafy API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

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

    application.add_api_route(
        "/ready",
        readiness,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    application.include_router(auth_router, prefix="/v1")
    application.include_router(workspaces_router, prefix="/v1")
    application.include_router(graph_browser_router, prefix="/v1")
    application.include_router(graph_folders_router, prefix="/v1")
    application.include_router(saved_graphs_router, prefix="/v1")
    application.include_router(collaboration_router, prefix="/v1")
    application.include_router(node_secrets_router, prefix="/v1")
    application.include_router(catalog_router, prefix="/v1")
    application.include_router(modules_router, prefix="/v1")
    application.include_router(templates_router, prefix="/v1")
    application.include_router(uploads_router, prefix="/v1")
    application.include_router(executions_router, prefix="/v1")
    application.include_router(artifacts_router, prefix="/v1")
    return application


app = create_app()

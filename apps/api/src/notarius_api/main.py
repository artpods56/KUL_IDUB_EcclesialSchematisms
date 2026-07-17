from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_validation_error_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from notarius_core.application.saved_graphs import SavedGraphService

from notarius_persistence.database import create_database
from notarius_persistence.unit_of_work import (
    SqlAlchemySavedGraphUnitOfWork,
    SqlAlchemyUnitOfWork,
)
from notarius_storage import create_file_storage

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.composition import build_workbench_components
from notarius_api.services.node_secrets import NodeSecretService
from notarius_api.settings import Settings, get_settings
from notarius_api.v1.routes.saved_graphs import router as saved_graphs_router
from notarius_api.v1.routes.node_secrets import router as node_secrets_router
from notarius_api.v1.routes.workbench import router as workbench_router


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
    if request.method != "PUT" or "/secrets/" not in request.url.path:
        return await default_validation_error_handler(request, exception)
    redacted_errors = [
        {key: value for key, value in error.items() if key not in {"input", "ctx"}}
        for error in exception.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": redacted_errors})


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
            prefect_task_retries=resolved_settings.prefect_task_retries,
            prefect_task_retry_delay_seconds=(
                resolved_settings.prefect_task_retry_delay_seconds
            ),
            storage_backend=resolved_settings.storage_backend,
            bucket=resolved_settings.storage_bucket,
            saved_graphs=saved_graphs,
            node_secrets=node_secrets,
        )
        app.state.workbench_plugin_registry = components.plugin_registry
        app.state.image_uploads = components.uploads
        app.state.graph_modules = components.modules
        app.state.run_graph = components.run_graph
        app.state.materializations = components.materializations
        app.state.run_result_presenter = components.presenter
        app.state.artifacts = components.artifacts
        app.state.saved_graphs = saved_graphs
        app.state.node_secrets = node_secrets
        try:
            yield
        finally:
            del app.state.node_secrets
            del app.state.saved_graphs
            del app.state.artifacts
            del app.state.run_result_presenter
            del app.state.materializations
            del app.state.run_graph
            del app.state.graph_modules
            del app.state.image_uploads
            del app.state.workbench_plugin_registry
            await database.dispose()

    application = FastAPI(
        title="Notarius API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(
        RequestValidationError,
        _request_validation_error_handler,
    )
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    application.include_router(saved_graphs_router, prefix="/v1")
    application.include_router(node_secrets_router, prefix="/v1")
    application.include_router(workbench_router, prefix="/v1")
    return application


app = create_app()

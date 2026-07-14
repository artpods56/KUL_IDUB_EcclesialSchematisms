from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from notarius_core.application.saved_graphs import SavedGraphService

from notarius_persistence.database import create_database
from notarius_persistence.unit_of_work import SqlAlchemySavedGraphUnitOfWork

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.workbench import WorkbenchService
from notarius_api.settings import Settings, get_settings
from notarius_api.v1.routes.saved_graphs import router as saved_graphs_router
from notarius_api.v1.routes.workbench import router as workbench_router


class HealthResponse(BaseModel):
    status: Literal["ok"]


async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    database = create_database(resolved_settings.resolved_database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = build_plugin_registry(builtin_plugins())
        app.state.workbench = WorkbenchService(
            plugin_registry=registry,
            workspace=resolved_settings.workspace,
        )
        app.state.saved_graphs = SavedGraphService(
            lambda: SqlAlchemySavedGraphUnitOfWork(database.sessions)
        )
        try:
            yield
        finally:
            del app.state.saved_graphs
            del app.state.workbench
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
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    application.include_router(saved_graphs_router, prefix="/v1")
    application.include_router(workbench_router, prefix="/v1")
    return application


app = create_app()

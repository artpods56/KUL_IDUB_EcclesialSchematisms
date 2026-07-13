import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from notarius_api.builtins import builtin_plugins
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.services.workbench import WorkbenchService
from notarius_api.v1.routes.workbench import router as workbench_router


class HealthResponse(BaseModel):
    status: Literal["ok"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = build_plugin_registry(builtin_plugins())
    app.state.workbench = WorkbenchService(plugin_registry=registry)
    try:
        yield
    finally:
        del app.state.workbench


async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Notarius API",
        version="0.1.0",
        lifespan=lifespan,
    )
    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "NOTARIUS_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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
    application.include_router(workbench_router, prefix="/v1")
    return application


app = create_app()

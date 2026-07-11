import asyncio
import os
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faststream.nats import NatsBroker
from pydantic import BaseModel

from notarius_api import dependencies as deps
from notarius_api.v1.router import router as v1_router
from notarius_core.domain.errors import ConflictError, NotFoundError, ValidationError
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

app = FastAPI(title="Notarius Studio API")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "NOTARIUS_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthComponentResponse(BaseModel):
    status: Literal["ok", "not_configured", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: HealthComponentResponse
    nats: HealthComponentResponse | None = None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        database=HealthComponentResponse(status="ok"),
    )


@app.get("/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    return HealthResponse(
        status="ok",
        database=HealthComponentResponse(status="ok"),
    )


@app.get("/health/ready", response_model=HealthResponse)
async def health_ready(
    response: Response,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> HealthResponse:
    database = await _database_health(uow)
    nats = await _nats_health()
    if database.status == "error" or nats.status == "error":
        response.status_code = 503
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(status=status, database=database, nats=nats)


async def _database_health(uow: StudioUnitOfWorkPort) -> HealthComponentResponse:
    try:
        async with uow:
            await uow.workflow_definitions.list()
    except Exception as exc:
        return HealthComponentResponse(
            status="error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )
    return HealthComponentResponse(status="ok")


async def _nats_health() -> HealthComponentResponse:
    nats_url = os.getenv("NATS_URL")
    if nats_url is None or nats_url == "":
        return HealthComponentResponse(
            status="not_configured",
            detail="NATS_URL is not set; local outbox draining is available.",
        )

    timeout_seconds = float(os.getenv("NOTARIUS_HEALTH_NATS_TIMEOUT_SECONDS", "2"))
    try:
        await asyncio.wait_for(
            _check_nats_connection(nats_url),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return HealthComponentResponse(
            status="error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )
    return HealthComponentResponse(status="ok")


async def _check_nats_connection(nats_url: str) -> None:
    broker = NatsBroker(nats_url)
    async with broker:
        return None


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(v1_router, prefix="/v1")

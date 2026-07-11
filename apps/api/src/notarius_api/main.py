import os
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from notarius_api.v1.routes.prototype import router as prototype_router


class HealthResponse(BaseModel):
    status: Literal["ok"]


app = FastAPI(
    title="Notarius Prototype API",
    version="0.1.0",
)

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


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


app.include_router(prototype_router, prefix="/v1")

import asyncio
from typing import Literal

import httpx
from fastapi import HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text

from grafy_api.app_state import get_app_settings, get_resources


class HealthResponse(BaseModel):
    status: Literal["ok"]


def health() -> HealthResponse:
    return HealthResponse(status="ok")


async def readiness(request: Request) -> HealthResponse:
    try:
        async with asyncio.timeout(3.0):
            resources = get_resources(request.app)
            settings = get_app_settings(request.app)

            async with resources.database.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))

            if settings.execution_backend == "prefect":
                prefect_api_url = settings.prefect_api_url
                if prefect_api_url is None:
                    raise RuntimeError("Prefect API URL is not configured")

                async with httpx.AsyncClient(
                    timeout=2.0,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    response = await client.get(f"{prefect_api_url.rstrip('/')}/health")
                response.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable",
        ) from exc

    return HealthResponse(status="ok")

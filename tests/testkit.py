from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from notarius_api.main import create_app

from notarius_api.settings import Settings
from starlette.testclient import TestClient

type AppDependency = Callable[..., Any]
type DependencyOverride = Callable[..., Any]


@dataclass(slots=True)
class AsyncApiHarness:
    app: FastAPI
    client: AsyncClient

    def override(
        self,
        dependency: AppDependency,
        override: DependencyOverride,
    ) -> None:
        self.app.dependency_overrides[dependency] = override

    def clear_override(self, dependency: AppDependency) -> None:
        _ = self.app.dependency_overrides.pop(dependency, None)

def app_with_overrides(
    *,
    settings: Settings | None = None,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> FastAPI:
    resolved_settings = settings if settings is not None else Settings()

    app = create_app(resolved_settings)

    if overrides:
        app.dependency_overrides.update(overrides)

    return app


def client_with_overrides(
    *,
    settings: Settings | None = None,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> TestClient:
    """Simple httpx client for synchronous testing."""
    return TestClient(
        app_with_overrides(settings=settings, overrides=overrides)
    )

@asynccontextmanager
async def async_client_with_overrides(
    *,
    settings: Settings,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> AsyncIterator[AsyncApiHarness]:
    """Httpx client for asynchronous testing."""
    app = app_with_overrides(settings=settings, overrides=overrides)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)

        async with AsyncClient(
                transport=transport,
                base_url="http://test"
        ) as client:
            yield AsyncApiHarness(
                app=app,
                client=client
            )

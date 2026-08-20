"""Test harness for the Grafy API.

This module provides typed, reusable helpers that replace the two
distinct injection mechanisms with a single coherent harness:

* ``create_app(settings)`` — application composition (lifespan resources)
* ``app.dependency_overrides`` — request-scoped FastAPI ``Depends()`` graph

The harness keeps those concerns separate so tests can replace
either one without confusion.
"""

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from grafy_api.main import create_app
from grafy_api.settings import Settings

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

type AppDependency = Callable[..., Any]
"""A FastAPI dependency callable used as ``Depends(callable)`` in a route."""

type DependencyOverride = Callable[..., Any]
"""A replacement callable registered via ``app.dependency_overrides[dep]``."""

# ---------------------------------------------------------------------------
# Async API harness
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AsyncApiHarness:
    """A running FastAPI application with an async HTTP client.

    The harness enters the ASGI lifespan before yielding so
    ``app.state.resources`` is fully populated.
    """

    app: FastAPI
    """The live FastAPI application (lifespan has already run)."""

    client: AsyncClient
    """An ``httpx.AsyncClient`` backed by ``ASGITransport`` for this app."""

    def override(
        self,
        dependency: AppDependency,
        override: DependencyOverride,
    ) -> None:
        """Replace one FastAPI dependency for the lifetime of this harness."""
        self.app.dependency_overrides[dependency] = override

    def clear_override(self, dependency: AppDependency) -> None:
        """Remove an earlier override, restoring the original dependency."""
        self.app.dependency_overrides.pop(dependency, None)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def app_with_overrides(
    *,
    settings: Settings,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> FastAPI:
    """Create a FastAPI app with the given *settings* and optional overrides.

    ``settings`` is forwarded directly to ``create_app()``, so every
    application-level resource (database, storage, plugin registry)
    is built from those values.

    ``overrides`` are bulk-applied to ``app.dependency_overrides``
    after the app is created.  They only affect the FastAPI
    ``Depends(...)`` graph, not the resources constructed during
    lifespan.
    """
    app = create_app(settings)

    if overrides:
        app.dependency_overrides.update(overrides)

    return app


# ---------------------------------------------------------------------------
# Async client context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def async_client_with_overrides(
    *,
    settings: Settings,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> AsyncIterator[AsyncApiHarness]:
    """Create an app, enter its lifespan, and yield a typed harness.

    Usage::

        async with async_client_with_overrides(settings=test_settings) as api:
            api.override(run_graph_service, lambda: fake_run_graph)
            response = await api.client.get("/ready")
    """
    app = app_with_overrides(
        settings=settings,
        overrides=overrides,
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield AsyncApiHarness(
                app=app,
                client=client,
            )


# ---------------------------------------------------------------------------
# Synchronous TestClient context manager
# ---------------------------------------------------------------------------


@contextmanager
def client_with_overrides(
    *,
    settings: Settings,
    overrides: dict[AppDependency, DependencyOverride] | None = None,
) -> Iterator[TestClient]:
    """Create an app, enter its lifespan, and yield a ``TestClient``.

    The context manager form is required because Starlette only
    invokes lifespan events when ``TestClient`` is entered via
    ``with``.  Constructing ``TestClient(app)`` without entering it
    skips startup, so ``app.state.resources`` would be missing.

    Usage::

        with client_with_overrides(settings=test_settings) as client:
            response = client.get("/ready")
    """
    app = app_with_overrides(
        settings=settings,
        overrides=overrides,
    )

    with TestClient(app) as client:
        yield client

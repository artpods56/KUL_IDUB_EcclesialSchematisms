"""Shared test identity constants and helpers for the Grafy API test suite."""

from uuid import UUID

from fastapi import FastAPI

from grafy_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from grafy_persistence.database import create_database
from grafy_persistence.orm import metadata
from grafy_persistence.unit_of_work import SqlAlchemySavedGraphUnitOfWork

from grafy_api.v1.routes.auth.dependencies import browser_actor


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")
TEST_USER_ID = UUID(int=1)
TEST_COMMAND_HMAC_KEY = "test-api-command-hmac-key"


def workspace_api_path(suffix: str) -> str:
    normalized = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"/v1/workspaces/{WORKSPACE_ID}{normalized}"


def install_browser_actor_override(application: FastAPI) -> None:
    def test_browser_actor() -> ActorContext:
        return ActorContext(
            user_id=TEST_USER_ID,
            credential_reference="test-session",
        )

    application.dependency_overrides[browser_actor] = test_browser_actor


async def create_schema(database_url: str) -> None:
    """Create the database schema and seed the shared test user/workspace."""

    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemySavedGraphUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.identity.add_user(
                User(
                    id=TEST_USER_ID,
                    email="owner@example.test",
                    display_name="Owner",
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_ID,
                    slug="local",
                    name="Local workspace",
                    kind="shared",
                )
            )
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=WORKSPACE_ID,
                    user_id=TEST_USER_ID,
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.commit()
    finally:
        await database.dispose()


__all__ = [
    "TEST_COMMAND_HMAC_KEY",
    "TEST_USER_ID",
    "WORKSPACE_ID",
    "create_schema",
    "install_browser_actor_override",
    "workspace_api_path",
]
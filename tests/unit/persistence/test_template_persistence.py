from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from notarius_core.domain.identity import User, Workspace, WorkspaceKind
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphDocument
from notarius_core.domain.templates import Template, TemplateState
from notarius_persistence.database import Database, create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000801")
OTHER_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000802")
USER_ID = UUID("00000000-0000-0000-0000-000000000803")


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = create_database(f"sqlite+aiosqlite:///{tmp_path / 'templates.sqlite3'}")
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.identity.add_user(User(id=USER_ID))
        await unit_of_work.identity.add_workspace(
            Workspace(
                id=WORKSPACE_ID,
                slug="my-graphs",
                name="My graphs",
                kind=WorkspaceKind.SHARED,
            )
        )
        await unit_of_work.identity.add_workspace(
            Workspace(
                id=OTHER_WORKSPACE_ID,
                slug="team-graphs",
                name="Team graphs",
                kind=WorkspaceKind.SHARED,
            )
        )
        await unit_of_work.commit()
    try:
        yield database
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_template_repository_scopes_searches_and_persists_snapshot_metadata(
    database: Database,
) -> None:
    source = SavedGraph(
        workspace_id=WORKSPACE_ID,
        created_by_user_id=USER_ID,
        name="Source graph",
        document=SavedGraphDocument(),
    )
    template = Template(
        workspace_id=WORKSPACE_ID,
        source_graph_id=source.id,
        source_revision=source.revision,
        source_graph_name=source.name,
        snapshot_document=source.document,
        created_by_user_id=USER_ID,
        name="Research starter",
        description="Prepare a field survey",
    )
    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        await unit_of_work.graphs.add(source)
        await unit_of_work.graphs.add_revision(source.snapshot())
        await unit_of_work.templates.add(template)
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert await unit_of_work.templates.get(WORKSPACE_ID, template.id) is not None
        assert await unit_of_work.templates.get(OTHER_WORKSPACE_ID, template.id) is None
        assert await unit_of_work.templates.list(
            WORKSPACE_ID,
            query="survey",
            include_archived=False,
        ) == [template]
        assert (
            await unit_of_work.templates.list(
                OTHER_WORKSPACE_ID,
                query=None,
                include_archived=True,
            )
            == []
        )
        stored_template = await unit_of_work.templates.get(WORKSPACE_ID, template.id)
        assert stored_template is not None
        stored_template.archive()
        await unit_of_work.commit()

    async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
        assert (
            await unit_of_work.templates.list(
                WORKSPACE_ID,
                query=None,
                include_archived=False,
            )
            == []
        )
        archived = await unit_of_work.templates.list(
            WORKSPACE_ID,
            query=None,
            include_archived=True,
        )
        assert len(archived) == 1
        assert archived[0].state is TemplateState.ARCHIVED

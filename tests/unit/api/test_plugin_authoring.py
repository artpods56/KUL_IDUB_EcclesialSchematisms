import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from typing_extensions import override

from grafy_api.plugin_authoring import (
    PluginAuthoringConflictError,
    PluginAuthoringService,
)
from grafy_api.plugin_oci import PluginOciImageBuilder
from grafy_api.plugin_publication import (
    PluginPublicationConflictError,
    PluginPublicationWorkflow,
)
from grafy_api.plugin_publishing import PluginDirectoryPublisher
from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_core.domain.errors import UserDisabledError
from grafy_core.domain.plugin_releases import (
    PluginCatalogManifest,
    PluginRuntimeArtifact,
)
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork
from grafy_storage import LocalFileObjectStore
from tests.support.identity import (
    TEST_USER_ID,
    WORKSPACE_ID as SEEDED_WORKSPACE_ID,
    create_schema,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ID = SEEDED_WORKSPACE_ID
ACTOR_ID = TEST_USER_ID


class RecordingImageBuilder(PluginOciImageBuilder):
    def __init__(self) -> None:
        self.build_count = 0

    @override
    async def build_and_store(
        self,
        *,
        workspace_id: UUID,
        catalog: PluginCatalogManifest,
        source_archive: bytes,
        source_digest: str,
        contract_digest: str,
        profile_digest: str,
    ) -> PluginRuntimeArtifact:
        del source_archive
        self.build_count += 1
        return PluginRuntimeArtifact(
            object_key=(
                f"plugin-releases/{workspace_id}/{catalog.slug}/runtime/"
                f"{source_digest}.oci.tar"
            ),
            archive_digest=source_digest,
            manifest_digest=contract_digest,
            config_digest=profile_digest,
        )


def test_agent_authoring_scaffolds_reviews_fences_and_uses_shared_publisher(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'authoring.sqlite3'}"
    asyncio.run(create_schema(database_url))
    database = create_database(database_url)
    storage = LocalFileObjectStore(tmp_path / "objects")
    releases = PluginReleaseService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        storage,
        bucket="authoring-test",
    )
    image_builder = RecordingImageBuilder()
    plugin_root = tmp_path / "plugins"
    publication = PluginPublicationWorkflow(
        PluginDirectoryPublisher(
            (plugin_root,),
            runtime_profile="python-uv",
        ),
        image_builder,
        releases,
    )
    authoring = PluginAuthoringService(
        authoring_root=plugin_root,
        allowed_roots=(plugin_root,),
        sdk_project=REPOSITORY_ROOT / "libs" / "core",
        publication=publication,
        releases=releases,
        storage=storage,
        bucket="authoring-test",
    )

    with pytest.raises(UserDisabledError, match="is disabled"):
        asyncio.run(
            publication.publish(
                workspace_id=WORKSPACE_ID,
                directory=plugin_root / "missing",
                expected_slug="generated-notes",
                published_by_user_id=UUID(int=999),
            )
        )
    assert image_builder.build_count == 0

    reservation = authoring.scaffold(
        workspace_id=WORKSPACE_ID,
        slug="generated-notes",
        operator_slug="create",
        title="Generated notes",
        actor_user_id=ACTOR_ID,
    )
    project = reservation.project_directory
    assert (project / "pyproject.toml").is_file()
    assert (project / "uv.lock").is_file()
    assert (project / "src" / "grafy_plugin" / "__init__.py").is_file()
    assert (project / "tests" / "test_plugin.py").is_file()
    assert list((project / "wheels").glob("grafy_core-*.whl"))
    assert "generated.node" not in (
        project / "src" / "grafy_plugin" / "nodes.py"
    ).read_text(encoding="utf-8")

    with pytest.raises(PluginAuthoringConflictError, match="active authoring"):
        authoring.reserve(
            workspace_id=WORKSPACE_ID,
            slug="generated-notes",
            actor_user_id=ACTOR_ID,
        )

    first_review = asyncio.run(
        authoring.review(
            workspace_id=WORKSPACE_ID,
            slug="generated-notes",
            actor_user_id=ACTOR_ID,
            session_id=reservation.session_id,
        )
    )
    assert first_review.base_revision is None
    assert first_review.changes
    assert {change.kind for change in first_review.changes} == {"added"}
    assert "working-copy/src/grafy_plugin/nodes.py" in first_review.unified_diff
    assert first_review.node_contract_changed is True
    assert first_review.artifact_contract_changed is True
    assert first_review.capabilities_changed is True
    assert first_review.runtime_profile_changed is True

    nodes = project / "src" / "grafy_plugin" / "nodes.py"
    nodes.write_text(
        nodes.read_text(encoding="utf-8") + "\n# changed after review\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PluginPublicationConflictError,
        match="changed after review",
    ):
        asyncio.run(
            authoring.publish(
                workspace_id=WORKSPACE_ID,
                slug="generated-notes",
                actor_user_id=ACTOR_ID,
                session_id=reservation.session_id,
            )
        )
    assert asyncio.run(releases.list_current(WORKSPACE_ID)) == []

    second_review = asyncio.run(
        authoring.review(
            workspace_id=WORKSPACE_ID,
            slug="generated-notes",
            actor_user_id=ACTOR_ID,
            session_id=reservation.session_id,
        )
    )
    agent_release = asyncio.run(
        authoring.publish(
            workspace_id=WORKSPACE_ID,
            slug="generated-notes",
            actor_user_id=ACTOR_ID,
            session_id=reservation.session_id,
        )
    )
    assert agent_release.revision == 1
    assert agent_release.source_digest == second_review.source_digest
    assert agent_release.published_by_user_id == ACTOR_ID
    assert not (project / ".grafy" / "authoring.json").exists()

    human_release = asyncio.run(
        publication.publish(
            workspace_id=WORKSPACE_ID,
            directory=project,
            expected_slug="generated-notes",
            published_by_user_id=ACTOR_ID,
        )
    )
    assert human_release == agent_release
    assert image_builder.build_count == 2
    asyncio.run(database.dispose())

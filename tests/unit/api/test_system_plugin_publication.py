from hashlib import sha256
from pathlib import Path

import pytest
from typing_extensions import override

from grafy_api.plugin_admission import (
    ReleaseExecutionAdmission,
    isolated_release_admission,
)
from grafy_api.plugin_egress import (
    PluginEgressBrokerPolicy,
    PluginEgressDestination,
)
from grafy_api.network_policy import legacy_network_policy
from grafy_api.plugin_oci import runtime_profile
from grafy_api.plugin_oci import PluginOciImageBuilder
from grafy_api.plugin_publication import SystemPluginPublicationWorkflow
from grafy_api.plugin_publishing import (
    PluginPublishingError,
    VerifiedPluginCandidate,
)
from grafy_api.system_host_bindings import SystemHostPluginBinding
from grafy_api.system_plugin_inventory import (
    CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH,
    SystemPluginInventory,
    load_system_plugin_inventory,
)
from grafy_core.application.plugin_releases import PluginReleaseService
from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.domain.plugin_releases import (
    PlatformPluginActor,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginExecutionPolicy,
    PluginNodeContract,
    PluginNodeHttpEgressContract,
    PluginReleaseNamespace,
    PluginReleaseError,
    PluginReleaseScope,
    PluginRuntimeArtifact,
    plugin_contract_digest,
)
from grafy_persistence.database import create_database
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork
from grafy_storage import LocalFileObjectStore
from tests.support.identity import TEST_USER_ID, WORKSPACE_ID, create_schema


class RecordingSystemImageBuilder(PluginOciImageBuilder):
    def __init__(self) -> None:
        self.namespaces: list[PluginReleaseNamespace] = []
        self.loader_targets: list[str] = []

    @override
    async def build_and_store(
        self,
        *,
        namespace: PluginReleaseNamespace,
        candidate: VerifiedPluginCandidate,
    ) -> PluginRuntimeArtifact:
        self.namespaces.append(namespace)
        self.loader_targets.append(candidate.loader_target)
        return PluginRuntimeArtifact(
            object_key=(
                f"plugin-releases/{namespace.storage_path}/{candidate.catalog.slug}/"
                f"runtime/{candidate.source_digest}.oci.tar"
            ),
            archive_digest=candidate.source_digest,
            manifest_digest=plugin_contract_digest(candidate.catalog),
            config_digest="a" * 64,
        )


def _catalog(
    *,
    slug: str = "builtin.text",
    operator_id: str = "text.echo",
    required_capabilities: tuple[PluginRuntimeCapability, ...] = (),
) -> PluginCatalogManifest:
    http_egress = None
    if PluginRuntimeCapability.NETWORK_EGRESS in set(required_capabilities):
        http_egress = PluginNodeHttpEgressContract(configured_inputs=("base_url",))
    return PluginCatalogManifest(
        slug=slug,
        title=slug,
        nodes=(
            PluginNodeContract(
                operator_id=operator_id,
                operator_version=1,
                title="Echo",
                description="Echo a value.",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(),
                outputs=(),
                required_capabilities=required_capabilities,
                http_egress=http_egress,
            ),
        ),
    )


def _verified(
    source: bytes,
    *,
    catalog: PluginCatalogManifest | None = None,
    capabilities: PluginCapabilityManifest | None = None,
    loader_target: str = "grafy_plugin_text.plugin:TEXT",
) -> VerifiedPluginCandidate:
    return VerifiedPluginCandidate(
        catalog=catalog or _catalog(),
        capabilities=capabilities or PluginCapabilityManifest(),
        loader_target=loader_target,
        source_archive=source,
        lock_digest=sha256(b"lock").hexdigest(),
        runtime_profile="python-uv",
    )


def _workflow(
    image_builder: RecordingSystemImageBuilder,
    releases: PluginReleaseService,
    inventory: SystemPluginInventory,
    *,
    bindings: tuple[SystemHostPluginBinding, ...] = (),
) -> SystemPluginPublicationWorkflow:
    return SystemPluginPublicationWorkflow(
        image_builder,
        releases,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
            system_host_bindings=bindings,
        ),
        inventory,
    )


@pytest.mark.asyncio
async def test_system_publication_stages_then_explicitly_promotes_and_rolls_back(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'system.sqlite3'}"
    await create_schema(database_url)
    database = create_database(database_url)
    releases = PluginReleaseService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        LocalFileObjectStore(tmp_path / "objects"),
        bucket="plugins",
    )
    image_builder = RecordingSystemImageBuilder()
    inventory = load_system_plugin_inventory(CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH)
    workflow = _workflow(image_builder, releases, inventory)
    actor = PlatformPluginActor("ci:system-release")

    first = await workflow.stage_verified(
        _verified(b"first"),
        platform_actor=actor,
    )
    second = await workflow.stage_verified(
        _verified(b"second"),
        platform_actor=actor,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert first.runtime_artifact is not None
    assert first.contract_digest == plugin_contract_digest(first.catalog)
    assert await releases.list_current_system() == []
    assert all(namespace.workspace_id is None for namespace in image_builder.namespaces)
    assert image_builder.loader_targets == [
        inventory.entry_for(first.slug).loader_target,
        inventory.entry_for(second.slug).loader_target,
    ]

    with pytest.raises(PluginPublishingError, match="exact deployment host binding"):
        await workflow.promote(
            slug=second.slug,
            revision=second.revision,
            platform_actor=actor,
            expected_generation=0,
        )
    assert await releases.list_current_system() == []

    inventory_entry = inventory.entry_for(second.slug)
    second_binding = SystemHostPluginBinding.from_release(
        second,
        selection_generation=1,
        loader_target=inventory_entry.loader_target,
        host_build_digest="f" * 64,
    )
    promotion_workflow = _workflow(
        image_builder,
        releases,
        inventory,
        bindings=(second_binding,),
    )
    selected = await promotion_workflow.promote(
        slug=second.slug,
        revision=second.revision,
        platform_actor=actor,
        expected_generation=0,
    )
    assert selected.selected_release_id == second.id
    assert selected.selected_revision == 2
    assert selected.generation == 1
    assert await releases.list_current_system() == [second]

    selected_again = await promotion_workflow.promote(
        slug=second.slug,
        revision=second.revision,
        platform_actor=actor,
        expected_generation=selected.generation,
    )
    assert selected_again.generation == 1

    mismatched_workflow = _workflow(
        image_builder,
        releases,
        inventory,
        bindings=(second_binding.model_copy(update={"selection_generation": 2}),),
    )
    with pytest.raises(PluginPublishingError, match="generation"):
        await mismatched_workflow.promote(
            slug=second.slug,
            revision=second.revision,
            platform_actor=actor,
            expected_generation=selected_again.generation,
        )
    unchanged = await releases.get_selection(
        WORKSPACE_ID,
        second.slug,
        scope=PluginReleaseScope.SYSTEM,
    )
    assert unchanged is not None
    assert unchanged.selected_release_id == second.id
    assert unchanged.generation == 1

    first_binding = SystemHostPluginBinding.from_release(
        first,
        selection_generation=2,
        loader_target=inventory_entry.loader_target,
        host_build_digest="e" * 64,
    )
    rollback_workflow = _workflow(
        image_builder,
        releases,
        inventory,
        bindings=(first_binding,),
    )
    rolled_back = await rollback_workflow.promote(
        slug=first.slug,
        revision=first.revision,
        platform_actor=actor,
        expected_generation=selected.generation,
    )
    assert rolled_back.selected_release_id == first.id
    assert rolled_back.selected_revision == 1
    assert rolled_back.generation == 2
    assert await releases.list_current_system() == [first]

    await database.dispose()


@pytest.mark.asyncio
async def test_system_publication_rejects_unauthorized_identity_before_image_build(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'authority.sqlite3'}"
    await create_schema(database_url)
    database = create_database(database_url)
    releases = PluginReleaseService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        LocalFileObjectStore(tmp_path / "objects"),
        bucket="plugins",
    )
    image_builder = RecordingSystemImageBuilder()
    inventory = load_system_plugin_inventory(CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH)
    workflow = _workflow(image_builder, releases, inventory)

    with pytest.raises(PluginPublishingError, match="allowlisted prefixes"):
        await workflow.stage_verified(
            _verified(b"bad", catalog=_catalog(operator_id="external.evil.echo")),
            platform_actor=PlatformPluginActor("ci:system-release"),
        )

    assert image_builder.namespaces == []
    await database.dispose()


@pytest.mark.asyncio
async def test_isolated_llm_system_release_promotes_without_a_host_manifest(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'llm.sqlite3'}"
    await create_schema(database_url)
    database = create_database(database_url)
    releases = PluginReleaseService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        LocalFileObjectStore(tmp_path / "objects"),
        bucket="plugins",
    )
    image_builder = RecordingSystemImageBuilder()
    inventory = load_system_plugin_inventory(CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH)
    destination = PluginEgressDestination.parse("https://api.openai.com:443")
    workflow = SystemPluginPublicationWorkflow(
        image_builder,
        releases,
        isolated_release_admission(
            profile=runtime_profile("python-uv"),
            egress_policy=PluginEgressBrokerPolicy(
                broker_image="registry.example/grafy-egress@sha256:" + "a" * 64,
                destinations=(destination,),
            ),
            network_policy=legacy_network_policy(
                http_destinations=(destination,),
            ),
        ),
        inventory,
    )
    entry = inventory.entry_for("external.llm")
    candidate = _verified(
        b"llm",
        catalog=_catalog(
            slug=entry.slug,
            operator_id="llm.openai_compatible.chat_completion",
            required_capabilities=entry.capabilities,
        ),
        capabilities=PluginCapabilityManifest(capabilities=entry.capabilities),
        loader_target=entry.loader_target,
    )
    actor = PlatformPluginActor("ci:system-release")

    release = await workflow.stage_verified(candidate, platform_actor=actor)
    selection = await workflow.promote(
        slug=release.slug,
        revision=release.revision,
        platform_actor=actor,
        expected_generation=0,
    )

    assert release.execution_policy is PluginExecutionPolicy.ISOLATED_ONLY
    assert selection.selected_release_id == release.id
    assert image_builder.loader_targets == [entry.loader_target]
    await database.dispose()


@pytest.mark.asyncio
async def test_direct_workspace_publish_cannot_reuse_historical_system_identity(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'collision.sqlite3'}"
    await create_schema(database_url)
    database = create_database(database_url)
    releases = PluginReleaseService(
        lambda: SqlAlchemyUnitOfWork(database.sessions),
        LocalFileObjectStore(tmp_path / "objects"),
        bucket="plugins",
    )
    image_builder = RecordingSystemImageBuilder()
    inventory = load_system_plugin_inventory(CHECKED_IN_SYSTEM_PLUGIN_INVENTORY_PATH)
    workflow = _workflow(image_builder, releases, inventory)
    gis_entry = inventory.entry_for("external.gis")
    system_catalog = _catalog(
        slug="external.gis",
        operator_id="gis.node",
        required_capabilities=gis_entry.capabilities,
    )
    await workflow.stage_verified(
        _verified(
            b"retained-system",
            catalog=system_catalog,
            capabilities=PluginCapabilityManifest(capabilities=gis_entry.capabilities),
            loader_target=gis_entry.loader_target,
        ),
        platform_actor=PlatformPluginActor("ci:system-release"),
    )

    workspace_catalog = _catalog(slug="gis", operator_id="gis.node")
    with pytest.raises(PluginReleaseError, match="retained System Plugin identity"):
        await releases.publish(
            workspace_id=WORKSPACE_ID,
            catalog=workspace_catalog,
            capabilities=PluginCapabilityManifest(),
            source_archive=b"workspace",
            lock_digest=sha256(b"workspace-lock").hexdigest(),
            runtime_profile="python-uv",
            runtime_artifact=None,
            published_by_user_id=TEST_USER_ID,
        )

    await database.dispose()

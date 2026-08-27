from collections.abc import Mapping
from hashlib import sha256
from uuid import UUID

import pytest
from pydantic import ValidationError

from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeContract,
    PluginArtifactTypeKey,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginNodeContract,
    PluginPortContract,
    PluginPortDirection,
    PluginRelease,
    PluginDistribution,
    PluginExecutionPolicy,
    PluginReleaseNamespace,
    PluginReleaseScope,
    PluginRuntimeArtifact,
    PluginSecretInputContract,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.domain.plugin_installations import (
    InstalledPluginRelease,
    PluginInstallation,
)
from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.domain.plugin_revocations import (
    PluginReleaseRevocation,
    PluginReleaseRevocationReason,
)
from grafy_core.domain.plugin_selection import (
    PluginFamilyLifecycle,
    PluginReleaseSelection,
)
from grafy_core.artifacts import ArtifactRef, NodeConfig, NodeInput, NodeOutput
from grafy_core.domain.modules import GraphModuleDefinition
from grafy_core.nodes import NodeExecutionContext, PortShape
from grafy_core.operators.modules import MODULE_BOUNDARY_REGISTRATIONS
from grafy_plugin_arithmetic import ARITHMETIC
from grafy_core.artifact_contracts import INTEGER_VALUE, RASTER_IMAGE, TEXT_VALUE
from grafy_core.table_contracts import TABLE_DATA
from grafy_plugin_text import TEXT
from grafy_core.plugins import Plugin, PluginRegistry
from grafy_core.ports.modules import GraphModuleExecutionResult

from grafy_api.plugin_admission import (
    ReleaseExecutionAdmission,
    ReleaseExecutionRoute,
)
from grafy_api.v1.routes.catalog.models import (
    NodeRegistryResponse,
    PluginCatalogReleaseState,
    PluginNonRunnableReason,
    PluginSpecResponse,
    plugin_release_readiness,
)
from grafy_api.v1.routes.catalog.services import GraphModuleCatalogListing


WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000661")
OTHER_WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000662")


class NotesConfig(NodeConfig):
    pass


class NotesInput(NodeInput):
    pass


class NotesOutput(NodeOutput):
    pass


HOST_NOTES = Plugin(slug="notes", title="Host notes")


@HOST_NOTES.function_node(
    operator_id="notes.transform",
    version=1,
    title="Host transform",
)
async def host_notes_transform(
    config: NotesConfig,
    inputs: NotesInput,
) -> NotesOutput:
    del config, inputs
    return NotesOutput()


class UnusedModuleExecutor:
    async def execute_module(
        self,
        definition: GraphModuleDefinition,
        context: NodeExecutionContext,
        inputs: Mapping[str, ArtifactRef],
        /,
    ) -> GraphModuleExecutionResult:
        raise AssertionError(
            f"Unexpected module execution for {definition.reference} with "
            f"{len(inputs)} inputs in {context}"
        )


def _port(
    name: str,
    direction: PluginPortDirection,
    artifact_type: str,
) -> PluginPortContract:
    return PluginPortContract(
        name=name,
        direction=direction,
        artifact_type=PluginArtifactTypeKey(
            id=artifact_type,
            schema_version=1,
        ),
        shape=PortShape.ONE,
        accepted_shapes=(PortShape.ONE,),
    )


def _release(
    *,
    workspace_id: UUID = WORKSPACE_ID,
    executable: bool = True,
    protocol_digest: str | None = None,
    runtime_profile: str = "python-uv",
    capabilities: tuple[PluginRuntimeCapability, ...] = (),
    input_type: str = "table.data",
    output_type: str = "scalar.text",
    own_output_type: bool = False,
    secret_inputs: bool = False,
) -> InstalledPluginRelease:
    if secret_inputs and PluginRuntimeCapability.NODE_SECRETS not in capabilities:
        capabilities = (*capabilities, PluginRuntimeCapability.NODE_SECRETS)
    artifact_types = (
        (
            PluginArtifactTypeContract(
                key=PluginArtifactTypeKey(id=output_type, schema_version=1),
                title="Owned output",
                payload_schema={"type": "object"},
            ),
        )
        if own_output_type
        else ()
    )
    artifact_type_dependencies = [
        PluginArtifactTypeContract.from_spec(spec)
        for spec in (TABLE_DATA, TEXT_VALUE, RASTER_IMAGE)
        if spec.key.id in {input_type, output_type}
    ]
    if input_type == "other.private_type":
        artifact_type_dependencies.append(
            PluginArtifactTypeContract(
                key=PluginArtifactTypeKey(
                    id="other.private_type",
                    schema_version=1,
                ),
                title="Cross-Plugin private type",
            )
        )
    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        artifact_types=artifact_types,
        artifact_type_dependencies=tuple(artifact_type_dependencies),
        nodes=(
            PluginNodeContract(
                operator_id="notes.transform",
                operator_version=1,
                title="Transform",
                description="Transform one artifact",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(_port("source", "input", input_type),),
                outputs=(_port("result", "output", output_type),),
                secret_inputs=(
                    (
                        PluginSecretInputContract(
                            name="token",
                            title="Token",
                        ),
                    )
                    if secret_inputs
                    else ()
                ),
                required_capabilities=tuple(
                    sorted(set(capabilities), key=lambda capability: capability.value)
                ),
            ),
        ),
    )
    declared_capabilities = set(capabilities)
    if secret_inputs:
        declared_capabilities.add(PluginRuntimeCapability.NODE_SECRETS)
    capability_manifest = PluginCapabilityManifest(
        capabilities=tuple(declared_capabilities)
    )
    runtime_artifact = (
        PluginRuntimeArtifact(
            object_key="plugin-releases/notes/runtime/image.oci.tar",
            archive_digest="1" * 64,
            manifest_digest="2" * 64,
            config_digest="3" * 64,
        )
        if executable
        else None
    )
    release = PluginRelease(
        slug="notes",
        revision=1,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capability_manifest,
        capability_digest=capability_manifest.digest,
        protocol_digest=protocol_digest or plugin_protocol_digest(),
        profile_digest=plugin_profile_digest(runtime_profile),
        source_object_key="plugin-releases/notes/source.tar.gz",
        source_digest="4" * 64,
        lock_digest="5" * 64,
        runtime_profile=runtime_profile,
        loader_target="grafy_plugin:PLUGIN",
        runtime_image_digest=(
            runtime_artifact.manifest_digest if runtime_artifact is not None else None
        ),
        runtime_artifact=runtime_artifact,
        published_by_user_id=workspace_id,
    )
    return InstalledPluginRelease(
        release=release,
        installation=PluginInstallation.from_release(
            release,
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.WORKSPACE,
                workspace_id=workspace_id,
            ),
            execution_policy=PluginExecutionPolicy.ISOLATED_ONLY,
            distribution=None,
            installed_by_user_id=workspace_id,
            installed_by_platform_actor=None,
        ),
    )


def _system_release(
    plugin: Plugin,
    *,
    catalog: PluginCatalogManifest | None = None,
) -> InstalledPluginRelease:
    catalog = catalog or PluginCatalogManifest.from_plugin(plugin)
    capabilities = PluginCapabilityManifest(capabilities=plugin.capabilities)
    runtime_artifact = PluginRuntimeArtifact(
        object_key="plugin-releases/system/notes/runtime/image.oci.tar",
        archive_digest="6" * 64,
        manifest_digest="7" * 64,
        config_digest="8" * 64,
    )
    release = PluginRelease(
        slug=catalog.slug,
        revision=3,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/system/notes/source.tar.gz",
        source_digest="9" * 64,
        lock_digest="a" * 64,
        runtime_profile="python-uv",
        loader_target="grafy_plugin:PLUGIN",
        runtime_image_digest=runtime_artifact.manifest_digest,
        runtime_artifact=runtime_artifact,
        published_by_platform_actor="test:system-catalog",
    )
    return InstalledPluginRelease(
        release=release,
        installation=PluginInstallation.from_release(
            release,
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.SYSTEM,
                workspace_id=None,
            ),
            execution_policy=PluginExecutionPolicy.ISOLATED_ONLY,
            distribution=PluginDistribution.BUNDLED,
            installed_by_user_id=None,
            installed_by_platform_actor="test:system-catalog",
        ),
    )


def _system_notes_release() -> InstalledPluginRelease:
    return _system_release(HOST_NOTES)


@pytest.mark.parametrize(
    ("release", "admission", "reason"),
    [
        (
            _release(executable=False),
            ReleaseExecutionAdmission(
                isolated_adapter_available=True,
                runtime_profile="python-uv",
            ),
            "missing_runtime_artifact",
        ),
        (
            _release(protocol_digest=sha256(b"grafy-plugin-invocation@1").hexdigest()),
            ReleaseExecutionAdmission(
                isolated_adapter_available=True,
                runtime_profile="python-uv",
            ),
            "incompatible_protocol",
        ),
        (
            _release(runtime_profile="python-uv-gdal"),
            ReleaseExecutionAdmission(
                isolated_adapter_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_runtime_profile",
        ),
        (
            _release(capabilities=(PluginRuntimeCapability.NETWORK_EGRESS,)),
            ReleaseExecutionAdmission(
                isolated_adapter_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_capabilities",
        ),
        (
            _release(secret_inputs=True),
            ReleaseExecutionAdmission(
                isolated_adapter_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_capabilities",
        ),
        (
            _release(),
            ReleaseExecutionAdmission(
                isolated_adapter_available=False,
                runtime_profile="python-uv",
            ),
            "plugin_runtime_unavailable",
        ),
    ],
)
def test_release_readiness_returns_a_stable_fail_closed_reason(
    release: InstalledPluginRelease,
    admission: ReleaseExecutionAdmission,
    reason: PluginNonRunnableReason,
) -> None:
    readiness = plugin_release_readiness(release, admission)

    assert readiness.runnable is False
    assert readiness.reason == reason
    assert readiness.detail


def test_release_readiness_accepts_table_core_scalars_and_owned_inline_types() -> None:
    release = _release(
        output_type="notes.table_summary",
        own_output_type=True,
    )

    readiness = plugin_release_readiness(
        release,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    assert readiness.runnable is True
    assert readiness.reason is None
    assert readiness.detail is None

    decision = ReleaseExecutionAdmission(
        isolated_adapter_available=True,
        runtime_profile="python-uv",
    ).decide(release)
    assert decision is ReleaseExecutionRoute.ISOLATED


def test_release_admission_accepts_exact_serialized_foreign_dependencies() -> None:
    release = _release(input_type="other.private_type")

    readiness = plugin_release_readiness(
        release,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    assert readiness.runnable is True
    assert readiness.reason is None


@pytest.mark.parametrize(
    "lifecycle",
    [PluginFamilyLifecycle.DEPRECATED, PluginFamilyLifecycle.WITHDRAWN],
)
def test_release_readiness_disables_non_published_families_for_new_insertion(
    lifecycle: PluginFamilyLifecycle,
) -> None:
    release = _release()
    selection = PluginReleaseSelection.from_release(release)
    selection.lifecycle = lifecycle

    readiness = plugin_release_readiness(
        release,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
        state=PluginCatalogReleaseState(selection=selection),
    )

    assert readiness.runnable is False
    assert readiness.reason == lifecycle.value
    assert readiness.detail is not None
    assert "new insertion" in readiness.detail


def test_release_readiness_prioritizes_exact_revocation_over_family_lifecycle() -> None:
    release = _release()
    selection = PluginReleaseSelection.from_release(release)
    selection.lifecycle = PluginFamilyLifecycle.WITHDRAWN
    revocation = PluginReleaseRevocation.from_release(
        release,
        reason=PluginReleaseRevocationReason.SECURITY,
        revoked_by_user_id=UUID("00000000-0000-4000-8000-000000000663"),
    )

    readiness = plugin_release_readiness(
        release,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
        state=PluginCatalogReleaseState(
            selection=selection,
            revocation=revocation,
        ),
    )

    assert readiness.runnable is False
    assert readiness.reason == "revoked"
    assert readiness.detail is not None


def test_exact_foreign_dependency_makes_cross_plugin_type_portable() -> None:
    release = _release(
        input_type="other.private_type",
        output_type="notes.table_summary",
        own_output_type=True,
    )

    readiness = plugin_release_readiness(
        release,
        ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    assert readiness.runnable is True
    assert readiness.reason is None
    assert readiness.detail is None


def test_system_release_replaces_matching_host_plugin_and_node_presentation() -> None:
    registry = PluginRegistry()
    registry.install(HOST_NOTES)
    release = _system_notes_release()

    response = NodeRegistryResponse.from_registry(
        registry,
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [release],
        workspace_id=WORKSPACE_ID,
        release_admission=ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    notes_plugins = [
        plugin for plugin in response.plugins if plugin.slug == release.slug
    ]
    assert len(notes_plugins) == 1
    plugin = notes_plugins[0]
    assert plugin.entry_kind == "plugin"
    assert plugin.scope is PluginReleaseScope.SYSTEM
    assert plugin.distribution is PluginDistribution.BUNDLED
    assert plugin.plugin_release is not None
    assert plugin.plugin_release.scope is PluginReleaseScope.SYSTEM
    assert plugin.plugin_release.slug == release.slug
    assert plugin.plugin_release.revision == release.revision
    assert plugin.revision == release.revision

    notes_nodes = [node for node in response.nodes if node.plugin_slug == release.slug]
    assert len(notes_nodes) == len(release.catalog.nodes)
    assert all(node.plugin_release is not None for node in notes_nodes)
    assert all(
        node.plugin_release.scope is PluginReleaseScope.SYSTEM
        for node in notes_nodes
        if node.plugin_release is not None
    )


def test_system_release_rejects_a_different_host_node_contract() -> None:
    registry = PluginRegistry()
    registry.install(HOST_NOTES)
    catalog = PluginCatalogManifest.from_plugin(HOST_NOTES)
    changed_node = catalog.nodes[0].model_copy(update={"title": "Changed"})
    changed_catalog = catalog.model_copy(update={"nodes": (changed_node,)})

    with pytest.raises(ValueError, match="conflicts with host Plugin 'notes'"):
        NodeRegistryResponse.from_registry(
            registry,
            GraphModuleCatalogListing(entries=[], unavailable=[]),
            UnusedModuleExecutor(),
            [_system_release(HOST_NOTES, catalog=changed_catalog)],
            workspace_id=WORKSPACE_ID,
        )


def test_host_runtime_registry_does_not_synthesize_plugin_catalog_entries() -> None:
    registry = PluginRegistry()
    registry.install(HOST_NOTES)

    response = NodeRegistryResponse.from_registry(
        registry,
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [],
        workspace_id=WORKSPACE_ID,
    )

    assert [entry.slug for entry in response.plugins] == ["graph.module"]
    assert response.nodes == []


def test_catalog_keeps_withdrawn_release_visible_disabled_and_exactly_pinned() -> None:
    release = _release()
    selection = PluginReleaseSelection.from_release(release)
    selection.lifecycle = PluginFamilyLifecycle.WITHDRAWN

    response = NodeRegistryResponse.from_registry(
        PluginRegistry(),
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [release],
        workspace_id=WORKSPACE_ID,
        release_admission=ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
        plugin_release_states={
            release.id: PluginCatalogReleaseState(selection=selection)
        },
    )

    plugin = next(entry for entry in response.plugins if entry.slug == release.slug)
    node = next(entry for entry in response.nodes if entry.plugin_slug == release.slug)
    assert plugin.runnable is False
    assert plugin.non_runnable_reason == "withdrawn"
    assert plugin.plugin_release is not None
    assert plugin.plugin_release.revision == release.revision
    assert node.runnable is False
    assert node.non_runnable_reason == "withdrawn"
    assert node.plugin_release is not None
    assert node.plugin_release.revision == release.revision


def test_system_release_collapses_exact_host_artifact_and_conversion_contracts() -> (
    None
):
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    release = _system_release(TEXT)

    response = NodeRegistryResponse.from_registry(
        registry,
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [_system_release(ARITHMETIC), release],
        workspace_id=WORKSPACE_ID,
        release_admission=ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    artifact_keys = [
        (artifact.key.id, artifact.key.schema_version)
        for artifact in response.artifact_types
    ]
    conversion_keys = [
        (conversion.key.id, conversion.key.version)
        for conversion in response.artifact_conversions
    ]
    assert len(artifact_keys) == len(set(artifact_keys))
    assert len(conversion_keys) == len(set(conversion_keys))
    assert conversion_keys == [("builtin.scalar.integer_to_text", 1)]
    assert registry.artifact_conversions == ()
    assert all(
        artifact_keys.count((contract.key.id, contract.key.schema_version)) == 1
        for contract in release.catalog.artifact_types
    )
    assert all(
        conversion_keys.count((contract.key.id, contract.key.version)) == 1
        for contract in release.catalog.artifact_conversions
    )


def test_system_release_accepts_exact_installed_foreign_artifact_dependency() -> (
    None
):
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    release = _system_release(TEXT)

    response = NodeRegistryResponse.from_registry(
        registry,
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [release],
        workspace_id=WORKSPACE_ID,
        release_admission=ReleaseExecutionAdmission(
            isolated_adapter_available=True,
            runtime_profile="python-uv",
        ),
    )

    dependency_keys = {
        (dependency.key.id, dependency.key.schema_version)
        for dependency in release.catalog.artifact_type_dependencies
    }
    response_keys = {
        (artifact.key.id, artifact.key.schema_version)
        for artifact in response.artifact_types
    }
    assert dependency_keys <= response_keys


def test_system_releases_reject_different_exact_foreign_dependency_contract() -> (
    None
):
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    text_catalog = PluginCatalogManifest.from_plugin(TEXT)
    changed_dependency = PluginArtifactTypeContract.from_spec(
        INTEGER_VALUE
    ).model_copy(update={"title": "Conflicting integer contract"})
    changed_text_catalog = text_catalog.model_copy(
        update={"artifact_type_dependencies": (changed_dependency,)}
    )

    with pytest.raises(ValueError, match="different exact contracts"):
        NodeRegistryResponse.from_registry(
            registry,
            GraphModuleCatalogListing(entries=[], unavailable=[]),
            UnusedModuleExecutor(),
            [
                _system_release(ARITHMETIC),
                _system_release(TEXT, catalog=changed_text_catalog),
            ],
            workspace_id=WORKSPACE_ID,
        )


def test_system_release_rejects_same_key_with_different_host_artifact_contract() -> (
    None
):
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    catalog = PluginCatalogManifest.from_plugin(TEXT)
    changed_artifact = catalog.artifact_types[0].model_copy(
        update={"title": "Changed host contract"}
    )
    changed_catalog = catalog.model_copy(
        update={
            "artifact_types": (changed_artifact, *catalog.artifact_types[1:]),
        }
    )

    with pytest.raises(ValueError, match="conflicts with the host catalog contract"):
        NodeRegistryResponse.from_registry(
            registry,
            GraphModuleCatalogListing(entries=[], unavailable=[]),
            UnusedModuleExecutor(),
            [
                _system_release(ARITHMETIC),
                _system_release(TEXT, catalog=changed_catalog),
            ],
            workspace_id=WORKSPACE_ID,
        )


def test_module_provider_is_a_separate_entry_kind_without_plugin_scope() -> None:
    registry = PluginRegistry()
    registry.register_module_boundaries(MODULE_BOUNDARY_REGISTRATIONS)
    response = NodeRegistryResponse.from_registry(
        registry,
        GraphModuleCatalogListing(entries=[], unavailable=[]),
        UnusedModuleExecutor(),
        [],
        workspace_id=WORKSPACE_ID,
    )

    module = next(
        plugin for plugin in response.plugins if plugin.slug == "graph.module"
    )
    assert module.entry_kind == "module"
    assert module.scope is None
    assert module.distribution is None
    assert module.plugin_release is None
    assert {node.operator_id for node in response.nodes} == {
        "module.input",
        "module.output",
    }
    assert {node.plugin_slug for node in response.nodes} == {"graph.module"}


def test_plugin_catalog_entry_requires_an_exact_release() -> None:
    with pytest.raises(ValidationError, match="must declare an exact release"):
        PluginSpecResponse(
            slug="notes",
            title="Notes",
            scope=PluginReleaseScope.WORKSPACE,
        )


def test_catalog_rejects_a_foreign_workspace_release() -> None:
    with pytest.raises(ValueError, match="foreign releases.*notes@1"):
        NodeRegistryResponse.from_registry(
            PluginRegistry(),
            GraphModuleCatalogListing(entries=[], unavailable=[]),
            UnusedModuleExecutor(),
            [_release(workspace_id=OTHER_WORKSPACE_ID)],
            workspace_id=WORKSPACE_ID,
        )


def test_catalog_rejects_a_system_workspace_slug_collision() -> None:
    system_release = _system_notes_release()
    workspace_release = _release()

    with pytest.raises(
        ValueError,
        match="Workspace Plugin releases conflict with System Plugins: notes",
    ):
        NodeRegistryResponse.from_registry(
            PluginRegistry(),
            GraphModuleCatalogListing(entries=[], unavailable=[]),
            UnusedModuleExecutor(),
            [system_release, workspace_release],
            workspace_id=WORKSPACE_ID,
        )

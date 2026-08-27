"""Explicit Plugin registry fixtures with no production discovery fallback."""

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.domain.plugin_installations import (
    InstalledPluginRelease,
    PluginInstallation,
)
from grafy_core.domain.plugin_releases import (
    PluginArtifactConversionContract,
    PluginArtifactTypeContract,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginDistribution,
    PluginExecutionPolicy,
    PluginRelease,
    PluginReleaseNamespace,
    PluginReleaseScope,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.canonical_conversions import CANONICAL_ARTIFACT_CONVERSIONS_BY_KEY
from grafy_core.domain.plugin_revocations import PluginReleaseRevocation
from grafy_core.domain.plugin_selection import PluginReleaseSelection
from grafy_core.operators.modules import MODULE_BOUNDARY_REGISTRATIONS
from grafy_core.plugins import Plugin, PluginRegistry, UnknownOperatorError
from grafy_plugin_arithmetic import ARITHMETIC
from grafy_plugin_image import IMAGES
from grafy_plugin_prompt import PROMPTS
from grafy_plugin_schema import SCHEMAS
from grafy_plugin_sequence import SEQUENCES
from grafy_plugin_table import TABLES
from grafy_plugin_text import TEXT

from grafy_api.system_host_bindings import LoadedSystemPlugin, SystemHostPluginBinding
from grafy_api.v1.models import ArtifactTypeBindingModel, PluginReleasePinModel
from grafy_api.v1.routes.executions.models import RunInputPlugRequest, RunNodeRequest


TEST_SYSTEM_PLUGINS: tuple[Plugin, ...] = (
    IMAGES,
    SEQUENCES,
    ARITHMETIC,
    TEXT,
    SCHEMAS,
    PROMPTS,
    TABLES,
)


def build_explicit_plugin_registry(
    plugins: Iterable[Plugin] = TEST_SYSTEM_PLUGINS,
) -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_module_boundaries(MODULE_BOUNDARY_REGISTRATIONS)
    for plugin in plugins:
        registry.install(plugin)
    registry.freeze()
    return registry


class SelectedSystemReleaseLookup:
    """Exact persisted-release view for a synthetic selected deployment."""

    def __init__(
        self,
        releases: tuple[InstalledPluginRelease, ...],
        selections: tuple[PluginReleaseSelection, ...],
    ) -> None:
        self._releases = {
            (release.scope, release.slug, release.revision): release
            for release in releases
        }
        self._selections = {
            (selection.scope, selection.slug): selection for selection in selections
        }
        self.release_reads = 0

    async def get_by_revision(
        self,
        workspace_id: UUID,
        slug: str,
        revision: int,
        *,
        scope: PluginReleaseScope = PluginReleaseScope.WORKSPACE,
    ) -> InstalledPluginRelease | None:
        self.release_reads += 1
        release = self._releases.get((scope, slug, revision))
        if release is None:
            return None
        expected_workspace_id = (
            workspace_id if scope is PluginReleaseScope.WORKSPACE else None
        )
        if release.workspace_id != expected_workspace_id:
            return None
        return release

    async def get_selection(
        self,
        workspace_id: UUID,
        slug: str,
        *,
        scope: PluginReleaseScope = PluginReleaseScope.WORKSPACE,
    ) -> PluginReleaseSelection | None:
        del workspace_id
        return self._selections.get((scope, slug))

    async def list_current_system(self) -> list[InstalledPluginRelease]:
        return [
            release
            for release in self._releases.values()
            if release.scope is PluginReleaseScope.SYSTEM
        ]

    async def list_current(
        self,
        workspace_id: UUID,
    ) -> list[InstalledPluginRelease]:
        return [
            release
            for release in self._releases.values()
            if release.scope is PluginReleaseScope.WORKSPACE
            and release.workspace_id == workspace_id
        ]

    async def get_revocation(
        self,
        *,
        workspace_id: UUID,
        slug: str,
        revision: int,
    ) -> PluginReleaseRevocation | None:
        del workspace_id, slug, revision
        return None

    async def get_system_revocation(
        self,
        *,
        slug: str,
        revision: int,
    ) -> PluginReleaseRevocation | None:
        del slug, revision
        return None


@dataclass(frozen=True, slots=True)
class SelectedSystemPluginDeployment:
    registry: PluginRegistry
    releases: tuple[InstalledPluginRelease, ...]
    selections: tuple[PluginReleaseSelection, ...]
    host_bindings: tuple[SystemHostPluginBinding, ...]
    loaded_plugins: tuple[LoadedSystemPlugin, ...]
    release_lookup: SelectedSystemReleaseLookup

    def pin_node(self, node: RunNodeRequest) -> RunNodeRequest:
        try:
            registration = self.registry.node_registration(
                node.operator_id,
                node.operator_version,
            )
        except UnknownOperatorError:
            return node
        if registration.plugin_slug == "graph.module":
            return node
        release = next(
            release
            for release in self.releases
            if release.slug == registration.plugin_slug
        )
        return node.model_copy(
            update={
                "plugin_release": PluginReleasePinModel(
                    scope=PluginReleaseScope.SYSTEM,
                    slug=release.slug,
                    revision=release.revision,
                )
            }
        )


def build_selected_system_plugin_deployment(
    plugins: Iterable[Plugin] = TEST_SYSTEM_PLUGINS,
) -> SelectedSystemPluginDeployment:
    selected_plugins = tuple(plugins)
    registry = build_explicit_plugin_registry(selected_plugins)
    releases: list[InstalledPluginRelease] = []
    selections: list[PluginReleaseSelection] = []
    host_bindings: list[SystemHostPluginBinding] = []
    loaded_plugins: list[LoadedSystemPlugin] = []
    canonical_conversion_contracts = {
        (key.id, key.version): PluginArtifactConversionContract.from_conversion(
            conversion
        )
        for key, conversion in CANONICAL_ARTIFACT_CONVERSIONS_BY_KEY.items()
    }
    for index, plugin in enumerate(selected_plugins, start=1):
        plugin_catalog = PluginCatalogManifest.from_plugin(plugin)
        catalog = plugin_catalog.model_copy(
            update={
                "artifact_types": tuple(
                    PluginArtifactTypeContract.from_spec(artifact_type)
                    for artifact_type in registry.artifact_types
                    if registry.artifact_type_owner(artifact_type.key) == plugin.slug
                ),
                "artifact_type_dependencies": tuple(
                    PluginArtifactTypeContract.from_spec(artifact_type)
                    for artifact_type in registry.artifact_types
                    if registry.artifact_type_owner(artifact_type.key) != plugin.slug
                ),
                "artifact_conversions": tuple(
                    contract
                    for contract in plugin_catalog.artifact_conversions
                    if canonical_conversion_contracts.get(
                        (contract.key.id, contract.key.version)
                    )
                    == contract
                ),
            }
        )
        required_capabilities = tuple(
            sorted(
                {
                    capability
                    for node in catalog.nodes
                    for capability in node.required_capabilities
                },
                key=lambda capability: capability.value,
            )
        )
        if any(node.secret_inputs for node in catalog.nodes):
            required_capabilities = tuple(
                sorted(
                    {*required_capabilities, PluginRuntimeCapability.NODE_SECRETS},
                    key=lambda capability: capability.value,
                )
            )
        capabilities = PluginCapabilityManifest(
            capabilities=required_capabilities,
        )
        digest_character = format(index, "x")
        runtime_artifact = PluginRuntimeArtifact(
            object_key=f"test-system-plugins/{plugin.slug}/r1.oci.tar",
            archive_digest=digest_character * 64,
            manifest_digest=digest_character * 64,
            config_digest=digest_character * 64,
        )
        loader_target = "tests.support.system_plugins:" + plugin.slug.replace(
            ".", "_"
        ).replace("-", "_")
        release_record = PluginRelease(
            slug=plugin.slug,
            revision=1,
            catalog=catalog,
            contract_digest=plugin_contract_digest(catalog),
            capabilities=capabilities,
            capability_digest=capabilities.digest,
            protocol_digest=plugin_protocol_digest(),
            profile_digest=plugin_profile_digest("python-uv"),
            source_object_key=f"test-system-plugins/{plugin.slug}/r1.tar.gz",
            source_digest=digest_character * 64,
            lock_digest=digest_character * 64,
            runtime_profile="python-uv",
            loader_target=loader_target,
            runtime_image_digest=runtime_artifact.manifest_digest,
            runtime_artifact=runtime_artifact,
            published_by_platform_actor="test:system",
        )
        release = InstalledPluginRelease(
            release=release_record,
            installation=PluginInstallation.from_release(
                release_record,
                namespace=PluginReleaseNamespace(
                    scope=PluginReleaseScope.SYSTEM,
                    workspace_id=None,
                ),
                execution_policy=PluginExecutionPolicy.HOST_ELIGIBLE,
                distribution=PluginDistribution.BUNDLED,
                installed_by_user_id=None,
                installed_by_platform_actor="test:system",
            ),
        )
        selection = PluginReleaseSelection.from_release(release)
        releases.append(release)
        selections.append(selection)
        host_build_digest = digest_character * 64
        host_bindings.append(
            SystemHostPluginBinding.from_release(
                release,
                selection_generation=selection.generation,
                loader_target=loader_target,
                host_build_digest=host_build_digest,
            )
        )
        loaded_plugins.append(
            LoadedSystemPlugin(
                slug=release.slug,
                loader_target=loader_target,
                host_build_digest=host_build_digest,
            )
        )
    frozen_releases = tuple(releases)
    frozen_selections = tuple(selections)
    return SelectedSystemPluginDeployment(
        registry=registry,
        releases=frozen_releases,
        selections=frozen_selections,
        host_bindings=tuple(host_bindings),
        loaded_plugins=tuple(loaded_plugins),
        release_lookup=SelectedSystemReleaseLookup(
            frozen_releases,
            frozen_selections,
        ),
    )


def pin_selected_system_nodes(
    nodes: Iterable[RunNodeRequest],
    plugins: Iterable[Plugin] = TEST_SYSTEM_PLUGINS,
) -> list[RunNodeRequest]:
    slugs_by_operator = {
        registration.key: plugin.slug
        for plugin in plugins
        for registration in plugin.nodes
    }
    pinned_nodes: list[RunNodeRequest] = []
    for node in nodes:
        slug = slugs_by_operator.get((node.operator_id, node.operator_version))
        if slug is None:
            pinned_nodes.append(node)
            continue
        pinned_nodes.append(
            node.model_copy(
                update={
                    "plugin_release": PluginReleasePinModel(
                        scope=PluginReleaseScope.SYSTEM,
                        slug=slug,
                        revision=1,
                    )
                }
            )
        )
    return pinned_nodes


def selected_system_run_node(
    *,
    id: str,
    operator_id: str,
    operator_version: int,
    config: dict[str, object] | None = None,
    input_plugs: list[RunInputPlugRequest] | None = None,
    artifact_type_bindings: list[ArtifactTypeBindingModel] | None = None,
    plugin_slug: str | None = None,
) -> RunNodeRequest:
    resolved_slug = plugin_slug
    if resolved_slug is None:
        resolved_slug = next(
            (
                plugin.slug
                for plugin in TEST_SYSTEM_PLUGINS
                if any(
                    registration.key == (operator_id, operator_version)
                    for registration in plugin.nodes
                )
            ),
            None,
        )
    if resolved_slug is None:
        raise ValueError(
            f"Test operator {operator_id}@{operator_version} requires an explicit "
            "selected System Plugin slug"
        )
    return RunNodeRequest(
        id=id,
        operator_id=operator_id,
        operator_version=operator_version,
        config={} if config is None else config,
        input_plugs=[] if input_plugs is None else input_plugs,
        artifact_type_bindings=(
            [] if artifact_type_bindings is None else artifact_type_bindings
        ),
        plugin_release=PluginReleasePinModel(
            scope=PluginReleaseScope.SYSTEM,
            slug=resolved_slug,
            revision=1,
        ),
    )


__all__ = [
    "TEST_SYSTEM_PLUGINS",
    "SelectedSystemPluginDeployment",
    "SelectedSystemReleaseLookup",
    "build_explicit_plugin_registry",
    "build_selected_system_plugin_deployment",
    "pin_selected_system_nodes",
    "selected_system_run_node",
]

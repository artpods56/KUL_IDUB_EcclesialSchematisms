"""Explicit Plugin registry fixtures with no production discovery fallback."""

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from grafy_core.domain.plugin_installations import InstalledPluginRelease
from grafy_core.domain.plugin_releases import PluginReleaseScope
from grafy_core.domain.plugin_revocations import PluginReleaseRevocation
from grafy_core.domain.plugin_selection import PluginReleaseSelection
from grafy_core.operators.modules import MODULE_BOUNDARY_REGISTRATIONS
from grafy_core.plugins import Plugin, PluginRegistry, UnknownOperatorError
from grafy_workbench import BUILTIN_FAMILIES

from grafy_api.system_host_bindings import LoadedSystemPlugin, SystemHostPluginBinding
from grafy_api.v1.models import ArtifactTypeBindingModel
from grafy_api.v1.routes.executions.models import RunInputPlugRequest, RunNodeRequest


TEST_SYSTEM_PLUGINS: tuple[Plugin, ...] = BUILTIN_FAMILIES
TEST_BUILD_DIGEST = "a" * 64


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
            return node.model_copy(update={"kind": "module", "plugin_release": None})
        return node.model_copy(
            update={
                "kind": "builtin",
                "plugin_release": None,
            }
        )


def build_selected_system_plugin_deployment(
    plugins: Iterable[Plugin] = TEST_SYSTEM_PLUGINS,
) -> SelectedSystemPluginDeployment:
    registry = build_explicit_plugin_registry(plugins)
    return SelectedSystemPluginDeployment(
        registry=registry,
        releases=(),
        selections=(),
        host_bindings=(),
        loaded_plugins=(),
        release_lookup=SelectedSystemReleaseLookup((), ()),
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
                    "kind": "builtin",
                    "plugin_release": None,
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
    kind: str | None = None,
    plugin_release: object | None = None,
) -> RunNodeRequest:
    del plugin_slug, kind, plugin_release
    return RunNodeRequest(
        kind="builtin",
        id=id,
        operator_id=operator_id,
        operator_version=operator_version,
        config={} if config is None else config,
        input_plugs=[] if input_plugs is None else input_plugs,
        artifact_type_bindings=(
            [] if artifact_type_bindings is None else artifact_type_bindings
        ),
    )


__all__ = [
    "TEST_BUILD_DIGEST",
    "TEST_SYSTEM_PLUGINS",
    "SelectedSystemPluginDeployment",
    "SelectedSystemReleaseLookup",
    "build_explicit_plugin_registry",
    "build_selected_system_plugin_deployment",
    "pin_selected_system_nodes",
    "selected_system_run_node",
]

from dataclasses import dataclass
from uuid import UUID

from grafy_core.application.modules import (
    ModuleLibraryService,
    validate_optional_input_targets,
)
from grafy_core.application.saved_graphs import SavedGraphService
from grafy_core.domain.errors import NotFoundError
from grafy_core.domain.module_library import ModulePublicationState
from grafy_core.domain.modules import (
    MODULE_INPUT_OPERATOR_ID,
    MODULE_OUTPUT_OPERATOR_ID,
    GraphModuleDefinition,
    GraphModuleDefinitionError,
    GraphModuleReference,
)
from grafy_core.domain.saved_graphs import SavedGraphDocument
from grafy_core.plugins import PluginRegistry

GRAPH_MODULE_PLUGIN_SLUG = "graph.module"

_MODULE_BOUNDARY_OPERATOR_IDS = frozenset(
    {
        MODULE_INPUT_OPERATOR_ID,
        MODULE_OUTPUT_OPERATOR_ID,
    }
)


class GraphModuleCatalogError(RuntimeError):
    """A saved graph cannot be resolved as an executable graph module."""


@dataclass(frozen=True, slots=True)
class GraphModuleCatalogEntry:
    definition: GraphModuleDefinition
    catalog_visible: bool
    module_id: UUID | None = None
    publication_state: ModulePublicationState | None = None
    is_current_library_release: bool = False


@dataclass(frozen=True, slots=True)
class UnavailableGraphModule:
    graph_id: UUID
    revision: int
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class GraphModuleCatalogListing:
    entries: list[GraphModuleCatalogEntry]
    unavailable: list[UnavailableGraphModule]


def document_has_module_boundary(document: SavedGraphDocument) -> bool:
    return any(
        node.operator_id in _MODULE_BOUNDARY_OPERATOR_IDS for node in document.nodes
    )


class GraphModuleCatalog:
    """Resolves published module releases for browse and any pin for execution."""

    def __init__(
        self,
        saved_graphs: SavedGraphService | None,
        plugin_registry: PluginRegistry,
        module_library: ModuleLibraryService | None = None,
    ) -> None:
        self._saved_graphs = saved_graphs
        self._plugin_registry = plugin_registry
        self._module_library = module_library

    async def list(self, workspace_id: UUID) -> GraphModuleCatalogListing:
        if self._module_library is None or self._saved_graphs is None:
            return GraphModuleCatalogListing(entries=[], unavailable=[])
        entries: list[GraphModuleCatalogEntry] = []
        for (
            module,
            release,
            definition,
        ) in await self._module_library.catalog_definitions(workspace_id):
            # catalog_definitions already validates and skips invalid
            # definitions, so no re-validation is needed here.
            is_current = module.current_library_release == release.revision
            entries.append(
                GraphModuleCatalogEntry(
                    definition=definition,
                    catalog_visible=is_current,
                    module_id=module.id,
                    publication_state=module.publication_state,
                    is_current_library_release=is_current,
                )
            )
        return GraphModuleCatalogListing(entries=entries, unavailable=[])

    async def get_definition(
        self,
        reference: GraphModuleReference,
        *,
        workspace_id: UUID,
    ) -> GraphModuleDefinition:
        if self._module_library is not None:
            try:
                return await self._module_library.resolve_definition(
                    reference,
                    workspace_id=workspace_id,
                )
            except NotFoundError as exc:
                raise NotFoundError(
                    "Saved graph module",
                    f"{reference.graph_id}@{reference.revision}",
                ) from exc
            except GraphModuleDefinitionError as exc:
                raise GraphModuleCatalogError(
                    f"Saved graph {reference.graph_id} revision {reference.revision} "
                    f"is not a valid module: {exc}"
                ) from exc
        # Lightweight configuration: no module library, resolve from saved graphs
        # directly through the same canonical validation rules.
        if self._saved_graphs is None:
            raise GraphModuleCatalogError(
                "Saved graph modules are not configured for this workbench"
            )
        try:
            revision = await self._saved_graphs.get_revision(
                workspace_id,
                reference.graph_id,
                reference.revision,
            )
            definition = GraphModuleDefinition.from_saved_graph_revision(revision)
            await validate_optional_input_targets(
                definition,
                workspace_id=workspace_id,
                graphs=self._saved_graphs,
                plugin_registry=self._plugin_registry,
            )
        except NotFoundError as exc:
            raise NotFoundError(
                "Saved graph module",
                f"{reference.graph_id}@{reference.revision}",
            ) from exc
        except GraphModuleDefinitionError as exc:
            raise GraphModuleCatalogError(
                f"Saved graph {reference.graph_id} revision {reference.revision} "
                f"is not a valid module: {exc}"
            ) from exc
        return definition


__all__ = [
    "GRAPH_MODULE_PLUGIN_SLUG",
    "GraphModuleCatalog",
    "GraphModuleCatalogEntry",
    "GraphModuleCatalogError",
    "GraphModuleCatalogListing",
    "UnavailableGraphModule",
    "document_has_module_boundary",
]

from dataclasses import dataclass

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.modules import (
    GraphModuleDefinition,
    GraphModuleDefinitionError,
    GraphModuleReference,
)

from notarius_api.services.execution.errors import GraphExecutionError


GRAPH_MODULE_PLUGIN_SLUG = "graph.module"


@dataclass(frozen=True, slots=True)
class GraphModuleCatalogEntry:
    definition: GraphModuleDefinition
    catalog_visible: bool


class GraphModuleCatalog:
    """Discovers valid saved-graph revisions exposed as graph modules."""

    def __init__(self, saved_graphs: SavedGraphService | None) -> None:
        self._saved_graphs = saved_graphs

    async def list(self) -> list[GraphModuleCatalogEntry]:
        if self._saved_graphs is None:
            return []
        entries: list[GraphModuleCatalogEntry] = []
        for graph in await self._saved_graphs.list():
            for revision in await self._saved_graphs.list_revisions(graph.id):
                try:
                    definition = GraphModuleDefinition.from_saved_graph_revision(
                        revision
                    )
                except GraphModuleDefinitionError:
                    continue
                entries.append(
                    GraphModuleCatalogEntry(
                        definition=definition,
                        catalog_visible=revision.revision == graph.revision,
                    )
                )
        return entries

    async def get_definition(
        self,
        reference: GraphModuleReference,
    ) -> GraphModuleDefinition:
        if self._saved_graphs is None:
            raise GraphExecutionError(
                "Saved graph modules are not configured for this workbench"
            )
        revision = await self._saved_graphs.get_revision(
            reference.graph_id,
            reference.revision,
        )
        try:
            return GraphModuleDefinition.from_saved_graph_revision(revision)
        except GraphModuleDefinitionError as exc:
            raise GraphExecutionError(
                f"Saved graph {reference.graph_id} revision {reference.revision} "
                f"is not a valid module: {exc}"
            ) from exc


__all__ = [
    "GRAPH_MODULE_PLUGIN_SLUG",
    "GraphModuleCatalog",
    "GraphModuleCatalogEntry",
]

from dataclasses import dataclass

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.modules import (
    GraphModuleDefinition,
    GraphModuleDefinitionError,
    GraphModuleReference,
    GraphModuleReferenceError,
)
from notarius_core.domain.errors import NotFoundError
from notarius_core.plugins import PluginRegistry, UnknownOperatorError

from notarius_api.services.execution.errors import GraphExecutionError


GRAPH_MODULE_PLUGIN_SLUG = "graph.module"


@dataclass(frozen=True, slots=True)
class GraphModuleCatalogEntry:
    definition: GraphModuleDefinition
    catalog_visible: bool


class GraphModuleCatalog:
    """Discovers valid saved-graph revisions exposed as graph modules."""

    def __init__(
        self,
        saved_graphs: SavedGraphService | None,
        plugin_registry: PluginRegistry,
    ) -> None:
        self._saved_graphs = saved_graphs
        self._plugin_registry = plugin_registry

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
                    await self._validate_optional_input_targets(definition)
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
            definition = GraphModuleDefinition.from_saved_graph_revision(revision)
            await self._validate_optional_input_targets(definition)
        except GraphModuleDefinitionError as exc:
            raise GraphExecutionError(
                f"Saved graph {reference.graph_id} revision {reference.revision} "
                f"is not a valid module: {exc}"
            ) from exc
        return definition

    async def _validate_optional_input_targets(
        self,
        definition: GraphModuleDefinition,
    ) -> None:
        nodes_by_id = {node.id: node for node in definition.document.nodes}
        for public_port in definition.input_ports:
            if public_port.required:
                continue
            for edge in definition.document.edges:
                if not edge.enabled or edge.from_node != public_port.boundary_node_id:
                    continue
                target_node = nodes_by_id[edge.to_node]
                try:
                    target_reference = GraphModuleReference.try_from_operator_identity(
                        target_node.operator_id,
                        target_node.operator_version,
                    )
                except GraphModuleReferenceError as exc:
                    raise GraphModuleDefinitionError(
                        f"Graph module {definition.reference.graph_id} revision "
                        f"{definition.reference.revision} optional public input "
                        f"{public_port.name!r} edge {edge.id!r} targets invalid "
                        f"graph module operator {target_node.operator_id}@"
                        f"{target_node.operator_version}: {exc}"
                    ) from exc

                if target_reference is not None:
                    if self._saved_graphs is None:
                        raise GraphModuleDefinitionError(
                            f"Graph module {definition.reference.graph_id} revision "
                            f"{definition.reference.revision} optional public input "
                            f"{public_port.name!r} edge {edge.id!r} cannot resolve "
                            "its target graph module because saved graphs are not "
                            "configured"
                        )
                    try:
                        target_revision = await self._saved_graphs.get_revision(
                            target_reference.graph_id,
                            target_reference.revision,
                        )
                        target_definition = (
                            GraphModuleDefinition.from_saved_graph_revision(
                                target_revision
                            )
                        )
                        target_port = target_definition.input_port(edge.to_port)
                    except (GraphModuleDefinitionError, NotFoundError) as exc:
                        raise GraphModuleDefinitionError(
                            f"Graph module {definition.reference.graph_id} revision "
                            f"{definition.reference.revision} optional public input "
                            f"{public_port.name!r} edge {edge.id!r} cannot resolve "
                            f"target input {target_node.id!r}.{edge.to_port!r} "
                            f"({target_node.operator_id}@"
                            f"{target_node.operator_version}): {exc}"
                        ) from exc
                    target_required = target_port.required
                else:
                    try:
                        registration = self._plugin_registry.node_registration(
                            target_node.operator_id,
                            target_node.operator_version,
                        )
                    except UnknownOperatorError as exc:
                        raise GraphModuleDefinitionError(
                            f"Graph module {definition.reference.graph_id} revision "
                            f"{definition.reference.revision} optional public input "
                            f"{public_port.name!r} edge {edge.id!r} targets unknown "
                            f"operator {target_node.operator_id}@"
                            f"{target_node.operator_version}"
                        ) from exc
                    target_port = registration.node_class.input_contract.ports.get(
                        edge.to_port
                    )
                    if target_port is None:
                        raise GraphModuleDefinitionError(
                            f"Graph module {definition.reference.graph_id} revision "
                            f"{definition.reference.revision} optional public input "
                            f"{public_port.name!r} edge {edge.id!r} targets missing "
                            f"input {target_node.id!r}.{edge.to_port!r} "
                            f"({target_node.operator_id}@"
                            f"{target_node.operator_version})"
                        )
                    target_required = target_port.required

                if target_required:
                    raise GraphModuleDefinitionError(
                        f"Graph module {definition.reference.graph_id} revision "
                        f"{definition.reference.revision} optional public input "
                        f"{public_port.name!r} edge {edge.id!r} targets required "
                        f"input {target_node.id!r}.{edge.to_port!r} "
                        f"({target_node.operator_id}@{target_node.operator_version})"
                    )


__all__ = [
    "GRAPH_MODULE_PLUGIN_SLUG",
    "GraphModuleCatalog",
    "GraphModuleCatalogEntry",
]

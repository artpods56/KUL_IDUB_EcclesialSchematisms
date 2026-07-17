from collections.abc import Callable
from typing import cast
from uuid import UUID

from pydantic import ValidationError

from notarius_core.domain.errors import (
    ConcurrentWriteError,
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.domain.node_secrets import (
    InvalidNodeSecretDependenciesError,
    JsonValue,
    node_secret_dependency_sha256,
)
from notarius_core.domain.saved_graphs import (
    SavedGraph,
    SavedGraphDocument,
    SavedGraphRevision,
)
from notarius_core.plugins import PluginRegistry
from notarius_core.ports.saved_graphs import SavedGraphUnitOfWorkPort


class SavedGraphService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], SavedGraphUnitOfWorkPort],
        plugin_registry: PluginRegistry,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._plugin_registry = plugin_registry

    async def create(
        self,
        *,
        name: str,
        document: SavedGraphDocument,
    ) -> SavedGraph:
        graph = SavedGraph(name=name, document=document)
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.graphs.add(graph)
            await unit_of_work.graphs.add_revision(graph.snapshot())
            await unit_of_work.commit()
        return graph

    async def get(self, graph_id: UUID) -> SavedGraph:
        async with self._unit_of_work_factory() as unit_of_work:
            graph = await unit_of_work.graphs.get(graph_id)
        if graph is None:
            raise NotFoundError("Saved graph", str(graph_id))
        return graph

    async def get_revision(
        self,
        graph_id: UUID,
        revision: int,
    ) -> SavedGraphRevision:
        async with self._unit_of_work_factory() as unit_of_work:
            snapshot = await unit_of_work.graphs.get_revision(graph_id, revision)
        if snapshot is None:
            raise NotFoundError("Saved graph revision", f"{graph_id}@{revision}")
        return snapshot

    async def list_revisions(self, graph_id: UUID) -> list[SavedGraphRevision]:
        async with self._unit_of_work_factory() as unit_of_work:
            graph = await unit_of_work.graphs.get(graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            return await unit_of_work.graphs.list_revisions(graph_id)

    async def list(self) -> list[SavedGraph]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.graphs.list()

    async def replace(
        self,
        graph_id: UUID,
        *,
        name: str,
        document: SavedGraphDocument,
        expected_revision: int,
    ) -> SavedGraph:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.graphs.lock_revision(graph_id, expected_revision)
            graph = await unit_of_work.graphs.get(graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            graph.ensure_revision(expected_revision)

            registrations = {
                registration.key: registration
                for registration in self._plugin_registry.nodes
            }
            valid_secret_bindings: set[tuple[str, str, str, int, str]] = set()
            for node in document.nodes:
                registration = registrations.get(
                    (node.operator_id, node.operator_version)
                )
                if registration is None:
                    continue
                try:
                    config = (
                        registration.node_class.config_contract.model.model_validate(
                            node.config_dict()
                        ).model_dump(mode="json")
                    )
                except ValidationError:
                    continue
                for declaration in registration.secret_inputs:
                    try:
                        dependencies = {
                            dependency: cast(JsonValue, config[dependency])
                            for dependency in declaration.config_dependencies
                        }
                        dependency_sha256 = node_secret_dependency_sha256(dependencies)
                    except (KeyError, InvalidNodeSecretDependenciesError):
                        continue
                    valid_secret_bindings.add(
                        (
                            node.id,
                            declaration.name,
                            node.operator_id,
                            node.operator_version,
                            dependency_sha256,
                        )
                    )

            stored_secrets = await unit_of_work.node_secrets.list_for_graph(graph_id)
            for secret in stored_secrets:
                binding = (
                    secret.node_id,
                    secret.name,
                    secret.operator_id,
                    secret.operator_version,
                    secret.dependency_sha256,
                )
                if binding in valid_secret_bindings:
                    continue
                await unit_of_work.node_secrets.remove(
                    graph_id,
                    secret.node_id,
                    secret.name,
                )
            graph.replace(
                name=name,
                document=document,
                expected_revision=expected_revision,
            )
            await unit_of_work.graphs.add_revision(graph.snapshot())
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                ) from exc
        return graph

    async def delete(self, graph_id: UUID, *, expected_revision: int) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.graphs.lock_revision(graph_id, expected_revision)
            graph = await unit_of_work.graphs.get(graph_id)
            if graph is None:
                raise NotFoundError("Saved graph", str(graph_id))
            graph.ensure_revision(expected_revision)
            await unit_of_work.graphs.remove(graph)
            try:
                await unit_of_work.commit()
            except ConcurrentWriteError as exc:
                raise SavedGraphRevisionConflictError(
                    graph_id=graph_id,
                    expected_revision=graph.revision,
                    actual_revision=None,
                ) from exc

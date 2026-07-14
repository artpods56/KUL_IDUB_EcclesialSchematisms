from types import TracebackType
from typing import Self
from uuid import UUID

import pytest

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.errors import (
    ConcurrentWriteError,
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphDocument,
    SavedGraphNode,
)


class FakeSavedGraphRepository:
    def __init__(self, graphs: dict[UUID, SavedGraph]) -> None:
        self._graphs = graphs

    async def add(self, graph: SavedGraph) -> None:
        self._graphs[graph.id] = graph

    async def get(self, graph_id: UUID) -> SavedGraph | None:
        return self._graphs.get(graph_id)

    async def list(self) -> list[SavedGraph]:
        return list(self._graphs.values())

    async def remove(self, graph: SavedGraph) -> None:
        self._graphs.pop(graph.id, None)


class FakeSavedGraphUnitOfWork:
    def __init__(
        self,
        graphs: dict[UUID, SavedGraph],
        commit_error: ConcurrentWriteError | None,
    ) -> None:
        self.graphs = FakeSavedGraphRepository(graphs)
        self._commit_error = commit_error
        self.commit_count = 0
        self.rollback_count = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if exc_type is not None:
            await self.rollback()

    async def commit(self) -> None:
        self.commit_count += 1
        if self._commit_error is not None:
            raise self._commit_error

    async def rollback(self) -> None:
        self.rollback_count += 1


class FakeSavedGraphUnitOfWorkFactory:
    def __init__(self) -> None:
        self.graphs: dict[UUID, SavedGraph] = {}
        self.commit_error: ConcurrentWriteError | None = None
        self.created: list[FakeSavedGraphUnitOfWork] = []

    def __call__(self) -> FakeSavedGraphUnitOfWork:
        unit_of_work = FakeSavedGraphUnitOfWork(
            self.graphs,
            self.commit_error,
        )
        self.created.append(unit_of_work)
        return unit_of_work


def _document(node_id: str = "draft-node") -> SavedGraphDocument:
    return SavedGraphDocument(
        nodes=(
            SavedGraphNode(
                id=node_id,
                operator_id="example.operator",
                operator_version=1,
                config={"draft": True},
                position=GraphPoint(x=1.0, y=2.0),
            ),
        )
    )


@pytest.mark.asyncio
async def test_create_adds_graph_and_commits_once() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    service = SavedGraphService(factory)

    graph = await service.create(name="  My draft  ", document=_document())

    assert graph.name == "My draft"
    assert factory.graphs == {graph.id: graph}
    assert factory.created[-1].commit_count == 1
    assert factory.created[-1].rollback_count == 0


@pytest.mark.asyncio
async def test_list_and_get_are_read_only() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    saved = SavedGraph(name="Saved", document=_document())
    factory.graphs[saved.id] = saved
    service = SavedGraphService(factory)

    listed = await service.list()
    loaded = await service.get(saved.id)

    assert listed == [saved]
    assert loaded is saved
    assert [unit_of_work.commit_count for unit_of_work in factory.created] == [0, 0]


@pytest.mark.asyncio
async def test_get_raises_not_found_for_unknown_graph() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    service = SavedGraphService(factory)
    graph_id = UUID("00000000-0000-0000-0000-000000000404")

    with pytest.raises(NotFoundError, match=str(graph_id)):
        await service.get(graph_id)

    assert factory.created[-1].commit_count == 0


@pytest.mark.asyncio
async def test_replace_updates_graph_and_commits_once() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Original", document=SavedGraphDocument())
    factory.graphs[graph.id] = graph
    service = SavedGraphService(factory)
    replacement = _document("replacement")

    replaced = await service.replace(
        graph.id,
        name="Replacement",
        document=replacement,
        expected_revision=1,
    )

    assert replaced is graph
    assert graph.name == "Replacement"
    assert graph.document == replacement
    assert graph.revision == 2
    assert factory.created[-1].commit_count == 1


@pytest.mark.asyncio
async def test_replace_preserves_domain_revision_conflict() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Current", document=SavedGraphDocument(), revision=2)
    factory.graphs[graph.id] = graph
    service = SavedGraphService(factory)

    with pytest.raises(SavedGraphRevisionConflictError) as raised:
        await service.replace(
            graph.id,
            name="Stale update",
            document=_document(),
            expected_revision=1,
        )

    assert raised.value.expected_revision == 1
    assert raised.value.actual_revision == 2
    assert factory.created[-1].commit_count == 0
    assert factory.created[-1].rollback_count == 1


@pytest.mark.asyncio
async def test_replace_translates_concurrent_commit_to_revision_conflict() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Current", document=SavedGraphDocument())
    factory.graphs[graph.id] = graph
    concurrent_error = ConcurrentWriteError("concurrent update")
    factory.commit_error = concurrent_error
    service = SavedGraphService(factory)

    with pytest.raises(SavedGraphRevisionConflictError) as raised:
        await service.replace(
            graph.id,
            name="Competing update",
            document=_document(),
            expected_revision=1,
        )

    assert raised.value.graph_id == graph.id
    assert raised.value.expected_revision == 1
    assert raised.value.actual_revision is None
    assert raised.value.__cause__ is concurrent_error
    assert factory.created[-1].commit_count == 1
    assert factory.created[-1].rollback_count == 1


@pytest.mark.asyncio
async def test_delete_removes_graph_and_commits_once() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Disposable", document=SavedGraphDocument())
    factory.graphs[graph.id] = graph
    service = SavedGraphService(factory)

    await service.delete(graph.id, expected_revision=1)

    assert graph.id not in factory.graphs
    assert factory.created[-1].commit_count == 1


@pytest.mark.asyncio
async def test_delete_raises_not_found_without_committing() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    service = SavedGraphService(factory)
    graph_id = UUID("00000000-0000-0000-0000-000000000404")

    with pytest.raises(NotFoundError, match=str(graph_id)):
        await service.delete(graph_id, expected_revision=1)

    assert factory.created[-1].commit_count == 0
    assert factory.created[-1].rollback_count == 1


@pytest.mark.asyncio
async def test_delete_rejects_stale_client_revision_without_removing_graph() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Newer graph", document=SavedGraphDocument(), revision=2)
    factory.graphs[graph.id] = graph
    service = SavedGraphService(factory)

    with pytest.raises(SavedGraphRevisionConflictError) as raised:
        await service.delete(graph.id, expected_revision=1)

    assert raised.value.actual_revision == 2
    assert graph.id in factory.graphs
    assert factory.created[-1].commit_count == 0


@pytest.mark.asyncio
async def test_delete_translates_concurrent_commit_to_revision_conflict() -> None:
    factory = FakeSavedGraphUnitOfWorkFactory()
    graph = SavedGraph(name="Competing delete", document=SavedGraphDocument())
    factory.graphs[graph.id] = graph
    concurrent_error = ConcurrentWriteError("concurrent delete")
    factory.commit_error = concurrent_error
    service = SavedGraphService(factory)

    with pytest.raises(SavedGraphRevisionConflictError) as raised:
        await service.delete(graph.id, expected_revision=1)

    assert raised.value.graph_id == graph.id
    assert raised.value.expected_revision == graph.revision
    assert raised.value.actual_revision is None
    assert raised.value.__cause__ is concurrent_error

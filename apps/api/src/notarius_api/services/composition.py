"""Composition root for workbench-facing application components."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from notarius_core.artifacts import InMemoryUnitOfWork
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.operators.arithmetic import (
    IntegerValueOutputWriter,
    IntegerValueResolver,
)
from notarius_core.operators.text import TextValueOutputWriter, TextValueResolver
from notarius_core.plugins import PluginRegistry, PluginRuntimeContext
from notarius_core.ports.materialized_outputs import WorkbenchUnitOfWorkPort
from notarius_core.ports.node_secrets import (
    NodeSecretResolverPort,
    UnavailableNodeSecretResolver,
)
from notarius_core.ports.storage import FileStoragePort
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.materialization import InputMaterializer
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriterRegistry,
    OutputPersister,
)
from notarius_core.runtime.resolvers import Resolver, ResolverRegistry
from notarius_storage import LocalFileObjectStore

from notarius_api.services.artifacts import ArtifactService
from notarius_api.services.execution.compiler import GraphCompiler
from notarius_api.services.execution.coordinator import GraphExecutionCoordinator
from notarius_api.services.execution.edge_values import EdgeValueResolver
from notarius_api.services.execution.inline import InlineExecutionEngine
from notarius_api.services.execution.manager import RunExecutionManager
from notarius_api.services.execution.node_execution import NodeExecutionService
from notarius_api.services.execution.prefect import PrefectExecutionEngine
from notarius_api.services.execution.preflight import GraphRunPreflight
from notarius_api.services.execution.run_graph import RunGraph
from notarius_api.services.invocation_cache import PersistentInvocationCache
from notarius_api.services.materializations import MaterializationService
from notarius_api.services.modules import GraphModuleCatalog
from notarius_api.services.run_presenter import RunResultPresenter
from notarius_api.services.uploads import ImageUploadService


_WORKBENCH_BUCKET = "workbench-artifacts"


@dataclass(frozen=True, slots=True)
class WorkbenchComponents:
    plugin_registry: PluginRegistry
    uploads: ImageUploadService
    modules: GraphModuleCatalog
    run_graph: RunGraph
    execution_manager: RunExecutionManager
    materializations: MaterializationService
    presenter: RunResultPresenter
    artifacts: ArtifactService


def build_workbench_components(
    *,
    plugin_registry: PluginRegistry,
    execution_backend: Literal["prefect", "inline"],
    prefect_task_retries: int = 0,
    prefect_task_retry_delay_seconds: float = 0,
    workspace: Path | None = None,
    unit_of_work: WorkbenchUnitOfWorkPort | None = None,
    storage: FileStoragePort | None = None,
    storage_backend: str = "local",
    bucket: str = _WORKBENCH_BUCKET,
    saved_graphs: SavedGraphService | None = None,
    node_secrets: NodeSecretResolverPort | None = None,
) -> WorkbenchComponents:
    resolved_workspace = (
        (
            workspace
            or Path(
                os.getenv(
                    "NOTARIUS_WORKSPACE",
                    ".notarius-artifacts/workbench",
                )
            )
        )
        .expanduser()
        .resolve()
    )
    uploads_dir = resolved_workspace / "uploads"
    uploads = ImageUploadService(uploads_dir)
    resolved_unit_of_work = unit_of_work or InMemoryUnitOfWork()
    resolved_storage = storage or LocalFileObjectStore(resolved_workspace / "objects")
    resolved_node_secrets = node_secrets or UnavailableNodeSecretResolver()
    plugin_context = PluginRuntimeContext(
        workspace=resolved_workspace,
        uploads_dir=uploads_dir,
        storage=resolved_storage,
        uow=resolved_unit_of_work,
        bucket=bucket,
        storage_backend=storage_backend,
        node_secrets=resolved_node_secrets,
    )

    resolvers = [
        cast(Resolver[object], IntegerValueResolver(uow=resolved_unit_of_work)),
        cast(Resolver[object], TextValueResolver(uow=resolved_unit_of_work)),
    ]
    resolvers.extend(plugin_registry.build_resolvers(plugin_context))
    resolver_registry = ResolverRegistry(resolvers)

    writers: list[ArtifactOutputWriter] = [
        IntegerValueOutputWriter(uow=resolved_unit_of_work),
        TextValueOutputWriter(uow=resolved_unit_of_work),
    ]
    writers.extend(plugin_registry.build_writers(plugin_context))
    writer_registry = ArtifactWriterRegistry(writers)

    artifacts = ArtifactService(resolved_unit_of_work, resolved_storage)
    modules = GraphModuleCatalog(saved_graphs)
    materializations = MaterializationService(
        resolved_unit_of_work,
        artifacts,
        saved_graphs,
    )
    presenter = RunResultPresenter(artifacts)
    compiler = GraphCompiler(
        plugin_registry=plugin_registry,
        plugin_context=plugin_context,
        module_catalog=modules,
    )
    edge_values = EdgeValueResolver(
        resolvers=resolver_registry,
        writers=writer_registry,
        artifacts=artifacts,
    )
    runtime = NodeRuntime(
        materializer=InputMaterializer(resolver_registry),
        persister=OutputPersister(writer_registry),
        invocation_cache=PersistentInvocationCache(
            unit_of_work=resolved_unit_of_work,
            storage=resolved_storage,
        ),
    )
    node_execution = NodeExecutionService(
        runtime=runtime,
        edge_values=edge_values,
        node_secrets=resolved_node_secrets,
    )
    coordinator = GraphExecutionCoordinator(node_execution=node_execution)
    if execution_backend == "prefect":
        engine = PrefectExecutionEngine(
            coordinator=coordinator,
            task_retries=prefect_task_retries,
            task_retry_delay_seconds=prefect_task_retry_delay_seconds,
        )
    else:
        engine = InlineExecutionEngine(coordinator=coordinator)
    preflight = GraphRunPreflight(
        plugin_registry=plugin_registry,
        saved_graphs=saved_graphs,
    )
    run_graph = RunGraph(
        preflight=preflight,
        compiler=compiler,
        engine=engine,
        materializations=materializations,
    )
    execution_manager = RunExecutionManager(run_graph)
    return WorkbenchComponents(
        plugin_registry=plugin_registry,
        uploads=uploads,
        modules=modules,
        run_graph=run_graph,
        execution_manager=execution_manager,
        materializations=materializations,
        presenter=presenter,
        artifacts=artifacts,
    )


__all__ = ["WorkbenchComponents", "build_workbench_components"]

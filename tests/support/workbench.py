"""Workbench dependency-override helpers shared across API tests."""

from fastapi import FastAPI

from grafy_api.services.composition import WorkbenchComponents
from grafy_api.v1.routes.artifacts.dependencies import artifact_service
from grafy_api.v1.routes.catalog.dependencies import (
    graph_module_catalog,
    graph_module_executor,
    plugin_registry,
)
from grafy_api.v1.routes.executions.dependencies import (
    execution_admission_limiter,
    execution_history_service,
    materialization_service,
    run_execution_manager,
    run_graph_service,
    run_result_presenter,
)
from grafy_api.v1.routes.uploads.dependencies import image_upload_service

from tests.support.identity import install_browser_actor_override


def install_workbench_dependency_overrides(
    application: FastAPI,
    components: WorkbenchComponents,
) -> None:
    """Route every workbench endpoint to one cohesive component graph."""

    application.dependency_overrides.update(
        {
            plugin_registry: lambda: components.plugin_registry,
            image_upload_service: lambda: components.uploads,
            run_graph_service: lambda: components.run_graph,
            execution_admission_limiter: lambda: components.execution_admission,
            run_execution_manager: lambda: components.execution_manager,
            execution_history_service: lambda: components.execution_history,
            materialization_service: lambda: components.materializations,
            run_result_presenter: lambda: components.presenter,
            artifact_service: lambda: components.artifacts,
            graph_module_catalog: lambda: components.modules,
            graph_module_executor: lambda: components.run_graph,
        }
    )
    install_browser_actor_override(application)


__all__ = ["install_workbench_dependency_overrides"]
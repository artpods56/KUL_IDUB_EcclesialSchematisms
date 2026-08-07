from typing import Annotated

from fastapi import Depends, Request

from notarius_core.plugins import PluginRegistry
from notarius_core.ports.modules import GraphModuleExecutorPort

from .services import GraphModuleCatalog


def plugin_registry(request: Request) -> PluginRegistry:
    registry = getattr(request.app.state, "workbench_plugin_registry", None)
    if not isinstance(registry, PluginRegistry):
        raise RuntimeError("Workbench plugin registry is not initialized")
    return registry


PluginRegistryDependency = Annotated[
    PluginRegistry,
    Depends(plugin_registry),
]


def graph_module_catalog(request: Request) -> GraphModuleCatalog:
    catalog = getattr(request.app.state, "graph_modules", None)
    if not isinstance(catalog, GraphModuleCatalog):
        raise RuntimeError("Graph module catalog is not initialized")
    return catalog


GraphModuleCatalogDependency = Annotated[
    GraphModuleCatalog,
    Depends(graph_module_catalog),
]


def graph_module_executor(request: Request) -> GraphModuleExecutorPort:
    executor = getattr(request.app.state, "run_graph", None)
    if not isinstance(executor, GraphModuleExecutorPort):
        raise RuntimeError("Graph module executor is not initialized")
    return executor


GraphModuleExecutorDependency = Annotated[
    GraphModuleExecutorPort,
    Depends(graph_module_executor),
]


__all__ = [
    "GraphModuleCatalogDependency",
    "GraphModuleExecutorDependency",
    "PluginRegistryDependency",
    "graph_module_catalog",
    "graph_module_executor",
    "plugin_registry",
]

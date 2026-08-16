from typing import Annotated

from fastapi import Depends, Request

from grafy_core.plugins import PluginRegistry
from grafy_core.ports.modules import GraphModuleExecutorPort

from grafy_api.app_state import get_resources

from .services import GraphModuleCatalog


def plugin_registry(request: Request) -> PluginRegistry:
    return get_resources(request.app).plugin_registry


PluginRegistryDependency = Annotated[
    PluginRegistry,
    Depends(plugin_registry),
]


def graph_module_catalog(request: Request) -> GraphModuleCatalog:
    return get_resources(request.app).graph_modules


GraphModuleCatalogDependency = Annotated[
    GraphModuleCatalog,
    Depends(graph_module_catalog),
]


def graph_module_executor(request: Request) -> GraphModuleExecutorPort:
    return get_resources(request.app).run_graph


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

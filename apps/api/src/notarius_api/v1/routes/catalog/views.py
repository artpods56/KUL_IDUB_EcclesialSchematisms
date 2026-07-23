from fastapi import APIRouter

from .dependencies import (
    GraphModuleCatalogDependency,
    GraphModuleExecutorDependency,
    PluginRegistryDependency,
)
from .models import NodeRegistryResponse


router = APIRouter(tags=["workbench"])


@router.get("/nodes", response_model=NodeRegistryResponse)
async def list_nodes(
    registry: PluginRegistryDependency,
    modules: GraphModuleCatalogDependency,
    module_executor: GraphModuleExecutorDependency,
) -> NodeRegistryResponse:
    module_listing = await modules.list()
    return NodeRegistryResponse.from_registry(
        registry,
        module_listing,
        module_executor,
    )


__all__ = ["router"]

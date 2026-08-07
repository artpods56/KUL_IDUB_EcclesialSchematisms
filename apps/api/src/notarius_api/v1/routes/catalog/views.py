from fastapi import APIRouter

from .dependencies import (
    GraphModuleCatalogDependency,
    GraphModuleExecutorDependency,
    PluginRegistryDependency,
)
from .models import NodeRegistryResponse
from notarius_api.v1.routes.workspace_scope import LegacyWorkspaceDependency


router = APIRouter(tags=["workbench"])


@router.get("/nodes", response_model=NodeRegistryResponse)
async def list_nodes(
    registry: PluginRegistryDependency,
    modules: GraphModuleCatalogDependency,
    module_executor: GraphModuleExecutorDependency,
    workspace_id: LegacyWorkspaceDependency,
) -> NodeRegistryResponse:
    module_listing = await modules.list(workspace_id)
    return NodeRegistryResponse.from_registry(
        registry,
        module_listing,
        module_executor,
    )


__all__ = ["router"]

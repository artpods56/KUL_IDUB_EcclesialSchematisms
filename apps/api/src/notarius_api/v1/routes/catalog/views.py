from fastapi import APIRouter

from notarius_core.domain.identity import WorkspaceCapability

from notarius_api.v1.routes.auth.dependencies import require_workspace_capability

from .dependencies import (
    GraphModuleCatalogDependency,
    GraphModuleExecutorDependency,
    PluginRegistryDependency,
)
from .models import NodeRegistryResponse


router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["workbench"])


@router.get("/nodes", response_model=NodeRegistryResponse)
async def list_nodes(
    registry: PluginRegistryDependency,
    modules: GraphModuleCatalogDependency,
    module_executor: GraphModuleExecutorDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> NodeRegistryResponse:
    module_listing = await modules.list(access.workspace_id)
    return NodeRegistryResponse.from_registry(
        registry,
        module_listing,
        module_executor,
    )


__all__ = ["router"]

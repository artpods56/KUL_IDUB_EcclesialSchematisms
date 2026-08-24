from fastapi import APIRouter, Request

from grafy_core.domain.identity import WorkspaceCapability

from grafy_api.app_state import get_app_settings, get_resources
from grafy_api.v1.routes.auth.dependencies import require_workspace_capability

from .dependencies import (
    GraphModuleCatalogDependency,
    GraphModuleExecutorDependency,
    PluginReleaseServiceDependency,
    PluginRegistryDependency,
)
from .models import NodeRegistryResponse, PluginCatalogExecutionSupport


router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["workbench"])


@router.get("/nodes", response_model=NodeRegistryResponse)
async def list_nodes(
    request: Request,
    registry: PluginRegistryDependency,
    modules: GraphModuleCatalogDependency,
    plugin_releases: PluginReleaseServiceDependency,
    module_executor: GraphModuleExecutorDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> NodeRegistryResponse:
    resources = get_resources(request.app)
    settings = get_app_settings(request.app)
    module_listing = await modules.list(access.workspace_id)
    workspace_plugin_releases = (
        []
        if plugin_releases is None
        else await plugin_releases.list_current(access.workspace_id)
    )
    return NodeRegistryResponse.from_registry(
        registry,
        module_listing,
        module_executor,
        workspace_plugin_releases,
        PluginCatalogExecutionSupport(
            runtime_available=resources.plugin_runtime is not None,
            runtime_profile=(
                settings.plugin_runtime_profile
                if resources.plugin_runtime is not None
                else None
            ),
        ),
    )


__all__ = ["router"]

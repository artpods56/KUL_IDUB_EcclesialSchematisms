"""Typed FastAPI application state for identity and workbench resources."""

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI

from notarius_core.application.collaboration import CollaborationService
from notarius_core.application.identity import IdentityService
from notarius_core.application.modules import ModuleLibraryService
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.application.templates import TemplateService
from notarius_core.plugins import PluginRegistry
from notarius_persistence.database import Database
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork

from notarius_api.v1.routes.artifacts.services import ArtifactService
from notarius_api.v1.routes.auth.services import AuthService
from notarius_api.v1.routes.catalog.services import GraphModuleCatalog
from notarius_api.v1.routes.collaboration.hub import GraphRoomHub
from notarius_api.v1.routes.executions.runtime.manager import RunExecutionManager
from notarius_api.v1.routes.executions.runtime.admission import (
    ExecutionAdmissionLimiter,
)
from notarius_api.v1.routes.executions.runtime.run_graph import RunGraph
from notarius_api.v1.routes.executions.services import (
    ExecutionHistoryService,
    MaterializationService,
    RunResultPresenter,
)
from notarius_api.v1.routes.node_secrets.services import NodeSecretService
from notarius_api.v1.routes.uploads.services import ImageUploadService


@dataclass(slots=True)
class AppIdentity:
    """Auth and workspace-identity services attached for the app lifetime."""

    identity_uow_factory: Callable[[], SqlAlchemyUnitOfWork]
    identity_service: IdentityService
    auth_service: AuthService


@dataclass(slots=True)
class AppResources:
    """Workbench services constructed during API lifespan and torn down once."""

    database: Database
    plugin_registry: PluginRegistry
    uploads: ImageUploadService
    graph_modules: GraphModuleCatalog
    module_library: ModuleLibraryService
    templates: TemplateService
    run_graph: RunGraph
    execution_admission: ExecutionAdmissionLimiter
    execution_manager: RunExecutionManager
    execution_history: ExecutionHistoryService
    materializations: MaterializationService
    presenter: RunResultPresenter
    artifacts: ArtifactService
    saved_graphs: SavedGraphService
    collaboration: CollaborationService
    node_secrets: NodeSecretService
    graph_room_hub: GraphRoomHub

    async def cleanup(self) -> None:
        await self.graph_room_hub.shutdown()
        await self.execution_manager.shutdown()
        await self.artifacts.close()
        await self.database.dispose()


def get_identity(app: FastAPI) -> AppIdentity:
    identity = getattr(app.state, "identity", None)
    if not isinstance(identity, AppIdentity):
        raise RuntimeError("Application identity services are not initialized")
    return identity


def get_resources(app: FastAPI) -> AppResources:
    resources = getattr(app.state, "resources", None)
    if not isinstance(resources, AppResources):
        raise RuntimeError("Application resources are not initialized")
    return resources

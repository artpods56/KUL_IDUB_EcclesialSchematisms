from typing import Annotated

from fastapi import Depends, Request

from .runtime.manager import RunExecutionManager
from .runtime.run_graph import RunGraph
from .services import (
    ExecutionHistoryService,
    MaterializationService,
    RunResultPresenter,
)


def run_graph_service(request: Request) -> RunGraph:
    service = getattr(request.app.state, "run_graph", None)
    if not isinstance(service, RunGraph):
        raise RuntimeError("Run graph service is not initialized")
    return service


RunGraphDependency = Annotated[RunGraph, Depends(run_graph_service)]


def run_execution_manager(request: Request) -> RunExecutionManager:
    manager = getattr(request.app.state, "execution_manager", None)
    if not isinstance(manager, RunExecutionManager):
        raise RuntimeError("Run execution manager is not initialized")
    return manager


RunExecutionManagerDependency = Annotated[
    RunExecutionManager,
    Depends(run_execution_manager),
]


def execution_history_service(request: Request) -> ExecutionHistoryService:
    service = getattr(request.app.state, "execution_history", None)
    if not isinstance(service, ExecutionHistoryService):
        raise RuntimeError("Execution history service is not initialized")
    return service


ExecutionHistoryDependency = Annotated[
    ExecutionHistoryService,
    Depends(execution_history_service),
]


def materialization_service(request: Request) -> MaterializationService:
    service = getattr(request.app.state, "materializations", None)
    if not isinstance(service, MaterializationService):
        raise RuntimeError("Materialization service is not initialized")
    return service


MaterializationDependency = Annotated[
    MaterializationService,
    Depends(materialization_service),
]


def run_result_presenter(request: Request) -> RunResultPresenter:
    presenter = getattr(request.app.state, "run_result_presenter", None)
    if not isinstance(presenter, RunResultPresenter):
        raise RuntimeError("Run result presenter is not initialized")
    return presenter


RunResultPresenterDependency = Annotated[
    RunResultPresenter,
    Depends(run_result_presenter),
]


__all__ = [
    "ExecutionHistoryDependency",
    "MaterializationDependency",
    "RunExecutionManagerDependency",
    "RunGraphDependency",
    "RunResultPresenterDependency",
    "execution_history_service",
    "materialization_service",
    "run_execution_manager",
    "run_graph_service",
    "run_result_presenter",
]

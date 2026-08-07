from typing import Annotated

from fastapi import Depends, Request

from notarius_api.app_state import get_resources

from .runtime.manager import RunExecutionManager
from .runtime.run_graph import RunGraph
from .services import (
    ExecutionHistoryService,
    MaterializationService,
    RunResultPresenter,
)


def run_graph_service(request: Request) -> RunGraph:
    return get_resources(request.app).run_graph


RunGraphDependency = Annotated[RunGraph, Depends(run_graph_service)]


def run_execution_manager(request: Request) -> RunExecutionManager:
    return get_resources(request.app).execution_manager


RunExecutionManagerDependency = Annotated[
    RunExecutionManager,
    Depends(run_execution_manager),
]


def execution_history_service(request: Request) -> ExecutionHistoryService:
    return get_resources(request.app).execution_history


ExecutionHistoryDependency = Annotated[
    ExecutionHistoryService,
    Depends(execution_history_service),
]


def materialization_service(request: Request) -> MaterializationService:
    return get_resources(request.app).materializations


MaterializationDependency = Annotated[
    MaterializationService,
    Depends(materialization_service),
]


def run_result_presenter(request: Request) -> RunResultPresenter:
    return get_resources(request.app).presenter


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

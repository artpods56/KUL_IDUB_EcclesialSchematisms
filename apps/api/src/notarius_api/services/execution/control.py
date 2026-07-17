"""Cooperative cancellation and outer-node progress for one graph execution."""

import asyncio


class RunExecutionCancelled(asyncio.CancelledError):
    """Raised when a managed graph execution observes cancellation intent."""


class RunExecutionControl:
    """Mutable, in-process control shared by one top-level execution tree."""

    def __init__(self) -> None:
        self._cancel_requested = False
        self._active_node_id: str | None = None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def active_node_id(self) -> str | None:
        return self._active_node_id

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def check_cancelled(self) -> None:
        if self._cancel_requested:
            raise RunExecutionCancelled("Graph execution was cancelled")

    def start_outer_node(self, node_id: str) -> None:
        self.check_cancelled()
        self._active_node_id = node_id

    def finish_outer_node(self, node_id: str) -> None:
        if self._active_node_id == node_id:
            self._active_node_id = None


__all__ = ["RunExecutionCancelled", "RunExecutionControl"]

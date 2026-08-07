from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request


LEGACY_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")


def legacy_workspace_id(request: Request) -> UUID:
    workspace_id = getattr(
        request.app.state,
        "legacy_workspace_id",
        LEGACY_WORKSPACE_ID,
    )
    if not isinstance(workspace_id, UUID):
        raise RuntimeError("Legacy workbench workspace is not initialized")
    return workspace_id


LegacyWorkspaceDependency = Annotated[
    UUID,
    Depends(legacy_workspace_id),
]


__all__ = [
    "LEGACY_WORKSPACE_ID",
    "LegacyWorkspaceDependency",
    "legacy_workspace_id",
]

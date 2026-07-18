from typing import Annotated

from fastapi import Depends, Request

from notarius_api.services.artifacts import ArtifactService


def artifact_service(request: Request) -> ArtifactService:
    service = getattr(request.app.state, "artifacts", None)
    if not isinstance(service, ArtifactService):
        raise RuntimeError("Artifact service is not initialized")
    return service


ArtifactDependency = Annotated[
    ArtifactService,
    Depends(artifact_service),
]


__all__ = ["ArtifactDependency", "artifact_service"]

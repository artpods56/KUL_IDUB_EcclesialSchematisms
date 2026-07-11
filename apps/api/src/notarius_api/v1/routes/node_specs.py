from typing import Annotated

from fastapi import APIRouter, Depends

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import NodeSpecResponse
from notarius_core.application.workflows import NodeSpecRegistry

router = APIRouter(tags=["node-specs"])


@router.get("/node-specs", response_model=list[NodeSpecResponse])
async def list_node_specs(
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> list[NodeSpecResponse]:
    return [
        NodeSpecResponse.from_domain(spec)
        for spec in sorted(node_specs.values(), key=lambda item: (item.id, item.version))
    ]

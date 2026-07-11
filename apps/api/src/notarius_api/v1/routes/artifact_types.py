from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    ArtifactPayloadSchemaResponse,
    ArtifactTypePortUseResponse,
    ArtifactTypeResponse,
)
from notarius_core.application.workflows import NodeSpecRegistry
from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.models import NodeSpec, PortSpec
from notarius_worker.operators import builtin_artifact_payload_models

router = APIRouter(tags=["artifact-types"])


@router.get("/artifact-types", response_model=list[ArtifactTypeResponse])
async def list_artifact_types(
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> list[ArtifactTypeResponse]:
    artifact_types: dict[tuple[str, int], ArtifactTypeResponse] = {}
    for spec in sorted(node_specs.values(), key=lambda item: (item.id, item.version)):
        for port in spec.inputs:
            entry = _artifact_type_entry(artifact_types, port)
            entry.consumed_by.append(_port_use_response(spec, port))
            entry.sequence = entry.sequence or port.sequence
        for port in spec.outputs:
            entry = _artifact_type_entry(artifact_types, port)
            entry.produced_by.append(_port_use_response(spec, port))
            entry.sequence = entry.sequence or port.sequence

    return sorted(
        artifact_types.values(),
        key=lambda item: (item.artifact_type, item.schema_version),
    )


@router.get(
    "/artifact-payload-schemas",
    response_model=list[ArtifactPayloadSchemaResponse],
)
async def list_artifact_payload_schemas() -> list[ArtifactPayloadSchemaResponse]:
    payload_models = builtin_artifact_payload_models()
    return [
        _payload_schema_response(artifact_type, schema_version, payload_models)
        for artifact_type, schema_version in sorted(payload_models)
    ]


@router.get(
    "/artifact-payload-schemas/{artifact_type}/{schema_version}",
    response_model=ArtifactPayloadSchemaResponse,
)
async def get_artifact_payload_schema(
    artifact_type: str,
    schema_version: int,
) -> ArtifactPayloadSchemaResponse:
    payload_models = builtin_artifact_payload_models()
    key = (artifact_type, schema_version)
    if key not in payload_models:
        raise NotFoundError(
            "ArtifactPayloadSchema",
            f"{artifact_type}@v{schema_version}",
        )

    return _payload_schema_response(artifact_type, schema_version, payload_models)


def _artifact_type_entry(
    artifact_types: dict[tuple[str, int], ArtifactTypeResponse],
    port: PortSpec,
) -> ArtifactTypeResponse:
    key = (port.artifact_type, port.schema_version)
    entry = artifact_types.get(key)
    if entry is None:
        entry = ArtifactTypeResponse(
            artifact_type=port.artifact_type,
            schema_version=port.schema_version,
            sequence=port.sequence,
        )
        artifact_types[key] = entry
    return entry


def _port_use_response(spec: NodeSpec, port: PortSpec) -> ArtifactTypePortUseResponse:
    return ArtifactTypePortUseResponse(
        operator_id=spec.id,
        operator_version=spec.version,
        port_name=port.name,
        sequence=port.sequence,
        required=port.required,
    )


def _payload_schema_response(
    artifact_type: str,
    schema_version: int,
    payload_models: dict[tuple[str, int], type[BaseModel]],
) -> ArtifactPayloadSchemaResponse:
    model_type = payload_models[(artifact_type, schema_version)]
    return ArtifactPayloadSchemaResponse(
        artifact_type=artifact_type,
        schema_version=schema_version,
        json_schema=model_type.model_json_schema(),
    )

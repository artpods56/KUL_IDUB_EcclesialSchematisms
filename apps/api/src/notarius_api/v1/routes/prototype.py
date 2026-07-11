from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema

from notarius_core.prototype.artifacts import (
    MISTRAL_OCR_RESPONSE,
    OCR_PAGE_RESULT,
    SOURCE_PAGE_IMAGE,
    TABLE_CSV_BUNDLE,
    TABLE_FRAGMENT,
    TABLE_PAGE,
    ArtifactTypeKey,
    ArtifactTypeSpec,
)
from notarius_core.prototype.arithmetic import (
    ARITHMETIC_RESULT,
    INTEGER_VALUE,
    AddSubtractNode,
    MultiplyNode,
    NumberNode,
)
from notarius_core.prototype.nodes import (
    ConfigContract,
    InputContract,
    InputPortSpec,
    OutputContract,
    OutputPortSpec,
    PortShape,
)
from notarius_core.prototype.mistral_ocr import MistralOcrNode
from notarius_core.prototype.ocr import TesseractOcrNode
from notarius_core.prototype.sources import (
    ImageSequenceMergeNode,
    LocalUploadImageSourceNode,
)
from notarius_core.prototype.tables import (
    BuildTableCsvBundleNode,
    ExtractTableFragmentsNode,
    MergeTablePagesNode,
)

from notarius_api.schemas.prototype import (
    PrototypeArtifactTypeKeyResponse,
    PrototypeArtifactTypeSpecResponse,
    PrototypeFieldProjectionResponse,
    PrototypeNodeGroup,
    PrototypeNodeRegistryResponse,
    PrototypeNodeSpecResponse,
    PrototypePortDirection,
    PrototypePortResponse,
    PrototypeRunRequest,
    PrototypeRunResponse,
    PrototypeSampleRequest,
    PrototypeSelectionItemResponse,
    PrototypeUploadRequest,
)
from notarius_api.services.prototype_workbench import (
    WorkbenchGraphError,
    get_workbench_service,
)

router = APIRouter(prefix="/prototype", tags=["prototype"])


class PrototypeNodeDefinition(Protocol):
    operator_id: ClassVar[str]
    operator_version: ClassVar[int]
    config_contract: ClassVar[ConfigContract[Any]]
    input_contract: ClassVar[InputContract[Any]]
    output_contract: ClassVar[OutputContract[Any]]


@dataclass(frozen=True, slots=True)
class PrototypeNodeRegistration:
    node_class: type[PrototypeNodeDefinition]
    title: str
    group: PrototypeNodeGroup
    description: str


PROTOTYPE_NODE_REGISTRY: tuple[PrototypeNodeRegistration, ...] = (
    PrototypeNodeRegistration(
        node_class=LocalUploadImageSourceNode,
        title="Local Upload Image Source",
        group="source",
        description="Imports a configured set of staged local images into a source.page_image@1 sequence.",
    ),
    PrototypeNodeRegistration(
        node_class=ImageSequenceMergeNode,
        title="Image Sequence Merge",
        group="transform",
        description="Concatenates source.page_image@1 sequences without copying artifact payload bytes.",
    ),
    PrototypeNodeRegistration(
        node_class=TesseractOcrNode,
        title="Tesseract OCR",
        group="ocr",
        description="Recognizes plain text from an ordered source page image sequence.",
    ),
    PrototypeNodeRegistration(
        node_class=MistralOcrNode,
        title="Mistral OCR 4",
        group="ocr",
        description="Runs Mistral OCR with block extraction and separate Markdown tables while preserving the full provider response.",
    ),
    PrototypeNodeRegistration(
        node_class=ExtractTableFragmentsNode,
        title="Extract Markdown Tables",
        group="transform",
        description="Extracts rectangular table fragments from provider tables with page-Markdown fallback.",
    ),
    PrototypeNodeRegistration(
        node_class=MergeTablePagesNode,
        title="Merge Page Tables",
        group="transform",
        description="Concatenates table fragments for each source image and provider page.",
    ),
    PrototypeNodeRegistration(
        node_class=BuildTableCsvBundleNode,
        title="Export Table CSVs",
        group="transform",
        description="Builds fragment, page, long-form, rectangular, and compatible-header CSV exports.",
    ),
    PrototypeNodeRegistration(
        node_class=NumberNode,
        title="Number",
        group="arithmetic",
        description="Produces a configured generic integer value.",
    ),
    PrototypeNodeRegistration(
        node_class=AddSubtractNode,
        title="Add & subtract",
        group="arithmetic",
        description="Produces addition and subtraction fields from two integer inputs.",
    ),
    PrototypeNodeRegistration(
        node_class=MultiplyNode,
        title="Multiply",
        group="arithmetic",
        description="Multiplies two generic integer inputs.",
    ),
)

PROTOTYPE_ARTIFACT_TYPES: tuple[ArtifactTypeSpec, ...] = (
    SOURCE_PAGE_IMAGE,
    OCR_PAGE_RESULT,
    MISTRAL_OCR_RESPONSE,
    TABLE_FRAGMENT,
    TABLE_PAGE,
    TABLE_CSV_BUNDLE,
    INTEGER_VALUE,
    ARITHMETIC_RESULT,
)


@router.get("/nodes", response_model=PrototypeNodeRegistryResponse)
async def list_prototype_nodes() -> PrototypeNodeRegistryResponse:
    return PrototypeNodeRegistryResponse(
        artifact_types=[
            _artifact_type_spec_response(spec) for spec in PROTOTYPE_ARTIFACT_TYPES
        ],
        nodes=[
            _node_registration_response(registration)
            for registration in PROTOTYPE_NODE_REGISTRY
        ],
    )


@router.post("/uploads", response_model=PrototypeSelectionItemResponse)
async def prototype_upload(
    request: PrototypeUploadRequest,
) -> PrototypeSelectionItemResponse:
    try:
        return await get_workbench_service().save_upload(
            filename=request.filename,
            content_base64=request.content_base64,
        )
    except WorkbenchGraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/samples", response_model=list[PrototypeSelectionItemResponse])
async def prototype_samples(
    request: PrototypeSampleRequest,
) -> list[PrototypeSelectionItemResponse]:
    return await get_workbench_service().create_sample_pages(request.count)


@router.post("/run", response_model=PrototypeRunResponse)
async def prototype_run(request: PrototypeRunRequest) -> PrototypeRunResponse:
    try:
        return await get_workbench_service().run_graph(request)
    except WorkbenchGraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}/content")
async def prototype_artifact_content(artifact_id: UUID) -> Response:
    service = get_workbench_service()
    artifact = await service.get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        content = await service.load_artifact_content(artifact)
    except WorkbenchGraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers: dict[str, str] = {}
    download_name = artifact.metadata.get("download_name")
    if isinstance(download_name, str) and download_name != "":
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return Response(
        content=content,
        media_type=artifact.content_type,
        headers=headers,
    )


def _node_registration_response(
    registration: PrototypeNodeRegistration,
) -> PrototypeNodeSpecResponse:
    return PrototypeNodeSpecResponse(
        operator_id=registration.node_class.operator_id,
        operator_version=registration.node_class.operator_version,
        title=registration.title,
        group=registration.group,
        description=registration.description,
        config_schema=_model_json_schema(
            registration.node_class.config_contract.model
        ),
        input_schema=_model_json_schema(
            registration.node_class.input_contract.model
        ),
        output_schema=_model_json_schema(
            registration.node_class.output_contract.model
        ),
        inputs=[
            _input_port_response(port)
            for port in registration.node_class.input_contract.ports.values()
        ],
        outputs=[
            _output_port_response(port)
            for port in registration.node_class.output_contract.ports.values()
        ],
    )


def _input_port_response(port: InputPortSpec) -> PrototypePortResponse:
    return _port_response(
        name=port.name,
        direction="input",
        artifact_type=port.accepts,
        shape=port.shape,
        variadic=port.variadic,
        required=port.required,
    )


def _output_port_response(port: OutputPortSpec) -> PrototypePortResponse:
    return _port_response(
        name=port.name,
        direction="output",
        artifact_type=port.produces,
        shape=port.shape,
        variadic=False,
        required=port.required,
    )


def _port_response(
    *,
    name: str,
    direction: PrototypePortDirection,
    artifact_type: ArtifactTypeKey,
    shape: PortShape,
    variadic: bool,
    required: bool,
) -> PrototypePortResponse:
    return PrototypePortResponse(
        name=name,
        direction=direction,
        artifact_type=_artifact_type_key_response(artifact_type),
        shape=shape,
        variadic=variadic,
        required=required,
    )


def _artifact_type_spec_response(
    spec: ArtifactTypeSpec,
) -> PrototypeArtifactTypeSpecResponse:
    return PrototypeArtifactTypeSpecResponse(
        key=_artifact_type_key_response(spec.key),
        title=spec.title,
        payload_schema=spec.payload_schema,
        field_projections=[
            PrototypeFieldProjectionResponse(
                path=list(projection.path),
                target_artifact_type=_artifact_type_key_response(projection.target),
                title=projection.title,
            )
            for projection in spec.field_projections
        ],
    )


def _artifact_type_key_response(
    key: ArtifactTypeKey,
) -> PrototypeArtifactTypeKeyResponse:
    return PrototypeArtifactTypeKeyResponse(
        id=key.id,
        schema_version=key.schema_version,
    )


def _model_json_schema(model: type[BaseModel]) -> dict[str, object]:
    try:
        return cast(dict[str, object], model.model_json_schema())
    except PydanticInvalidForJsonSchema as exc:
        return {
            "title": model.__name__,
            "type": "object",
            "x-schema-error": str(exc),
            "properties": {
                name: {
                    "title": name,
                    "x-python-type": str(field.annotation),
                }
                for name, field in model.model_fields.items()
            },
        }

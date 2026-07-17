from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema

from notarius_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec
from notarius_core.domain.errors import (
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.nodes import (
    ArtifactTypeVariable,
    InputPortSpec,
    OutputPortSpec,
    PortShape,
)
from notarius_core.operators.modules import GraphModuleNode
from notarius_core.plugins import NodeRegistration, PluginOrigin, PluginRegistry
from notarius_core.ports.modules import GraphModuleExecutorPort

from notarius_api.schemas.workbench import (
    ArtifactConversionKeyResponse,
    ArtifactConversionSpecResponse,
    ArtifactTypeKeyResponse,
    ArtifactTypeSpecResponse,
    FieldProjectionResponse,
    GraphMaterializationsResponse,
    ImageUploadItemResponse,
    NodeRegistryResponse,
    NodeSecretInputResponse,
    NodeSpecResponse,
    PluginSpecResponse,
    PortDirection,
    PortResponse,
    RunRequest,
    RunExecutionResponse,
    RunResponse,
    SampleRequest,
    UploadRequest,
)
from notarius_api.services.artifacts import ArtifactService
from notarius_api.services.errors import WorkbenchOperationError
from notarius_api.services.execution.run_graph import RunGraph
from notarius_api.services.execution.manager import RunExecutionManager
from notarius_api.services.materializations import MaterializationService
from notarius_api.services.modules import (
    GRAPH_MODULE_PLUGIN_SLUG,
    GraphModuleCatalog,
    GraphModuleCatalogEntry,
)
from notarius_api.services.run_presenter import RunResultPresenter
from notarius_api.services.uploads import ImageUploadService

router = APIRouter(tags=["workbench"])


def workbench_plugin_registry(request: Request) -> PluginRegistry:
    registry = getattr(request.app.state, "workbench_plugin_registry", None)
    if not isinstance(registry, PluginRegistry):
        raise RuntimeError("Workbench plugin registry is not initialized")
    return registry


PluginRegistryDependency = Annotated[
    PluginRegistry,
    Depends(workbench_plugin_registry),
]


def image_upload_service(request: Request) -> ImageUploadService:
    service = getattr(request.app.state, "image_uploads", None)
    if not isinstance(service, ImageUploadService):
        raise RuntimeError("Image upload service is not initialized")
    return service


ImageUploadDependency = Annotated[
    ImageUploadService,
    Depends(image_upload_service),
]


def run_graph_service(request: Request) -> RunGraph:
    service = getattr(request.app.state, "run_graph", None)
    if not isinstance(service, RunGraph):
        raise RuntimeError("Run graph service is not initialized")
    return service


RunGraphDependency = Annotated[
    RunGraph,
    Depends(run_graph_service),
]


def run_execution_manager(request: Request) -> RunExecutionManager:
    manager = getattr(request.app.state, "execution_manager", None)
    if not isinstance(manager, RunExecutionManager):
        raise RuntimeError("Run execution manager is not initialized")
    return manager


RunExecutionManagerDependency = Annotated[
    RunExecutionManager,
    Depends(run_execution_manager),
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


def artifact_service(request: Request) -> ArtifactService:
    service = getattr(request.app.state, "artifacts", None)
    if not isinstance(service, ArtifactService):
        raise RuntimeError("Artifact service is not initialized")
    return service


ArtifactDependency = Annotated[
    ArtifactService,
    Depends(artifact_service),
]


def graph_module_catalog(request: Request) -> GraphModuleCatalog:
    catalog = getattr(request.app.state, "graph_modules", None)
    if not isinstance(catalog, GraphModuleCatalog):
        raise RuntimeError("Graph module catalog is not initialized")
    return catalog


GraphModuleCatalogDependency = Annotated[
    GraphModuleCatalog,
    Depends(graph_module_catalog),
]


@router.get("/nodes", response_model=NodeRegistryResponse)
async def list_nodes(
    registry: PluginRegistryDependency,
    modules: GraphModuleCatalogDependency,
    module_executor: RunGraphDependency,
) -> NodeRegistryResponse:
    module_entries = await modules.list()
    return NodeRegistryResponse(
        plugins=[
            PluginSpecResponse(
                slug=plugin.slug,
                title=plugin.title,
                origin=plugin.origin,
            )
            for plugin in registry.plugins
        ]
        + [
            PluginSpecResponse(
                slug=GRAPH_MODULE_PLUGIN_SLUG,
                title="Modules",
                origin=PluginOrigin.MODULE,
            )
        ],
        artifact_types=[
            _artifact_type_spec_response(spec) for spec in registry.artifact_types
        ],
        artifact_conversions=[
            ArtifactConversionSpecResponse(
                key=ArtifactConversionKeyResponse(
                    id=conversion.key.id,
                    version=conversion.key.version,
                ),
                source_artifact_type=_artifact_type_key_response(conversion.source),
                target_artifact_type=_artifact_type_key_response(conversion.target),
                title=conversion.title,
            )
            for conversion in registry.artifact_conversions
        ],
        nodes=[
            _node_registration_response(registration) for registration in registry.nodes
        ]
        + [_graph_module_response(entry, module_executor) for entry in module_entries],
    )


@router.post("/uploads", response_model=ImageUploadItemResponse)
async def upload_file(
    request: UploadRequest,
    service: ImageUploadDependency,
) -> ImageUploadItemResponse:
    try:
        return await service.save_image_upload(
            filename=request.filename,
            content_base64=request.content_base64,
        )
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/samples", response_model=list[ImageUploadItemResponse])
async def create_samples(
    request: SampleRequest,
    service: ImageUploadDependency,
) -> list[ImageUploadItemResponse]:
    return await service.create_sample_images(request.count)


@router.post("/runs", response_model=RunResponse)
async def run_graph(
    request: RunRequest,
    service: RunGraphDependency,
    presenter: RunResultPresenterDependency,
) -> RunResponse:
    try:
        execution = await service.run(request)
        return await presenter.run_response(execution)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/executions",
    response_model=RunExecutionResponse,
    status_code=202,
)
async def start_graph_execution(
    request: RunRequest,
    manager: RunExecutionManagerDependency,
    presenter: RunResultPresenterDependency,
) -> RunExecutionResponse:
    execution = await manager.start(request)
    return await presenter.execution_response(execution)


@router.get(
    "/executions/{execution_id}",
    response_model=RunExecutionResponse,
)
async def get_graph_execution(
    execution_id: UUID,
    manager: RunExecutionManagerDependency,
    presenter: RunResultPresenterDependency,
) -> RunExecutionResponse:
    try:
        execution = await manager.get(execution_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await presenter.execution_response(execution)


@router.delete(
    "/executions/{execution_id}",
    response_model=RunExecutionResponse,
)
async def cancel_graph_execution(
    execution_id: UUID,
    manager: RunExecutionManagerDependency,
    presenter: RunResultPresenterDependency,
) -> RunExecutionResponse:
    try:
        execution = await manager.cancel(execution_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await presenter.execution_response(execution)


@router.get(
    "/graphs/{graph_id}/materializations",
    response_model=GraphMaterializationsResponse,
)
async def get_graph_materializations(
    graph_id: UUID,
    graph_revision: Annotated[int, Query(ge=1)],
    service: MaterializationDependency,
    presenter: RunResultPresenterDependency,
) -> GraphMaterializationsResponse:
    try:
        materializations = await service.list_for_graph(
            graph_id,
            graph_revision,
        )
        return await presenter.materializations_response(
            graph_id,
            graph_revision,
            materializations,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: UUID,
    service: ArtifactDependency,
) -> Response:
    artifact = await service.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        content = await service.load_content(artifact)
    except WorkbenchOperationError as exc:
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
    registration: NodeRegistration,
) -> NodeSpecResponse:
    return NodeSpecResponse(
        operator_id=registration.node_class.operator_id,
        operator_version=registration.node_class.operator_version,
        plugin_slug=registration.plugin_slug,
        title=registration.title,
        description=registration.description,
        config_schema=_model_json_schema(registration.node_class.config_contract.model),
        input_schema=_model_json_schema(registration.node_class.input_contract.model),
        output_schema=_model_json_schema(registration.node_class.output_contract.model),
        inputs=[
            _input_port_response(port)
            for port in registration.node_class.input_contract.ports.values()
        ],
        outputs=[
            _output_port_response(port)
            for port in registration.node_class.output_contract.ports.values()
        ],
        secret_inputs=[
            NodeSecretInputResponse(
                name=secret_input.name,
                config_dependencies=list(secret_input.config_dependencies),
                title=secret_input.title,
                description=secret_input.description,
            )
            for secret_input in registration.secret_inputs
        ],
    )


def _graph_module_response(
    entry: GraphModuleCatalogEntry,
    module_executor: GraphModuleExecutorPort,
) -> NodeSpecResponse:
    definition = entry.definition
    node = GraphModuleNode(definition, module_executor)
    return NodeSpecResponse(
        operator_id=node.operator_id,
        operator_version=node.operator_version,
        plugin_slug=GRAPH_MODULE_PLUGIN_SLUG,
        title=node.title,
        description=node.description,
        config_schema=_model_json_schema(node.config_contract.model),
        input_schema=_model_json_schema(node.input_contract.model),
        output_schema=_model_json_schema(node.output_contract.model),
        inputs=[
            _input_port_response(port) for port in node.input_contract.ports.values()
        ],
        outputs=[
            _output_port_response(port) for port in node.output_contract.ports.values()
        ],
        module_graph_id=definition.reference.graph_id,
        module_graph_revision=definition.reference.revision,
        catalog_visible=entry.catalog_visible,
    )


def _input_port_response(port: InputPortSpec) -> PortResponse:
    return _port_response(
        name=port.name,
        title=port.title,
        description=port.description,
        direction="input",
        artifact_type_contract=port.accepts,
        shape=port.shape,
        accepted_shapes=port.accepted_shapes,
        instance_plugs=port.instance_plugs,
        variadic=port.variadic,
        required=port.required,
    )


def _output_port_response(port: OutputPortSpec) -> PortResponse:
    return _port_response(
        name=port.name,
        title=port.title,
        description=port.description,
        direction="output",
        artifact_type_contract=port.produces,
        shape=port.shape,
        accepted_shapes=(port.shape,),
        instance_plugs=False,
        variadic=False,
        required=port.required,
    )


def _port_response(
    *,
    name: str,
    title: str | None,
    description: str | None,
    direction: PortDirection,
    artifact_type_contract: ArtifactTypeKey | ArtifactTypeVariable,
    shape: PortShape,
    accepted_shapes: tuple[PortShape, ...],
    instance_plugs: bool,
    variadic: bool,
    required: bool,
) -> PortResponse:
    if isinstance(artifact_type_contract, ArtifactTypeVariable):
        artifact_type = None
        artifact_type_variable = artifact_type_contract.name
    else:
        artifact_type = _artifact_type_key_response(artifact_type_contract)
        artifact_type_variable = None
    return PortResponse(
        name=name,
        title=title,
        description=description,
        direction=direction,
        artifact_type=artifact_type,
        artifact_type_variable=artifact_type_variable,
        shape=shape,
        accepted_shapes=list(accepted_shapes),
        instance_plugs=instance_plugs,
        variadic=variadic,
        required=required,
    )


def _artifact_type_spec_response(
    spec: ArtifactTypeSpec,
) -> ArtifactTypeSpecResponse:
    return ArtifactTypeSpecResponse(
        key=_artifact_type_key_response(spec.key),
        title=spec.title,
        payload_schema=spec.payload_schema,
        field_projections=[
            FieldProjectionResponse(
                path=list(projection.path),
                target_artifact_type=_artifact_type_key_response(projection.target),
                title=projection.title,
            )
            for projection in spec.field_projections
        ],
    )


def _artifact_type_key_response(
    key: ArtifactTypeKey,
) -> ArtifactTypeKeyResponse:
    return ArtifactTypeKeyResponse(
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

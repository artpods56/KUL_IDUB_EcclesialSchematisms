from typing import Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from pydantic.errors import PydanticInvalidForJsonSchema

from notarius_core.artifacts import ArtifactFieldProjection, ArtifactTypeSpec
from notarius_core.conversions import ArtifactConversion, ArtifactConversionKey
from notarius_core.nodes import (
    ArtifactTypeVariable,
    InputPortSpec,
    OutputPortSpec,
    PortShape,
)
from notarius_core.operators.modules import GraphModuleNode
from notarius_core.plugins import (
    InstalledPlugin,
    NodeRegistration,
    NodeSecretInput,
    PluginOrigin,
    PluginRegistry,
)
from notarius_core.ports.modules import GraphModuleExecutorPort

from notarius_api.v1.models import (
    ApiResponse,
    ArtifactTypeKeyResponse,
    ArtifactTypeVariableIdentifier,
)

from .services import (
    GRAPH_MODULE_PLUGIN_SLUG,
    GraphModuleCatalogEntry,
    GraphModuleCatalogListing,
    UnavailableGraphModule,
)


PortDirection = Literal["input", "output"]


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


class FieldProjectionResponse(ApiResponse):
    path: list[str]
    target_artifact_type: ArtifactTypeKeyResponse
    title: str

    @classmethod
    def from_projection(cls, projection: ArtifactFieldProjection) -> Self:
        return cls(
            path=list(projection.path),
            target_artifact_type=ArtifactTypeKeyResponse.from_key(projection.target),
            title=projection.title,
        )


class ArtifactTypeSpecResponse(ApiResponse):
    key: ArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, object]
    field_projections: list[FieldProjectionResponse]

    @classmethod
    def from_spec(cls, spec: ArtifactTypeSpec) -> Self:
        return cls(
            key=ArtifactTypeKeyResponse.from_key(spec.key),
            title=spec.title,
            payload_schema=spec.payload_schema,
            field_projections=[
                FieldProjectionResponse.from_projection(projection)
                for projection in spec.field_projections
            ],
        )


class ArtifactConversionKeyResponse(ApiResponse):
    id: str
    version: int

    @classmethod
    def from_key(cls, key: ArtifactConversionKey) -> Self:
        return cls(id=key.id, version=key.version)


class ArtifactConversionSpecResponse(ApiResponse):
    key: ArtifactConversionKeyResponse
    source_artifact_type: ArtifactTypeKeyResponse
    target_artifact_type: ArtifactTypeKeyResponse
    title: str

    @classmethod
    def from_spec[SourceT, TargetT](
        cls,
        spec: ArtifactConversion[SourceT, TargetT],
    ) -> Self:
        return cls(
            key=ArtifactConversionKeyResponse.from_key(spec.key),
            source_artifact_type=ArtifactTypeKeyResponse.from_key(spec.source),
            target_artifact_type=ArtifactTypeKeyResponse.from_key(spec.target),
            title=spec.title,
        )


class PluginSpecResponse(ApiResponse):
    slug: str
    title: str
    origin: PluginOrigin

    @classmethod
    def from_plugin(cls, plugin: InstalledPlugin) -> Self:
        return cls(
            slug=plugin.slug,
            title=plugin.title,
            origin=plugin.origin,
        )


class PortResponse(ApiResponse):
    name: str
    title: str | None = None
    description: str | None = None
    direction: PortDirection
    artifact_type: ArtifactTypeKeyResponse | None = None
    artifact_type_variable: ArtifactTypeVariableIdentifier | None = None
    shape: PortShape
    accepted_shapes: list[PortShape]
    instance_plugs: bool = False
    variadic: bool = False
    required: bool = True

    @model_validator(mode="after")
    def validate_artifact_type_contract(self) -> Self:
        if (self.artifact_type is None) == (self.artifact_type_variable is None):
            raise ValueError(
                "Port must declare exactly one of artifact_type or "
                "artifact_type_variable"
            )
        return self

    @classmethod
    def from_input_port(cls, port: InputPortSpec) -> Self:
        artifact_type: ArtifactTypeKeyResponse | None
        artifact_type_variable: str | None
        if isinstance(port.accepts, ArtifactTypeVariable):
            artifact_type = None
            artifact_type_variable = port.accepts.name
        else:
            artifact_type = ArtifactTypeKeyResponse.from_key(port.accepts)
            artifact_type_variable = None
        return cls(
            name=port.name,
            title=port.title,
            description=port.description,
            direction="input",
            artifact_type=artifact_type,
            artifact_type_variable=artifact_type_variable,
            shape=port.shape,
            accepted_shapes=list(port.accepted_shapes),
            instance_plugs=port.instance_plugs,
            variadic=port.variadic,
            required=port.required,
        )

    @classmethod
    def from_output_port(cls, port: OutputPortSpec) -> Self:
        artifact_type: ArtifactTypeKeyResponse | None
        artifact_type_variable: str | None
        if isinstance(port.produces, ArtifactTypeVariable):
            artifact_type = None
            artifact_type_variable = port.produces.name
        else:
            artifact_type = ArtifactTypeKeyResponse.from_key(port.produces)
            artifact_type_variable = None
        return cls(
            name=port.name,
            title=port.title,
            description=port.description,
            direction="output",
            artifact_type=artifact_type,
            artifact_type_variable=artifact_type_variable,
            shape=port.shape,
            accepted_shapes=[port.shape],
            instance_plugs=False,
            variadic=False,
            required=port.required,
        )


class NodeSecretInputResponse(ApiResponse):
    name: str
    config_dependencies: list[str]
    title: str
    description: str | None = None

    @classmethod
    def from_spec(cls, spec: NodeSecretInput) -> Self:
        return cls(
            name=spec.name,
            config_dependencies=list(spec.config_dependencies),
            title=spec.title,
            description=spec.description,
        )


class NodeSpecResponse(ApiResponse):
    operator_id: str
    operator_version: int
    plugin_slug: str
    title: str
    description: str
    config_schema: dict[str, object]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    inputs: list[PortResponse]
    outputs: list[PortResponse]
    secret_inputs: list[NodeSecretInputResponse] = Field(default_factory=list)
    module_graph_id: UUID | None = None
    module_graph_revision: int | None = Field(default=None, ge=1)
    catalog_visible: bool = True

    @model_validator(mode="after")
    def validate_module_identity(self) -> Self:
        if (self.module_graph_id is None) != (self.module_graph_revision is None):
            raise ValueError(
                "module_graph_id and module_graph_revision must be provided together"
            )
        return self

    @classmethod
    def from_registration(cls, registration: NodeRegistration) -> Self:
        node_class = registration.node_class
        return cls(
            operator_id=node_class.operator_id,
            operator_version=node_class.operator_version,
            plugin_slug=registration.plugin_slug,
            title=registration.title,
            description=registration.description,
            config_schema=_model_json_schema(node_class.config_contract.model),
            input_schema=_model_json_schema(node_class.input_contract.model),
            output_schema=_model_json_schema(node_class.output_contract.model),
            inputs=[
                PortResponse.from_input_port(port)
                for port in node_class.input_contract.ports.values()
            ],
            outputs=[
                PortResponse.from_output_port(port)
                for port in node_class.output_contract.ports.values()
            ],
            secret_inputs=[
                NodeSecretInputResponse.from_spec(spec)
                for spec in registration.secret_inputs
            ],
        )

    @classmethod
    def from_graph_module(
        cls,
        entry: GraphModuleCatalogEntry,
        module_executor: GraphModuleExecutorPort,
    ) -> Self:
        definition = entry.definition
        node = GraphModuleNode(definition, module_executor)
        return cls(
            operator_id=node.operator_id,
            operator_version=node.operator_version,
            plugin_slug=GRAPH_MODULE_PLUGIN_SLUG,
            title=node.title,
            description=node.description,
            config_schema=_model_json_schema(node.config_contract.model),
            input_schema=_model_json_schema(node.input_contract.model),
            output_schema=_model_json_schema(node.output_contract.model),
            inputs=[
                PortResponse.from_input_port(port)
                for port in node.input_contract.ports.values()
            ],
            outputs=[
                PortResponse.from_output_port(port)
                for port in node.output_contract.ports.values()
            ],
            module_graph_id=definition.reference.graph_id,
            module_graph_revision=definition.reference.revision,
            catalog_visible=entry.catalog_visible,
        )


class UnavailableGraphModuleResponse(ApiResponse):
    graph_id: UUID
    revision: int = Field(ge=1, strict=True)
    name: str
    reason: str

    @classmethod
    def from_module(cls, module: UnavailableGraphModule) -> Self:
        return cls(
            graph_id=module.graph_id,
            revision=module.revision,
            name=module.name,
            reason=module.reason,
        )


class NodeRegistryResponse(ApiResponse):
    plugins: list[PluginSpecResponse]
    artifact_types: list[ArtifactTypeSpecResponse]
    artifact_conversions: list[ArtifactConversionSpecResponse]
    nodes: list[NodeSpecResponse]
    unavailable_modules: list[UnavailableGraphModuleResponse] = Field(
        default_factory=list
    )

    @classmethod
    def from_registry(
        cls,
        registry: PluginRegistry,
        module_listing: GraphModuleCatalogListing,
        module_executor: GraphModuleExecutorPort,
    ) -> Self:
        return cls(
            plugins=[
                PluginSpecResponse.from_plugin(plugin) for plugin in registry.plugins
            ]
            + [
                PluginSpecResponse(
                    slug=GRAPH_MODULE_PLUGIN_SLUG,
                    title="Modules",
                    origin=PluginOrigin.MODULE,
                )
            ],
            artifact_types=[
                ArtifactTypeSpecResponse.from_spec(spec)
                for spec in registry.artifact_types
            ],
            artifact_conversions=[
                ArtifactConversionSpecResponse.from_spec(spec)
                for spec in registry.artifact_conversions
            ],
            nodes=[
                NodeSpecResponse.from_registration(registration)
                for registration in registry.nodes
            ]
            + [
                NodeSpecResponse.from_graph_module(entry, module_executor)
                for entry in module_listing.entries
            ],
            unavailable_modules=[
                UnavailableGraphModuleResponse.from_module(module)
                for module in module_listing.unavailable
            ],
        )


__all__ = [
    "ArtifactConversionKeyResponse",
    "ArtifactConversionSpecResponse",
    "ArtifactTypeSpecResponse",
    "FieldProjectionResponse",
    "NodeRegistryResponse",
    "NodeSecretInputResponse",
    "NodeSpecResponse",
    "PluginSpecResponse",
    "PortDirection",
    "PortResponse",
    "UnavailableGraphModuleResponse",
]

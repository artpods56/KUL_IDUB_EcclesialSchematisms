from dataclasses import dataclass
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from pydantic.errors import PydanticInvalidForJsonSchema

from grafy_core.artifacts import (
    ArtifactExportFormat,
    ArtifactFieldProjection,
    ArtifactTypeSpec,
)
from grafy_core.conversions import ArtifactConversion, ArtifactConversionKey
from grafy_core.domain.module_library import ModulePublicationState
from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeContract,
    PluginNodeContract,
    PluginPortContract,
    PluginRelease,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import (
    ArtifactTypeVariable,
    InputPortSpec,
    OutputPortSpec,
    PortShape,
)
from grafy_core.operators.modules import GraphModuleNode
from grafy_core.plugins import (
    InstalledPlugin,
    NodeRegistration,
    NodeSecretInput,
    PluginOrigin,
    PluginRegistry,
)
from grafy_core.ports.modules import GraphModuleExecutorPort

from grafy_api.v1.models import (
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
PluginNonRunnableReason = Literal[
    "missing_runtime_artifact",
    "incompatible_protocol",
    "unsupported_runtime_profile",
    "unsupported_capabilities",
    "unsupported_artifact_type",
    "plugin_runtime_unavailable",
]

_SUPPORTED_CORE_PLUGIN_BUNDLES = frozenset(
    {
        ("scalar.integer", 1),
        ("scalar.text", 1),
        ("table.data", 1),
    }
)


@dataclass(frozen=True, slots=True)
class PluginCatalogExecutionSupport:
    runtime_available: bool = False
    runtime_profile: str | None = None
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PluginReleaseReadiness:
    runnable: bool
    reason: PluginNonRunnableReason | None = None
    detail: str | None = None


def plugin_release_readiness(
    release: PluginRelease,
    support: PluginCatalogExecutionSupport,
) -> PluginReleaseReadiness:
    if not release.executable:
        return PluginReleaseReadiness(
            runnable=False,
            reason="missing_runtime_artifact",
            detail="This release has no immutable runtime image.",
        )
    if release.protocol_digest != plugin_protocol_digest():
        return PluginReleaseReadiness(
            runnable=False,
            reason="incompatible_protocol",
            detail="This release uses an incompatible invocation protocol.",
        )
    if (
        support.runtime_profile is None
        or release.runtime_profile != support.runtime_profile
        or release.profile_digest != plugin_profile_digest(support.runtime_profile)
    ):
        return PluginReleaseReadiness(
            runnable=False,
            reason="unsupported_runtime_profile",
            detail=(
                f"Runtime profile {release.runtime_profile!r} is not available in "
                "this deployment."
            ),
        )
    unsupported_capabilities = sorted(
        set(release.capabilities.capabilities) - support.capabilities
    )
    has_secret_inputs = any(
        contract.secret_inputs for contract in release.catalog.nodes
    )
    if unsupported_capabilities or has_secret_inputs:
        rendered = ", ".join(unsupported_capabilities)
        if has_secret_inputs:
            rendered = f"{rendered}, secret inputs" if rendered else "secret inputs"
        return PluginReleaseReadiness(
            runnable=False,
            reason="unsupported_capabilities",
            detail=f"Unsupported Plugin capabilities: {rendered}.",
        )
    release_owned_types = {
        (artifact.key.id, artifact.key.schema_version)
        for artifact in release.catalog.artifact_types
    }
    unsupported_types: set[str] = set()
    for contract in release.catalog.nodes:
        for port in (*contract.inputs, *contract.outputs):
            if port.artifact_type is None:
                unsupported_types.add(f"type variable {port.artifact_type_variable!r}")
                continue
            key = (port.artifact_type.id, port.artifact_type.schema_version)
            if key not in _SUPPORTED_CORE_PLUGIN_BUNDLES | release_owned_types:
                unsupported_types.add(f"{key[0]}@{key[1]}")
    if unsupported_types:
        return PluginReleaseReadiness(
            runnable=False,
            reason="unsupported_artifact_type",
            detail=(
                "No portable Plugin bundle is available for: "
                + ", ".join(sorted(unsupported_types))
                + "."
            ),
        )
    if not support.runtime_available:
        return PluginReleaseReadiness(
            runnable=False,
            reason="plugin_runtime_unavailable",
            detail="The isolated Plugin runtime is not available.",
        )
    return PluginReleaseReadiness(runnable=True)


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


class ArtifactExportFormatResponse(ApiResponse):
    format: str
    content_type: str
    filename: str

    @classmethod
    def from_export_format(cls, export_format: ArtifactExportFormat) -> Self:
        return cls(
            format=export_format.format,
            content_type=export_format.content_type,
            filename=export_format.filename,
        )


class ArtifactTypeSpecResponse(ApiResponse):
    key: ArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, object]
    field_projections: list[FieldProjectionResponse]
    export_formats: list[ArtifactExportFormatResponse] = Field(
        default_factory=list,
    )

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
            export_formats=[
                ArtifactExportFormatResponse.from_export_format(export_format)
                for export_format in spec.export_formats
            ],
        )

    @classmethod
    def from_plugin_contract(cls, contract: PluginArtifactTypeContract) -> Self:
        return cls(
            key=ArtifactTypeKeyResponse(
                id=contract.key.id,
                schema_version=contract.key.schema_version,
            ),
            title=contract.title,
            payload_schema=contract.payload_schema,
            field_projections=[
                FieldProjectionResponse(
                    path=list(projection.path),
                    target_artifact_type=ArtifactTypeKeyResponse(
                        id=projection.target.id,
                        schema_version=projection.target.schema_version,
                    ),
                    title=projection.title,
                )
                for projection in contract.field_projections
            ],
            export_formats=[
                ArtifactExportFormatResponse(
                    format=export_format.format,
                    content_type=export_format.content_type,
                    filename=export_format.filename,
                )
                for export_format in contract.export_formats
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
    revision: int | None = Field(default=None, ge=1)
    runnable: bool = True
    non_runnable_reason: PluginNonRunnableReason | None = None
    non_runnable_detail: str | None = None

    @classmethod
    def from_plugin(cls, plugin: InstalledPlugin) -> Self:
        return cls(
            slug=plugin.slug,
            title=plugin.title,
            origin=plugin.origin,
        )

    @classmethod
    def from_plugin_release(
        cls,
        release: PluginRelease,
        readiness: PluginReleaseReadiness,
    ) -> Self:
        return cls(
            slug=release.slug,
            title=release.catalog.title,
            origin=PluginOrigin.WORKSPACE,
            revision=release.revision,
            runnable=readiness.runnable,
            non_runnable_reason=readiness.reason,
            non_runnable_detail=readiness.detail,
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

    @classmethod
    def from_plugin_contract(cls, port: PluginPortContract) -> Self:
        artifact_type = (
            None
            if port.artifact_type is None
            else ArtifactTypeKeyResponse(
                id=port.artifact_type.id,
                schema_version=port.artifact_type.schema_version,
            )
        )
        return cls(
            name=port.name,
            title=port.title,
            description=port.description,
            direction=port.direction,
            artifact_type=artifact_type,
            artifact_type_variable=port.artifact_type_variable,
            shape=port.shape,
            accepted_shapes=list(port.accepted_shapes),
            instance_plugs=port.instance_plugs,
            variadic=port.variadic,
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
    module_id: UUID | None = None
    publication_state: ModulePublicationState | None = None
    is_current_library_release: bool | None = None
    catalog_visible: bool = True
    plugin_revision: int | None = Field(default=None, ge=1)
    runnable: bool = True
    non_runnable_reason: PluginNonRunnableReason | None = None
    non_runnable_detail: str | None = None

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
            module_id=entry.module_id,
            publication_state=entry.publication_state,
            is_current_library_release=entry.is_current_library_release,
            catalog_visible=entry.catalog_visible,
        )

    @classmethod
    def from_plugin_release(
        cls,
        release: PluginRelease,
        contract: PluginNodeContract,
        readiness: PluginReleaseReadiness,
    ) -> Self:
        return cls(
            operator_id=contract.operator_id,
            operator_version=contract.operator_version,
            plugin_slug=release.slug,
            title=contract.title,
            description=contract.description,
            config_schema=contract.config_schema,
            input_schema=contract.input_schema,
            output_schema=contract.output_schema,
            inputs=[
                PortResponse.from_plugin_contract(port) for port in contract.inputs
            ],
            outputs=[
                PortResponse.from_plugin_contract(port) for port in contract.outputs
            ],
            secret_inputs=[
                NodeSecretInputResponse(
                    name=secret.name,
                    config_dependencies=list(secret.config_dependencies),
                    title=secret.title,
                    description=secret.description,
                )
                for secret in contract.secret_inputs
            ],
            plugin_revision=release.revision,
            runnable=readiness.runnable,
            non_runnable_reason=readiness.reason,
            non_runnable_detail=readiness.detail,
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
        plugin_releases: list[PluginRelease],
        plugin_execution_support: PluginCatalogExecutionSupport | None = None,
    ) -> Self:
        execution_support = plugin_execution_support or PluginCatalogExecutionSupport()
        release_readiness = {
            release.slug: plugin_release_readiness(release, execution_support)
            for release in plugin_releases
        }
        installed_plugin_slugs = {plugin.slug for plugin in registry.plugins}
        release_slugs = {release.slug for release in plugin_releases}
        duplicate_slugs = (installed_plugin_slugs | {GRAPH_MODULE_PLUGIN_SLUG}) & (
            release_slugs
        )
        if duplicate_slugs:
            rendered = ", ".join(sorted(duplicate_slugs))
            raise ValueError(
                "Workspace Plugin releases conflict with reserved or host Plugins: "
                f"{rendered}"
            )
        installed_artifact_keys = {
            (spec.key.id, spec.key.schema_version) for spec in registry.artifact_types
        }
        release_artifact_keys = [
            (contract.key.id, contract.key.schema_version)
            for release in plugin_releases
            for contract in release.catalog.artifact_types
        ]
        duplicate_artifact_keys = installed_artifact_keys & set(release_artifact_keys)
        seen: set[tuple[str, int]] = set()
        for key in release_artifact_keys:
            if key in seen:
                duplicate_artifact_keys.add(key)
            else:
                seen.add(key)
        if duplicate_artifact_keys:
            rendered = ", ".join(
                f"{artifact_id}@{schema_version}"
                for artifact_id, schema_version in sorted(duplicate_artifact_keys)
            )
            raise ValueError(
                "Workspace Plugin releases conflict with catalog artifact types: "
                f"{rendered}"
            )
        return cls(
            plugins=[
                PluginSpecResponse.from_plugin(plugin) for plugin in registry.plugins
            ]
            + [
                PluginSpecResponse(
                    slug=GRAPH_MODULE_PLUGIN_SLUG,
                    title="Workspace library",
                    origin=PluginOrigin.MODULE,
                )
            ]
            + [
                PluginSpecResponse.from_plugin_release(
                    release,
                    release_readiness[release.slug],
                )
                for release in plugin_releases
            ],
            artifact_types=[
                ArtifactTypeSpecResponse.from_spec(spec)
                for spec in registry.artifact_types
            ]
            + [
                ArtifactTypeSpecResponse.from_plugin_contract(contract)
                for release in plugin_releases
                for contract in release.catalog.artifact_types
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
            ]
            + [
                NodeSpecResponse.from_plugin_release(
                    release,
                    contract,
                    release_readiness[release.slug],
                )
                for release in plugin_releases
                for contract in release.catalog.nodes
            ],
            unavailable_modules=[
                UnavailableGraphModuleResponse.from_module(module)
                for module in module_listing.unavailable
            ],
        )


__all__ = [
    "ArtifactConversionKeyResponse",
    "ArtifactConversionSpecResponse",
    "ArtifactExportFormatResponse",
    "ArtifactTypeSpecResponse",
    "FieldProjectionResponse",
    "NodeRegistryResponse",
    "NodeSecretInputResponse",
    "NodeSpecResponse",
    "PluginCatalogExecutionSupport",
    "PluginNonRunnableReason",
    "PluginReleaseReadiness",
    "PluginSpecResponse",
    "PortDirection",
    "PortResponse",
    "UnavailableGraphModuleResponse",
    "plugin_release_readiness",
]

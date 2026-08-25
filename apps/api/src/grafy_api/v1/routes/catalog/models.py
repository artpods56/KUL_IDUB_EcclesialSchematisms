from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from pydantic.errors import PydanticInvalidForJsonSchema

from grafy_core.artifacts import (
    ArtifactBundleFormat,
    ArtifactExportFormat,
    ArtifactFieldProjection,
    ArtifactTypeSpec,
    MaterializedJsonType,
)
from grafy_core.canonical_conversions import CANONICAL_ARTIFACT_CONVERSIONS
from grafy_core.conversions import ArtifactConversion, ArtifactConversionKey
from grafy_core.domain.module_library import ModulePublicationState
from grafy_core.domain.plugin_releases import (
    PluginArtifactConversionContract,
    PluginArtifactTypeContract,
    PluginDistribution,
    PluginNodeContract,
    PluginPortContract,
    PluginRelease,
    PluginReleaseScope,
)
from grafy_core.domain.plugin_revocations import PluginReleaseRevocation
from grafy_core.domain.plugin_selection import (
    PluginFamilyLifecycle,
    PluginReleaseSelection,
)
from grafy_core.nodes import (
    ArtifactTypeVariable,
    InputPortSpec,
    OutputPortSpec,
    PortShape,
)
from grafy_core.operators.modules import GraphModuleNode
from grafy_core.plugins import NodeRegistration, NodeSecretInput, PluginRegistry
from grafy_core.ports.modules import GraphModuleExecutorPort

from grafy_api.plugin_admission import (
    PluginNonRunnableReason,
    ReleaseExecutionAdmission,
    ReleaseExecutionRejection,
)
from grafy_api.v1.models import (
    ApiResponse,
    ArtifactTypeKeyResponse,
    ArtifactTypeVariableIdentifier,
    PluginReleasePinModel,
)

from .services import (
    GRAPH_MODULE_PLUGIN_SLUG,
    GraphModuleCatalogEntry,
    GraphModuleCatalogListing,
    UnavailableGraphModule,
)


PortDirection = Literal["input", "output"]
CatalogEntryKind = Literal["plugin", "module"]
CatalogNonRunnableReason = PluginNonRunnableReason | Literal[
    "deprecated",
    "withdrawn",
]


@dataclass(frozen=True, slots=True)
class PluginCatalogReleaseState:
    selection: PluginReleaseSelection | None = None
    revocation: PluginReleaseRevocation | None = None


@dataclass(frozen=True, slots=True)
class PluginReleaseReadiness:
    runnable: bool
    reason: CatalogNonRunnableReason | None = None
    detail: str | None = None


def plugin_release_readiness(
    release: PluginRelease,
    admission: ReleaseExecutionAdmission | None,
    *,
    state: PluginCatalogReleaseState | None = None,
    node_contract: PluginNodeContract | None = None,
) -> PluginReleaseReadiness:
    if admission is None:
        return PluginReleaseReadiness(
            runnable=False,
            reason="plugin_runtime_unavailable",
            detail="Plugin release execution is not configured for this workbench.",
        )
    release_state = state or PluginCatalogReleaseState()
    decision = admission.decide(
        release,
        node_contract=node_contract,
        selection=release_state.selection,
        revocation=release_state.revocation,
    )
    if (
        isinstance(decision, ReleaseExecutionRejection)
        and decision.reason == "revoked"
    ):
        return PluginReleaseReadiness(
            runnable=False,
            reason=decision.reason,
            detail=decision.detail,
        )
    if (
        release_state.selection is not None
        and release_state.selection.lifecycle is not PluginFamilyLifecycle.PUBLISHED
    ):
        lifecycle = release_state.selection.lifecycle
        return PluginReleaseReadiness(
            runnable=False,
            reason=lifecycle.value,
            detail=(
                f"This Plugin family is {lifecycle.value} and unavailable for "
                "new insertion. Existing exact pins remain retained."
            ),
        )
    if isinstance(decision, ReleaseExecutionRejection):
        return PluginReleaseReadiness(
            runnable=False,
            reason=decision.reason,
            detail=decision.detail,
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


class ArtifactBundleContractResponse(ApiResponse):
    format: ArtifactBundleFormat
    version: int = Field(ge=1, strict=True)


class ArtifactTypeSpecResponse(ApiResponse):
    key: ArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, object]
    field_projections: list[FieldProjectionResponse]
    materialized_json_type: MaterializedJsonType | None = None
    export_formats: list[ArtifactExportFormatResponse] = Field(
        default_factory=list,
    )
    bundle: ArtifactBundleContractResponse

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
            materialized_json_type=spec.materialized_json_type,
            export_formats=[
                ArtifactExportFormatResponse.from_export_format(export_format)
                for export_format in spec.export_formats
            ],
            bundle=ArtifactBundleContractResponse(
                format=spec.bundle.format,
                version=spec.bundle.version,
            ),
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
            materialized_json_type=contract.materialized_json_type,
            export_formats=[
                ArtifactExportFormatResponse(
                    format=export_format.format,
                    content_type=export_format.content_type,
                    filename=export_format.filename,
                )
                for export_format in contract.export_formats
            ],
            bundle=ArtifactBundleContractResponse(
                format=contract.bundle.format,
                version=contract.bundle.version,
            ),
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

    @classmethod
    def from_plugin_contract(
        cls,
        contract: PluginArtifactConversionContract,
    ) -> Self:
        return cls(
            key=ArtifactConversionKeyResponse(
                id=contract.key.id,
                version=contract.key.version,
            ),
            source_artifact_type=ArtifactTypeKeyResponse(
                id=contract.source.id,
                schema_version=contract.source.schema_version,
            ),
            target_artifact_type=ArtifactTypeKeyResponse(
                id=contract.target.id,
                schema_version=contract.target.schema_version,
            ),
            title=contract.title,
        )


class PluginSpecResponse(ApiResponse):
    slug: str
    title: str
    entry_kind: CatalogEntryKind = "plugin"
    scope: PluginReleaseScope | None = None
    distribution: PluginDistribution | None = None
    plugin_release: PluginReleasePinModel | None = None
    revision: int | None = Field(default=None, ge=1)
    runnable: bool = True
    non_runnable_reason: CatalogNonRunnableReason | None = None
    non_runnable_detail: str | None = None

    @model_validator(mode="after")
    def validate_catalog_identity(self) -> Self:
        if self.entry_kind == "module":
            if self.scope is not None:
                raise ValueError("Module catalog entries cannot declare Plugin scope")
            if self.distribution is not None:
                raise ValueError(
                    "Module catalog entries cannot declare Plugin distribution"
                )
            if self.plugin_release is not None or self.revision is not None:
                raise ValueError(
                    "Module catalog entries cannot declare a Plugin release"
                )
            return self
        if self.scope is None:
            raise ValueError("Plugin catalog entries must declare Plugin scope")
        if self.scope is PluginReleaseScope.SYSTEM and self.distribution is None:
            raise ValueError("System Plugin catalog entries must declare distribution")
        if self.scope is PluginReleaseScope.WORKSPACE and self.distribution is not None:
            raise ValueError(
                "Workspace Plugin catalog entries cannot declare System distribution"
            )
        if self.plugin_release is None or self.revision is None:
            raise ValueError("Plugin catalog entries must declare an exact release")
        if self.plugin_release.scope is not self.scope:
            raise ValueError("plugin_release scope must match Plugin scope")
        if self.plugin_release.slug != self.slug:
            raise ValueError("plugin_release slug must match Plugin slug")
        if self.plugin_release.revision != self.revision:
            raise ValueError("plugin_release revision must match revision")
        return self

    @classmethod
    def from_plugin_release(
        cls,
        release: PluginRelease,
        readiness: PluginReleaseReadiness,
    ) -> Self:
        return cls(
            slug=release.slug,
            title=release.catalog.title,
            scope=release.scope,
            distribution=release.distribution,
            plugin_release=PluginReleasePinModel(
                scope=release.scope,
                slug=release.slug,
                revision=release.revision,
            ),
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
    plugin_release: PluginReleasePinModel | None = None
    runnable: bool = True
    non_runnable_reason: CatalogNonRunnableReason | None = None
    non_runnable_detail: str | None = None

    @model_validator(mode="after")
    def validate_module_identity(self) -> Self:
        if (self.module_graph_id is None) != (self.module_graph_revision is None):
            raise ValueError(
                "module_graph_id and module_graph_revision must be provided together"
            )
        if (self.plugin_release is None) != (self.plugin_revision is None):
            raise ValueError(
                "plugin_release and plugin_revision must be provided together"
            )
        if (
            self.plugin_release is not None
            and self.plugin_release.slug != self.plugin_slug
        ):
            raise ValueError("plugin_release slug must match plugin_slug")
        if (
            self.plugin_release is not None
            and self.plugin_release.revision != self.plugin_revision
        ):
            raise ValueError("plugin_release revision must match plugin_revision")
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
            plugin_release=PluginReleasePinModel(
                scope=release.scope,
                slug=release.slug,
                revision=release.revision,
            ),
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
        *,
        workspace_id: UUID,
        release_admission: ReleaseExecutionAdmission | None = None,
        plugin_release_states: Mapping[UUID, PluginCatalogReleaseState] | None = None,
    ) -> Self:
        release_states = plugin_release_states or {}
        system_releases = [
            release
            for release in plugin_releases
            if release.scope is PluginReleaseScope.SYSTEM
        ]
        workspace_releases = [
            release
            for release in plugin_releases
            if release.scope is PluginReleaseScope.WORKSPACE
        ]
        foreign_workspace_releases = [
            release
            for release in workspace_releases
            if release.workspace_id != workspace_id
        ]
        if foreign_workspace_releases:
            rendered = ", ".join(
                sorted(
                    f"{release.slug}@{release.revision} owned by {release.workspace_id}"
                    for release in foreign_workspace_releases
                )
            )
            raise ValueError(
                f"Workspace Plugin catalog received foreign releases: {rendered}"
            )
        system_slugs = {release.slug for release in system_releases}
        workspace_slugs = {release.slug for release in workspace_releases}
        if len(system_slugs) != len(system_releases):
            raise ValueError("System Plugin catalog contains duplicate current slugs")
        if len(workspace_slugs) != len(workspace_releases):
            raise ValueError(
                "Workspace Plugin catalog contains duplicate current slugs"
            )
        cross_scope_slugs = system_slugs & workspace_slugs
        if cross_scope_slugs:
            rendered = ", ".join(sorted(cross_scope_slugs))
            raise ValueError(
                f"Workspace Plugin releases conflict with System Plugins: {rendered}"
            )
        reserved_slugs = {GRAPH_MODULE_PLUGIN_SLUG} & (system_slugs | workspace_slugs)
        if reserved_slugs:
            raise ValueError(
                f"Plugin releases conflict with reserved Module provider: "
                f"{GRAPH_MODULE_PLUGIN_SLUG}"
            )

        installed_plugin_slugs = {plugin.slug for plugin in registry.plugins}
        workspace_host_collisions = installed_plugin_slugs & workspace_slugs
        if workspace_host_collisions:
            rendered = ", ".join(sorted(workspace_host_collisions))
            raise ValueError(
                "Workspace Plugin releases conflict with transitional host "
                f"Plugins: {rendered}"
            )

        host_nodes = {registration.key: registration for registration in registry.nodes}
        release_node_owners: dict[tuple[str, int], PluginRelease] = {}
        for release in plugin_releases:
            for contract in release.catalog.nodes:
                key = (contract.operator_id, contract.operator_version)
                other_release = release_node_owners.get(key)
                if other_release is not None:
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"operator {key[0]}@{key[1]} conflicts with "
                        f"{other_release.scope.value} Plugin "
                        f"{other_release.slug!r}"
                    )
                host_registration = host_nodes.get(key)
                overlays_matching_host = (
                    release.scope is PluginReleaseScope.SYSTEM
                    and host_registration is not None
                    and host_registration.plugin_slug == release.slug
                    and PluginNodeContract.from_registration(host_registration)
                    == contract
                )
                if host_registration is not None and not overlays_matching_host:
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"operator {key[0]}@{key[1]} conflicts with host Plugin "
                        f"{host_registration.plugin_slug!r}"
                    )
                release_node_owners[key] = release

        module_node_owners: dict[tuple[str, int], GraphModuleCatalogEntry] = {}
        for entry in module_listing.entries:
            definition = entry.definition
            module_node = GraphModuleNode(definition, module_executor)
            key = (module_node.operator_id, module_node.operator_version)
            if (
                key in host_nodes
                or key in release_node_owners
                or key in module_node_owners
            ):
                raise ValueError(
                    f"Module {definition.reference.graph_id}@"
                    f"{definition.reference.revision} operator {key[0]}@{key[1]} "
                    "conflicts with another catalog entry"
                )
            module_node_owners[key] = entry

        node_readiness = {
            (release.slug, contract.operator_id, contract.operator_version): (
                plugin_release_readiness(
                    release,
                    release_admission,
                    state=release_states.get(release.id),
                    node_contract=contract,
                )
            )
            for release in plugin_releases
            for contract in release.catalog.nodes
        }
        release_readiness: dict[str, PluginReleaseReadiness] = {}
        for release in plugin_releases:
            node_states = [
                node_readiness[
                    (release.slug, contract.operator_id, contract.operator_version)
                ]
                for contract in release.catalog.nodes
            ]
            if any(readiness.runnable for readiness in node_states):
                release_readiness[release.slug] = PluginReleaseReadiness(
                    runnable=True
                )
            else:
                release_readiness[release.slug] = plugin_release_readiness(
                    release,
                    release_admission,
                    state=release_states.get(release.id),
                )
        installed_artifacts = {
            (spec.key.id, spec.key.schema_version): spec
            for spec in registry.artifact_types
        }
        release_artifact_owners = [
            (
                (contract.key.id, contract.key.schema_version),
                release,
            )
            for release in plugin_releases
            for contract in release.catalog.artifact_types
        ]
        seen_artifact_owners: dict[tuple[str, int], PluginRelease] = {}
        for key, release in release_artifact_owners:
            other_release = seen_artifact_owners.get(key)
            if other_release is not None:
                raise ValueError(
                    f"{release.scope.value.title()} Plugin {release.slug!r} artifact "
                    f"type {key[0]}@{key[1]} conflicts with "
                    f"{other_release.scope.value} Plugin {other_release.slug!r}"
                )
            installed = installed_artifacts.get(key)
            if installed is not None:
                release_contract = next(
                    contract
                    for contract in release.catalog.artifact_types
                    if (contract.key.id, contract.key.schema_version) == key
                )
                overlays_matching_host = (
                    release.scope is PluginReleaseScope.SYSTEM
                    and registry.artifact_type_owner(installed.key) == release.slug
                    and release_contract
                    == PluginArtifactTypeContract.from_spec(installed)
                )
                if not overlays_matching_host:
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"artifact type {key[0]}@{key[1]} conflicts with the "
                        "host catalog contract"
                    )
            seen_artifact_owners[key] = release

        canonical_conversion_contracts = {
            (conversion.key.id, conversion.key.version): (
                PluginArtifactConversionContract.from_conversion(conversion)
            )
            for conversion in CANONICAL_ARTIFACT_CONVERSIONS
        }
        for release in plugin_releases:
            for contract in release.catalog.artifact_conversions:
                key = (contract.key.id, contract.key.version)
                canonical_contract = canonical_conversion_contracts.get(key)
                if canonical_contract is None:
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"declares non-canonical artifact conversion "
                        f"{key[0]}@{key[1]}"
                    )
                if contract != canonical_contract:
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"artifact conversion {key[0]}@{key[1]} conflicts with "
                        "the deployment-owned canonical contract"
                    )

        serialized_artifact_contracts: dict[
            tuple[str, int], PluginArtifactTypeContract
        ] = {}
        for release in plugin_releases:
            for contract in (
                *release.catalog.artifact_types,
                *release.catalog.artifact_type_dependencies,
            ):
                key = (contract.key.id, contract.key.schema_version)
                existing = serialized_artifact_contracts.get(key)
                if existing is not None and existing != contract:
                    raise ValueError(
                        f"Plugin releases declare different exact contracts for "
                        f"artifact type {key[0]}@{key[1]}"
                    )
                installed = installed_artifacts.get(key)
                if (
                    installed is not None
                    and contract != PluginArtifactTypeContract.from_spec(installed)
                ):
                    raise ValueError(
                        f"{release.scope.value.title()} Plugin {release.slug!r} "
                        f"artifact type {key[0]}@{key[1]} conflicts with the "
                        "host catalog contract"
                    )
                serialized_artifact_contracts.setdefault(key, contract)

        visible_host_nodes = [
            registration
            for registration in registry.nodes
            if registration.plugin_slug == GRAPH_MODULE_PLUGIN_SLUG
        ]
        return cls(
            plugins=[
                PluginSpecResponse(
                    slug=GRAPH_MODULE_PLUGIN_SLUG,
                    title="Workspace library",
                    entry_kind="module",
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
                ArtifactTypeSpecResponse.from_plugin_contract(contract)
                for contract in serialized_artifact_contracts.values()
            ],
            artifact_conversions=[
                ArtifactConversionSpecResponse.from_spec(conversion)
                for conversion in CANONICAL_ARTIFACT_CONVERSIONS
            ],
            nodes=[
                NodeSpecResponse.from_registration(registration)
                for registration in visible_host_nodes
            ]
            + [
                NodeSpecResponse.from_graph_module(entry, module_executor)
                for entry in module_listing.entries
            ]
            + [
                NodeSpecResponse.from_plugin_release(
                    release,
                    contract,
                    node_readiness[
                        (
                            release.slug,
                            contract.operator_id,
                            contract.operator_version,
                        )
                    ],
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
    "ArtifactBundleContractResponse",
    "ArtifactConversionKeyResponse",
    "ArtifactConversionSpecResponse",
    "ArtifactExportFormatResponse",
    "ArtifactTypeSpecResponse",
    "CatalogEntryKind",
    "CatalogNonRunnableReason",
    "FieldProjectionResponse",
    "NodeRegistryResponse",
    "NodeSecretInputResponse",
    "NodeSpecResponse",
    "PluginCatalogReleaseState",
    "PluginNonRunnableReason",
    "PluginReleaseReadiness",
    "PluginSpecResponse",
    "PortDirection",
    "PortResponse",
    "UnavailableGraphModuleResponse",
    "plugin_release_readiness",
]

"""Append-only Workspace Plugin releases and their catalog contracts."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import re
from typing import ClassVar, Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.errors import PydanticInvalidForJsonSchema

from grafy_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec
from grafy_core.nodes import (
    ArtifactTypeVariable,
    InputPortSpec,
    OutputPortSpec,
    PortShape,
)
from grafy_core.plugins import NodeRegistration, Plugin


PluginPortDirection = Literal["input", "output"]

PLUGIN_INVOCATION_PROTOCOL = "grafy-plugin-invocation@2"


def plugin_protocol_digest() -> str:
    """Digest of the artifact invocation protocol a release was inspected for."""

    return sha256(PLUGIN_INVOCATION_PROTOCOL.encode("utf-8")).hexdigest()


def plugin_profile_digest(runtime_profile: str) -> str:
    return sha256(runtime_profile.strip().encode("utf-8")).hexdigest()


def plugin_contract_digest(catalog: PluginCatalogManifest) -> str:
    return sha256(catalog.model_dump_json().encode("utf-8")).hexdigest()


class PluginReleaseError(ValueError):
    """A Plugin release or catalog contract is invalid."""


class PluginReleaseValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class PluginRuntimeArtifact(PluginReleaseValue):
    """Immutable provider-neutral reference to one stored OCI image layout."""

    format: Literal["oci-archive"] = "oci-archive"
    media_type: Literal["application/vnd.oci.image.manifest.v1+json"] = (
        "application/vnd.oci.image.manifest.v1+json"
    )
    object_key: str = Field(min_length=1, max_length=2_048)
    archive_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("object_key")
    @classmethod
    def validate_object_key(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/") or "\\" in value:
            raise ValueError("Plugin runtime artifact object key must be safe")
        return value


class PluginArtifactTypeKey(PluginReleaseValue):
    id: str = Field(min_length=1, max_length=255)
    schema_version: int = Field(ge=1, strict=True)

    @classmethod
    def from_key(cls, key: ArtifactTypeKey) -> Self:
        return cls(id=key.id, schema_version=key.schema_version)


class PluginFieldProjection(PluginReleaseValue):
    path: tuple[str, ...] = Field(min_length=1)
    target: PluginArtifactTypeKey
    title: str = Field(min_length=1, max_length=255)


class PluginExportFormat(PluginReleaseValue):
    format: str = Field(min_length=1, max_length=64)
    content_type: str = Field(min_length=1, max_length=255)
    filename: str = Field(min_length=1, max_length=255)


class PluginArtifactTypeContract(PluginReleaseValue):
    key: PluginArtifactTypeKey
    title: str = Field(min_length=1, max_length=255)
    payload_schema: dict[str, object] = Field(default_factory=dict)
    field_projections: tuple[PluginFieldProjection, ...] = ()
    export_formats: tuple[PluginExportFormat, ...] = ()

    @classmethod
    def from_spec(cls, spec: ArtifactTypeSpec) -> Self:
        return cls(
            key=PluginArtifactTypeKey.from_key(spec.key),
            title=spec.title,
            payload_schema=spec.payload_schema,
            field_projections=tuple(
                PluginFieldProjection(
                    path=projection.path,
                    target=PluginArtifactTypeKey.from_key(projection.target),
                    title=projection.title,
                )
                for projection in spec.field_projections
            ),
            export_formats=tuple(
                PluginExportFormat(
                    format=export_format.format,
                    content_type=export_format.content_type,
                    filename=export_format.filename,
                )
                for export_format in spec.export_formats
            ),
        )


class PluginPortContract(PluginReleaseValue):
    name: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=4_000)
    direction: PluginPortDirection
    artifact_type: PluginArtifactTypeKey | None = None
    artifact_type_variable: str | None = Field(default=None, max_length=255)
    shape: PortShape
    accepted_shapes: tuple[PortShape, ...] = Field(min_length=1)
    instance_plugs: bool = False
    variadic: bool = False
    required: bool = True

    @model_validator(mode="after")
    def validate_artifact_type_contract(self) -> Self:
        if (self.artifact_type is None) == (self.artifact_type_variable is None):
            raise ValueError(
                "Plugin port must declare exactly one artifact type or type variable"
            )
        return self

    @classmethod
    def from_input_port(cls, port: InputPortSpec) -> Self:
        artifact_type: PluginArtifactTypeKey | None
        artifact_type_variable: str | None
        if isinstance(port.accepts, ArtifactTypeVariable):
            artifact_type = None
            artifact_type_variable = port.accepts.name
        else:
            artifact_type = PluginArtifactTypeKey.from_key(port.accepts)
            artifact_type_variable = None
        return cls(
            name=port.name,
            title=port.title,
            description=port.description,
            direction="input",
            artifact_type=artifact_type,
            artifact_type_variable=artifact_type_variable,
            shape=port.shape,
            accepted_shapes=port.accepted_shapes,
            instance_plugs=port.instance_plugs,
            variadic=port.variadic,
            required=port.required,
        )

    @classmethod
    def from_output_port(cls, port: OutputPortSpec) -> Self:
        artifact_type: PluginArtifactTypeKey | None
        artifact_type_variable: str | None
        if isinstance(port.produces, ArtifactTypeVariable):
            artifact_type = None
            artifact_type_variable = port.produces.name
        else:
            artifact_type = PluginArtifactTypeKey.from_key(port.produces)
            artifact_type_variable = None
        return cls(
            name=port.name,
            title=port.title,
            description=port.description,
            direction="output",
            artifact_type=artifact_type,
            artifact_type_variable=artifact_type_variable,
            shape=port.shape,
            accepted_shapes=(port.shape,),
            required=port.required,
        )


class PluginSecretInputContract(PluginReleaseValue):
    name: str = Field(min_length=1, max_length=255)
    config_dependencies: tuple[str, ...] = ()
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4_000)


class PluginNodeContract(PluginReleaseValue):
    operator_id: str = Field(min_length=1, max_length=255)
    operator_version: int = Field(ge=1, strict=True)
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(max_length=4_000)
    config_schema: dict[str, object]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    inputs: tuple[PluginPortContract, ...]
    outputs: tuple[PluginPortContract, ...]
    secret_inputs: tuple[PluginSecretInputContract, ...] = ()

    @classmethod
    def from_registration(cls, registration: NodeRegistration) -> Self:
        node_class = registration.node_class
        return cls(
            operator_id=node_class.operator_id,
            operator_version=node_class.operator_version,
            title=registration.title,
            description=registration.description,
            config_schema=_model_json_schema(node_class.config_contract.model),
            input_schema=_model_json_schema(node_class.input_contract.model),
            output_schema=_model_json_schema(node_class.output_contract.model),
            inputs=tuple(
                PluginPortContract.from_input_port(port)
                for port in node_class.input_contract.ports.values()
            ),
            outputs=tuple(
                PluginPortContract.from_output_port(port)
                for port in node_class.output_contract.ports.values()
            ),
            secret_inputs=tuple(
                PluginSecretInputContract(
                    name=secret.name,
                    config_dependencies=secret.config_dependencies,
                    title=secret.title,
                    description=secret.description,
                )
                for secret in registration.secret_inputs
            ),
        )


class PluginCatalogManifest(PluginReleaseValue):
    slug: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$", max_length=100)
    title: str = Field(min_length=1, max_length=160)
    artifact_types: tuple[PluginArtifactTypeContract, ...] = ()
    nodes: tuple[PluginNodeContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog_identity(self) -> Self:
        node_keys = [(node.operator_id, node.operator_version) for node in self.nodes]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError(
                "Plugin catalog nodes must have unique operator identities"
            )
        operator_prefix = f"{self.slug}."
        for node in self.nodes:
            if not node.operator_id.startswith(operator_prefix):
                raise ValueError(
                    f"Plugin {self.slug!r} node {node.operator_id!r} must use "
                    f"the {operator_prefix!r} operator prefix"
                )
        artifact_keys = [
            (artifact.key.id, artifact.key.schema_version)
            for artifact in self.artifact_types
        ]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("Plugin artifact types must have unique identities")
        for artifact in self.artifact_types:
            if not artifact.key.id.startswith(operator_prefix):
                raise ValueError(
                    f"Plugin {self.slug!r} owned artifact type "
                    f"{artifact.key.id!r} must use the {operator_prefix!r} prefix"
                )
        return self

    @classmethod
    def from_plugin(cls, plugin: Plugin) -> Self:
        return cls(
            slug=plugin.slug,
            title=plugin.title,
            artifact_types=tuple(
                PluginArtifactTypeContract.from_spec(spec)
                for spec in plugin.artifact_types
            ),
            nodes=tuple(
                PluginNodeContract.from_registration(registration)
                for registration in plugin.nodes
            ),
        )


@dataclass(frozen=True, slots=True)
class PluginReleaseIdentity:
    """Exact immutable identity of one resolved Plugin release.

    Invocation fingerprints and retained provenance use this value so two
    releases containing the same operator version never share cache entries
    or diagnostics.
    """

    slug: str
    revision: int
    source_digest: str
    contract_digest: str
    protocol_digest: str

    @classmethod
    def from_release(cls, release: PluginRelease) -> Self:
        return cls(
            slug=release.slug,
            revision=release.revision,
            source_digest=release.source_digest,
            contract_digest=release.contract_digest,
            protocol_digest=release.protocol_digest,
        )

    def fingerprint_document(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "revision": self.revision,
            "source_digest": self.source_digest,
        }


class PluginCapabilityManifest(PluginReleaseValue):
    capabilities: tuple[str, ...] = ()

    @field_validator("capabilities")
    @classmethod
    def normalize_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: set[str] = set()
        for capability in value:
            candidate = capability.strip()
            if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,254}", candidate) is None:
                raise ValueError(f"Invalid Plugin capability {capability!r}")
            normalized.add(candidate)
        return tuple(sorted(normalized))

    @property
    def digest(self) -> str:
        payload = self.model_dump_json().encode("utf-8")
        return sha256(payload).hexdigest()


class PluginReleaseDescriptor(PluginReleaseValue):
    """Content identity inputs for idempotent immutable publication."""

    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_profile: str = Field(min_length=1, max_length=100)
    runtime_artifact: PluginRuntimeArtifact | None = None

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass
class PluginRelease:
    """One immutable release of a Workspace-scoped Plugin family.

    The release descriptor references independent immutable objects: the source
    archive, the inspected catalog contract, the deployment runtime profile, the
    artifact invocation protocol, and the declared capability manifest. The
    runtime image digest stays absent until the image-building slice fills it.
    """

    workspace_id: UUID
    slug: str
    revision: int
    catalog: PluginCatalogManifest
    contract_digest: str
    capabilities: PluginCapabilityManifest
    capability_digest: str
    protocol_digest: str
    profile_digest: str
    source_object_key: str
    source_digest: str
    lock_digest: str
    runtime_profile: str
    runtime_image_digest: str | None = None
    runtime_artifact: PluginRuntimeArtifact | None = None
    descriptor_digest: str | None = None
    published_by_user_id: UUID | None = None
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.catalog.slug != self.slug:
            raise PluginReleaseError("Plugin release slug must match its catalog")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise PluginReleaseError("Plugin release revision must be positive")
        if self.contract_digest != plugin_contract_digest(self.catalog):
            raise PluginReleaseError(
                "Plugin contract digest must match the serialized catalog contract"
            )
        self.capability_digest = _sha256(
            self.capability_digest,
            "Plugin capability digest",
        )
        if self.capability_digest != self.capabilities.digest:
            raise PluginReleaseError(
                "Plugin capability digest must match the capability manifest"
            )
        self.protocol_digest = _sha256(
            self.protocol_digest,
            "Plugin invocation protocol digest",
        )
        self.profile_digest = _sha256(
            self.profile_digest,
            "Plugin runtime profile digest",
        )
        if self.profile_digest != plugin_profile_digest(self.runtime_profile):
            raise PluginReleaseError(
                "Plugin profile digest must match the runtime profile"
            )
        self.source_digest = _sha256(self.source_digest, "Plugin source digest")
        self.lock_digest = _sha256(self.lock_digest, "Plugin lock digest")
        if self.runtime_image_digest is not None:
            self.runtime_image_digest = _sha256(
                self.runtime_image_digest,
                "Plugin runtime image digest",
            )
        if self.runtime_artifact is not None:
            if self.runtime_image_digest != self.runtime_artifact.manifest_digest:
                raise PluginReleaseError(
                    "Plugin runtime image digest must match its OCI artifact"
                )
        self.runtime_profile = self.runtime_profile.strip()
        if self.runtime_profile == "":
            raise PluginReleaseError("Plugin runtime profile must not be blank")
        if len(self.runtime_profile) > 100:
            raise PluginReleaseError(
                "Plugin runtime profile must be at most 100 characters"
            )
        if self.source_object_key == "":
            raise PluginReleaseError("Plugin source object key must not be blank")
        if self.source_object_key.startswith(
            "/"
        ) or ".." in self.source_object_key.split("/"):
            raise PluginReleaseError(
                "Plugin source object key must be a safe relative path"
            )
        expected_descriptor_digest = self.descriptor.digest
        if self.descriptor_digest is None:
            self.descriptor_digest = expected_descriptor_digest
        else:
            self.descriptor_digest = _sha256(
                self.descriptor_digest,
                "Plugin release descriptor digest",
            )
            if self.descriptor_digest != expected_descriptor_digest:
                raise PluginReleaseError(
                    "Plugin release descriptor digest must match its inputs"
                )
        if self.published_at.tzinfo is None:
            raise PluginReleaseError("Plugin published_at must be timezone-aware")

    @property
    def descriptor(self) -> PluginReleaseDescriptor:
        return PluginReleaseDescriptor(
            source_digest=self.source_digest,
            contract_digest=self.contract_digest,
            capability_digest=self.capability_digest,
            protocol_digest=self.protocol_digest,
            profile_digest=self.profile_digest,
            lock_digest=self.lock_digest,
            runtime_profile=self.runtime_profile,
            runtime_artifact=self.runtime_artifact,
        )

    @property
    def executable(self) -> bool:
        return self.runtime_artifact is not None


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
                    "x-python-type": str(model_field.annotation),
                }
                for name, model_field in model.model_fields.items()
            },
        }


def _sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PluginReleaseError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "PLUGIN_INVOCATION_PROTOCOL",
    "PluginArtifactTypeContract",
    "PluginArtifactTypeKey",
    "PluginCapabilityManifest",
    "PluginCatalogManifest",
    "PluginExportFormat",
    "PluginFieldProjection",
    "PluginNodeContract",
    "PluginPortContract",
    "PluginPortDirection",
    "PluginRelease",
    "PluginReleaseDescriptor",
    "PluginReleaseError",
    "PluginReleaseIdentity",
    "PluginRuntimeArtifact",
    "PluginSecretInputContract",
    "plugin_contract_digest",
    "plugin_profile_digest",
    "plugin_protocol_digest",
]

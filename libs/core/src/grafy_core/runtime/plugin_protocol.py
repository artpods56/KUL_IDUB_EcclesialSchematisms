"""Provider-neutral artifact bundle protocol for Workspace Plugin invocations.

The models in this module are the complete serialized host/guest contract.
They contain domain identities, relative bundle paths, limits, and typed
success or failure envelopes; process, mount, image, database, and storage
details belong to outer adapters.
"""

import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grafy_core.artifacts import JsonObject
from grafy_core.domain.plugin_releases import (
    PLUGIN_INVOCATION_PROTOCOL,
    PluginArtifactTypeKey,
)


PluginArtifactShape = Literal["one", "many"]
PluginInvocationStatus = Literal["succeeded", "failed"]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PluginProtocolCompatibilityError(ValueError):
    """A manifest uses a protocol version this runtime cannot execute."""


class PluginFailureCode(StrEnum):
    CONTRACT_FAILURE = "contract_failure"
    MATERIALIZATION_FAILURE = "materialization_failure"
    OPERATOR_FAILURE = "operator_failure"
    OUTPUT_VALIDATION = "output_validation"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    INTERNAL_ADAPTER_FAILURE = "internal_adapter_failure"


class PluginProtocolValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class _PluginProtocolHeader(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    protocol_version: str


def _require_supported_protocol(payload: bytes) -> None:
    header = _PluginProtocolHeader.model_validate_json(payload)
    if header.protocol_version != PLUGIN_INVOCATION_PROTOCOL:
        raise PluginProtocolCompatibilityError(
            f"Unsupported Plugin invocation protocol "
            f"{header.protocol_version!r}; expected {PLUGIN_INVOCATION_PROTOCOL!r}"
        )


def _validate_relative_bundle_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("Plugin bundle paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        value == ""
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            "Plugin bundle paths must be normalized relative paths without traversal"
        )
    return value


class PluginInvocationRelease(PluginProtocolValue):
    slug: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$",
        max_length=100,
    )
    revision: int = Field(ge=1, strict=True)
    source_digest: Sha256Digest
    contract_digest: Sha256Digest
    protocol_digest: Sha256Digest


class PluginInvocationArtifactTypeBinding(PluginProtocolValue):
    variable: str = Field(min_length=1, max_length=255)
    artifact_type: PluginArtifactTypeKey


class PluginInvocationLimits(PluginProtocolValue):
    wall_time_seconds: int = Field(default=60, ge=1, le=3_600, strict=True)
    max_input_bytes: int = Field(
        default=64 * 1_024 * 1_024,
        ge=1,
        le=16 * 1_024 * 1_024 * 1_024,
        strict=True,
    )
    max_output_bytes: int = Field(
        default=64 * 1_024 * 1_024,
        ge=1,
        le=16 * 1_024 * 1_024 * 1_024,
        strict=True,
    )
    max_files: int = Field(default=1_024, ge=1, le=100_000, strict=True)
    max_log_bytes: int = Field(
        default=256 * 1_024,
        ge=1,
        le=64 * 1_024 * 1_024,
        strict=True,
    )
    max_table_rows: int = Field(default=1_000_000, ge=0, le=100_000_000, strict=True)
    max_table_columns: int = Field(default=10_000, ge=0, le=10_000, strict=True)
    max_table_chunks: int = Field(default=10_000, ge=0, le=100_000, strict=True)


class PluginInputArtifactBundle(PluginProtocolValue):
    artifact_id: UUID
    relative_path: str = Field(min_length=1, max_length=1_024)
    byte_count: int = Field(ge=0, strict=True)
    content_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        _validate_relative_bundle_path(self.relative_path)
        if not self.relative_path.startswith("inputs/"):
            raise ValueError("Plugin input bundle paths must be beneath inputs/")
        return self


class PluginOutputArtifactBundle(PluginProtocolValue):
    relative_path: str = Field(min_length=1, max_length=1_024)
    byte_count: int = Field(ge=0, strict=True)
    content_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        _validate_relative_bundle_path(self.relative_path)
        if not self.relative_path.startswith("outputs/"):
            raise ValueError("Plugin output bundle paths must be beneath outputs/")
        return self


class PluginInputArtifactGroup(PluginProtocolValue):
    shape: PluginArtifactShape
    artifacts: tuple[PluginInputArtifactBundle, ...]

    @model_validator(mode="after")
    def validate_cardinality(self) -> Self:
        if self.shape == "one" and len(self.artifacts) != 1:
            raise ValueError("A shape='one' input group must contain one artifact")
        return self


class PluginInputBinding(PluginProtocolValue):
    port: str = Field(min_length=1, max_length=255)
    artifact_type: PluginArtifactTypeKey
    groups: tuple[PluginInputArtifactGroup, ...] = Field(min_length=1)


class PluginOutputDeclaration(PluginProtocolValue):
    port: str = Field(min_length=1, max_length=255)
    artifact_type: PluginArtifactTypeKey
    shape: PluginArtifactShape
    required: bool = True


class PluginOutputBinding(PluginProtocolValue):
    port: str = Field(min_length=1, max_length=255)
    artifact_type: PluginArtifactTypeKey
    shape: PluginArtifactShape
    artifacts: tuple[PluginOutputArtifactBundle, ...]

    @model_validator(mode="after")
    def validate_cardinality(self) -> Self:
        if self.shape == "one" and len(self.artifacts) != 1:
            raise ValueError("A shape='one' output must contain one artifact")
        return self


class PluginInvocationEnvelope(PluginProtocolValue):
    protocol_version: Literal["grafy-plugin-invocation@2"] = PLUGIN_INVOCATION_PROTOCOL
    invocation_id: UUID
    execution_scope_id: UUID
    workspace_id: UUID
    workflow_run_id: UUID | None = None
    node_id: str | None = Field(default=None, min_length=1, max_length=255)
    invocation_index: int | None = Field(default=None, ge=0, strict=True)
    release: PluginInvocationRelease
    operator_id: str = Field(min_length=1, max_length=255)
    operator_version: int = Field(ge=1, strict=True)
    artifact_type_bindings: tuple[PluginInvocationArtifactTypeBinding, ...] = ()
    config: JsonObject
    inputs: tuple[PluginInputBinding, ...]
    outputs: tuple[PluginOutputDeclaration, ...]
    limits: PluginInvocationLimits

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        input_ports = [binding.port for binding in self.inputs]
        if len(input_ports) != len(set(input_ports)):
            raise ValueError("Plugin invocation input port bindings must be unique")
        output_ports = [declaration.port for declaration in self.outputs]
        if len(output_ports) != len(set(output_ports)):
            raise ValueError("Plugin invocation output declarations must be unique")
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Plugin artifact type variable bindings must be unique")
        paths = [
            artifact.relative_path
            for binding in self.inputs
            for group in binding.groups
            for artifact in group.artifacts
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("Plugin invocation input bundle paths must be unique")
        json.dumps(
            self.config,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> Self:
        _require_supported_protocol(payload)
        return cls.model_validate_json(payload)


class PluginFailureEnvelope(PluginProtocolValue):
    code: PluginFailureCode
    message: str = Field(min_length=1, max_length=1_000)
    release_slug: str = Field(min_length=1, max_length=100)
    release_revision: int = Field(ge=1, strict=True)
    operator_id: str = Field(min_length=1, max_length=255)
    operator_version: int = Field(ge=1, strict=True)
    node_id: str | None = Field(default=None, min_length=1, max_length=255)
    invocation_index: int | None = Field(default=None, ge=0, strict=True)


class PluginInvocationResultEnvelope(PluginProtocolValue):
    protocol_version: Literal["grafy-plugin-invocation@2"] = PLUGIN_INVOCATION_PROTOCOL
    invocation_id: UUID
    status: PluginInvocationStatus
    outputs: tuple[PluginOutputBinding, ...] = ()
    failure: PluginFailureEnvelope | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status == "succeeded" and self.failure is not None:
            raise ValueError("A successful Plugin result cannot contain a failure")
        if self.status == "failed" and self.failure is None:
            raise ValueError("A failed Plugin result requires a failure envelope")
        if self.status == "failed" and self.outputs:
            raise ValueError("A failed Plugin result cannot expose output bundles")
        output_ports = [binding.port for binding in self.outputs]
        if len(output_ports) != len(set(output_ports)):
            raise ValueError("Plugin result output port bindings must be unique")
        paths = [
            artifact.relative_path
            for binding in self.outputs
            for artifact in binding.artifacts
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("Plugin result output bundle paths must be unique")
        return self

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> Self:
        _require_supported_protocol(payload)
        return cls.model_validate_json(payload)


__all__ = [
    "PluginArtifactShape",
    "PluginFailureCode",
    "PluginFailureEnvelope",
    "PluginInputArtifactBundle",
    "PluginInputArtifactGroup",
    "PluginInputBinding",
    "PluginInvocationArtifactTypeBinding",
    "PluginInvocationEnvelope",
    "PluginInvocationLimits",
    "PluginInvocationRelease",
    "PluginInvocationResultEnvelope",
    "PluginOutputArtifactBundle",
    "PluginOutputBinding",
    "PluginOutputDeclaration",
    "PluginProtocolCompatibilityError",
]

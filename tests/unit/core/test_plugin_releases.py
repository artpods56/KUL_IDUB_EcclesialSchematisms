from typing import Annotated, cast
from uuid import UUID

from pydantic import Field
import pytest

from grafy_core.artifacts import NoConfig, NodeInput, NodeOutput
from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeContract,
    PluginArtifactTypeKey,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginRelease,
    PluginReleaseError,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import InPort, OutPort
from grafy_core.operators.text import TEXT_VALUE, TextValue
from grafy_core.plugins import Plugin


PLUGIN = Plugin(slug="test.notes", title="Test notes")


class EchoInput(NodeInput):
    text: Annotated[TextValue, InPort(TEXT_VALUE), Field(description="Input text")]


class EchoOutput(NodeOutput):
    text: Annotated[TextValue, OutPort(TEXT_VALUE), Field(description="Output text")]


@PLUGIN.function_node(operator_id="test.notes.echo", version=1, title="Echo")
async def echo(_config: NoConfig, inputs: EchoInput) -> EchoOutput:
    return EchoOutput(text=inputs.text)


def test_catalog_manifest_is_derived_from_function_node_contracts() -> None:
    manifest = PluginCatalogManifest.from_plugin(PLUGIN)

    assert manifest.slug == "test.notes"
    assert len(manifest.nodes) == 1
    node = manifest.nodes[0]
    assert node.operator_id == "test.notes.echo"
    assert node.inputs[0].artifact_type is not None
    assert node.inputs[0].artifact_type.id == "scalar.text"
    assert node.outputs[0].artifact_type is not None
    assert node.outputs[0].artifact_type.id == "scalar.text"
    assert (
        PluginCatalogManifest.model_validate_json(manifest.model_dump_json())
        == manifest
    )


def test_catalog_manifest_rejects_nodes_outside_plugin_namespace() -> None:
    payload = PluginCatalogManifest.from_plugin(PLUGIN).model_dump(mode="json")
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    node = cast(dict[str, object], nodes[0])
    node["operator_id"] = "other.echo"

    with pytest.raises(ValueError, match="operator prefix"):
        PluginCatalogManifest.model_validate(payload)


def test_catalog_manifest_rejects_owned_types_outside_plugin_namespace() -> None:
    payload = PluginCatalogManifest.from_plugin(PLUGIN).model_copy(
        update={
            "artifact_types": (
                PluginArtifactTypeContract(
                    key=PluginArtifactTypeKey(id="other.summary", schema_version=1),
                    title="Summary",
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="owned artifact type"):
        PluginCatalogManifest.model_validate(payload.model_dump())


def test_runtime_artifact_makes_release_executable_and_part_of_its_identity() -> None:
    catalog = PluginCatalogManifest.from_plugin(PLUGIN)
    capabilities = PluginCapabilityManifest()
    artifact = PluginRuntimeArtifact(
        object_key="plugin-releases/test.notes/runtime/image.oci.tar",
        archive_digest="1" * 64,
        manifest_digest="2" * 64,
        config_digest="3" * 64,
    )
    source_only = PluginRelease(
        workspace_id=UUID("00000000-0000-4000-8000-000000000853"),
        slug=catalog.slug,
        revision=1,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=plugin_protocol_digest(),
        profile_digest=plugin_profile_digest("python-uv"),
        source_object_key="plugin-releases/test.notes/source.tar.gz",
        source_digest="4" * 64,
        lock_digest="5" * 64,
        runtime_profile="python-uv",
    )
    executable = PluginRelease(
        workspace_id=source_only.workspace_id,
        slug=catalog.slug,
        revision=2,
        catalog=catalog,
        contract_digest=source_only.contract_digest,
        capabilities=capabilities,
        capability_digest=capabilities.digest,
        protocol_digest=source_only.protocol_digest,
        profile_digest=source_only.profile_digest,
        source_object_key=source_only.source_object_key,
        source_digest=source_only.source_digest,
        lock_digest=source_only.lock_digest,
        runtime_profile=source_only.runtime_profile,
        runtime_image_digest=artifact.manifest_digest,
        runtime_artifact=artifact,
    )

    assert source_only.executable is False
    assert executable.executable is True
    assert executable.descriptor.runtime_artifact == artifact
    assert executable.descriptor_digest != source_only.descriptor_digest


def test_release_rejects_runtime_image_that_does_not_match_oci_manifest() -> None:
    catalog = PluginCatalogManifest.from_plugin(PLUGIN)
    capabilities = PluginCapabilityManifest()
    artifact = PluginRuntimeArtifact(
        object_key="plugin-releases/test.notes/runtime/image.oci.tar",
        archive_digest="1" * 64,
        manifest_digest="2" * 64,
        config_digest="3" * 64,
    )

    with pytest.raises(PluginReleaseError, match="must match its OCI artifact"):
        PluginRelease(
            workspace_id=UUID("00000000-0000-4000-8000-000000000853"),
            slug=catalog.slug,
            revision=1,
            catalog=catalog,
            contract_digest=plugin_contract_digest(catalog),
            capabilities=capabilities,
            capability_digest=capabilities.digest,
            protocol_digest=plugin_protocol_digest(),
            profile_digest=plugin_profile_digest("python-uv"),
            source_object_key="plugin-releases/test.notes/source.tar.gz",
            source_digest="4" * 64,
            lock_digest="5" * 64,
            runtime_profile="python-uv",
            runtime_image_digest="6" * 64,
            runtime_artifact=artifact,
        )


@pytest.mark.parametrize(
    "object_key",
    ["/absolute/image.oci.tar", "runtime/../image.oci.tar", r"runtime\image.oci.tar"],
)
def test_runtime_artifact_rejects_unsafe_object_keys(object_key: str) -> None:
    with pytest.raises(ValueError, match="object key must be safe"):
        PluginRuntimeArtifact(
            object_key=object_key,
            archive_digest="1" * 64,
            manifest_digest="2" * 64,
            config_digest="3" * 64,
        )

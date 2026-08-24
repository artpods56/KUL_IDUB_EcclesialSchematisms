from hashlib import sha256
from uuid import UUID

import pytest

from grafy_core.domain.plugin_releases import (
    PluginArtifactTypeContract,
    PluginArtifactTypeKey,
    PluginCapabilityManifest,
    PluginCatalogManifest,
    PluginNodeContract,
    PluginPortContract,
    PluginPortDirection,
    PluginRelease,
    PluginRuntimeArtifact,
    plugin_contract_digest,
    plugin_profile_digest,
    plugin_protocol_digest,
)
from grafy_core.nodes import PortShape

from grafy_api.v1.routes.catalog.models import (
    PluginCatalogExecutionSupport,
    PluginNonRunnableReason,
    plugin_release_readiness,
)


WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000661")


def _port(
    name: str,
    direction: PluginPortDirection,
    artifact_type: str,
) -> PluginPortContract:
    return PluginPortContract(
        name=name,
        direction=direction,
        artifact_type=PluginArtifactTypeKey(
            id=artifact_type,
            schema_version=1,
        ),
        shape=PortShape.ONE,
        accepted_shapes=(PortShape.ONE,),
    )


def _release(
    *,
    executable: bool = True,
    protocol_digest: str | None = None,
    runtime_profile: str = "python-uv",
    capabilities: tuple[str, ...] = (),
    input_type: str = "table.data",
    output_type: str = "scalar.text",
    own_output_type: bool = False,
) -> PluginRelease:
    artifact_types = (
        (
            PluginArtifactTypeContract(
                key=PluginArtifactTypeKey(id=output_type, schema_version=1),
                title="Owned output",
                payload_schema={"type": "object"},
            ),
        )
        if own_output_type
        else ()
    )
    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        artifact_types=artifact_types,
        nodes=(
            PluginNodeContract(
                operator_id="notes.transform",
                operator_version=1,
                title="Transform",
                description="Transform one artifact",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(_port("source", "input", input_type),),
                outputs=(_port("result", "output", output_type),),
            ),
        ),
    )
    capability_manifest = PluginCapabilityManifest(capabilities=capabilities)
    runtime_artifact = (
        PluginRuntimeArtifact(
            object_key="plugin-releases/notes/runtime/image.oci.tar",
            archive_digest="1" * 64,
            manifest_digest="2" * 64,
            config_digest="3" * 64,
        )
        if executable
        else None
    )
    return PluginRelease(
        workspace_id=WORKSPACE_ID,
        slug="notes",
        revision=1,
        catalog=catalog,
        contract_digest=plugin_contract_digest(catalog),
        capabilities=capability_manifest,
        capability_digest=capability_manifest.digest,
        protocol_digest=protocol_digest or plugin_protocol_digest(),
        profile_digest=plugin_profile_digest(runtime_profile),
        source_object_key="plugin-releases/notes/source.tar.gz",
        source_digest="4" * 64,
        lock_digest="5" * 64,
        runtime_profile=runtime_profile,
        runtime_image_digest=(
            runtime_artifact.manifest_digest if runtime_artifact is not None else None
        ),
        runtime_artifact=runtime_artifact,
    )


@pytest.mark.parametrize(
    ("release", "support", "reason"),
    [
        (
            _release(executable=False),
            PluginCatalogExecutionSupport(
                runtime_available=True,
                runtime_profile="python-uv",
            ),
            "missing_runtime_artifact",
        ),
        (
            _release(protocol_digest=sha256(b"grafy-plugin-invocation@1").hexdigest()),
            PluginCatalogExecutionSupport(
                runtime_available=True,
                runtime_profile="python-uv",
            ),
            "incompatible_protocol",
        ),
        (
            _release(runtime_profile="python-uv-gdal"),
            PluginCatalogExecutionSupport(
                runtime_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_runtime_profile",
        ),
        (
            _release(capabilities=("network.egress",)),
            PluginCatalogExecutionSupport(
                runtime_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_capabilities",
        ),
        (
            _release(input_type="image.raster"),
            PluginCatalogExecutionSupport(
                runtime_available=True,
                runtime_profile="python-uv",
            ),
            "unsupported_artifact_type",
        ),
        (
            _release(),
            PluginCatalogExecutionSupport(
                runtime_available=False,
                runtime_profile="python-uv",
            ),
            "plugin_runtime_unavailable",
        ),
    ],
)
def test_release_readiness_returns_a_stable_fail_closed_reason(
    release: PluginRelease,
    support: PluginCatalogExecutionSupport,
    reason: PluginNonRunnableReason,
) -> None:
    readiness = plugin_release_readiness(release, support)

    assert readiness.runnable is False
    assert readiness.reason == reason
    assert readiness.detail


def test_release_readiness_accepts_table_core_scalars_and_owned_inline_types() -> None:
    release = _release(
        output_type="notes.table_summary",
        own_output_type=True,
    )

    readiness = plugin_release_readiness(
        release,
        PluginCatalogExecutionSupport(
            runtime_available=True,
            runtime_profile="python-uv",
        ),
    )

    assert readiness.runnable is True
    assert readiness.reason is None
    assert readiness.detail is None


def test_release_owned_type_does_not_make_a_cross_plugin_type_portable() -> None:
    release = _release(
        input_type="other.private_type",
        output_type="notes.table_summary",
        own_output_type=True,
    )

    readiness = plugin_release_readiness(
        release,
        PluginCatalogExecutionSupport(
            runtime_available=True,
            runtime_profile="python-uv",
        ),
    )

    assert readiness.reason == "unsupported_artifact_type"
    assert readiness.detail is not None
    assert "other.private_type@1" in readiness.detail

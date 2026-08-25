from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.domain.plugin_releases import (
    PluginCatalogManifest,
    PluginNodeContract,
    PluginReleaseNamespace,
    PluginReleaseScope,
)
from grafy_storage import LocalFileObjectStore
from grafy_core.runtime.plugin_loader import WORKSPACE_PLUGIN_LOADER_TARGET

from grafy_api.plugin_oci import (
    PYTHON_UV_BASE_IMAGE_DIGEST,
    PluginOciBuildError,
    PluginOciImageBuilder,
    _dockerfile,
    runtime_profile,
)


def test_runtime_profile_is_pinned_and_unknown_names_are_rejected() -> None:
    profile = runtime_profile("python-uv")

    assert profile.python_version == "3.14"
    assert profile.base_image_digest == PYTHON_UV_BASE_IMAGE_DIGEST
    assert profile.pinned_base_image.endswith(f"@sha256:{PYTHON_UV_BASE_IMAGE_DIGEST}")
    assert profile.cpu_count == 1.0
    assert profile.memory_bytes == 512 * 1_024 * 1_024
    assert profile.pid_limit == 128

    with pytest.raises(ValueError, match="Unknown Plugin runtime profile"):
        runtime_profile("team-image:latest")


def test_dockerfile_sync_uses_bundled_plugin_wheels() -> None:
    dockerfile = _dockerfile(
        runtime_profile("python-uv"),
        release_namespace="workspace/example",
        source_digest="1" * 64,
        contract_digest="2" * 64,
        profile_digest="3" * 64,
        loader_manifest_digest="4" * 64,
    )

    assert (
        "RUN uv sync --locked --no-dev --no-editable \\\n"
        "    --find-links /opt/grafy/plugin/wheels \\\n"
        "    && test -x /opt/grafy/plugin/.venv/bin/python"
    ) in dockerfile
    assert (
        "COPY --chmod=0444 plugin-loader.json "
        "/opt/grafy/plugin/plugin-loader.json"
    ) in dockerfile
    assert 'io.grafy.plugin.loader.digest="sha256:' + "4" * 64 in dockerfile


def test_native_capabilities_require_exact_deployment_owned_profile_image() -> None:
    with pytest.raises(ValueError, match="requires an exact deployment-owned"):
        runtime_profile("python-uv-gdal")

    profile = runtime_profile(
        "python-uv-gdal",
        native_base_image="registry.example/grafy-python-gdal",
        native_base_image_digest="d" * 64,
    )

    assert profile.pinned_base_image == (
        "registry.example/grafy-python-gdal@sha256:" + "d" * 64
    )
    assert profile.native_capabilities == frozenset(
        {PluginRuntimeCapability.NATIVE_GDAL}
    )
    dockerfile = _dockerfile(
        profile,
        release_namespace="system/external.gis",
        source_digest="1" * 64,
        contract_digest="2" * 64,
        profile_digest="3" * 64,
        loader_manifest_digest="4" * 64,
    )
    assert dockerfile.startswith(f"FROM {profile.pinned_base_image}\n")
    assert f'io.grafy.plugin.base.digest="sha256:{"d" * 64}"' in dockerfile
    assert 'io.grafy.plugin.profile.digest="sha256:' + "3" * 64 in dockerfile
    assert "apt-get" not in dockerfile


@pytest.mark.asyncio
async def test_oci_builder_rejects_source_bytes_outside_frozen_identity(
    tmp_path: Path,
) -> None:
    catalog = PluginCatalogManifest(
        slug="notes",
        title="Notes",
        nodes=(
            PluginNodeContract(
                operator_id="notes.echo",
                operator_version=1,
                title="Echo",
                description="Echo",
                config_schema={"type": "object"},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                inputs=(),
                outputs=(),
            ),
        ),
    )
    archive = b"frozen source"
    builder = PluginOciImageBuilder(
        LocalFileObjectStore(tmp_path / "objects"),
        bucket="plugins",
        profile=runtime_profile("python-uv"),
    )

    with pytest.raises(PluginOciBuildError, match="publication digest"):
        await builder.build_and_store(
            namespace=PluginReleaseNamespace(
                scope=PluginReleaseScope.WORKSPACE,
                workspace_id=UUID("00000000-0000-4000-8000-000000000853"),
            ),
            catalog=catalog,
            loader_target=WORKSPACE_PLUGIN_LOADER_TARGET,
            source_archive=archive,
            source_digest=sha256(b"different").hexdigest(),
            contract_digest="1" * 64,
            profile_digest="2" * 64,
        )

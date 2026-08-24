from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from grafy_core.domain.plugin_releases import PluginCatalogManifest, PluginNodeContract
from grafy_storage import LocalFileObjectStore

from grafy_api.plugin_oci import (
    PYTHON_UV_BASE_IMAGE_DIGEST,
    PluginOciBuildError,
    PluginOciImageBuilder,
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
            workspace_id=UUID("00000000-0000-4000-8000-000000000853"),
            catalog=catalog,
            source_archive=archive,
            source_digest=sha256(b"different").hexdigest(),
            contract_digest="1" * 64,
            profile_digest="2" * 64,
        )

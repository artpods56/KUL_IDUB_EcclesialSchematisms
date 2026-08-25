import asyncio
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess

import pytest

from grafy_core.domain.plugin_identity import PluginReleaseScope
from grafy_core.domain.plugin_releases import (
    PluginReleaseNamespace,
    plugin_contract_digest,
    plugin_profile_digest,
)
from grafy_core.runtime.plugin_loader import PluginGuestLoaderManifest
from grafy_storage import LocalFileObjectStore

from grafy_api.plugin_oci import PluginOciImageBuilder, runtime_profile
from grafy_api.plugin_publishing import PluginDirectoryPublisher


SYSTEM_LOADER_TARGET = "grafy_plugin_arithmetic.plugin:ARITHMETIC"


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ("docker", "info"),
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _system_arithmetic_project(repository: Path, destination: Path) -> Path:
    shutil.copytree(
        repository / "plugins" / "arithmetic",
        destination,
        ignore=shutil.ignore_patterns(
            ".venv",
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
        ),
    )
    wheels = destination / "wheels"
    for wheel in wheels.glob("grafy_core-*.whl"):
        wheel.unlink()
    subprocess.run(
        (
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheels),
            str(repository / "libs" / "core"),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        (
            "uv",
            "lock",
            "--find-links",
            "wheels",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=destination,
    )
    return destination


@pytest.mark.asyncio
async def test_retained_system_oci_executes_its_exact_family_loader(
    tmp_path: Path,
) -> None:
    if not _docker_available():
        pytest.skip("local Docker daemon is unavailable")
    repository = Path(__file__).resolve().parents[3]
    project = await asyncio.to_thread(
        _system_arithmetic_project,
        repository,
        tmp_path / "system-arithmetic",
    )
    verified = await asyncio.to_thread(
        PluginDirectoryPublisher(
            (tmp_path,),
            runtime_profile="python-uv",
        ).verify,
        project,
        expected_slug="builtin.arithmetic",
        loader_target=SYSTEM_LOADER_TARGET,
    )
    source_digest = sha256(verified.source_archive).hexdigest()
    contract_digest = plugin_contract_digest(verified.catalog)
    profile_digest = plugin_profile_digest(verified.runtime_profile)
    storage = LocalFileObjectStore(tmp_path / "objects")
    artifact = await PluginOciImageBuilder(
        storage,
        bucket="runtime-test",
        profile=runtime_profile(verified.runtime_profile),
    ).build_and_store(
        namespace=PluginReleaseNamespace(
            scope=PluginReleaseScope.SYSTEM,
            workspace_id=None,
        ),
        catalog=verified.catalog,
        loader_target=SYSTEM_LOADER_TARGET,
        source_archive=verified.source_archive,
        source_digest=source_digest,
        contract_digest=contract_digest,
        profile_digest=profile_digest,
    )
    archive_path = tmp_path / "system-arithmetic.oci.tar"
    stream = await storage.load("runtime-test", artifact.object_key)
    try:
        archive_path.write_bytes(stream.read())
    finally:
        stream.close()
    subprocess.run(
        ("docker", "load", "--input", str(archive_path)),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    image_reference = f"sha256:{artifact.manifest_digest}"
    loader_manifest = PluginGuestLoaderManifest(
        scope=PluginReleaseScope.SYSTEM,
        slug="builtin.arithmetic",
        loader_target=SYSTEM_LOADER_TARGET,
    )
    try:
        labels = subprocess.run(
            (
                "docker",
                "image",
                "inspect",
                "--format",
                "{{ index .Config.Labels \"io.grafy.plugin.loader.digest\" }}",
                image_reference,
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert labels.stdout.strip() == f"sha256:{loader_manifest.digest}"
        script = (
            "from grafy_core.domain.plugin_identity import PluginReleaseScope;"
            "from grafy_core.domain.plugin_releases import plugin_protocol_digest;"
            "from grafy_core.runtime.plugin_guest import load_guest_plugin;"
            "from grafy_core.runtime.plugin_protocol import PluginInvocationRelease;"
            "release=PluginInvocationRelease("
            "scope=PluginReleaseScope.SYSTEM,workspace_id=None,"
            "slug='builtin.arithmetic',revision=1,source_digest='"
            + source_digest
            + "',contract_digest='"
            + contract_digest
            + "',protocol_digest=plugin_protocol_digest(),descriptor_digest='"
            + "d" * 64
            + "');"
            "plugin,catalog=load_guest_plugin(release);"
            "assert plugin.slug == 'builtin.arithmetic';"
            "assert catalog.slug == plugin.slug"
        )
        executed = subprocess.run(
            (
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                image_reference,
                "-c",
                script,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert executed.returncode == 0, executed.stderr
    finally:
        subprocess.run(
            ("docker", "image", "rm", "--force", image_reference),
            check=False,
            capture_output=True,
            timeout=30,
        )

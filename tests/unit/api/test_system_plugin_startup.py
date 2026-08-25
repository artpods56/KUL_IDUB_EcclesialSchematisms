from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

from asgi_lifespan import LifespanManager
from pydantic import SecretStr
import pytest

from grafy_core.domain.plugin_releases import (
    PluginCatalogManifest,
    plugin_contract_digest,
)
from grafy_persistence.database import create_database
from grafy_persistence.orm import metadata
from grafy_plugin_text import TEXT

from grafy_api import main
from grafy_api.main import create_app
from grafy_api.settings import Settings
from grafy_api.system_host_bindings import (
    LoadedSystemPlugin,
    SystemHostPluginBinding,
)
from grafy_api.system_plugin_loader import (
    LoadedSystemPluginDeployment,
    SystemPluginDeploymentError,
)


_LOADER_TARGET = "grafy_plugin_text.plugin:TEXT"
_HOST_BUILD_DIGEST = "f" * 64


@pytest.fixture
async def startup_settings(tmp_path: Path) -> Settings:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'startup.sqlite3'}"
    database = create_database(database_url)
    async with database.engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    await database.dispose()
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        workspace=tmp_path / "workbench",
        database_url=SecretStr(database_url),
        command_hmac_key=SecretStr("test-system-plugin-startup-hmac-key"),
        require_single_api_owner=False,
        graph_room_heartbeat_seconds=0,
    )


def _text_deployment() -> LoadedSystemPluginDeployment:
    catalog = PluginCatalogManifest.from_plugin(TEXT)
    binding = SystemHostPluginBinding(
        release_id=UUID("00000000-0000-0000-0000-000000000901"),
        slug=TEXT.slug,
        revision=3,
        selection_generation=2,
        descriptor_digest="a" * 64,
        contract_digest=plugin_contract_digest(catalog),
        source_digest="b" * 64,
        runtime_archive_digest="c" * 64,
        loader_target=_LOADER_TARGET,
        host_build_digest=_HOST_BUILD_DIGEST,
        catalog=catalog,
    )
    return LoadedSystemPluginDeployment(
        plugins=(TEXT,),
        loaded_plugins=(
            LoadedSystemPlugin(
                slug=TEXT.slug,
                loader_target=_LOADER_TARGET,
                host_build_digest=_HOST_BUILD_DIGEST,
            ),
        ),
        bindings=(binding,),
    )


@pytest.mark.asyncio
async def test_absent_manifest_starts_with_module_boundaries_only(
    startup_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock()
    monkeypatch.setattr(main, "load_system_plugin_deployment_file", loader)
    application = create_app(startup_settings)

    async with LifespanManager(application):
        registry = application.state.resources.plugin_registry

        assert registry.plugins == ()
        assert {(node.key, node.plugin_slug) for node in registry.nodes} == {
            (("module.input", 1), "graph.module"),
            (("module.output", 1), "graph.module"),
        }
        assert application.state.resources.release_admission.system_host_bindings == ()

    loader.assert_not_called()


@pytest.mark.asyncio
async def test_configured_manifest_registers_only_exact_declared_system_plugins(
    startup_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "deployment" / "system-plugins.json"
    deployment = _text_deployment()
    loader = Mock(return_value=deployment)
    monkeypatch.setattr(main, "load_system_plugin_deployment_file", loader)
    configured = startup_settings.model_copy(
        update={"system_plugin_deployment_manifest": manifest_path}
    )
    application = create_app(configured)

    async with LifespanManager(application):
        resources = application.state.resources
        registry = resources.plugin_registry

        assert [(plugin.slug, plugin.title) for plugin in registry.plugins] == [
            (TEXT.slug, TEXT.title)
        ]
        assert {(node.key, node.plugin_slug) for node in registry.nodes} == {
            *((registration.key, TEXT.slug) for registration in TEXT.nodes),
            (("module.input", 1), "graph.module"),
            (("module.output", 1), "graph.module"),
        }
        assert resources.release_admission.system_host_bindings == deployment.bindings

    loader.assert_called_once_with(manifest_path.resolve())


@pytest.mark.asyncio
async def test_configured_manifest_failure_aborts_startup_without_resources(
    startup_settings: Settings,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "deployment" / "tampered.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text('{"plugins":[{"unexpected":true}]}', encoding="utf-8")
    configured = startup_settings.model_copy(
        update={"system_plugin_deployment_manifest": manifest_path}
    )
    application = create_app(configured)

    with pytest.raises(
        SystemPluginDeploymentError,
        match=f"Invalid System Plugin deployment manifest {manifest_path}",
    ):
        async with LifespanManager(application):
            pass

    assert not hasattr(application.state, "resources")

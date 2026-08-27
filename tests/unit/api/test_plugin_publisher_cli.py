import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from grafy_api import cli
from grafy_core.domain.plugin_releases import PlatformPluginActor


class FakeDatabase:
    sessions = object()

    async def dispose(self) -> None:
        pass


class FakePluginReleaseService:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class FakePluginOciImageBuilder:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class RecordingPluginChecker:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def verify(
        self,
        directory: Path,
        *,
        expected_slug: str | None = None,
        loader_target: str,
    ) -> SimpleNamespace:
        assert directory == Path("examples/plugin-notes")
        assert expected_slug is None
        assert loader_target == "grafy_plugin:PLUGIN"
        return SimpleNamespace(
            loader_target="verified_plugin:PLUGIN",
            catalog=SimpleNamespace(
                slug="notes",
                nodes=(object(), object()),
                artifact_types=(object(),),
            ),
            capabilities=SimpleNamespace(capabilities=()),
            source_archive=b"verified source",
            source_digest=(
                "a4d4e45e121b5a09d0219a01e0dc212a76cb9198f016a79e9901c720cf32487f"
            ),
            lock_digest="1" * 64,
            runtime_profile="python-uv",
        )


def create_fake_database(database_url: str) -> FakeDatabase:
    del database_url
    return FakeDatabase()


def configured_fake_storage(settings: object) -> object:
    del settings
    return object()


def test_check_valid_plugin_is_read_only_and_reports_the_verified_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = SimpleNamespace(
        plugin_runtime_profile="python-uv",
        resolved_plugin_roots=(Path("examples"),),
        resolved_plugin_wheelhouse=None,
    )

    def fail_if_database_is_opened(_database_url: str) -> object:
        raise AssertionError("plugin check must not open the database")

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_database", fail_if_database_is_opened)
    monkeypatch.setattr(cli, "PluginDirectoryPublisher", RecordingPluginChecker)
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "check", "examples/plugin-notes"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert '"slug": "notes"' in output
    assert '"loader_target": "verified_plugin:PLUGIN"' in output
    assert '"node_count": 2' in output
    assert '"artifact_type_count": 1' in output
    assert '"source_digest": "' in output


def test_check_incomplete_plugin_reports_the_missing_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "incomplete"
    (project / "src" / "grafy_plugin").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname = 'incomplete'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        plugin_runtime_profile="python-uv",
        resolved_plugin_roots=(tmp_path,),
        resolved_plugin_wheelhouse=None,
    )

    def fail_if_database_is_opened(_database_url: str) -> object:
        raise AssertionError("plugin check must not open the database")

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_database", fail_if_database_is_opened)
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "check", str(project)],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    assert (
        "Plugin check failed: Plugin source archive is missing 'uv.lock'"
        in capsys.readouterr().err
    )


def test_check_rejects_a_plugin_outside_the_configured_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    project = tmp_path / "outside"
    project.mkdir()
    settings = SimpleNamespace(
        plugin_runtime_profile="python-uv",
        resolved_plugin_roots=(allowed_root,),
        resolved_plugin_wheelhouse=None,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "check", str(project)],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    message = capsys.readouterr().err
    assert "Plugin check failed:" in message
    assert "outside configured roots" in message


def test_check_rejects_a_symlinked_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "symlinked"
    project.mkdir()
    outside_source = tmp_path / "outside-source"
    outside_source.mkdir()
    (project / "src").symlink_to(outside_source, target_is_directory=True)
    (project / "tests").mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    settings = SimpleNamespace(
        plugin_runtime_profile="python-uv",
        resolved_plugin_roots=(tmp_path,),
        resolved_plugin_wheelhouse=None,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "check", str(project)],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    message = capsys.readouterr().err
    assert "Plugin check failed:" in message
    assert "unsupported symlinked directory" in message
    assert str(outside_source) in message


class RecordingSystemPublisher:
    observed_loader_target: str | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def verify(
        self,
        directory: object,
        *,
        expected_slug: str | None = None,
        loader_target: str,
    ) -> object:
        del directory, expected_slug
        type(self).observed_loader_target = loader_target
        return object()


class FakeSystemPublicationWorkflow:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def stage_verified(
        self,
        verified: object,
        *,
        platform_actor: PlatformPluginActor,
    ) -> SimpleNamespace:
        del verified, platform_actor
        return SimpleNamespace(slug="builtin.text", revision=1)


def test_system_stage_command_requires_the_sandbox_image_at_parse_time(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grafy",
            "plugin",
            "stage-system",
            "/plugins/notes",
            "--slug",
            "notes",
            "--platform-actor",
            "ci:test",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2
    assert "--sandbox-image" in capsys.readouterr().err


def test_system_stage_inspects_with_checked_in_loader_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = SimpleNamespace(
        resolved_database_url="sqlite+aiosqlite://",
        storage_bucket="plugins",
        plugin_runtime_profile="python-uv",
        plugin_runtime_native_base_image=None,
        plugin_runtime_native_base_image_digest=None,
        plugin_docker_binary="docker",
        resolved_plugin_roots=(),
        resolved_plugin_wheelhouse=None,
        resolved_plugin_egress_policy=object(),
        resolved_network_policy=object(),
    )
    RecordingSystemPublisher.observed_loader_target = None
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_database", create_fake_database)
    monkeypatch.setattr(cli, "configured_file_storage", configured_fake_storage)
    monkeypatch.setattr(cli, "PluginReleaseService", FakePluginReleaseService)
    monkeypatch.setattr(cli, "PluginOciImageBuilder", FakePluginOciImageBuilder)
    monkeypatch.setattr(cli, "isolated_release_admission", lambda **kwargs: object())
    monkeypatch.setattr(
        cli,
        "DockerPluginDirectoryPublisher",
        RecordingSystemPublisher,
    )
    monkeypatch.setattr(
        cli,
        "SystemPluginPublicationWorkflow",
        FakeSystemPublicationWorkflow,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grafy",
            "plugin",
            "stage-system",
            "plugins/text",
            "--slug",
            "builtin.text",
            "--platform-actor",
            "ci:test",
            "--sandbox-image",
            "grafy-publisher:test",
        ],
    )

    cli.main()

    assert (
        RecordingSystemPublisher.observed_loader_target
        == "grafy_plugin_text.plugin:TEXT"
    )
    assert "Staged System Plugin builtin.text release 1" in capsys.readouterr().out


def test_system_promotion_is_a_distinct_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "promote-system", "--help"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--revision" in output
    assert "--expected-generation" in output
    assert "--deployment-manifest" in output


def test_system_deployment_builder_exposes_exact_or_all_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "build-system-deployment", "--help"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--output" in output
    assert "--slug" in output
    assert "--revision" in output


def test_system_revocation_requires_exact_reason_and_platform_actor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["grafy", "plugin", "revoke-system", "--help"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--revision" in output
    assert "--reason" in output
    assert "security" in output
    assert "--platform-actor" in output

import sys
from types import SimpleNamespace

import pytest

from grafy_api import cli
from grafy_core.domain.plugin_releases import (
    PlatformPluginActor,
    PluginDistribution,
    PluginExecutionPolicy,
)


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


def create_fake_database(database_url: str) -> FakeDatabase:
    del database_url
    return FakeDatabase()


def configured_fake_storage(settings: object) -> object:
    del settings
    return object()


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
        execution_policy: PluginExecutionPolicy,
        distribution: PluginDistribution,
        platform_actor: PlatformPluginActor,
    ) -> SimpleNamespace:
        del verified, execution_policy, distribution, platform_actor
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
            "--execution-policy",
            "isolated-only",
            "--distribution",
            "published",
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
    )
    RecordingSystemPublisher.observed_loader_target = None
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_database", create_fake_database)
    monkeypatch.setattr(cli, "configured_file_storage", configured_fake_storage)
    monkeypatch.setattr(cli, "PluginReleaseService", FakePluginReleaseService)
    monkeypatch.setattr(cli, "PluginOciImageBuilder", FakePluginOciImageBuilder)
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
            "--execution-policy",
            "host-eligible",
            "--distribution",
            "bundled",
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

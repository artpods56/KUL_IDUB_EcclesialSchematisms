from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from grafy_api.settings import Settings


def test_default_workspace_reuses_legacy_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy_workspace = tmp_path / ".notarius-artifacts" / "workbench"
    legacy_workspace.mkdir(parents=True)

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.workspace == Path(".notarius-artifacts/workbench")


def test_default_workspace_prefers_grafy_when_both_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".notarius-artifacts" / "workbench").mkdir(parents=True)
    (tmp_path / ".grafy-artifacts" / "workbench").mkdir(parents=True)

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.workspace == Path(".grafy-artifacts/workbench")


def test_plugin_roots_resolve_from_the_deployment_allowlist(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        plugin_roots=(tmp_path / "team-plugins", tmp_path / "examples"),
    )

    assert settings.resolved_plugin_roots == (
        (tmp_path / "team-plugins").resolve(),
        (tmp_path / "examples").resolve(),
    )


def test_agent_authoring_paths_are_deployment_owned(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        plugin_authoring_root=tmp_path / "team-plugins",
        plugin_sdk_project=tmp_path / "sdk",
    )

    assert (
        settings.resolved_plugin_authoring_root == (tmp_path / "team-plugins").resolve()
    )
    assert settings.resolved_plugin_sdk_project == (tmp_path / "sdk").resolve()


def test_database_url_reuses_legacy_database(tmp_path: Path) -> None:
    legacy_database = tmp_path / "notarius.sqlite3"
    legacy_database.touch()

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        workspace=tmp_path,
    )

    assert settings.resolved_database_url == (
        f"sqlite+aiosqlite:///{legacy_database.resolve()}"
    )


def test_execution_defaults_with_bounded_map_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAFY_MAP_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("GRAFY_MAX_ACTIVE_EXECUTIONS", raising=False)
    # Defaults must be tested independently from a developer's local .env.
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.map_max_concurrency == 4
    assert settings.max_active_executions == 2
    assert settings.max_pending_graphs == 20
    assert settings.max_active_plugin_invocations == 4
    assert settings.max_live_plugin_sandboxes == 4
    assert settings.max_distinct_plugin_releases_per_graph == 4


def test_map_max_concurrency_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFY_MAP_MAX_CONCURRENCY", "7")

    assert Settings().map_max_concurrency == 7


def test_map_max_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(map_max_concurrency=0)


def test_max_active_executions_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFY_MAX_ACTIVE_EXECUTIONS", "6")

    assert Settings().max_active_executions == 6


@pytest.mark.parametrize("value", [0, 33])
def test_max_active_executions_is_bounded(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_active_executions=value)


def test_max_pending_graphs_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFY_MAX_PENDING_GRAPHS", "80")

    assert Settings().max_pending_graphs == 80


@pytest.mark.parametrize("value", [0, 1_001])
def test_max_pending_graphs_is_bounded(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_pending_graphs=value)


def test_plugin_capacity_dimensions_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFY_MAX_ACTIVE_PLUGIN_INVOCATIONS", "6")
    monkeypatch.setenv("GRAFY_MAX_LIVE_PLUGIN_SANDBOXES", "8")
    monkeypatch.setenv("GRAFY_MAX_DISTINCT_PLUGIN_RELEASES_PER_GRAPH", "7")

    settings = Settings()
    assert settings.max_active_plugin_invocations == 6
    assert settings.max_live_plugin_sandboxes == 8
    assert settings.max_distinct_plugin_releases_per_graph == 7


def test_distinct_plugin_release_limit_cannot_exceed_live_sandboxes() -> None:
    with pytest.raises(ValidationError):
        Settings(
            max_live_plugin_sandboxes=4,
            max_distinct_plugin_releases_per_graph=5,
        )


def test_oidc_signing_algorithms_are_strictly_allowlisted() -> None:
    with pytest.raises(ValidationError):
        Settings(oidc_allowed_signing_algorithms=("none",))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("oidc_client_id", ""),
        ("oidc_auth_wrapping_key", SecretStr("")),
        ("oidc_client_secret", SecretStr("")),
    ],
)
def test_oidc_configuration_rejects_empty_security_values(
    field: str,
    value: str | SecretStr,
) -> None:
    configured: dict[str, object] = {
        "oidc_issuer": "https://issuer.example.test",
        "oidc_client_id": "grafy-web",
        "oidc_auth_wrapping_key": SecretStr("wrapping-key"),
    }
    configured[field] = value

    with pytest.raises(ValidationError):
        Settings.model_validate(configured)


def test_database_url_is_redacted_from_serialized_settings() -> None:
    database_url = "sqlite+aiosqlite:///sensitive-database-name.sqlite3"
    settings = Settings(database_url=SecretStr(database_url))

    assert settings.resolved_database_url == database_url
    assert database_url not in repr(settings)
    dumped_url = settings.model_dump()["database_url"]
    assert isinstance(dumped_url, SecretStr)
    assert dumped_url.get_secret_value() == database_url
    assert database_url not in str(settings.model_dump())


def test_s3_credentials_are_redacted_from_serialized_settings() -> None:
    access_key = "sensitive-access-key"
    secret_key = "sensitive-secret-key"
    settings = Settings(
        storage_backend="s3",
        s3_access_key_id=SecretStr(access_key),
        s3_secret_access_key=SecretStr(secret_key),
    )

    assert access_key not in repr(settings)
    assert secret_key not in repr(settings)
    assert access_key not in str(settings.model_dump())
    assert secret_key not in str(settings.model_dump())


def test_credential_encryption_key_is_redacted_from_serialized_settings() -> None:
    encryption_key = "sensitive-node-secret-encryption-key"
    settings = Settings(
        credential_encryption_key=SecretStr(encryption_key),
    )

    assert encryption_key not in repr(settings)
    assert encryption_key not in str(settings.model_dump())


def test_command_hmac_key_is_redacted_and_resolved() -> None:
    hmac_key = "sensitive-command-hmac-key"
    settings = Settings(
        command_hmac_key=SecretStr(hmac_key),
        command_hmac_key_version=2,
    )

    assert settings.resolved_command_hmac_key() == hmac_key.encode("utf-8")
    assert settings.command_hmac_key_version == 2
    assert hmac_key not in repr(settings)
    assert hmac_key not in str(settings.model_dump())


def test_command_hmac_key_fails_closed_when_missing() -> None:
    settings = Settings(command_hmac_key=None)
    with pytest.raises(ValueError, match="GRAFY_COMMAND_HMAC_KEY"):
        settings.resolved_command_hmac_key()


def test_command_hmac_key_fails_closed_when_empty() -> None:
    settings = Settings(command_hmac_key=SecretStr(""))
    with pytest.raises(ValueError, match="must not be empty"):
        settings.resolved_command_hmac_key()

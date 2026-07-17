import pytest
from pydantic import SecretStr, ValidationError

from notarius_api.settings import Settings


def test_execution_defaults_to_prefect_with_bounded_map_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOTARIUS_EXECUTION_BACKEND", raising=False)
    monkeypatch.delenv("NOTARIUS_MAP_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("NOTARIUS_PREFECT_TASK_RETRIES", raising=False)
    monkeypatch.delenv(
        "NOTARIUS_PREFECT_TASK_RETRY_DELAY_SECONDS",
        raising=False,
    )
    settings = Settings()

    assert settings.execution_backend == "prefect"
    assert settings.map_max_concurrency == 4
    assert settings.prefect_task_retries == 0
    assert settings.prefect_task_retry_delay_seconds == 0


def test_execution_backend_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTARIUS_EXECUTION_BACKEND", "inline")

    assert Settings().execution_backend == "inline"


def test_execution_backend_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"execution_backend": "worker"})


def test_map_max_concurrency_can_be_selected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTARIUS_MAP_MAX_CONCURRENCY", "7")

    assert Settings().map_max_concurrency == 7


def test_map_max_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(map_max_concurrency=0)


def test_prefect_task_retry_settings_must_not_be_negative() -> None:
    with pytest.raises(ValidationError):
        Settings(prefect_task_retries=-1)
    with pytest.raises(ValidationError):
        Settings(prefect_task_retry_delay_seconds=-0.1)


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

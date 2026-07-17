from pydantic import SecretStr

from notarius_api.settings import Settings


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

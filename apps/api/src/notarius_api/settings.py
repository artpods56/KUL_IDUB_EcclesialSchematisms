from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_OIDC_ALLOWED_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
    }
)


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOTARIUS_",
        extra="ignore",
    )

    workspace: Path = Path(".notarius-artifacts/workbench")
    public_origin: str = "http://localhost:3000"
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: SecretStr | None = None
    oidc_allowed_signing_algorithms: tuple[str, ...] = ("RS256",)
    oidc_auth_wrapping_key: SecretStr | None = None
    oidc_auth_wrapping_key_version: int = Field(default=1, ge=1)
    oidc_login_transaction_ttl_seconds: int = Field(default=300, ge=30, le=900)
    auth_session_idle_seconds: int = Field(default=1800, ge=60)
    auth_session_absolute_seconds: int = Field(default=28800, ge=300)
    personal_access_token_max_lifetime_seconds: int = Field(
        default=2592000,
        ge=60,
    )
    auth_cookie_secure: bool = True
    oidc_callback_path: str = "/api/v1/auth/oidc/callback"
    auth_rate_window_seconds: int = Field(default=60, ge=1, le=3600)
    auth_login_start_rate_limit: int = Field(default=10, ge=1)
    auth_callback_rate_limit: int = Field(default=20, ge=1)
    auth_session_failure_rate_limit: int = Field(default=30, ge=1)
    auth_pat_creation_rate_limit: int = Field(default=10, ge=1)
    auth_outstanding_login_limit: int = Field(default=2, ge=1, le=10)
    auth_outstanding_login_network_limit: int = Field(default=8, ge=1, le=40)
    auth_cleanup_interval_seconds: int = Field(default=60, ge=1, le=3600)
    database_url: SecretStr | None = None
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    execution_backend: Literal["prefect", "inline"] = "prefect"
    map_max_concurrency: int = Field(default=4, ge=1)
    prefect_task_retries: int = Field(default=0, ge=0)
    prefect_task_retry_delay_seconds: float = Field(default=0, ge=0)
    storage_backend: Literal["local", "s3"] = "local"
    storage_bucket: str = Field(default="workbench-artifacts", min_length=1)
    s3_endpoint_url: str | None = None
    s3_region: str = Field(default="us-east-1", min_length=1)
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_force_path_style: bool = False
    credential_encryption_key: SecretStr | None = None
    command_hmac_key: SecretStr | None = None
    command_hmac_key_version: int = Field(default=1, ge=1)

    @field_validator("public_origin", "oidc_issuer")
    @classmethod
    def _validate_origin_or_issuer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OIDC origin and issuer must be absolute HTTP URLs")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "OIDC origin and issuer must not contain query or fragment"
            )
        return value.rstrip("/")

    @field_validator("oidc_callback_path")
    @classmethod
    def _validate_callback_path(cls, value: str) -> str:
        if not value.startswith("/") or value.startswith("//"):
            raise ValueError("OIDC callback path must be absolute and relative")
        if "?" in value or "#" in value:
            raise ValueError("OIDC callback path must not contain a query or fragment")
        if value != "/api/v1/auth/oidc/callback":
            raise ValueError("OIDC callback path must be the registered callback")
        return value

    @model_validator(mode="after")
    def _validate_oidc_configuration(self) -> "Settings":
        configured = (
            self.oidc_issuer,
            self.oidc_client_id,
            self.oidc_auth_wrapping_key,
        )
        if any(value is not None for value in configured) and not all(
            value is not None for value in configured
        ):
            raise ValueError(
                "oidc_issuer, oidc_client_id, and oidc_auth_wrapping_key must be "
                "configured together"
            )
        if not self.oidc_allowed_signing_algorithms:
            raise ValueError("At least one OIDC signing algorithm is required")
        if any(
            algorithm.strip() == "" or algorithm != algorithm.strip()
            for algorithm in self.oidc_allowed_signing_algorithms
        ):
            raise ValueError("OIDC signing algorithms must be non-empty values")
        if any(
            algorithm not in _OIDC_ALLOWED_ALGORITHMS
            for algorithm in self.oidc_allowed_signing_algorithms
        ):
            raise ValueError("OIDC signing algorithm is not allowed")
        if self.auth_session_idle_seconds >= self.auth_session_absolute_seconds:
            raise ValueError(
                "Auth session idle lifetime must be below absolute lifetime"
            )
        return self

    @property
    def resolved_database_url(self) -> str:
        if self.database_url is not None:
            return self.database_url.get_secret_value()
        database_path = (self.workspace / "notarius.sqlite3").resolve()
        return f"sqlite+aiosqlite:///{database_path}"

    @property
    def allowed_cors_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        )

    @property
    def oidc_callback_url(self) -> str:
        return f"{self.public_origin}{self.oidc_callback_path}"

    @property
    def oidc_is_configured(self) -> bool:
        return self.oidc_issuer is not None

    def resolved_command_hmac_key(self) -> bytes:
        """Return the deployment HMAC key, failing closed when unset or empty."""
        configured = self.command_hmac_key
        if configured is None:
            raise ValueError(
                "NOTARIUS_COMMAND_HMAC_KEY must be configured for collaboration"
            )
        value = configured.get_secret_value()
        if value == "":
            raise ValueError("NOTARIUS_COMMAND_HMAC_KEY must not be empty")
        return value.encode("utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()

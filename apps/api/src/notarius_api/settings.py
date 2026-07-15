from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOTARIUS_",
        extra="ignore",
    )

    workspace: Path = Path(".notarius-artifacts/workbench")
    database_url: SecretStr | None = None
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    storage_backend: Literal["local", "s3"] = "local"
    storage_bucket: str = Field(default="workbench-artifacts", min_length=1)
    s3_endpoint_url: str | None = None
    s3_region: str = Field(default="us-east-1", min_length=1)
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_force_path_style: bool = False

    @property
    def resolved_database_url(self) -> str:
        if self.database_url is not None:
            return self.database_url.get_secret_value()
        database_path = (self.workspace / "notarius.sqlite3").resolve()
        return f"sqlite+aiosqlite:///{database_path}"

    @property
    def allowed_cors_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

from typing import ClassVar

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOTARIUS_MCP_",
        extra="ignore",
    )

    api_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    timeout_seconds: float = Field(default=15.0, gt=0)


__all__ = ["Settings"]

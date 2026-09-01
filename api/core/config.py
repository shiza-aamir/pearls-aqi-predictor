from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    app_name: str = "Pearls AQI Predictor API"
    app_version: str = "1.0.0"

    environment: str = "development"

    api_prefix: str = "/api/v1"

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

    allowed_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings()
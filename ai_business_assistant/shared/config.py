from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "local"
    project_name: str = "AI Business Assistant"

    log_level: str = "INFO"

    jwt_secret_key: str = Field(
        default="CHANGE_ME_CHANGE_ME_CHANGE_ME_CHANGE_ME",
        min_length=32,
        description="JWT signing secret. Override via environment variables.",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_seconds: int = 60 * 60

    postgres_dsn: str = "postgresql+asyncpg://aba:aba@postgres:5432/aba"
    redis_url: str = "redis://redis:6379/0"

    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672//"
    celery_result_backend: str = "redis://redis:6379/1"

    qdrant_url: str = "http://qdrant:6333"

    api_gateway_auth_service_url: str = "http://auth:8001"
    api_gateway_assistant_service_url: str = "http://assistant:8002"


@lru_cache
def get_settings() -> Settings:
    return Settings()

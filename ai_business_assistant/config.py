"""
Configuration management for Business AI Assistant.
Environment-based settings with Pydantic.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    # Application
    APP_NAME: str = "Business AI Assistant"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/business_ai"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    CACHE_TTL: int = 3600
    
    # Security
    SECRET_KEY: str = "change-me-in-production-use-random-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8
    
    # OAuth2
    OAUTH2_ENABLED: bool = False
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".csv", ".json", ".xlsx", ".xls"]
    
    # Data Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = Path("data")
    UPLOAD_DIR: Path = Path("data/uploads")
    EXPORT_DIR: Path = Path("data/exports")
    MODEL_DIR: Path = Path("models")
    
    # Warehouse
    WAREHOUSE_PATH: Path = Path("data/warehouse.duckdb")
    
    # ML Models
    MODEL_CACHE_DIR: Path = Path("models/cache")
    MODEL_VERSION: str = "1.0.0"
    
    # Celery (for async tasks)
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False
    
    # External APIs
    MARKET_DATA_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    
    # Webhooks
    WEBHOOK_SECRET: str = "webhook-secret-change-in-production"
    WEBHOOK_TIMEOUT: int = 30
    
    # Feature Flags
    ENABLE_GRAPHQL: bool = True
    ENABLE_WEBSOCKET: bool = True
    ENABLE_BATCH_PROCESSING: bool = True
    
    def abs_path(self, p: Path) -> Path:
        """Get absolute path relative to project root."""
        if p.is_absolute():
            return p
        return self.PROJECT_ROOT / p
    
    @property
    def abs_data_dir(self) -> Path:
        """Get absolute data directory path."""
        return self.abs_path(self.DATA_DIR)
    
    @property
    def abs_warehouse_path(self) -> Path:
        """Get absolute warehouse path."""
        return self.abs_path(self.WAREHOUSE_PATH)
    
    @property
    def abs_upload_dir(self) -> Path:
        """Get absolute upload directory path."""
        return self.abs_path(self.UPLOAD_DIR)
    
    @property
    def abs_export_dir(self) -> Path:
        """Get absolute export directory path."""
        return self.abs_path(self.EXPORT_DIR)
    
    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        for p in [
            self.abs_data_dir,
            self.abs_upload_dir,
            self.abs_export_dir,
            self.abs_path(self.MODEL_DIR),
            self.abs_path(self.MODEL_CACHE_DIR),
        ]:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

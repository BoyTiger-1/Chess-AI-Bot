"""Shared utilities and modules for Business AI Assistant."""

from ai_business_assistant.shared.database import get_db, init_db, close_db
from ai_business_assistant.shared.logging import get_logger, setup_logging
from ai_business_assistant.shared.model_cache import get_model_cache
from ai_business_assistant.shared.model_loader import (
    LazyModelLoader,
    ModelLoader,
    get_lazy_loader,
    get_model_loader,
    get_sentiment_pipeline,
    get_sklearn_model,
    preload_models,
    unload_models,
)
from ai_business_assistant.shared.redis_cache import get_redis, init_redis, close_redis

__all__ = [
    # Database
    "get_db",
    "init_db",
    "close_db",
    # Logging
    "get_logger",
    "setup_logging",
    # Model Cache
    "get_model_cache",
    # Model Loader
    "ModelLoader",
    "LazyModelLoader",
    "get_model_loader",
    "get_lazy_loader",
    "get_sentiment_pipeline",
    "get_sklearn_model",
    "preload_models",
    "unload_models",
    # Redis
    "get_redis",
    "init_redis",
    "close_redis",
]

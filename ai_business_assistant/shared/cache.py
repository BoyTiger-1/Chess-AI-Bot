from __future__ import annotations

from functools import lru_cache

import redis.asyncio as redis

from ai_business_assistant.shared.config import Settings, get_settings


@lru_cache
def get_redis(settings: Settings | None = None) -> redis.Redis:
    settings = settings or get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)

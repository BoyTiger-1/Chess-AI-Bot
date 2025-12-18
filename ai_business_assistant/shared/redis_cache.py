"""
Redis caching layer for high-performance data access.
"""

import json
from typing import Optional, Any
from datetime import timedelta

import redis.asyncio as redis

from ai_business_assistant.config import get_settings
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global Redis client
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
        # Test connection
        await redis_client.ping()
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Redis: {e}")
        raise


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


async def get_redis_health() -> bool:
    """Check Redis health."""
    try:
        if redis_client:
            await redis_client.ping()
            return True
        return False
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        if not redis_client:
            return None
        
        value = await redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Error getting from cache: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: Optional[int] = 3600):
    """Set value in cache with TTL in seconds."""
    try:
        if not redis_client:
            return False
        
        serialized = json.dumps(value, default=str)
        if ttl:
            await redis_client.setex(key, ttl, serialized)
        else:
            await redis_client.set(key, serialized)
        return True
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    try:
        if not redis_client:
            return False
        
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Error deleting from cache: {e}")
        return False


async def cache_delete_pattern(pattern: str) -> bool:
    """Delete all keys matching pattern."""
    try:
        if not redis_client:
            return False
        
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Error deleting cache pattern: {e}")
        return False


async def cache_exists(key: str) -> bool:
    """Check if key exists in cache."""
    try:
        if not redis_client:
            return False
        
        return await redis_client.exists(key) > 0
    except Exception as e:
        logger.error(f"Error checking cache existence: {e}")
        return False


async def cache_ttl(key: str) -> int:
    """Get TTL of key in seconds."""
    try:
        if not redis_client:
            return -2
        
        return await redis_client.ttl(key)
    except Exception as e:
        logger.error(f"Error getting cache TTL: {e}")
        return -2


async def cache_increment(key: str, amount: int = 1) -> Optional[int]:
    """Increment counter in cache."""
    try:
        if not redis_client:
            return None
        
        return await redis_client.incrby(key, amount)
    except Exception as e:
        logger.error(f"Error incrementing cache: {e}")
        return None


async def cache_expire(key: str, ttl: int) -> bool:
    """Set expiration on existing key."""
    try:
        if not redis_client:
            return False
        
        return await redis_client.expire(key, ttl)
    except Exception as e:
        logger.error(f"Error setting cache expiration: {e}")
        return False


def cache_key(*parts: str) -> str:
    """Generate cache key from parts."""
    return ":".join(str(p) for p in parts)

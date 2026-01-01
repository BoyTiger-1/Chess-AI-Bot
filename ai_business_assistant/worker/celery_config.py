from __future__ import annotations

from ai_business_assistant.shared.config import get_settings

settings = get_settings()

broker_url = settings.redis_url
result_backend = settings.celery_result_backend

task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "UTC"
enable_utc = True

# Task timeout configurations
task_soft_time_limit = 300  # 5 minutes
task_time_limit = 600       # 10 minutes

# Task result expiration
result_expires = 3600       # 1 hour

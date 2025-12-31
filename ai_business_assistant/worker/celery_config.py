"""
Celery configuration for Business AI Assistant.
"""

from ai_business_assistant.config import get_settings

settings = get_settings()

broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "UTC"
enable_utc = True

# Task routing
task_routes = {
    "ai_business_assistant.worker.tasks.*": {"queue": "default"},
    "ai_business_assistant.worker.tasks.ml_*": {"queue": "ml_tasks"},
}

# Reliability settings
task_acks_late = True
worker_prefetch_multiplier = 1
task_reject_on_worker_lost = True

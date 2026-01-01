from __future__ import annotations

from celery import Celery

from ai_business_assistant.worker import celery_config


celery_app = Celery("ai_business_assistant")
celery_app.config_from_object(celery_config)
celery_app.autodiscover_tasks(["ai_business_assistant.worker"])

if __name__ == "__main__":
    celery_app.start()

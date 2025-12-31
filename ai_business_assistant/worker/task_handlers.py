"""
Task result storage and status tracking for Celery tasks.
"""

import logging
from typing import Any, Dict, Optional
from celery.result import AsyncResult
from ai_business_assistant.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status and result of a Celery task.
    """
    result = AsyncResult(task_id, app=celery_app)
    
    status_info = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
    }
    
    if result.ready():
        if result.successful():
            status_info["result"] = result.result
        else:
            status_info["error"] = str(result.result)
            
    return status_info

def cancel_task(task_id: str) -> bool:
    """
    Revoke a running or pending task.
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {e}")
        return False

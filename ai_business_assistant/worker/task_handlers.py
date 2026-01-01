from __future__ import annotations

import json
import httpx
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from celery.result import AsyncResult
from sqlalchemy import select, delete
from ai_business_assistant.worker.celery_app import celery_app
from ai_business_assistant.shared.logging import get_logger
from ai_business_assistant.shared.db.session import get_sessionmaker
from ai_business_assistant.shared.db.models import TaskResult

logger = get_logger(__name__)

async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status and result of a Celery task, checking DB if not in Celery."""
    result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
        
        # Save to DB if finished
        await save_task_result(
            task_id=task_id,
            task_name=result.name or "unknown",
            status=result.status,
            result=result.result if result.successful() else None,
            error=None if result.successful() else str(result.result)
        )
            
    else:
        # Check DB if it was already completed and removed from Celery's backend
        session_maker = get_sessionmaker()
        async with session_maker() as session:
            db_result = (await session.execute(
                select(TaskResult).where(TaskResult.task_id == task_id)
            )).scalar_one_or_none()
            
            if db_result:
                response["status"] = db_result.status
                if db_result.result:
                    response["result"] = json.loads(db_result.result)
                if db_result.error:
                    response["error"] = db_result.error

    return response

async def save_task_result(task_id: str, task_name: str, status: str, result: Any = None, error: str = None, webhook_url: str = None):
    """Save task result to database."""
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        try:
            db_result = (await session.execute(
                select(TaskResult).where(TaskResult.task_id == task_id)
            )).scalar_one_or_none()
            
            if not db_result:
                db_result = TaskResult(
                    task_id=task_id,
                    task_name=task_name,
                    status=status,
                    result=json.dumps(result) if result else None,
                    error=error,
                    webhook_url=webhook_url
                )
                session.add(db_result)
            else:
                db_result.status = status
                db_result.result = json.dumps(result) if result else None
                db_result.error = error
                if webhook_url:
                    db_result.webhook_url = webhook_url
            
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to save task result to DB: {e}")

async def cleanup_old_results(days: int = 7):
    """Clean up old task results from the database."""
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        try:
            threshold = datetime.now() - timedelta(days=days)
            await session.execute(
                delete(TaskResult).where(TaskResult.created_at < threshold)
            )
            await session.commit()
            logger.info(f"Cleaned up task results older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old task results: {e}")

async def notify_webhook(webhook_url: str, payload: Dict[str, Any]):
    """Send a webhook notification when a task completes."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully sent webhook to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {e}")

def on_task_completion(task_id: str, webhook_url: Optional[str] = None):
    """Callback for task completion."""
    result = AsyncResult(task_id, app=celery_app)
    status = result.status
    task_result = result.result if result.successful() else None
    
    if webhook_url:
        import asyncio
        payload = {
            "task_id": task_id,
            "status": status,
            "result": task_result
        }
        # Note: This is a bit tricky from within Celery if we want to be truly async,
        # but for this purpose, we can use a helper task or just fire and forget.
        logger.info(f"Task {task_id} completed with status {status}. Notifying {webhook_url}")
        # In a real app, we might trigger another task to handle the webhook delivery.

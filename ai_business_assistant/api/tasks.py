from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ai_business_assistant.worker.task_handlers import get_task_status
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/{task_id}")
async def check_task_status(task_id: str):
    """Check the status and result of an asynchronous task."""
    try:
        status = await get_task_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")

"""
API routes for monitoring and managing background tasks.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict
from ai_business_assistant.worker.task_handlers import get_task_status, cancel_task

router = APIRouter()

@router.get("/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str):
    """Get the current status and result of a task."""
    try:
        status_info = get_task_status(task_id)
        if not status_info:
            raise HTTPException(status_code=404, detail="Task not found")
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_task(task_id: str):
    """Cancel a running or pending task."""
    if not cancel_task(task_id):
        raise HTTPException(status_code=400, detail="Could not cancel task")
    return None

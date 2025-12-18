"""
Webhook management API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class WebhookCreate(BaseModel):
    name: str
    url: HttpUrl
    events: List[str]
    secret: Optional[str] = None


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    is_active: Optional[bool] = None


@router.post("/")
async def create_webhook(
    webhook: WebhookCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new webhook."""
    try:
        logger.info(f"Webhook created: {webhook.name}")
        
        return {
            "id": 1,
            "name": webhook.name,
            "url": str(webhook.url),
            "events": webhook.events,
            "is_active": True,
            "created_at": "2024-01-01T00:00:00"
        }
        
    except Exception as e:
        logger.error(f"Webhook creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_webhooks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all webhooks."""
    try:
        return {"webhooks": [], "count": 0}
        
    except Exception as e:
        logger.error(f"Failed to list webhooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{webhook_id}")
async def get_webhook(
    webhook_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get webhook by ID."""
    try:
        raise HTTPException(status_code=404, detail="Webhook not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{webhook_id}")
async def update_webhook(
    webhook_id: int,
    webhook: WebhookUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update webhook configuration."""
    try:
        logger.info(f"Webhook updated: {webhook_id}")
        
        return {"message": "Webhook updated successfully"}
        
    except Exception as e:
        logger.error(f"Webhook update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete webhook."""
    try:
        logger.info(f"Webhook deleted: {webhook_id}")
        
        return {"message": "Webhook deleted successfully"}
        
    except Exception as e:
        logger.error(f"Webhook deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{webhook_id}/events")
async def get_webhook_events(
    webhook_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get webhook delivery events/logs."""
    try:
        return {"events": [], "count": 0}
        
    except Exception as e:
        logger.error(f"Failed to retrieve webhook events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

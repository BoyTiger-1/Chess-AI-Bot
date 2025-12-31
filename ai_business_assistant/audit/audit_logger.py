"""
Audit logger for tracking system actions.
"""

import logging
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ai_business_assistant.audit.audit_models import AuditLog

logger = logging.getLogger(__name__)

async def log_audit_event(
    db: AsyncSession,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    status: str = "success"
):
    """Log an audit event to the database."""
    try:
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            changes=changes,
            ip_address=ip_address,
            status=status
        )
        db.add(audit_entry)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")

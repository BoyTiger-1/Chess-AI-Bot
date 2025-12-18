"""
Webhook models for event notifications.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean

from ai_business_assistant.shared.database import Base


class Webhook(Base):
    """Webhook configuration model."""
    
    __tablename__ = "webhooks"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=False)
    
    events = Column(JSON)
    headers = Column(JSON)
    secret = Column(String(255))
    
    is_active = Column(Boolean, default=True)
    retry_count = Column(Integer, default=3)
    timeout = Column(Integer, default=30)
    
    last_triggered_at = Column(DateTime)
    last_success_at = Column(DateTime)
    last_failure_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WebhookEvent(Base):
    """Webhook event delivery log."""
    
    __tablename__ = "webhook_events"
    
    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"), nullable=False)
    
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON)
    
    status = Column(String(20))
    response_code = Column(Integer)
    response_body = Column(Text)
    
    attempts = Column(Integer, default=0)
    delivered_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)

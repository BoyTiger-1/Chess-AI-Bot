"""
Audit log models for database.
"""

from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey
from ai_business_assistant.shared.database import Base
from datetime import datetime
import uuid

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, index=True)
    action = Column(String, index=True)
    resource_type = Column(String, index=True)
    resource_id = Column(String)
    changes = Column(JSON)
    ip_address = Column(String)
    status = Column(String)

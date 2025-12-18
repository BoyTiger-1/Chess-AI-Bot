"""
Customer behavior and segmentation models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean

from ai_business_assistant.shared.database import Base


class Customer(Base):
    """Customer model for behavior analysis."""
    
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    external_id = Column(String(255), index=True)
    email = Column(String(255))
    name = Column(String(255))
    
    segment_id = Column(Integer, ForeignKey("customer_segments.id"))
    
    total_revenue = Column(Float, default=0.0)
    lifetime_value = Column(Float)
    acquisition_cost = Column(Float)
    
    churn_risk_score = Column(Float)
    engagement_score = Column(Float)
    satisfaction_score = Column(Float)
    
    purchase_count = Column(Integer, default=0)
    last_purchase_date = Column(DateTime)
    
    attributes = Column(JSON)
    preferences = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CustomerSegment(Base):
    """Customer segmentation model."""
    
    __tablename__ = "customer_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    segment_type = Column(String(50))
    criteria = Column(JSON)
    characteristics = Column(JSON)
    
    customer_count = Column(Integer, default=0)
    avg_lifetime_value = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CustomerEvent(Base):
    """Customer behavioral events and interactions."""
    
    __tablename__ = "customer_events"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    
    event_type = Column(String(100), nullable=False)
    event_name = Column(String(255))
    
    properties = Column(JSON)
    value = Column(Float)
    
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

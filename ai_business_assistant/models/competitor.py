"""
Competitive intelligence models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean

from ai_business_assistant.shared.database import Base


class Competitor(Base):
    """Competitor tracking model."""
    
    __tablename__ = "competitors"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    website = Column(String(500))
    description = Column(Text)
    
    industry = Column(String(100))
    size = Column(String(50))
    location = Column(String(255))
    
    market_share = Column(Float)
    revenue_estimate = Column(Float)
    
    strengths = Column(JSON)
    weaknesses = Column(JSON)
    positioning = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CompetitorProduct(Base):
    """Competitor product and pricing information."""
    
    __tablename__ = "competitor_products"
    
    id = Column(Integer, primary_key=True, index=True)
    competitor_id = Column(Integer, ForeignKey("competitors.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    description = Column(Text)
    
    price = Column(Float)
    pricing_model = Column(String(50))
    
    features = Column(JSON)
    specifications = Column(JSON)
    
    availability = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

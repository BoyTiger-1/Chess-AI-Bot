"""
Recommendation engine models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean

from ai_business_assistant.shared.database import Base


class Recommendation(Base):
    """Strategic recommendation model."""
    
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    category = Column(String(100))
    priority = Column(String(20))
    confidence = Column(Float)
    
    recommendation_type = Column(String(50))
    status = Column(String(20), default="pending")
    
    expected_impact = Column(JSON)
    required_resources = Column(JSON)
    timeline = Column(JSON)
    
    explanation = Column(Text)
    supporting_data = Column(JSON)
    
    related_entities = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    
    created_by_model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RecommendationFeedback(Base):
    """Feedback on recommendations for model improvement."""
    
    __tablename__ = "recommendation_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    rating = Column(Integer)
    implemented = Column(Boolean)
    outcome = Column(Text)
    
    comments = Column(Text)
    metadata = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)

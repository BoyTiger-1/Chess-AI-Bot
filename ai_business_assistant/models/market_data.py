"""
Market data and sentiment models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON

from ai_business_assistant.shared.database import Base


class MarketData(Base):
    """Market data model for storing market trends and metrics."""
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    symbol = Column(String(20), index=True)
    market = Column(String(50), index=True)
    sector = Column(String(100))
    
    price = Column(Float)
    volume = Column(Float)
    market_cap = Column(Float)
    
    change_percent = Column(Float)
    volatility = Column(Float)
    
    indicators = Column(JSON)
    metadata = Column(JSON)
    
    timestamp = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketSentiment(Base):
    """Market sentiment analysis results."""
    
    __tablename__ = "market_sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    symbol = Column(String(20), index=True)
    source = Column(String(100))
    
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    confidence = Column(Float)
    
    text = Column(Text)
    entities = Column(JSON)
    keywords = Column(JSON)
    
    timestamp = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

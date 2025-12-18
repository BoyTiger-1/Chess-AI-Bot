"""
Forecasting models for financial predictions.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON

from ai_business_assistant.shared.database import Base


class Forecast(Base):
    """Financial forecast model."""
    
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    metric = Column(String(100), nullable=False)
    model_type = Column(String(50))
    
    forecast_start = Column(DateTime)
    forecast_end = Column(DateTime)
    
    predictions = Column(JSON)
    confidence_intervals = Column(JSON)
    
    accuracy_metrics = Column(JSON)
    model_params = Column(JSON)
    
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ForecastScenario(Base):
    """Scenario-based forecast variations."""
    
    __tablename__ = "forecast_scenarios"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, ForeignKey("forecasts.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    scenario_type = Column(String(50))
    
    assumptions = Column(JSON)
    predictions = Column(JSON)
    probability = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

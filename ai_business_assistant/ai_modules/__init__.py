"""AI Intelligence Modules for Business Assistant.

This package provides core AI capabilities including:
- Real-time market analysis (sentiment, trends, volatility)
- Financial forecasting (ARIMA, Prophet, regression)
- Competitive intelligence
- Customer behavior modeling
- Strategic recommendation engine
"""

from .market_analysis import MarketAnalysisModule
from .forecasting import ForecastingModule
from .competitive_intelligence import CompetitiveIntelligenceModule
from .customer_behavior import CustomerBehaviorModule
from .recommendation_engine import RecommendationEngine

__all__ = [
    "MarketAnalysisModule",
    "ForecastingModule",
    "CompetitiveIntelligenceModule",
    "CustomerBehaviorModule",
    "RecommendationEngine",
]

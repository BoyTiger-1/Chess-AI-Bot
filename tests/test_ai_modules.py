"""
Unit tests for AI modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from ai_business_assistant.ai_modules.market_analysis import MarketAnalysisModule
from ai_business_assistant.ai_modules.forecasting import FinancialForecastingModule
from ai_business_assistant.ai_modules.competitive_intelligence import CompetitiveIntelligenceModule
from ai_business_assistant.ai_modules.customer_behavior import CustomerBehaviorModule


def test_market_sentiment_analysis():
    """Test sentiment analysis."""
    module = MarketAnalysisModule(use_transformer=False)
    
    text = "The market is showing strong positive momentum."
    result = module.analyze_sentiment(text)
    
    assert result.polarity >= -1 and result.polarity <= 1
    assert result.subjectivity >= 0 and result.subjectivity <= 1
    assert result.label in ["positive", "negative", "neutral"]


def test_trend_detection():
    """Test trend detection."""
    module = MarketAnalysisModule()
    
    prices = np.cumsum(np.random.randn(100)) + 100
    dates = pd.date_range(end=datetime.now(), periods=100).tolist()
    
    result = module.detect_trend(prices, dates=dates)
    
    assert result.trend_direction in ["upward", "downward", "sideways"]
    assert result.confidence >= 0 and result.confidence <= 1
    assert isinstance(result.change_points, list)


def test_volatility_modeling():
    """Test volatility calculation."""
    module = MarketAnalysisModule(volatility_window=20)
    
    prices = np.random.randn(100).cumsum() + 100
    
    result = module.model_volatility(prices)
    
    assert result.current_volatility >= 0
    assert result.historical_volatility >= 0
    assert result.risk_level in ["low", "medium", "high"]


def test_financial_forecasting_arima():
    """Test ARIMA forecasting."""
    module = FinancialForecastingModule()
    
    data = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'y': np.random.randn(100).cumsum() + 100
    })
    
    result = module.forecast_arima(data, periods=10)
    
    assert len(result.forecast) == 10
    assert len(result.confidence_intervals) == 10
    assert result.model_type == "arima"


def test_competitive_analysis():
    """Test competitive positioning analysis."""
    module = CompetitiveIntelligenceModule()
    
    competitors_data = pd.DataFrame({
        'name': ['Competitor A', 'Competitor B', 'Competitor C'],
        'market_share': [0.30, 0.25, 0.20],
        'price_index': [1.2, 1.0, 0.8],
        'quality_score': [8.5, 7.0, 6.5]
    })
    
    result = module.analyze_competitive_positioning(competitors_data)
    
    assert 'positioning_matrix' in result
    assert len(result['competitor_profiles']) == 3


def test_customer_segmentation():
    """Test customer segmentation."""
    module = CustomerBehaviorModule()
    
    customer_data = pd.DataFrame({
        'customer_id': range(100),
        'total_spend': np.random.uniform(100, 1000, 100),
        'frequency': np.random.randint(1, 20, 100),
        'recency_days': np.random.randint(1, 365, 100)
    })
    
    result = module.segment_customers(customer_data, n_segments=3)
    
    assert len(result.segments) == 3
    assert result.silhouette_score >= -1 and result.silhouette_score <= 1


def test_churn_prediction():
    """Test churn prediction."""
    module = CustomerBehaviorModule()
    
    customer_features = pd.DataFrame({
        'recency_days': [10, 30, 90, 180, 365],
        'frequency': [20, 15, 5, 2, 1],
        'total_spend': [1000, 800, 300, 100, 50],
        'avg_order_value': [50, 53, 60, 50, 50],
        'engagement_score': [0.9, 0.7, 0.4, 0.2, 0.1]
    })
    
    result = module.predict_churn(customer_features)
    
    assert len(result.churn_probabilities) == 5
    assert all(0 <= p <= 1 for p in result.churn_probabilities)

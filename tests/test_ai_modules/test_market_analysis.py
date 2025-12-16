"""Tests for Market Analysis Module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_business_assistant.ai_modules.market_analysis import (
    MarketAnalysisModule,
    SentimentResult,
    TrendResult,
    VolatilityResult,
)


class TestMarketAnalysisModule:
    """Test suite for MarketAnalysisModule."""
    
    @pytest.fixture
    def market_module(self):
        """Create market analysis module instance."""
        return MarketAnalysisModule(use_transformer=False)
    
    def test_analyze_sentiment_basic(self, market_module):
        """Test basic sentiment analysis."""
        texts = [
            "This is amazing and wonderful!",
            "Terrible experience, very disappointing.",
            "Just okay, nothing special.",
        ]
        
        results = market_module.analyze_sentiment(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].polarity > 0
        assert results[1].polarity < 0
    
    def test_aggregate_sentiment(self, market_module):
        """Test sentiment aggregation."""
        texts = ["Great!" for _ in range(7)] + ["Bad!" for _ in range(3)]
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10)]
        
        results = market_module.analyze_sentiment(texts, timestamps)
        aggregated = market_module.aggregate_sentiment(results)
        
        assert "mean_polarity" in aggregated
        assert "positive_ratio" in aggregated
        assert aggregated["volume"] == 10
        assert aggregated["positive_ratio"] > aggregated["negative_ratio"]
    
    def test_detect_trends(self, market_module):
        """Test trend detection."""
        upward_series = pd.Series(np.arange(100) + np.random.randn(100) * 5)
        
        trend = market_module.detect_trends(upward_series)
        
        assert isinstance(trend, TrendResult)
        assert trend.trend_direction in ["up", "down", "sideways"]
        assert 0 <= trend.confidence <= 1
    
    def test_calculate_volatility(self, market_module):
        """Test volatility calculation."""
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
        
        volatility = market_module.calculate_volatility(prices, returns=False)
        
        assert isinstance(volatility, VolatilityResult)
        assert volatility.current_volatility > 0
        assert volatility.risk_level in ["low", "medium", "high"]
        assert 0 <= volatility.confidence <= 1
    
    def test_empty_sentiment_analysis(self, market_module):
        """Test sentiment analysis with empty input."""
        results = market_module.analyze_sentiment([])
        assert len(results) == 0
    
    def test_aggregate_empty_sentiment(self, market_module):
        """Test aggregation with empty results."""
        aggregated = market_module.aggregate_sentiment([])
        assert aggregated["volume"] == 0
        assert aggregated["mean_polarity"] == 0.0

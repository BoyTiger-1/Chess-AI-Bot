"""Tests for Forecasting Module."""

import pytest
import pandas as pd
import numpy as np

from ai_business_assistant.ai_modules.forecasting import (
    ForecastingModule,
    ForecastResult,
    ScenarioResult,
)


class TestForecastingModule:
    """Test suite for ForecastingModule."""
    
    @pytest.fixture
    def forecast_module(self):
        """Create forecasting module instance."""
        return ForecastingModule()
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        values = 100 + np.cumsum(np.random.randn(100) * 2)
        return pd.Series(values, index=dates)
    
    def test_forecast_arima(self, forecast_module, sample_time_series):
        """Test ARIMA forecasting."""
        result = forecast_module.forecast_arima(sample_time_series, periods=10)
        
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 10
        assert len(result.lower_bound) == 10
        assert len(result.upper_bound) == 10
        assert 0 <= result.confidence <= 1
        assert "ARIMA" in result.model_type or "Fallback" in result.model_type
    
    def test_forecast_prophet(self, forecast_module, sample_time_series):
        """Test Prophet forecasting."""
        result = forecast_module.forecast_prophet(sample_time_series, periods=10)
        
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 10
        assert result.model_type == "Prophet" or "Fallback" in result.model_type
        assert "rmse" in result.metrics
    
    def test_forecast_regression(self, forecast_module):
        """Test regression-based forecasting."""
        target = pd.Series(np.random.randn(100) * 10 + 50)
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        future_features = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10),
        })
        
        result = forecast_module.forecast_regression(
            target,
            features,
            future_features,
            model_type="linear",
        )
        
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 10
        assert "Regression" in result.model_type
    
    def test_scenario_modeling(self, forecast_module, sample_time_series):
        """Test scenario modeling."""
        scenarios = {
            "optimistic": {"growth_rate": 0.1, "volatility": 0.05, "probability": 0.3},
            "base": {"growth_rate": 0.0, "volatility": 0.1, "probability": 0.5},
            "pessimistic": {"growth_rate": -0.1, "volatility": 0.15, "probability": 0.2},
        }
        
        results = forecast_module.scenario_modeling(sample_time_series, periods=10, scenarios=scenarios)
        
        assert len(results) == 3
        assert all(isinstance(r, ScenarioResult) for r in results)
        assert all(len(r.forecast) == 10 for r in results)
    
    def test_max_forecast_horizon(self, forecast_module, sample_time_series):
        """Test that max forecast horizon is respected."""
        result = forecast_module.forecast_arima(sample_time_series, periods=200)
        assert len(result.forecast) == forecast_module.max_forecast_horizon

"""
Time Series Analysis Engine
Provides comprehensive time series decomposition, forecasting, and advanced modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

logger = logging.getLogger(__name__)


class TimeSeriesMethod(Enum):
    """Time series analysis methods."""
    DECOMPOSITION = "decomposition"
    FORECASTING = "forecasting"
    STATIONARITY = "stationarity"
    SEASONALITY = "seasonality"
    TREND_ANALYSIS = "trend_analysis"
    CAUSALITY = "causality"
    COINTEGRATION = "cointegration"
    CHANGEPOINT = "change_point"
    REGIME_SWITCHING = "regime_switching"


@dataclass
class TimeSeriesResult:
    """Result of time series analysis."""
    method: TimeSeriesMethod
    is_stationary: Optional[bool] = None
    trend_component: Optional[np.ndarray] = None
    seasonal_component: Optional[np.ndarray] = None
    residual_component: Optional[np.ndarray] = None
    forecast: Optional[np.ndarray] = None
    confidence_interval: Optional[np.ndarray] = None
    model_summary: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class TimeSeriesAnalyzer:
    """
    Advanced time series analysis engine.
    Provides decomposition, forecasting, stationarity testing, and causal analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize time series analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def decompose_series(self, 
                        series: pd.Series,
                        method: str = "stl",
                        seasonal_periods: int = 12) -> TimeSeriesResult:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            series: Time series data
            method: "additive", "multiplicative", "stl"
            seasonal_periods: Number of periods in seasonal cycle
        """
        series_clean = series.dropna()
        
        if method == "stl":
            # STL decomposition (Seasonal and Trend decomposition using Loess)
            stl = STL(series_clean, seasonal=seasonal_periods)
            decomposition = stl.fit()
            
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
        else:
            # Classical decomposition
            decomposition = seasonal_decompose(
                series_clean, 
                model=method,
                period=seasonal_periods
            )
            
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
        
        # Calculate statistics
        trend_strength = self._calculate_component_strength(series_clean, trend, residual)
        seasonal_strength = self._calculate_component_strength(series_clean, seasonal, residual)
        
        # Determine if series is stationary
        is_stationary = self._test_stationarity(residual.dropna())
        
        # Generate recommendations
        recommendations = self._generate_decomposition_recommendations(
            trend_strength, seasonal_strength, is_stationary
        )
        
        return TimeSeriesResult(
            method=TimeSeriesMethod.DECOMPOSITION,
            trend_component=trend.values,
            seasonal_component=seasonal.values,
            residual_component=residual.values,
            is_stationary=is_stationary,
            statistics={
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "original_mean": series_clean.mean(),
                "original_std": series_clean.std(),
                "trend_mean": trend.mean(),
                "seasonal_mean": seasonal.mean(),
                "residual_std": residual.std()
            },
            recommendations=recommendations,
            model_summary={
                "method": method,
                "seasonal_periods": seasonal_periods,
                "decomposition_type": "stl" if method == "stl" else "classical",
                "sample_size": len(series_clean)
            }
        )
    
    def forecast_series(self, 
                       series: pd.Series,
                       model_type: str = "auto_arima",
                       forecast_horizon: int = 12,
                       seasonal_periods: int = 12,
                       seasonal: bool = True) -> TimeSeriesResult:
        """
        Forecast time series using various models.
        
        Args:
            series: Time series data
            model_type: "auto_arima", "sarimax", "exponential_smoothing", "var", "neural_prophet"
            forecast_horizon: Number of periods to forecast
            seasonal_periods: Seasonal period for SARIMA/ETS
            seasonal: Whether to include seasonal components
        """
        series_clean = series.dropna()
        
        if model_type == "auto_arima":
            result = self._auto_arima_forecast(series_clean, forecast_horizon, seasonal_periods, seasonal)
        elif model_type == "sarimax":
            result = self._sarimax_forecast(series_clean, forecast_horizon, seasonal_periods, seasonal)
        elif model_type == "exponential_smoothing":
            result = self._exponential_smoothing_forecast(series_clean, forecast_horizon, seasonal_periods)
        elif model_type == "var":
            result = self._var_forecast(series_clean, forecast_horizon)
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        return result
    
    def _auto_arima_forecast(self, 
                           series: pd.Series, 
                           horizon: int, 
                           seasonal_periods: int,
                           seasonal: bool) -> TimeSeriesResult:
        """Auto ARIMA forecasting."""
        try:
            from pmdarima import auto_arima
            
            # Auto ARIMA to find best parameters
            model = auto_arima(
                series,
                seasonal=seasonal,
                m=seasonal_periods if seasonal else 1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            # Generate forecast
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
            
            # Calculate model statistics
            aic = model.aic()
            bic = model.bic()
            
            # Generate future timestamps
            last_date = series.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(series.index)
                if freq is None:
                    freq = 'D'  # Default to daily
                future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), 
                                           periods=horizon, freq=freq)
            else:
                future_dates = range(len(series), len(series) + horizon)
            
            return TimeSeriesResult(
                method=TimeSeriesMethod.FORECASTING,
                forecast=forecast.values,
                confidence_interval=conf_int,
                model_summary={
                    "model_type": "auto_arima",
                    "order": model.order,
                    "seasonal_order": model.seasonal_order,
                    "aic": aic,
                    "bic": bic,
                    "forecast_horizon": horizon
                },
                statistics={
                    "mean_absolute_error": None,  # Would need fitted values
                    "root_mean_square_error": None,
                    "mean_absolute_percentage_error": None
                }
            )
            
        except ImportError:
            logger.warning("pmdarima not available, falling back to manual ARIMA")
            return self._sarimax_forecast(series, horizon, seasonal_periods, seasonal)
    
    def _sarimax_forecast(self, 
                        series: pd.Series, 
                        horizon: int, 
                        seasonal_periods: int,
                        seasonal: bool) -> TimeSeriesResult:
        """SARIMA forecasting."""
        try:
            # Simple ARIMA model (can be enhanced with auto-selection)
            if seasonal:
                model = SARIMAX(series, 
                              order=(1, 1, 1),
                              seasonal_order=(1, 1, 1, seasonal_periods))
            else:
                model = ARIMA(series, order=(1, 1, 1))
            
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            return TimeSeriesResult(
                method=TimeSeriesMethod.FORECASTING,
                forecast=forecast.values,
                confidence_interval=conf_int.values,
                model_summary={
                    "model_type": "sarimax" if seasonal else "arima",
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "log_likelihood": fitted_model.llf,
                    "forecast_horizon": horizon
                },
                statistics={
                    "residuals_std": np.std(fitted_model.resid),
                    "ljung_box_pvalue": self._ljung_box_test(fitted_model.resid)
                }
            )
            
        except Exception as e:
            logger.error(f"SARIMA forecasting failed: {e}")
            # Fallback to simple exponential smoothing
            return self._exponential_smoothing_forecast(series, horizon, seasonal_periods)
    
    def _exponential_smoothing_forecast(self, 
                                      series: pd.Series, 
                                      horizon: int, 
                                      seasonal_periods: int) -> TimeSeriesResult:
        """Exponential Smoothing forecasting."""
        try:
            # Try Holt-Winters with trend and seasonality
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            )
            
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            
            # Calculate confidence intervals (approximation)
            residuals = fitted_model.resid
            residual_std = np.std(residuals)
            
            # Simple confidence interval calculation
            t_critical = stats.t.ppf(1 - self.alpha/2, len(residuals))
            conf_int_lower = forecast - t_critical * residual_std
            conf_int_upper = forecast + t_critical * residual_std
            conf_int = np.column_stack([conf_int_lower, conf_int_upper])
            
            return TimeSeriesResult(
                method=TimeSeriesMethod.FORECASTING,
                forecast=forecast.values,
                confidence_interval=conf_int,
                model_summary={
                    "model_type": "exponential_smoothing",
                    "smoothing_level": fitted_model.params['smoothing_level'],
                    "smoothing_trend": fitted_model.params['smoothing_trend'],
                    "smoothing_seasonal": fitted_model.params['smoothing_seasonal'],
                    "aic": fitted_model.aic,
                    "forecast_horizon": horizon
                },
                statistics={
                    "sse": fitted_model.sse,
                    "residuals_std": residual_std
                }
            )
            
        except Exception as e:
            logger.error(f"Exponential smoothing failed: {e}")
            # Simple trend forecast as last resort
            return self._simple_trend_forecast(series, horizon)
    
    def _var_forecast(self, series: pd.Series, horizon: int) -> TimeSeriesResult:
        """Vector Autoregression forecasting."""
        try:
            # For multivariate series, select optimal lag
            if isinstance(series, pd.DataFrame):
                data = series
            else:
                # Convert to DataFrame with lag features
                df = pd.DataFrame(series, columns=['y'])
                for lag in range(1, 4):
                    df[f'y_lag_{lag}'] = df['y'].shift(lag)
                df = df.dropna()
                data = df
            
            # Find optimal lag order
            max_lags = min(len(data) // 4, 10)
            model = VAR(data)
            lag_order = model.select_order(maxlags=max_lags)
            selected_lag = lag_order.aic
            
            # Fit VAR model
            var_model = VAR(data)
            fitted_model = var_model.fit(maxlags=selected_lag, ic='aic')
            
            # Generate forecast
            forecast = fitted_model.forecast(data.values[-selected_lag:], steps=horizon)
            
            return TimeSeriesResult(
                method=TimeSeriesMethod.FORECASTING,
                forecast=forecast[:, 0] if forecast.ndim > 1 else forecast,
                model_summary={
                    "model_type": "vector_autoregression",
                    "lag_order": selected_lag,
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "forecast_horizon": horizon
                },
                statistics={
                    "log_likelihood": fitted_model.llf,
                    "residuals_corr": np.corrcoef(fitted_model.resid.T)
                }
            )
            
        except Exception as e:
            logger.error(f"VAR forecasting failed: {e}")
            return self._simple_trend_forecast(series, horizon)
    
    def _simple_trend_forecast(self, series: pd.Series, horizon: int) -> TimeSeriesResult:
        """Simple linear trend forecasting."""
        from sklearn.linear_model import LinearRegression
        
        y = series.values
        X = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future values
        future_X = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Simple confidence interval
        residual_std = np.std(y - model.predict(X))
        t_critical = stats.t.ppf(1 - self.alpha/2, len(y) - 2)
        conf_int_lower = forecast - t_critical * residual_std
        conf_int_upper = forecast + t_critical * residual_std
        conf_int = np.column_stack([conf_int_lower, conf_int_upper])
        
        return TimeSeriesResult(
            method=TimeSeriesMethod.FORECASTING,
            forecast=forecast,
            confidence_interval=conf_int,
            model_summary={
                "model_type": "linear_trend",
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r_squared": model.score(X, y),
                "forecast_horizon": horizon
            },
            statistics={
                "residual_std": residual_std,
                "trend_strength": abs(model.coef_[0]) / np.std(y)
            }
        )
    
    def test_stationarity(self, series: pd.Series, method: str = "adf") -> TimeSeriesResult:
        """
        Test for stationarity using various tests.
        
        Args:
            series: Time series data
            method: "adf", "kpss", "pp", "chi2"
        """
        series_clean = series.dropna()
        
        if method == "adf":
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series_clean, autolag='AIC')
            statistic, p_value, used_lags, n_obs, critical_values, icbest = adf_result
            
            is_stationary = p_value < self.alpha
            test_statistic = statistic
            critical_value = critical_values[f'{int((1-self.alpha)*100)}%']
            
        elif method == "kpss":
            # KPSS test
            kpss_result = kpss(series_clean, regression='ct')
            statistic, p_value, lags, critical_values = kpss_result
            
            is_stationary = p_value > self.alpha  # KPSS null hypothesis is stationarity
            test_statistic = statistic
            critical_value = critical_values[f'{int((1-self.alpha)*100)}%']
            
        else:
            raise ValueError(f"Stationarity test method '{method}' not supported")
        
        # Determine differencing order needed
        if not is_stationary:
            # Test first difference
            diff_series = series_clean.diff().dropna()
            if method == "adf":
                diff_result = adfuller(diff_series, autolag='AIC')
                diff_stationary = diff_result[1] < self.alpha
            else:
                diff_result = kpss(diff_series, regression='ct')
                diff_stationary = diff_result[1] > self.alpha
            
            differencing_order = 1 if diff_stationary else 2
        else:
            differencing_order = 0
        
        # Generate recommendations
        recommendations = []
        if not is_stationary:
            recommendations.append(f"Series appears non-stationary. Consider differencing (d={differencing_order})")
            recommendations.append("Apply transformations like log or Box-Cox to stabilize variance")
            recommendations.append("Consider seasonal differencing if strong seasonality is present")
        
        return TimeSeriesResult(
            method=TimeSeriesMethod.STATIONARITY,
            is_stationary=is_stationary,
            model_summary={
                "test_method": method,
                "test_statistic": test_statistic,
                "p_value": p_value if method == "adf" else p_value,
                "critical_value": critical_value,
                "used_lags": used_lags if method == "adf" else lags,
                "differencing_order": differencing_order,
                "n_observations": n_obs if method == "adf" else len(series_clean)
            },
            statistics={
                "is_stationary": is_stationary,
                "significance_level": self.alpha
            },
            recommendations=recommendations
        )
    
    def detect_seasonality(self, series: pd.Series) -> TimeSeriesResult:
        """
        Detect and analyze seasonality patterns.
        
        Args:
            series: Time series data
        """
        series_clean = series.dropna()
        n = len(series_clean)
        
        # FFT-based seasonality detection
        fft_vals = np.fft.fft(series_clean)
        fft_freq = np.fft.fftfreq(n)
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Identify seasonal periods
        max_periods = min(n // 2, 365)  # Don't look for periods longer than half the series
        significant_periods = []
        
        for period in range(2, max_periods + 1):
            frequency = 1 / period
            freq_idx = np.argmin(np.abs(fft_freq - frequency))
            
            if power_spectrum[freq_idx] > np.mean(power_spectrum) * 2:
                significance = power_spectrum[freq_idx] / np.mean(power_spectrum)
                significant_periods.append((period, significance))
        
        # Sort by significance
        significant_periods.sort(key=lambda x: x[1], reverse=True)
        
        # Test for autocorrelation at seasonal lags
        seasonal_lags = [12, 24, 48]  # Common seasonal lags
        seasonal_autocorr = {}
        
        for lag in seasonal_lags:
            if lag < n:
                acf_val, confint = acf(series_clean, nlags=lag, alpha=self.alpha)
                if abs(acf_val[lag]) > abs(confint[lag][0] - acf_val[lag]):
                    seasonal_autocorr[lag] = acf_val[lag]
        
        # Determine seasonality strength
        trend_strength = self._calculate_component_strength(series_clean, series_clean.rolling(window=12).mean().fillna(method='bfill'), series_clean - series_clean.rolling(window=12).mean().fillna(method='bfill'))
        
        seasonal_strength = 0
        if significant_periods:
            seasonal_strength = max(significance for _, significance in significant_periods)
        
        # Generate recommendations
        recommendations = []
        if significant_periods:
            recommendations.append(f"Strong seasonality detected with periods: {[p for p, s in significant_periods[:3]]}")
            recommendations.append("Consider SARIMA models with seasonal components")
            recommendations.append("Use seasonal decomposition before forecasting")
        else:
            recommendations.append("No strong seasonality detected")
            recommendations.append("Simple ARIMA models should be sufficient")
        
        return TimeSeriesResult(
            method=TimeSeriesMethod.SEASONALITY,
            model_summary={
                "dominant_periods": significant_periods[:10],
                "seasonal_autocorrelation": seasonal_autocorr,
                "seasonal_strength": seasonal_strength
            },
            statistics={
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "n_significant_periods": len(significant_periods)
            },
            recommendations=recommendations
        )
    
    def detect_trend(self, series: pd.Series) -> TimeSeriesResult:
        """
        Analyze trend components.
        
        Args:
            series: Time series data
        """
        series_clean = series.dropna()
        n = len(series_clean)
        
        # Mann-Kendall test for trend
        mk_result = self._mann_kendall_test(series_clean.values)
        
        # Linear trend estimation
        from sklearn.linear_model import LinearRegression
        
        X = np.arange(n).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, series_clean.values)
        
        trend_slope = lr.coef_[0]
        trend_intercept = lr.intercept_
        trend_strength = abs(trend_slope) / np.std(series_clean.values)
        
        # Theil-Sen robust trend estimation
        trend_slope_ts = self._theil_sen_estimator(series_clean.values)
        
        # Trend change point detection
        change_points = self._detect_trend_changes(series_clean.values)
        
        # Generate trend interpretation
        trend_interpretation = self._interpret_trend(trend_slope, mk_result['p_value'], trend_strength)
        
        recommendations = self._generate_trend_recommendations(trend_slope, mk_result['p_value'], trend_strength, change_points)
        
        return TimeSeriesResult(
            method=TimeSeriesMethod.TREND_ANALYSIS,
            trend_component=lr.predict(X),
            model_summary={
                "linear_slope": trend_slope,
                "linear_intercept": trend_intercept,
                "theil_sen_slope": trend_slope_ts,
                "r_squared": lr.score(X, series_clean.values),
                "change_points": change_points,
                "trend_direction": "increasing" if trend_slope > 0 else "decreasing"
            },
            statistics={
                "mann_kendall_tau": mk_result['tau'],
                "mann_kendall_pvalue": mk_result['p_value'],
                "trend_strength": trend_strength,
                "average_change_rate": trend_slope,
                "relative_trend_strength": trend_slope / np.mean(series_clean.values)
            },
            recommendations=recommendations
        )
    
    def _calculate_component_strength(self, 
                                    original: pd.Series,
                                    component: pd.Series,
                                    residual: pd.Series) -> float:
        """Calculate the strength of a component in decomposition."""
        component_var = np.var(component.dropna())
        residual_var = np.var(residual.dropna())
        
        if component_var + residual_var == 0:
            return 0.0
        
        strength = component_var / (component_var + residual_var)
        return min(strength, 1.0)
    
    def _test_stationarity(self, series: pd.Series) -> bool:
        """Simple stationarity test."""
        if len(series) < 10:
            return True
        
        try:
            adf_result = adfuller(series, autolag='AIC')
            return adf_result[1] < self.alpha
        except:
            return True
    
    def _generate_decomposition_recommendations(self, trend_strength: float, seasonal_strength: float, is_stationary: bool) -> List[str]:
        """Generate recommendations based on decomposition results."""
        recommendations = []
        
        if trend_strength > 0.3:
            recommendations.append("Strong trend detected - consider trend-aware models")
        
        if seasonal_strength > 0.3:
            recommendations.append("Strong seasonality detected - use seasonal decomposition or SARIMA")
        
        if not is_stationary:
            recommendations.append("Non-stationary residuals - consider differencing")
        
        return recommendations
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """Perform Ljung-Box test for autocorrelation."""
        try:
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return result['lb_pvalue'].iloc[-1]
        except:
            return 1.0
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, float]:
        """Mann-Kendall test for monotonic trend."""
        n = len(data)
        
        # Calculate S statistic
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        # Calculate Kendall's tau
        tau = S / (0.5 * n * (n - 1))
        
        return {
            'S': S,
            'Z': Z,
            'p_value': p_value,
            'tau': tau
        }
    
    def _theil_sen_estimator(self, data: np.ndarray) -> float:
        """Calculate Theil-Sen robust trend estimator."""
        n = len(data)
        slopes = []
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i != j:
                    slope = (data[j] - data[i]) / (j - i)
                    slopes.append(slope)
        
        return np.median(slopes)
    
    def _detect_trend_changes(self, data: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detect change points in trend using cumulative sum."""
        # Center the data
        centered_data = data - np.mean(data)
        
        # Calculate cumulative sum
        cusum = np.cumsum(centered_data)
        
        # Detect change points where cumulative sum deviates significantly
        cusum_std = np.std(cusum)
        change_points = []
        
        for i in range(1, len(cusum)):
            if abs(cusum[i] - cusum[i-1]) > threshold * cusum_std:
                change_points.append(i)
        
        return change_points
    
    def _interpret_trend(self, slope: float, p_value: float, strength: float) -> str:
        """Interpret trend analysis results."""
        if p_value > 0.05:
            return "No significant trend detected"
        
        direction = "increasing" if slope > 0 else "decreasing"
        strength_desc = "strong" if strength > 0.5 else "moderate" if strength > 0.2 else "weak"
        
        return f"{strength_desc.title()} {direction} trend"
    
    def _generate_trend_recommendations(self, slope: float, p_value: float, strength: float, change_points: List[int]) -> List[str]:
        """Generate trend-related recommendations."""
        recommendations = []
        
        if p_value > 0.05:
            recommendations.append("No significant trend detected - series may be stationary")
        elif slope > 0:
            recommendations.append("Upward trend detected - consider incorporating trend in models")
        else:
            recommendations.append("Downward trend detected - investigate potential causes")
        
        if strength > 0.5:
            recommendations.append("Strong trend - consider trend-robust forecasting methods")
        
        if len(change_points) > 2:
            recommendations.append("Multiple trend changes detected - check for regime changes or external factors")
        
        return recommendations
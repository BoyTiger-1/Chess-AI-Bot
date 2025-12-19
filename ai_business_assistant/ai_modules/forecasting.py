"""Financial Forecasting Module.

Provides time-series models (ARIMA, Prophet), regression models, and scenario modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


@dataclass
class ForecastResult:
    """Result of a forecasting operation."""
    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence: float
    model_type: str
    metrics: Dict[str, float]


@dataclass
class ScenarioResult:
    """Result of scenario modeling."""
    scenario_name: str
    forecast: pd.Series
    probability: float
    assumptions: Dict[str, Any]


class ForecastingModule:
    """Financial forecasting with multiple time-series and regression models.
    
    Assumptions:
    - Time series data is regularly sampled
    - Missing values are handled via interpolation
    - Seasonality is detected automatically (Prophet)
    - ARIMA assumes stationarity or can difference to achieve it
    
    Limitations:
    - Prophet requires at least 2 periods of historical data
    - ARIMA is sensitive to outliers
    - Regression models assume feature-target relationships are stable
    - Long-term forecasts have decreasing accuracy
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        max_forecast_horizon: int = 90,
    ):
        """Initialize the forecasting module.
        
        Args:
            confidence_level: Confidence level for prediction intervals
            max_forecast_horizon: Maximum number of periods to forecast
        """
        self.confidence_level = confidence_level
        self.max_forecast_horizon = max_forecast_horizon
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, StandardScaler] = {}
    
    def forecast_arima(
        self,
        time_series: pd.Series,
        periods: int,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ) -> ForecastResult:
        """Forecast using ARIMA or SARIMA model.
        
        Args:
            time_series: Historical time series data
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q). If None, will auto-select
            seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
            
        Returns:
            ForecastResult object
        """
        if periods > self.max_forecast_horizon:
            periods = self.max_forecast_horizon
        
        ts_clean = time_series.dropna()
        
        if order is None:
            order = self._auto_select_arima_order(ts_clean)
        
        try:
            if seasonal_order:
                model = SARIMAX(
                    ts_clean,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted_model = model.fit(disp=False)
            else:
                model = ARIMA(ts_clean, order=order)
                fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=periods)
            forecast_obj = fitted_model.get_forecast(steps=periods)
            conf_int = forecast_obj.conf_int(alpha=1 - self.confidence_level)
            
            forecast_index = pd.date_range(
                start=ts_clean.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=ts_clean.index.inferred_freq or "D",
            )
            
            forecast_series = pd.Series(forecast.values, index=forecast_index)
            lower = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index)
            upper = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index)
            
            metrics = {
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "rmse": float(np.sqrt(np.mean(fitted_model.resid ** 2))),
            }
            
            confidence = self._calculate_forecast_confidence(
                fitted_model.resid,
                periods,
            )
            
            return ForecastResult(
                forecast=forecast_series,
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
                model_type="SARIMA" if seasonal_order else "ARIMA",
                metrics=metrics,
            )
        
        except Exception as e:
            return self._fallback_forecast(ts_clean, periods, f"ARIMA failed: {e}")
    
    def forecast_prophet(
        self,
        time_series: pd.Series,
        periods: int,
        seasonality_mode: str = "multiplicative",
        include_holidays: bool = False,
    ) -> ForecastResult:
        """Forecast using Facebook Prophet.
        
        Args:
            time_series: Historical time series data
            periods: Number of periods to forecast
            seasonality_mode: 'additive' or 'multiplicative'
            include_holidays: Whether to include holiday effects
            
        Returns:
            ForecastResult object
        """
        if periods > self.max_forecast_horizon:
            periods = self.max_forecast_horizon
        
        df = pd.DataFrame({
            "ds": time_series.index,
            "y": time_series.values,
        }).dropna()
        
        try:
            try:
                from prophet import Prophet  # type: ignore
            except Exception as e:  # noqa: BLE001
                return self._fallback_forecast(time_series, periods, f"Prophet unavailable: {e}")

            model = Prophet(
                seasonality_mode=seasonality_mode,
                interval_width=self.confidence_level,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
            )

            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods, freq="D")
            forecast = model.predict(future)
            
            forecast_only = forecast.tail(periods)
            
            forecast_series = pd.Series(
                forecast_only["yhat"].values,
                index=pd.to_datetime(forecast_only["ds"]),
            )
            lower = pd.Series(
                forecast_only["yhat_lower"].values,
                index=pd.to_datetime(forecast_only["ds"]),
            )
            upper = pd.Series(
                forecast_only["yhat_upper"].values,
                index=pd.to_datetime(forecast_only["ds"]),
            )
            
            in_sample = forecast.iloc[:-periods]
            residuals = df["y"].values - in_sample["yhat"].values[-len(df):]
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            
            metrics = {
                "rmse": rmse,
                "mae": float(np.mean(np.abs(residuals))),
            }
            
            confidence = self._calculate_forecast_confidence(
                pd.Series(residuals),
                periods,
            )
            
            return ForecastResult(
                forecast=forecast_series,
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
                model_type="Prophet",
                metrics=metrics,
            )
        
        except Exception as e:
            return self._fallback_forecast(time_series, periods, f"Prophet failed: {e}")
    
    def forecast_regression(
        self,
        target: pd.Series,
        features: pd.DataFrame,
        future_features: pd.DataFrame,
        model_type: str = "gradient_boosting",
    ) -> ForecastResult:
        """Forecast using regression models with exogenous features.
        
        Args:
            target: Target variable (historical)
            features: Feature matrix (historical)
            future_features: Feature matrix for future periods
            model_type: 'linear', 'ridge', 'random_forest', or 'gradient_boosting'
            
        Returns:
            ForecastResult object
        """
        X = features.values
        y = target.values
        X_future = future_features.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_future_scaled = scaler.transform(X_future)
        
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_scaled, y)
        
        predictions = model.predict(X_future_scaled)
        
        residuals = y - model.predict(X_scaled)
        residual_std = np.std(residuals)
        
        z_score = 1.96 if self.confidence_level == 0.95 else 2.576
        margin = z_score * residual_std
        
        forecast_series = pd.Series(predictions, index=future_features.index)
        lower = pd.Series(predictions - margin, index=future_features.index)
        upper = pd.Series(predictions + margin, index=future_features.index)
        
        metrics = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "r2": float(model.score(X_scaled, y)),
        }
        
        confidence = self._calculate_forecast_confidence(
            pd.Series(residuals),
            len(predictions),
        )
        
        return ForecastResult(
            forecast=forecast_series,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence,
            model_type=f"Regression_{model_type}",
            metrics=metrics,
        )
    
    def scenario_modeling(
        self,
        time_series: pd.Series,
        periods: int,
        scenarios: Dict[str, Dict[str, Any]],
    ) -> List[ScenarioResult]:
        """Generate forecasts for multiple scenarios.
        
        Args:
            time_series: Historical time series data
            periods: Number of periods to forecast
            scenarios: Dictionary of scenario definitions
            
        Returns:
            List of ScenarioResult objects
        """
        results = []
        
        for scenario_name, params in scenarios.items():
            growth_rate = params.get("growth_rate", 0.0)
            volatility = params.get("volatility", 0.01)
            probability = params.get("probability", 1.0 / len(scenarios))
            
            base_forecast = self.forecast_prophet(time_series, periods)
            
            scenario_values = base_forecast.forecast.values * (1 + growth_rate)
            
            noise = np.random.normal(0, volatility, periods)
            scenario_values = scenario_values * (1 + noise)
            
            scenario_series = pd.Series(
                scenario_values,
                index=base_forecast.forecast.index,
            )
            
            results.append(ScenarioResult(
                scenario_name=scenario_name,
                forecast=scenario_series,
                probability=probability,
                assumptions=params,
            ))
        
        return results
    
    def _auto_select_arima_order(
        self,
        time_series: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
    ) -> Tuple[int, int, int]:
        """Automatically select ARIMA order using AIC."""
        d = self._determine_differencing_order(time_series)
        
        best_aic = np.inf
        best_order = (1, d, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        return best_order
    
    def _determine_differencing_order(self, time_series: pd.Series) -> int:
        """Determine the order of differencing needed for stationarity."""
        result = adfuller(time_series.dropna(), autolag="AIC")
        p_value = result[1]
        
        if p_value < 0.05:
            return 0
        
        diff_series = time_series.diff().dropna()
        result = adfuller(diff_series, autolag="AIC")
        p_value = result[1]
        
        if p_value < 0.05:
            return 1
        
        return 2
    
    def _calculate_forecast_confidence(
        self,
        residuals: pd.Series,
        forecast_horizon: int,
    ) -> float:
        """Calculate confidence score for forecast."""
        residual_std = residuals.std()
        residual_mean = residuals.mean()
        
        if residual_std == 0:
            base_confidence = 0.95
        else:
            base_confidence = max(0.5, 1.0 - abs(residual_mean) / residual_std)
        
        horizon_penalty = np.exp(-forecast_horizon / 30.0)
        
        confidence = base_confidence * horizon_penalty
        
        return float(np.clip(confidence, 0.1, 0.99))
    
    def _fallback_forecast(
        self,
        time_series: pd.Series,
        periods: int,
        reason: str,
    ) -> ForecastResult:
        """Fallback to simple moving average forecast."""
        window = min(30, len(time_series))
        last_values = time_series.tail(window)
        mean_value = last_values.mean()
        std_value = last_values.std()
        
        forecast_index = pd.date_range(
            start=time_series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq="D",
        )
        
        forecast = pd.Series([mean_value] * periods, index=forecast_index)
        lower = pd.Series([mean_value - 2 * std_value] * periods, index=forecast_index)
        upper = pd.Series([mean_value + 2 * std_value] * periods, index=forecast_index)
        
        return ForecastResult(
            forecast=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.3,
            model_type=f"Fallback_MA ({reason})",
            metrics={"rmse": float(std_value)},
        )


@dataclass
class LegacyForecastResult:
    """Legacy wrapper used by the simplified test suite."""

    forecast: List[float]
    confidence_intervals: List[Dict[str, float]]
    model_type: str


class FinancialForecastingModule:
    """Compatibility wrapper around ForecastingModule.

    The repository contains both an "advanced" series-based ForecastingModule and a
    simplified DataFrame-based interface referenced by a legacy test.
    """

    def __init__(self, *, confidence_level: float = 0.95, max_forecast_horizon: int = 90):
        self._inner = ForecastingModule(confidence_level=confidence_level, max_forecast_horizon=max_forecast_horizon)

    def forecast_arima(self, data: pd.DataFrame, *, periods: int = 10) -> LegacyForecastResult:
        if data.empty:
            return LegacyForecastResult(forecast=[], confidence_intervals=[], model_type="arima")

        if not {"ds", "y"}.issubset(set(data.columns)):
            raise ValueError("Expected DataFrame with columns: ds, y")

        df = data.copy()
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds", "y"]).sort_values("ds")

        ts = pd.Series(pd.to_numeric(df["y"], errors="coerce").values, index=pd.to_datetime(df["ds"]))
        result = self._inner.forecast_arima(ts, periods=periods)

        forecast = [float(x) for x in result.forecast.values.tolist()]
        ci = [
            {"lower": float(l), "upper": float(u)}
            for l, u in zip(result.lower_bound.values.tolist(), result.upper_bound.values.tolist())
        ]

        return LegacyForecastResult(forecast=forecast, confidence_intervals=ci, model_type="arima")

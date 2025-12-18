"""
Predictive Analytics Engine
Provides comprehensive predictive modeling including demand forecasting, revenue prediction, price optimization,
Monte Carlo simulations, and scenario planning with advanced machine learning models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ForecastingMethod(Enum):
    """Forecasting methods."""
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"


class PriceOptimization(Enum):
    """Price optimization methods."""
    ELASTICITY_BASED = "elasticity_based"
    COMPETITIVE_PRICING = "competitive_pricing"
    VALUE_BASED = "value_based"
    DYNAMIC_PRICING = "dynamic_pricing"
    PENETRATION_PRICING = "penetration_pricing"


class ScenarioType(Enum):
    """Scenario analysis types."""
    BASE_CASE = "base_case"
    BULL_CASE = "bull_case"
    BEAR_CASE = "bear_case"
    STRESS_TEST = "stress_test"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY = "sensitivity"


@dataclass
class ForecastResult:
    """Forecasting result."""
    method: ForecastingMethod
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    feature_importance: Dict[str, float]
    model_metrics: Dict[str, float]
    forecast_horizon: int
    timestamp: datetime = field(default_factory=datetime.now)
    model_artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceOptimizationResult:
    """Price optimization result."""
    optimal_prices: Dict[str, float]
    price_elasticity: Dict[str, float]
    profit_impact: Dict[str, float]
    competitive_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ScenarioResult:
    """Scenario analysis result."""
    scenario_type: ScenarioType
    outcomes: Dict[str, float]
    probability: float
    assumptions: List[str]
    key_drivers: List[str]
    risk_factors: List[str]
    mitigation_strategies: List[str]


class PredictiveAnalyticsEngine:
    """
    Advanced Predictive Analytics Engine
    Provides comprehensive forecasting, price optimization, and scenario planning capabilities.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize Predictive Analytics Engine."""
        self.confidence_level = confidence_level
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        
    def demand_forecasting(self,
                         historical_data: pd.DataFrame,
                         target_column: str,
                         features: List[str],
                         forecast_horizon: int,
                         method: ForecastingMethod = ForecastingMethod.ENSEMBLE,
                         seasonality_periods: int = 12) -> ForecastResult:
        """
        Advanced demand forecasting with multiple algorithms.
        
        Args:
            historical_data: Historical data with target and features
            target_column: Name of target column
            features: List of feature columns
            forecast_horizon: Number of periods to forecast
            method: Forecasting method
            seasonality_periods: Seasonal periods for time series
        """
        # Prepare data
        X = historical_data[features].fillna(method='ffill')
        y = historical_data[target_column].fillna(method='ffill')
        
        # Handle time series data
        time_features = self._extract_time_features(historical_data.index if hasattr(historical_data.index, 'to_series') else pd.to_datetime(historical_data.index))
        
        if time_features:
            for feature_name, feature_values in time_features.items():
                X[feature_name] = feature_values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['demand_forecasting'] = scaler
        
        # Train models based on method
        if method == ForecastingMethod.ENSEMBLE:
            return self._ensemble_demand_forecast(X_scaled, y, features, forecast_horizon, seasonality_periods)
        elif method == ForecastingMethod.LIGHTGBM:
            return self._lightgbm_forecast(X_scaled, y, features, forecast_horizon, 'demand')
        elif method == ForecastingMethod.XGBOOST:
            return self._xgboost_forecast(X_scaled, y, features, forecast_horizon, 'demand')
        else:
            return self._linear_forecast(X_scaled, y, features, forecast_horizon, 'demand')
    
    def revenue_forecasting(self,
                          historical_data: pd.DataFrame,
                          revenue_column: str,
                          features: List[str],
                          forecast_horizon: int,
                          confidence_intervals: bool = True) -> ForecastResult:
        """
        Revenue forecasting with confidence intervals.
        
        Args:
            historical_data: Historical revenue data
            revenue_column: Revenue column name
            features: Feature columns
            forecast_horizon: Forecast horizon
            confidence_intervals: Whether to calculate confidence intervals
        """
        # Prepare data
        X = historical_data[features].fillna(method='ffill')
        y = historical_data[revenue_column].fillna(method='ffill')
        
        # Add derived features
        X = self._add_revenue_features(X, historical_data)
        
        # Train ensemble model
        result = self._ensemble_revenue_forecast(X, y, features, forecast_horizon)
        
        # Calculate confidence intervals if requested
        if confidence_intervals:
            result.confidence_intervals = self._calculate_confidence_intervals(
                X, y, result.predictions, forecast_horizon
            )
        
        return result
    
    def customer_churn_prediction(self,
                                customer_data: pd.DataFrame,
                                churn_column: str,
                                features: Optional[List[str]] = None,
                                model_ensemble: bool = True) -> Dict[str, Any]:
        """
        Predict customer churn with advanced ensemble methods.
        
        Args:
            customer_data: Customer data
            churn_column: Churn indicator column
            features: Feature columns (auto-selected if None)
            model_ensemble: Whether to use ensemble models
        """
        # Auto-select features if not provided
        if features is None:
            features = [col for col in customer_data.columns if col != churn_column]
        
        # Prepare data
        X = customer_data[features].fillna(0)
        y = customer_data[churn_column]
        
        # Feature engineering
        X = self._engineer_churn_features(X, customer_data)
        
        # Train models
        if model_ensemble:
            return self._ensemble_churn_prediction(X, y, features)
        else:
            return self._single_churn_model(X, y, features)
    
    def customer_lifetime_value_prediction(self,
                                         transactions: pd.DataFrame,
                                         customer_features: pd.DataFrame,
                                         time_horizon: int = 24,
                                         model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Predict Customer Lifetime Value using advanced models.
        
        Args:
            transactions: Transaction history
            customer_features: Customer characteristics
            time_horizon: CLV prediction horizon in months
            model_type: Model type for CLV prediction
        """
        # Calculate CLV features
        clv_features = self._calculate_clv_features(transactions, customer_features, time_horizon)
        
        # Prepare features
        X = clv_features.drop(['customer_id', 'clv'], axis=1, errors='ignore')
        y = clv_features['clv'] if 'clv' in clv_features.columns else None
        
        if y is None:
            # Calculate historical CLV
            y = self._calculate_historical_clv(transactions)
            clv_features['clv'] = y
        
        # Remove rows with missing CLV
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Train CLV model
        if model_type == "ensemble":
            return self._ensemble_clv_prediction(X, y, clv_features['customer_id'][valid_mask])
        else:
            return self._single_clv_model(X, y, clv_features['customer_id'][valid_mask])
    
    def price_optimization(self,
                         price_data: pd.DataFrame,
                         demand_data: pd.DataFrame,
                         competitor_data: pd.DataFrame,
                         optimization_method: PriceOptimization = PriceOptimization.ELASTICITY_BASED) -> PriceOptimizationResult:
        """
        Advanced price optimization using multiple methods.
        
        Args:
            price_data: Historical price data
            demand_data: Demand response data
            competitor_data: Competitor pricing data
            optimization_method: Optimization approach
        """
        # Calculate price elasticity
        elasticity = self._calculate_price_elasticity(price_data, demand_data)
        
        # Competitive analysis
        competitive_analysis = self._analyze_competitive_pricing(competitor_data, price_data)
        
        # Optimal price calculation
        if optimization_method == PriceOptimization.ELASTICITY_BASED:
            optimal_prices = self._elasticity_based_optimization(price_data, demand_data, elasticity)
        elif optimization_method == PriceOptimization.COMPETITIVE_PRICING:
            optimal_prices = self._competitive_optimization(competitor_data, price_data)
        elif optimization_method == PriceOptimization.VALUE_BASED:
            optimal_prices = self._value_based_optimization(demand_data, price_data)
        else:
            optimal_prices = self._dynamic_pricing_optimization(price_data, demand_data, competitor_data)
        
        # Profit impact analysis
        profit_impact = self._analyze_profit_impact(price_data, demand_data, optimal_prices)
        
        # Risk assessment
        risk_assessment = self._assess_pricing_risks(optimal_prices, competitive_analysis)
        
        # Generate recommendations
        recommendations = self._generate_pricing_recommendations(
            optimal_prices, elasticity, competitive_analysis, risk_assessment
        )
        
        return PriceOptimizationResult(
            optimal_prices=optimal_prices,
            price_elasticity=elasticity,
            profit_impact=profit_impact,
            competitive_analysis=competitive_analysis,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
    
    def monte_carlo_simulation(self,
                             base_parameters: Dict[str, Any],
                             distribution_types: Dict[str, str],
                             num_simulations: int = 10000,
                             time_horizon: int = 12) -> Dict[str, Any]:
        """
        Monte Carlo simulation for uncertainty analysis.
        
        Args:
            base_parameters: Base case parameters
            distribution_types: Distribution types for each parameter
            num_simulations: Number of simulation runs
            time_horizon: Simulation time horizon
        """
        simulations = []
        
        for _ in range(num_simulations):
            # Sample from distributions
            sampled_params = {}
            for param, dist_type in distribution_types.items():
                if dist_type == "normal":
                    sampled_params[param] = np.random.normal(
                        base_parameters[param]['mean'],
                        base_parameters[param]['std']
                    )
                elif dist_type == "uniform":
                    sampled_params[param] = np.random.uniform(
                        base_parameters[param]['min'],
                        base_parameters[param]['max']
                    )
                elif dist_type == "triangular":
                    sampled_params[param] = np.random.triangular(
                        base_parameters[param]['min'],
                        base_parameters[param]['mode'],
                        base_parameters[param]['max']
                    )
                elif dist_type == "exponential":
                    sampled_params[param] = np.random.exponential(
                        base_parameters[param]['rate']
                    )
            
            # Run simulation with sampled parameters
            result = self._run_simulation(sampled_params, time_horizon)
            simulations.append(result)
        
        # Analyze results
        simulation_df = pd.DataFrame(simulations)
        
        # Calculate statistics
        statistics = {
            'mean': simulation_df.mean().to_dict(),
            'median': simulation_df.median().to_dict(),
            'std': simulation_df.std().to_dict(),
            'percentile_5': simulation_df.quantile(0.05).to_dict(),
            'percentile_95': simulation_df.quantile(0.95).to_dict(),
            'min': simulation_df.min().to_dict(),
            'max': simulation_df.max().to_dict()
        }
        
        # Risk metrics
        risk_metrics = self._calculate_simulation_risks(simulation_df)
        
        # Correlation analysis
        correlations = simulation_df.corr().to_dict()
        
        return {
            'simulations': simulations,
            'statistics': statistics,
            'risk_metrics': risk_metrics,
            'correlations': correlations,
            'recommendations': self._generate_simulation_recommendations(statistics, risk_metrics)
        }
    
    def scenario_analysis(self,
                        base_case: Dict[str, Any],
                        scenarios: Dict[str, Dict[str, Any]],
                        forecast_model: Optional[Any] = None) -> Dict[str, ScenarioResult]:
        """
        Comprehensive scenario analysis including base, bull, bear cases.
        
        Args:
            base_case: Base case parameters
            scenarios: Scenario definitions
            forecast_model: Optional forecasting model
        """
        scenario_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_type = self._determine_scenario_type(scenario_name)
            
            # Merge base case with scenario
            combined_params = {**base_case, **scenario_params}
            
            # Run scenario analysis
            outcome = self._run_scenario(combined_params, scenario_type)
            
            # Calculate scenario probability
            probability = self._calculate_scenario_probability(scenario_name, scenarios)
            
            # Generate scenario-specific insights
            assumptions = self._identify_scenario_assumptions(scenario_name, scenario_params)
            key_drivers = self._identify_key_drivers(scenario_name, scenario_params)
            risk_factors = self._identify_scenario_risks(scenario_name, scenario_params)
            mitigation_strategies = self._generate_mitigation_strategies(scenario_name, risk_factors)
            
            scenario_results[scenario_name] = ScenarioResult(
                scenario_type=scenario_type,
                outcomes=outcome,
                probability=probability,
                assumptions=assumptions,
                key_drivers=key_drivers,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies
            )
        
        return scenario_results
    
    def sensitivity_analysis(self,
                           model: Any,
                             base_parameters: Dict[str, float],
                           parameter_ranges: Dict[str, Tuple[float, float]],
                           target_variable: str,
                           analysis_type: str = "tornado") -> Dict[str, Any]:
        """
        Sensitivity analysis to identify key drivers and uncertainties.
        
        Args:
            model: Predictive model
            base_parameters: Base case parameters
            parameter_ranges: Parameter variation ranges
            target_variable: Variable to analyze sensitivity for
            analysis_type: Type of sensitivity analysis
        """
        if analysis_type == "tornado":
            return self._tornado_analysis(model, base_parameters, parameter_ranges, target_variable)
        elif analysis_type == "monte_carlo":
            return self._monte_carlo_sensitivity(model, base_parameters, parameter_ranges, target_variable)
        else:
            return self._one_factor_analysis(model, base_parameters, parameter_ranges, target_variable)
    
    def market_share_forecasting(self,
                               market_data: pd.DataFrame,
                               company_data: pd.DataFrame,
                               competitor_data: pd.DataFrame,
                               forecast_horizon: int,
                               growth_scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Market share forecasting with competitive dynamics.
        
        Args:
            market_data: Overall market data
            company_data: Company-specific data
            competitor_data: Competitor data
            forecast_horizon: Forecast horizon
            growth_scenarios: Growth scenarios to consider
        """
        # Market size forecasting
        market_forecast = self._forecast_market_size(market_data, forecast_horizon)
        
        # Company share evolution
        share_evolution = self._forecast_company_share(company_data, market_data, forecast_horizon)
        
        # Competitive dynamics
        competitive_dynamics = self._analyze_competitive_dynamics(competitor_data, forecast_horizon)
        
        # Scenario analysis
        if growth_scenarios is None:
            growth_scenarios = ["optimistic", "realistic", "pessimistic"]
        
        scenario_forecasts = {}
        for scenario in growth_scenarios:
            scenario_forecasts[scenario] = self._apply_growth_scenario(
                market_forecast, share_evolution, competitive_dynamics, scenario
            )
        
        # Market share predictions
        market_share_predictions = self._calculate_market_share_predictions(
            market_forecast, share_evolution, competitive_dynamics
        )
        
        # Risk factors
        risk_factors = self._identify_market_share_risks(market_data, competitor_data)
        
        return {
            'market_forecast': market_forecast,
            'share_evolution': share_evolution,
            'competitive_dynamics': competitive_dynamics,
            'scenario_forecasts': scenario_forecasts,
            'market_share_predictions': market_share_predictions,
            'risk_factors': risk_factors,
            'strategic_recommendations': self._generate_market_share_recommendations(market_share_predictions)
        }
    
    # Helper methods
    
    def _extract_time_features(self, time_index) -> Dict[str, np.ndarray]:
        """Extract time-based features."""
        if not isinstance(time_index, pd.DatetimeIndex):
            try:
                time_index = pd.to_datetime(time_index)
            except:
                return {}
        
        return {
            'month': time_index.month,
            'quarter': time_index.quarter,
            'year': time_index.year,
            'day_of_week': time_index.dayofweek,
            'day_of_year': time_index.dayofyear,
            'week_of_year': time_index.isocalendar().week,
            'is_weekend': (time_index.dayofweek >= 5).astype(int),
            'is_month_start': time_index.is_month_start.astype(int),
            'is_month_end': time_index.is_month_end.astype(int)
        }
    
    def _ensemble_demand_forecast(self, X: np.ndarray, y: pd.Series, features: List[str], 
                                horizon: int, seasonality: int) -> ForecastResult:
        """Ensemble demand forecasting with multiple models."""
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        trained_models = {}
        predictions = []
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Predict
                pred = model.predict(X_test)
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            
            # Feature importance (average across models)
            feature_importance = {}
            for i, feature in enumerate(features):
                importances = []
                for name, model in trained_models.items():
                    if hasattr(model, 'feature_importances_'):
                        importances.append(model.feature_importances_[i])
                    elif hasattr(model, 'coef_'):
                        importances.append(abs(model.coef_[i]) if len(model.coef_) > i else 0)
                feature_importance[feature] = np.mean(importances) if importances else 0
            
            # Future predictions (simplified)
            future_predictions = np.array([np.mean(y_test)] * horizon)
            
            # Confidence intervals
            predictions_std = np.std(predictions, axis=0)
            confidence_interval = 1.96 * np.mean(predictions_std)  # 95% CI
            
            return ForecastResult(
                method=ForecastingMethod.ENSEMBLE,
                predictions=future_predictions,
                confidence_intervals=np.column_stack([
                    future_predictions - confidence_interval,
                    future_predictions + confidence_interval
                ]),
                feature_importance=feature_importance,
                model_metrics={
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'models_used': len(trained_models)
                },
                forecast_horizon=horizon,
                model_artifacts={'trained_models': trained_models}
            )
        
        raise ValueError("No models trained successfully")
    
    def _lightgbm_forecast(self, X: np.ndarray, y: pd.Series, features: List[str], 
                          horizon: int, forecast_type: str) -> ForecastResult:
        """LightGBM forecasting."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        future_pred = model.predict(np.tile(np.mean(X_test, axis=0), (horizon, 1)))
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        return ForecastResult(
            method=ForecastingMethod.LIGHTGBM,
            predictions=future_pred,
            confidence_intervals=np.column_stack([
                future_pred - 1.96 * mae,
                future_pred + 1.96 * mae
            ]),
            feature_importance=feature_importance,
            model_metrics={'mae': mae, 'rmse': rmse, 'r2': r2},
            forecast_horizon=horizon,
            model_artifacts={'model': model}
        )
    
    def _xgboost_forecast(self, X: np.ndarray, y: pd.Series, features: List[str], 
                         horizon: int, forecast_type: str) -> ForecastResult:
        """XGBoost forecasting."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        future_pred = model.predict(np.tile(np.mean(X_test, axis=0), (horizon, 1)))
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        return ForecastResult(
            method=ForecastingMethod.XGBOOST,
            predictions=future_pred,
            confidence_intervals=np.column_stack([
                future_pred - 1.96 * mae,
                future_pred + 1.96 * mae
            ]),
            feature_importance=feature_importance,
            model_metrics={'mae': mae, 'rmse': rmse, 'r2': r2},
            forecast_horizon=horizon,
            model_artifacts={'model': model}
        )
    
    def _linear_forecast(self, X: np.ndarray, y: pd.Series, features: List[str], 
                        horizon: int, forecast_type: str) -> ForecastResult:
        """Linear regression forecasting."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear regression with polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_poly)
        future_X = np.tile(np.mean(X_test, axis=0), (horizon, 1))
        future_X_poly = poly.transform(future_X)
        future_pred = model.predict(future_X_poly)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (coefficients)
        feature_importance = dict(zip(poly.get_feature_names_out(features), np.abs(model.coef_)))
        
        return ForecastResult(
            method=ForecastingMethod.LINEAR_REGRESSION,
            predictions=future_pred,
            confidence_intervals=np.column_stack([
                future_pred - 1.96 * mae,
                future_pred + 1.96 * mae
            ]),
            feature_importance=feature_importance,
            model_metrics={'mae': mae, 'rmse': rmse, 'r2': r2},
            forecast_horizon=horizon,
            model_artifacts={'model': model, 'polynomial_features': poly}
        )
    
    def _add_revenue_features(self, X: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add revenue-specific features."""
        X_enhanced = X.copy()
        
        # Revenue per customer
        if 'revenue' in data.columns and 'customers' in data.columns:
            X_enhanced['revenue_per_customer'] = data['revenue'] / data['customers']
        
        # Growth rates
        for col in ['revenue', 'customers', 'transactions']:
            if col in data.columns:
                X_enhanced[f'{col}_growth'] = data[col].pct_change().fillna(0)
        
        # Moving averages
        for col in ['revenue', 'customers']:
            if col in data.columns:
                X_enhanced[f'{col}_ma_3'] = data[col].rolling(3).mean()
                X_enhanced[f'{col}_ma_6'] = data[col].rolling(6).mean()
        
        return X_enhanced
    
    def _ensemble_revenue_forecast(self, X: pd.DataFrame, y: pd.Series, 
                                 features: List[str], horizon: int) -> ForecastResult:
        """Ensemble revenue forecasting."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        predictions = []
        trained_models = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
                pred = model.predict(X_test_scaled)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Metrics
            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            
            # Future predictions
            future_X = np.tile(np.mean(X_test_scaled, axis=0), (horizon, 1))
            future_preds = []
            for model in trained_models.values():
                future_preds.append(model.predict(future_X))
            
            future_predictions = np.mean(future_preds, axis=0)
            
            return ForecastResult(
                method=ForecastingMethod.ENSEMBLE,
                predictions=future_predictions,
                confidence_intervals=np.column_stack([
                    future_predictions - 1.96 * mae,
                    future_predictions + 1.96 * mae
                ]),
                feature_importance={},
                model_metrics={'mae': mae, 'rmse': rmse, 'r2': r2},
                forecast_horizon=horizon,
                model_artifacts={'models': trained_models, 'scaler': scaler}
            )
        
        raise ValueError("No models trained successfully")
    
    def _calculate_confidence_intervals(self, X: np.ndarray, y: pd.Series, 
                                      predictions: np.ndarray, horizon: int) -> np.ndarray:
        """Calculate confidence intervals for predictions."""
        from sklearn.model_selection import cross_val_score
        
        # Estimate prediction uncertainty using cross-validation
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae_std = np.std(-scores)
        
        confidence_intervals = np.column_stack([
            predictions - 1.96 * mae_std,
            predictions + 1.96 * mae_std
        ])
        
        return confidence_intervals
    
    def _engineer_churn_features(self, X: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for churn prediction."""
        X_enhanced = X.copy()
        
        # Days since last activity
        if 'last_activity' in data.columns:
            X_enhanced['days_since_last_activity'] = (datetime.now() - pd.to_datetime(data['last_activity'])).dt.days
        
        # Frequency features
        for col in ['transactions', 'interactions', 'support_tickets']:
            if col in data.columns:
                X_enhanced[f'{col}_rate'] = data[col] / (data['tenure_days'] + 1)
        
        # Recency features
        recency_cols = [col for col in data.columns if 'recency' in col.lower()]
        for col in recency_cols:
            X_enhanced[f'{col}_ratio'] = data[col] / (data['tenure_days'] + 1)
        
        # Engagement score
        engagement_cols = ['transactions', 'interactions', 'logins']
        if all(col in data.columns for col in engagement_cols):
            X_enhanced['engagement_score'] = (
                data['transactions'] * 0.4 +
                data['interactions'] * 0.3 +
                data['logins'] * 0.3
            )
        
        return X_enhanced
    
    def _ensemble_churn_prediction(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict[str, Any]:
        """Ensemble churn prediction."""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual models
        models = {
            'lr': LogisticRegression(random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gbm': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        trained_models = {}
        predictions = []
        
        for name, model in models.items():
            try:
                if name == 'lr':
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_test)[:, 1]
                
                trained_models[name] = model
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, classification_report
            auc_score = roc_auc_score(y_test, ensemble_pred)
            
            # Feature importance (average across tree-based models)
            feature_importance = {}
            for i, feature in enumerate(features):
                importances = []
                for name, model in trained_models.items():
                    if hasattr(model, 'feature_importances_'):
                        importances.append(model.feature_importances_[i])
                feature_importance[feature] = np.mean(importances) if importances else 0
            
            return {
                'churn_probability': ensemble_pred,
                'churn_prediction': (ensemble_pred > 0.5).astype(int),
                'feature_importance': feature_importance,
                'model_performance': {
                    'auc_score': auc_score,
                    'models_used': len(trained_models)
                },
                'individual_predictions': dict(zip(trained_models.keys(), predictions)),
                'model_artifacts': trained_models
            }
        
        raise ValueError("No models trained successfully")
    
    def _single_churn_model(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict[str, Any]:
        """Single model churn prediction."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Use Random Forest as single model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'churn_probability': y_pred_proba,
            'churn_prediction': y_pred,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'model_performance': {
                'auc_score': auc_score,
                'accuracy': (y_pred == y_test).mean()
            },
            'model_artifacts': {'model': model}
        }
    
    def _calculate_clv_features(self, transactions: pd.DataFrame, customer_features: pd.DataFrame, 
                              time_horizon: int) -> pd.DataFrame:
        """Calculate CLV features."""
        # Aggregate transaction features
        transaction_features = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'transaction_date': ['min', 'max', 'count']
        }).reset_index()
        
        transaction_features.columns = ['customer_id'] + [f'transaction_{col[0]}_{col[1]}' for col in transaction_features.columns[1:]]
        
        # Calculate derived features
        transaction_features['transaction_avg_days_between'] = (
            transaction_features['transaction_date_max'] - transaction_features['transaction_date_min']
        ).dt.days / (transaction_features['transaction_date_count'] + 1)
        
        transaction_features['transaction_frequency'] = (
            transaction_features['transaction_date_count'] / 
            ((transaction_features['transaction_date_max'] - transaction_features['transaction_date_min']).dt.days + 1) * 30
        )
        
        # Merge with customer features
        clv_features = customer_features.merge(transaction_features, on='customer_id', how='left')
        
        return clv_features
    
    def _calculate_historical_clv(self, transactions: pd.DataFrame) -> pd.Series:
        """Calculate historical CLV."""
        clv = transactions.groupby('customer_id')['amount'].sum()
        return clv
    
    def _ensemble_clv_prediction(self, X: pd.DataFrame, y: pd.Series, customer_ids: pd.Series) -> Dict[str, Any]:
        """Ensemble CLV prediction."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        predictions = []
        trained_models = {}
        
        for name, model in models.items():
            try:
                if name in ['ridge', 'elastic']:
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                
                trained_models[name] = model
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Metrics
            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            
            return {
                'clv_predictions': ensemble_pred,
                'customer_ids': ids_test,
                'model_performance': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'models_used': len(trained_models)
                },
                'feature_importance': {},
                'model_artifacts': trained_models
            }
        
        raise ValueError("No models trained successfully")
    
    def _single_clv_model(self, X: pd.DataFrame, y: pd.Series, customer_ids: pd.Series) -> Dict[str, Any]:
        """Single model CLV prediction."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids, test_size=0.2, random_state=42
        )
        
        # Use Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'clv_predictions': y_pred,
            'customer_ids': ids_test,
            'model_performance': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            'model_artifacts': {'model': model}
        }
    
    def _calculate_price_elasticity(self, price_data: pd.DataFrame, demand_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price elasticity."""
        elasticity = {}
        
        for product in price_data['product'].unique():
            product_prices = price_data[price_data['product'] == product]['price']
            product_demand = demand_data[demand_data['product'] == product]['quantity']
            
            if len(product_prices) > 1 and len(product_demand) > 1:
                # Calculate elasticity using log-log regression
                log_prices = np.log(product_prices + 1e-10)
                log_demand = np.log(product_demand + 1e-10)
                
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(log_prices.values.reshape(-1, 1), log_demand.values)
                
                elasticity[product] = model.coef_[0]
        
        return elasticity
    
    def _analyze_competitive_pricing(self, competitor_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze competitive pricing."""
        analysis = {}
        
        # Price distribution analysis
        competitor_prices = competitor_data.groupby('product')['price'].agg(['mean', 'std', 'min', 'max'])
        company_prices = price_data.groupby('product')['price'].agg(['mean', 'std'])
        
        # Price positioning
        for product in competitor_prices.index:
            if product in company_prices.index:
                comp_price = competitor_prices.loc[product, 'mean']
                company_price = company_prices.loc[product, 'mean']
                
                analysis[product] = {
                    'company_price': company_price,
                    'competitor_avg_price': comp_price,
                    'price_gap': company_price - comp_price,
                    'position': 'premium' if company_price > comp_price * 1.1 else 'value' if company_price < comp_price * 0.9 else 'competitive'
                }
        
        return analysis
    
    def _elasticity_based_optimization(self, price_data: pd.DataFrame, demand_data: pd.DataFrame, 
                                     elasticity: Dict[str, float]) -> Dict[str, float]:
        """Elasticity-based price optimization."""
        optimal_prices = {}
        
        for product, elastic in elasticity.items():
            product_data = price_data[price_data['product'] == product]
            demand_info = demand_data[demand_data['product'] == product]
            
            if len(product_data) > 0 and len(demand_info) > 0:
                current_price = product_data['price'].mean()
                current_demand = demand_info['quantity'].mean()
                
                # Optimal price formula: P* = E / (E + 1) * P_current
                optimal_price = (elastic / (elastic + 1)) * current_price
                optimal_prices[product] = max(optimal_price, current_price * 0.5)  # Floor at 50% of current price
        
        return optimal_prices
    
    def _competitive_optimization(self, competitor_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Competitive-based price optimization."""
        optimal_prices = {}
        
        for product in competitor_data['product'].unique():
            comp_data = competitor_data[competitor_data['product'] == product]
            company_data = price_data[price_data['product'] == product]
            
            if len(comp_data) > 0 and len(company_data) > 0:
                # Price at 5-10% below average competitor price
                avg_competitor_price = comp_data['price'].mean()
                optimal_price = avg_competitor_price * 0.92  # 8% below
                optimal_prices[product] = optimal_price
        
        return optimal_prices
    
    def _value_based_optimization(self, demand_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Value-based price optimization."""
        optimal_prices = {}
        
        for product in demand_data['product'].unique():
            demand_info = demand_data[demand_data['product'] == product]
            price_info = price_data[price_data['product'] == product]
            
            if len(demand_info) > 0 and len(price_info) > 0:
                # Higher price for high-value customers
                value_score = demand_info['quantity'].mean()
                current_price = price_info['price'].mean()
                
                # Adjust price based on demand level
                if value_score > demand_info['quantity'].quantile(0.75):
                    optimal_price = current_price * 1.15  # 15% premium
                elif value_score < demand_info['quantity'].quantile(0.25):
                    optimal_price = current_price * 0.85  # 15% discount
                else:
                    optimal_price = current_price
                
                optimal_prices[product] = optimal_price
        
        return optimal_prices
    
    def _dynamic_pricing_optimization(self, price_data: pd.DataFrame, demand_data: pd.DataFrame, 
                                    competitor_data: pd.DataFrame) -> Dict[str, float]:
        """Dynamic pricing optimization."""
        # Combine all optimization methods
        optimal_prices = {}
        
        # Get elasticity-based prices
        elasticity = self._calculate_price_elasticity(price_data, demand_data)
        elasticity_prices = self._elasticity_based_optimization(price_data, demand_data, elasticity)
        
        # Get competitive-based prices
        competitive_prices = self._competitive_optimization(competitor_data, price_data)
        
        # Combine strategies (weighted average)
        for product in set(elasticity_prices.keys()) | set(competitive_prices.keys()):
            weights = [0.6, 0.4]  # More weight on elasticity
            prices = [elasticity_prices.get(product, 0), competitive_prices.get(product, 0)]
            optimal_prices[product] = np.average(prices, weights=weights)
        
        return optimal_prices
    
    def _analyze_profit_impact(self, price_data: pd.DataFrame, demand_data: pd.DataFrame, 
                             optimal_prices: Dict[str, float]) -> Dict[str, Any]:
        """Analyze profit impact of price changes."""
        impact = {}
        
        for product, new_price in optimal_prices.items():
            current_price_data = price_data[price_data['product'] == product]
            current_demand_data = demand_data[demand_data['product'] == product]
            
            if len(current_price_data) > 0 and len(current_demand_data) > 0:
                current_price = current_price_data['price'].mean()
                current_demand = current_demand_data['quantity'].mean()
                cost = current_demand_data.get('cost', pd.Series([current_price * 0.6])).mean()
                
                # Calculate profit impact
                current_profit = (current_price - cost) * current_demand
                
                # Estimate new demand (using elasticity if available)
                price_change = (new_price - current_price) / current_price
                demand_change = -0.5 * price_change  # Simplified elasticity
                new_demand = current_demand * (1 + demand_change)
                
                new_profit = (new_price - cost) * new_demand
                profit_change = new_profit - current_profit
                
                impact[product] = {
                    'current_profit': current_profit,
                    'projected_profit': new_profit,
                    'profit_change': profit_change,
                    'profit_change_pct': profit_change / current_profit if current_profit != 0 else 0,
                    'new_demand': new_demand,
                    'demand_change': new_demand - current_demand
                }
        
        return impact
    
    def _assess_pricing_risks(self, optimal_prices: Dict[str, float], competitive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess pricing risks."""
        risks = {}
        
        for product, price in optimal_prices.items():
            if product in competitive_analysis:
                comp_analysis = competitive_analysis[product]
                
                risk_factors = []
                
                # Price gap risk
                if comp_analysis['position'] == 'premium':
                    risk_factors.append('High-end pricing may limit market share')
                elif comp_analysis['position'] == 'value':
                    risk_factors.append('Low pricing may damage brand perception')
                
                # Competitor response risk
                risk_factors.append('Competitors may adjust pricing in response')
                
                # Market share risk
                price_gap = abs(comp_analysis['price_gap'])
                if price_gap > optimal_prices[product] * 0.2:
                    risk_factors.append('Significant price difference may affect competitiveness')
                
                risks[product] = {
                    'risk_level': 'high' if len(risk_factors) > 2 else 'medium' if len(risk_factors) > 1 else 'low',
                    'risk_factors': risk_factors,
                    'mitigation_strategies': [
                        'Monitor competitor pricing closely',
                        'Test price changes gradually',
                        'Focus on value differentiation'
                    ]
                }
        
        return risks
    
    def _generate_pricing_recommendations(self, optimal_prices: Dict[str, float], elasticity: Dict[str, float],
                                        competitive_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate pricing recommendations."""
        recommendations = []
        
        # Elasticity-based recommendations
        for product, elastic in elasticity.items():
            if abs(elastic) > 1:
                recommendations.append(f"{product}: High elasticity - consider promotional pricing")
            elif abs(elastic) < 0.5:
                recommendations.append(f"{product}: Low elasticity - opportunity for price increases")
        
        # Competitive positioning recommendations
        for product, analysis in competitive_analysis.items():
            if analysis['position'] == 'premium':
                recommendations.append(f"{product}: Premium positioning - ensure value proposition justifies price")
            elif analysis['position'] == 'value':
                recommendations.append(f"{product}: Value positioning - focus on cost efficiency")
        
        # Risk-based recommendations
        for product, risk in risk_assessment.items():
            if risk['risk_level'] == 'high':
                recommendations.append(f"{product}: High pricing risk - implement gradual price changes")
        
        # General recommendations
        recommendations.extend([
            'Implement dynamic pricing for seasonal products',
            'Monitor competitor pricing changes weekly',
            'Test price elasticity with controlled experiments',
            'Consider bundle pricing for related products',
            'Review pricing strategy quarterly'
        ])
        
        return recommendations
    
    def _run_simulation(self, params: Dict[str, Any], time_horizon: int) -> Dict[str, float]:
        """Run simulation with given parameters."""
        # This is a simplified simulation framework
        # In practice, this would implement specific business logic
        
        result = {}
        
        # Market growth simulation
        base_growth = params.get('market_growth', 0.05)
        growth_rate = np.random.normal(base_growth, 0.02)
        
        # Revenue simulation
        base_revenue = params.get('base_revenue', 1000000)
        result['revenue'] = base_revenue * (1 + growth_rate) ** time_horizon
        
        # Customer acquisition simulation
        base_acquisition = params.get('customer_acquisition', 1000)
        acquisition_rate = params.get('acquisition_rate', 0.1)
        result['new_customers'] = base_acquisition * acquisition_rate * time_horizon
        
        # Cost simulation
        base_costs = params.get('base_costs', 600000)
        cost_growth = params.get('cost_growth', 0.03)
        result['costs'] = base_costs * (1 + cost_growth) ** time_horizon
        
        # Profit calculation
        result['profit'] = result['revenue'] - result['costs']
        result['profit_margin'] = result['profit'] / result['revenue'] if result['revenue'] != 0 else 0
        
        return result
    
    def _calculate_simulation_risks(self, simulation_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risks from simulation results."""
        risks = {}
        
        for column in simulation_df.select_dtypes(include=[np.number]).columns:
            values = simulation_df[column]
            
            # Value at Risk (VaR)
            risks[f'{column}_var_5'] = np.percentile(values, 5)
            risks[f'{column}_var_95'] = np.percentile(values, 95)
            
            # Probability of negative outcomes
            if column in ['profit', 'revenue']:
                risks[f'{column}_prob_negative'] = (values < 0).mean()
            
            # Volatility
            risks[f'{column}_volatility'] = values.std() / values.mean() if values.mean() != 0 else 0
        
        return risks
    
    def _generate_simulation_recommendations(self, statistics: Dict[str, Any], risk_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Revenue recommendations
        if 'revenue_mean' in statistics:
            revenue_mean = statistics['revenue_mean']
            revenue_std = risk_metrics.get('revenue_volatility', 0)
            
            if revenue_std > revenue_mean * 0.3:
                recommendations.append('High revenue volatility - diversify revenue streams')
        
        # Profit recommendations
        if 'profit_mean' in statistics:
            profit_mean = statistics['profit_mean']
            prob_negative = risk_metrics.get('profit_prob_negative', 0)
            
            if prob_negative > 0.1:
                recommendations.append('High probability of losses - implement risk management')
        
        # General recommendations
        recommendations.extend([
            'Monitor key assumptions regularly',
            'Update simulations with new data',
            'Consider additional scenario variations',
            'Implement early warning systems',
            'Develop contingency plans for adverse scenarios'
        ])
        
        return recommendations
    
    def _determine_scenario_type(self, scenario_name: str) -> ScenarioType:
        """Determine scenario type from name."""
        scenario_name_lower = scenario_name.lower()
        
        if 'base' in scenario_name_lower:
            return ScenarioType.BASE_CASE
        elif 'bull' in scenario_name_lower or 'optimistic' in scenario_name_lower:
            return ScenarioType.BULL_CASE
        elif 'bear' in scenario_name_lower or 'pessimistic' in scenario_name_lower:
            return ScenarioType.BEAR_CASE
        elif 'stress' in scenario_name_lower:
            return ScenarioType.STRESS_TEST
        else:
            return ScenarioType.BASE_CASE
    
    def _run_scenario(self, params: Dict[str, Any], scenario_type: ScenarioType) -> Dict[str, Any]:
        """Run scenario analysis."""
        # Apply scenario-specific adjustments
        if scenario_type == ScenarioType.BULL_CASE:
            params['market_growth'] = params.get('market_growth', 0.05) * 1.5
            params['customer_acquisition'] = params.get('customer_acquisition', 1000) * 1.3
        elif scenario_type == ScenarioType.BEAR_CASE:
            params['market_growth'] = params.get('market_growth', 0.05) * 0.5
            params['customer_acquisition'] = params.get('customer_acquisition', 1000) * 0.7
        elif scenario_type == ScenarioType.STRESS_TEST:
            params['market_growth'] = -0.1  # Negative growth
            params['customer_acquisition'] = params.get('customer_acquisition', 1000) * 0.5
        
        # Run simulation
        return self._run_simulation(params, 12)  # 12-month horizon
    
    def _calculate_scenario_probability(self, scenario_name: str, scenarios: Dict[str, Dict[str, Any]]) -> float:
        """Calculate scenario probability."""
        # This is a simplified probability calculation
        # In practice, this would use historical data or expert judgment
        
        scenario_name_lower = scenario_name.lower()
        
        if 'base' in scenario_name_lower:
            return 0.5  # 50% probability for base case
        elif 'bull' in scenario_name_lower:
            return 0.25  # 25% probability for bull case
        elif 'bear' in scenario_name_lower:
            return 0.25  # 25% probability for bear case
        else:
            return 0.1  # 10% probability for other scenarios
    
    def _identify_scenario_assumptions(self, scenario_name: str, params: Dict[str, Any]) -> List[str]:
        """Identify scenario assumptions."""
        assumptions = []
        
        for param, value in params.items():
            if 'growth' in param:
                assumptions.append(f"Market {param.replace('_', ' ')}: {value:.1%}")
            elif 'acquisition' in param:
                assumptions.append(f"Customer {param.replace('_', ' ')}: {value}")
            elif 'price' in param:
                assumptions.append(f"{param.replace('_', ' ')}: ${value:.2f}")
        
        return assumptions
    
    def _identify_key_drivers(self, scenario_name: str, params: Dict[str, Any]) -> List[str]:
        """Identify key scenario drivers."""
        # Key drivers are parameters with highest impact
        drivers = []
        
        # High-impact parameters
        if 'market_growth' in params:
            drivers.append('Market Growth Rate')
        if 'customer_acquisition' in params:
            drivers.append('Customer Acquisition Rate')
        if 'pricing' in params:
            drivers.append('Pricing Strategy')
        
        return drivers
    
    def _identify_scenario_risks(self, scenario_name: str, params: Dict[str, Any]) -> List[str]:
        """Identify scenario-specific risks."""
        risks = []
        
        scenario_name_lower = scenario_name.lower()
        
        if 'bull' in scenario_name_lower:
            risks.extend([
                'Market may not sustain high growth',
                'Competition may intensify',
                'Resource constraints may limit growth'
            ])
        elif 'bear' in scenario_name_lower:
            risks.extend([
                'Market deterioration may accelerate',
                'Customer churn may increase',
                'Cash flow challenges'
            ])
        elif 'stress' in scenario_name_lower:
            risks.extend([
                'Multiple adverse events simultaneously',
                'Systemic market failure',
                'Business continuity threats'
            ])
        
        return risks
    
    def _generate_mitigation_strategies(self, scenario_name: str, risks: List[str]) -> List[str]:
        """Generate mitigation strategies."""
        strategies = []
        
        # General strategies
        strategies.extend([
            'Develop contingency plans',
            'Strengthen financial reserves',
            'Diversify revenue streams',
            'Enhance operational flexibility'
        ])
        
        # Scenario-specific strategies
        scenario_name_lower = scenario_name.lower()
        
        if 'bull' in scenario_name_lower:
            strategies.append('Prepare infrastructure for rapid scaling')
        elif 'bear' in scenario_name_lower:
            strategies.append('Implement cost reduction measures')
        elif 'stress' in scenario_name_lower:
            strategies.append('Establish crisis management protocols')
        
        return strategies
    
    def _tornado_analysis(self, model: Any, base_params: Dict[str, float], 
                         param_ranges: Dict[str, Tuple[float, float]], 
                         target_variable: str) -> Dict[str, Any]:
        """Tornado diagram sensitivity analysis."""
        # This would implement tornado analysis
        # For now, return simplified results
        
        sensitivities = {}
        for param, (low, high) in param_ranges.items():
            # Calculate sensitivity (simplified)
            base_value = base_params.get(param, (low + high) / 2)
            sensitivity = (high - low) / base_value if base_value != 0 else 0
            sensitivities[param] = {
                'low_value': low,
                'high_value': high,
                'base_value': base_value,
                'sensitivity': sensitivity,
                'impact_range': sensitivity * base_value
            }
        
        # Sort by sensitivity
        sorted_sensitivities = dict(sorted(sensitivities.items(), 
                                         key=lambda x: abs(x[1]['sensitivity']), 
                                         reverse=True))
        
        return {
            'tornado_data': sorted_sensitivities,
            'most_sensitive_parameters': list(sorted_sensitivities.keys())[:5],
            'total_sensitivity': sum(abs(s['sensitivity']) for s in sorted_sensitivities.values())
        }
    
    def _monte_carlo_sensitivity(self, model: Any, base_params: Dict[str, float], 
                               param_ranges: Dict[str, Tuple[float, float]], 
                               target_variable: str) -> Dict[str, Any]:
        """Monte Carlo sensitivity analysis."""
        # This would implement Monte Carlo sensitivity analysis
        return {
            'method': 'monte_carlo_sensitivity',
            'note': 'Monte Carlo sensitivity analysis implementation needed'
        }
    
    def _one_factor_analysis(self, model: Any, base_params: Dict[str, float], 
                           param_ranges: Dict[str, Tuple[float, float]], 
                           target_variable: str) -> Dict[str, Any]:
        """One-factor-at-a-time sensitivity analysis."""
        # This would implement one-factor-at-a-time analysis
        return {
            'method': 'one_factor_analysis',
            'note': 'One-factor sensitivity analysis implementation needed'
        }
    
    def _forecast_market_size(self, market_data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Forecast market size."""
        # Simplified market size forecasting
        if 'market_size' in market_data.columns:
            # Use exponential smoothing or trend analysis
            recent_growth = market_data['market_size'].pct_change().tail(4).mean()
            current_size = market_data['market_size'].iloc[-1]
            
            forecasts = []
            for i in range(horizon):
                forecast_size = current_size * (1 + recent_growth) ** (i + 1)
                forecasts.append(forecast_size)
            
            return {
                'current_size': current_size,
                'forecast_size': forecasts,
                'annual_growth_rate': recent_growth,
                'method': 'trend_extrapolation'
            }
        
        return {'error': 'market_size column not found'}
    
    def _forecast_company_share(self, company_data: pd.DataFrame, market_data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Forecast company market share evolution."""
        # Simplified share forecasting
        if 'market_share' in company_data.columns:
            share_data = company_data['market_share'].dropna()
            if len(share_data) > 1:
                # Calculate share trend
                share_trend = (share_data.iloc[-1] - share_data.iloc[0]) / len(share_data)
                current_share = share_data.iloc[-1]
                
                forecasts = []
                for i in range(horizon):
                    forecast_share = current_share + (share_trend * (i + 1))
                    forecast_share = max(0, min(1, forecast_share))  # Bound between 0 and 1
                    forecasts.append(forecast_share)
                
                return {
                    'current_share': current_share,
                    'forecast_shares': forecasts,
                    'share_trend': share_trend,
                    'method': 'trend_extrapolation'
                }
        
        return {'error': 'market_share data insufficient'}
    
    def _analyze_competitive_dynamics(self, competitor_data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Analyze competitive dynamics."""
        dynamics = {
            'number_of_competitors': len(competitor_data),
            'market_concentration': 'moderate',  # Simplified
            'competitive_intensity': 'high',  # Simplified
            'entry_barriers': 'medium'  # Simplified
        }
        
        return dynamics
    
    def _apply_growth_scenario(self, market_forecast: Dict, share_evolution: Dict, 
                             competitive_dynamics: Dict, scenario: str) -> Dict[str, Any]:
        """Apply growth scenario to forecasts."""
        scenario_multipliers = {
            'optimistic': 1.5,
            'realistic': 1.0,
            'pessimistic': 0.7
        }
        
        multiplier = scenario_multipliers.get(scenario, 1.0)
        
        # Adjust forecasts
        if 'forecast_size' in market_forecast:
            adjusted_forecasts = [size * multiplier for size in market_forecast['forecast_size']]
        else:
            adjusted_forecasts = []
        
        return {
            'scenario': scenario,
            'adjusted_market_forecasts': adjusted_forecasts,
            'multiplier': multiplier
        }
    
    def _calculate_market_share_predictions(self, market_forecast: Dict, share_evolution: Dict, 
                                          competitive_dynamics: Dict) -> Dict[str, Any]:
        """Calculate detailed market share predictions."""
        # Combine market and share forecasts
        if 'forecast_size' in market_forecast and 'forecast_shares' in share_evolution:
            horizon = min(len(market_forecast['forecast_size']), len(share_evolution['forecast_shares']))
            
            predictions = []
            for i in range(horizon):
                market_size = market_forecast['forecast_size'][i]
                share = share_evolution['forecast_shares'][i]
                predicted_revenue = market_size * share
                
                predictions.append({
                    'period': i + 1,
                    'market_size': market_size,
                    'market_share': share,
                    'predicted_revenue': predicted_revenue
                })
            
            return {'predictions': predictions}
        
        return {'error': 'Insufficient forecast data'}
    
    def _identify_market_share_risks(self, market_data: pd.DataFrame, competitor_data: pd.DataFrame) -> List[str]:
        """Identify market share risks."""
        risks = [
            'Competitive price wars',
            'New market entrants',
            'Technology disruption',
            'Changing customer preferences',
            'Economic downturns'
        ]
        
        return risks
    
    def _generate_market_share_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate market share recommendations."""
        recommendations = [
            'Focus on differentiation strategies',
            'Invest in customer retention programs',
            'Monitor competitor moves closely',
            'Consider strategic partnerships',
            'Develop innovation capabilities',
            'Optimize pricing strategies',
            'Enhance customer experience'
        ]
        
        return recommendations
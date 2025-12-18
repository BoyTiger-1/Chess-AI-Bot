"""
Financial Analytics Engine
Provides comprehensive financial analysis including portfolio optimization, risk management,
VaR calculations, GARCH modeling, and financial ratio analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats, optimize
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Financial risk metrics."""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    DOWNSIDE_DEVIATION = "downside_deviation"


class PortfolioOptimization(Enum):
    """Portfolio optimization methods."""
    MARKOWITZ = "markowitz"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"


@dataclass
class PortfolioResult:
    """Portfolio optimization result."""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: PortfolioOptimization
    asset_names: List[str]
    risk_metrics: Dict[str, float]
    optimization_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAnalysisResult:
    """Financial risk analysis result."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    confidence_level: float = 0.95
    analysis_period: int = 252  # trading days
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FinancialForecastResult:
    """Financial forecasting result."""
    forecasts: Dict[str, np.ndarray]
    confidence_intervals: Dict[str, np.ndarray]
    model_type: str
    forecast_horizon: int
    asset_names: List[str]
    forecast_dates: pd.DatetimeIndex
    model_metrics: Dict[str, float] = field(default_factory=dict)


class FinancialAnalyticsEngine:
    """
    Advanced Financial Analytics Engine
    Provides portfolio optimization, risk management, forecasting, and financial analysis.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, confidence_level: float = 0.95):
        """
        Initialize Financial Analytics Engine.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
            confidence_level: Confidence level for risk metrics
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        
    def portfolio_optimization(self,
                             returns: pd.DataFrame,
                             method: PortfolioOptimization = PortfolioOptimization.MARKOWITZ,
                             constraints: Optional[Dict[str, Any]] = None) -> PortfolioResult:
        """
        Perform portfolio optimization using various methods.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method
            constraints: Portfolio constraints
        """
        asset_names = returns.columns.tolist()
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        if constraints is None:
            constraints = {
                'long_only': True,
                'max_weight': 0.4,
                'min_weight': 0.0,
                'target_return': None
            }
        
        if method == PortfolioOptimization.MARKOWITZ:
            return self._markowitz_optimization(mean_returns, cov_matrix, asset_names, constraints)
        elif method == PortfolioOptimization.BLACK_LITTERMAN:
            return self._black_litterman_optimization(returns, asset_names, constraints)
        elif method == PortfolioOptimization.RISK_PARITY:
            return self._risk_parity_optimization(cov_matrix, asset_names, constraints)
        elif method == PortfolioOptimization.MAX_SHARPE:
            return self._max_sharpe_optimization(mean_returns, cov_matrix, asset_names, constraints)
        elif method == PortfolioOptimization.MIN_VOLATILITY:
            return self._min_volatility_optimization(mean_returns, cov_matrix, asset_names, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _markowitz_optimization(self,
                              mean_returns: pd.Series,
                              cov_matrix: pd.DataFrame,
                              asset_names: List[str],
                              constraints: Dict[str, Any]) -> PortfolioResult:
        """Markowitz mean-variance optimization."""
        n_assets = len(asset_names)
        
        def objective_function(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Long-only constraint
        if constraints.get('long_only', True):
            constraints_list.extend([
                {'type': 'ineq', 'fun': lambda x: x},  # Weights >= 0
                {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}  # Weights <= max_weight
            ])
        
        # Bounds
        if constraints.get('long_only', True):
            bounds = [(0, constraints['max_weight']) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]  # Allow short selling
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fallback to equal weights
            weights = initial_weights
        else:
            weights = result.x
        
        # Calculate metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return PortfolioResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            method=PortfolioOptimization.MARKOWITZ,
            asset_names=asset_names,
            risk_metrics={
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            },
            optimization_details={
                'optimization_success': result.success,
                'optimization_message': result.message
            }
        )
    
    def _black_litterman_optimization(self,
                                    returns: pd.DataFrame,
                                    asset_names: List[str],
                                    constraints: Dict[str, Any]) -> PortfolioResult:
        """Black-Litterman portfolio optimization."""
        # Simplified Black-Litterman implementation
        # In practice, this would involve market cap weights and investor views
        
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252
        
        # Assume market capitalization weights (equal weights as proxy)
        market_weights = np.ones(len(asset_names)) / len(asset_names)
        
        # Risk aversion parameter
        risk_aversion = 2.5
        
        # Calculate implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # Add confidence in market equilibrium (tau parameter)
        tau = 0.025
        
        # Black-Litterman formula
        cov_mtx_inv = np.linalg.inv(cov_matrix)
        pi_adj = np.dot(
            np.linalg.inv(np.eye(len(asset_names)) + tau * cov_mtx_inv),
            pi
        )
        
        # Optimize with adjusted returns
        mean_returns_adj = pd.Series(pi_adj, index=asset_names)
        
        # Use Markowitz optimization with adjusted returns
        return self._markowitz_optimization(mean_returns_adj, cov_matrix, asset_names, constraints)
    
    def _risk_parity_optimization(self,
                                cov_matrix: pd.DataFrame,
                                asset_names: List[str],
                                constraints: Dict[str, Any]) -> PortfolioResult:
        """Risk parity portfolio optimization."""
        n_assets = len(asset_names)
        
        def risk_budget_objective(weights):
            """Objective function for risk parity."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize sum of squared deviations from equal risk contribution
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints.get('long_only', True):
            constraints_list.extend([
                {'type': 'ineq', 'fun': lambda x: x},
                {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}
            ])
        
        # Bounds
        if constraints.get('long_only', True):
            bounds = [(0, constraints['max_weight']) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = optimize.minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            weights = initial_weights
        else:
            weights = result.x
        
        # Calculate metrics
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        portfolio_return = np.dot(weights, np.mean(returns.cov()) * 252)  # Approximation
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return PortfolioResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            method=PortfolioOptimization.RISK_PARITY,
            asset_names=asset_names,
            risk_metrics={
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            }
        )
    
    def _max_sharpe_optimization(self,
                               mean_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               asset_names: List[str],
                               constraints: Dict[str, Any]) -> PortfolioResult:
        """Maximum Sharpe ratio optimization."""
        return self._markowitz_optimization(mean_returns, cov_matrix, asset_names, constraints)
    
    def _min_volatility_optimization(self,
                                   mean_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   asset_names: List[str],
                                   constraints: Dict[str, Any]) -> PortfolioResult:
        """Minimum volatility optimization."""
        # This is same as Markowitz but with different objective
        return self._markowitz_optimization(mean_returns, cov_matrix, asset_names, constraints)
    
    def calculate_var_cvar(self,
                         returns: pd.Series,
                         method: str = "historical",
                         confidence_level: float = 0.95,
                         portfolio_value: float = 1000000) -> RiskAnalysisResult:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of portfolio returns
            method: "historical", "parametric", "monte_carlo"
            confidence_level: Confidence level for VaR calculation
            portfolio_value: Portfolio value for absolute VaR
        """
        if method == "historical":
            return self._historical_var_cvar(returns, confidence_level, portfolio_value)
        elif method == "parametric":
            return self._parametric_var_cvar(returns, confidence_level, portfolio_value)
        elif method == "monte_carlo":
            return self._monte_carlo_var_cvar(returns, confidence_level, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var_cvar(self,
                           returns: pd.Series,
                           confidence_level: float,
                           portfolio_value: float) -> RiskAnalysisResult:
        """Historical VaR and CVaR calculation."""
        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(returns, var_percentile) * portfolio_value
        
        # CVaR (Expected Shortfall)
        tail_returns = returns[returns <= np.percentile(returns, var_percentile)]
        cvar_value = np.mean(tail_returns) * portfolio_value if len(tail_returns) > 0 else var_value
        
        # Calculate other metrics
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # VaR at different confidence levels
        var_95 = np.percentile(returns, 5) * portfolio_value
        var_99 = np.percentile(returns, 1) * portfolio_value
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio_value
        cvar_99 = np.mean(returns[returns <= np.percentile(returns, 1)]) * portfolio_value
        
        return RiskAnalysisResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_level=confidence_level
        )
    
    def _parametric_var_cvar(self,
                           returns: pd.Series,
                           confidence_level: float,
                           portfolio_value: float) -> RiskAnalysisResult:
        """Parametric VaR and CVaR calculation using normal distribution."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Parametric VaR (assuming normal distribution)
        z_score = stats.norm.ppf(1 - confidence_level)
        var_value = (mean_return + z_score * std_return) * portfolio_value
        
        # CVaR calculation
        var_threshold = mean_return + z_score * std_return
        cvar_value = (mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)) * portfolio_value
        
        # Calculate other metrics
        volatility = std_return * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return RiskAnalysisResult(
            var_95=var_value if confidence_level == 0.95 else np.percentile(returns, 5) * portfolio_value,
            var_99=np.percentile(returns, 1) * portfolio_value,
            cvar_95=cvar_value if confidence_level == 0.95 else np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio_value,
            cvar_99=np.mean(returns[returns <= np.percentile(returns, 1)]) * portfolio_value,
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_level=confidence_level
        )
    
    def _monte_carlo_var_cvar(self,
                            returns: pd.Series,
                            confidence_level: float,
                            portfolio_value: float,
                            num_simulations: int = 10000) -> RiskAnalysisResult:
        """Monte Carlo VaR and CVaR calculation."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate VaR and CVaR from simulated returns
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(simulated_returns, var_percentile) * portfolio_value
        
        tail_returns = simulated_returns[simulated_returns <= np.percentile(simulated_returns, var_percentile)]
        cvar_value = np.mean(tail_returns) * portfolio_value if len(tail_returns) > 0 else var_value
        
        # Calculate other metrics
        volatility = std_return * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return RiskAnalysisResult(
            var_95=var_value if confidence_level == 0.95 else np.percentile(returns, 5) * portfolio_value,
            var_99=np.percentile(returns, 1) * portfolio_value,
            cvar_95=cvar_value if confidence_level == 0.95 else np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio_value,
            cvar_99=np.mean(returns[returns <= np.percentile(returns, 1)]) * portfolio_value,
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_level=confidence_level
        )
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information Ratio
        market_return = returns.mean() * 252  # Would need market benchmark
        tracking_error = returns.std() * np.sqrt(252)  # Simplified
        information_ratio = (annualized_return - market_return) / tracking_error if tracking_error > 0 else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
    
    def garch_modeling(self, 
                      returns: pd.Series,
                      model_type: str = "GARCH",
                      order: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
        """
        GARCH volatility modeling.
        
        Args:
            returns: Return series
            model_type: "GARCH", "EGARCH", "GJR-GARCH"
            order: (p, q) order for GARCH model
        """
        try:
            # Remove any null values
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                raise ValueError("Insufficient data for GARCH modeling")
            
            # Fit GARCH model
            model_spec = arch_model(returns_clean, vol=model_type, p=order[0], q=order[1])
            fitted_model = model_spec.fit(disp='off')
            
            # Extract results
            volatility_forecast = fitted_model.forecast(horizon=10)
            volatility_forecast_values = volatility_forecast.variance.values[-1, :]
            mean_forecast = volatility_forecast.mean.values[-1, :]
            
            # Model diagnostics
            aic = fitted_model.aic
            bic = fitted_model.bic
            log_likelihood = fitted_model.loglikelihood
            
            # Ljung-Box test for residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            residuals = fitted_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            return {
                'model_type': model_type,
                'order': order,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'volatility_forecast': volatility_forecast_values,
                'mean_forecast': mean_forecast,
                'model_summary': str(fitted_model.summary()),
                'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
                'arch_effects_pvalue': ljung_box['lb_pvalue'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"GARCH modeling failed: {e}")
            return {
                'error': str(e),
                'model_type': model_type,
                'order': order
            }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def correlation_analysis(self, 
                           returns: pd.DataFrame,
                           rolling_window: int = 60) -> Dict[str, Any]:
        """
        Advanced correlation analysis with rolling correlations and regime detection.
        
        Args:
            returns: DataFrame of asset returns
            rolling_window: Window for rolling correlations
        """
        # Rolling correlation matrix
        rolling_corr = returns.rolling(window=rolling_window).corr()
        
        # Average correlation
        avg_corr_matrix = returns.corr()
        
        # Diversification ratio
        portfolio_weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_vol = np.sqrt(np.dot(portfolio_weights, np.dot(avg_corr_matrix, portfolio_weights)) * 252)
        average_vol = np.sqrt(np.mean(np.diag(avg_corr_matrix) * 252))
        diversification_ratio = average_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Correlation regime detection using clustering
        # Flatten correlation matrix (excluding diagonal)
        corr_values = []
        for i in range(len(returns.columns)):
            for j in range(i + 1, len(returns.columns)):
                corr_values.append(avg_corr_matrix.iloc[i, j])
        
        # Detect correlation regimes
        if len(corr_values) > 10:
            # Use mixture model to detect correlation regimes
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(np.array(corr_values).reshape(-1, 1))
            
            regime_means = gmm.means_.flatten()
            regime_weights = gmm.weights_
            regime_std = np.sqrt(gmm.covariances_.flatten())
            
            # Classify regimes
            regimes = ['low', 'medium', 'high']
            regime_classification = {
                regimes[np.argmin(regime_means)]: 'Low correlation regime',
                regimes[np.argmax(regime_means)]: 'High correlation regime'
            }
        else:
            regime_means = np.array([0])
            regime_weights = np.array([1])
            regime_classification = {'single_regime': 'Single correlation regime'}
        
        return {
            'correlation_matrix': avg_corr_matrix,
            'rolling_correlations': rolling_corr,
            'diversification_ratio': diversification_ratio,
            'regime_means': regime_means,
            'regime_weights': regime_weights,
            'regime_classification': regime_classification,
            'average_correlation': np.mean(corr_values),
            'max_correlation': np.max(corr_values),
            'min_correlation': np.min(corr_values)
        }
    
    def stress_testing(self,
                      portfolio_returns: pd.Series,
                      stress_scenarios: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Perform portfolio stress testing.
        
        Args:
            portfolio_returns: Historical portfolio returns
            stress_scenarios: Custom stress scenarios
        """
        if stress_scenarios is None:
            # Default stress scenarios
            stress_scenarios = [
                {'name': 'Market Crash', 'shock': -0.20, 'volatility_multiplier': 2.0},
                {'name': 'Interest Rate Spike', 'shock': -0.10, 'volatility_multiplier': 1.5},
                {'name': 'Inflation Shock', 'shock': -0.15, 'volatility_multiplier': 1.8},
                {'name': 'Credit Crisis', 'shock': -0.25, 'volatility_multiplier': 2.5}
            ]
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            scenario_name = scenario['name']
            shock = scenario['shock']
            vol_multiplier = scenario['volatility_multiplier']
            
            # Calculate stressed returns
            current_vol = portfolio_returns.std()
            stressed_vol = current_vol * vol_multiplier
            
            # Generate stressed return distribution
            np.random.seed(42)
            stressed_returns = np.random.normal(shock, stressed_vol, 10000)
            
            # Calculate portfolio impact
            portfolio_value = 1000000  # $1M portfolio
            stressed_values = portfolio_value * (1 + stressed_returns)
            
            # Calculate metrics
            var_95 = np.percentile(stressed_values, 5)
            cvar_95 = np.mean(stressed_values[stressed_values <= var_95])
            max_loss = portfolio_value - min(stressed_values)
            
            stress_results[scenario_name] = {
                'expected_return': shock,
                'volatility': stressed_vol,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_loss': max_loss,
                'probability_loss': np.mean(stressed_returns < 0),
                'expected_shortfall': portfolio_value - cvar_95
            }
        
        # Summary statistics
        worst_scenario = min(stress_results.keys(), 
                           key=lambda x: stress_results[x]['var_95'])
        best_scenario = max(stress_results.keys(), 
                          key=lambda x: stress_results[x]['var_95'])
        
        return {
            'stress_scenarios': stress_results,
            'worst_case_scenario': worst_scenario,
            'best_case_scenario': best_scenario,
            'stress_summary': {
                'average_var_95': np.mean([s['var_95'] for s in stress_results.values()]),
                'worst_var_95': stress_results[worst_scenario]['var_95'],
                'stress_correlation': 'stress testing completed'
            }
        }
    
    def backtesting_framework(self,
                            strategy_returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = None) -> Dict[str, Any]:
        """
        Comprehensive backtesting framework for trading strategies.
        
        Args:
            strategy_returns: Returns from trading strategy
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for calculations
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Basic performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
        annualized_volatility = strategy_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Information ratio if benchmark provided
        if benchmark_returns is not None:
            excess_returns = strategy_returns - benchmark_returns
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            tracking_error = excess_returns.std() * np.sqrt(252)
            alpha = excess_returns.mean() * 252  # Annualized alpha
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            information_ratio = None
            tracking_error = None
            alpha = None
            beta = None
        
        # Drawdown analysis
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean()
        drawdown_duration = self._calculate_drawdown_duration(drawdowns)
        
        # Performance ratios
        sortino_ratio = self._calculate_sortino_ratio(strategy_returns, risk_free_rate)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/Loss analysis
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        win_rate = len(positive_returns) / len(strategy_returns)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Risk metrics
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        return {
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'alpha': alpha,
                'beta': beta
            },
            'risk_metrics': {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'max_drawdown_duration': drawdown_duration['max'],
                'avg_drawdown_duration': drawdown_duration['avg']
            },
            'trading_statistics': {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'total_trades': len(strategy_returns),
                'positive_periods': len(positive_returns),
                'negative_periods': len(negative_returns)
            },
            'drawdown_analysis': drawdowns.to_dict(),
            'cumulative_returns': cumulative_returns.to_dict()
        }
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown duration statistics."""
        in_drawdown = drawdowns < 0
        
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return {
            'max': max(durations) if durations else 0,
            'avg': np.mean(durations) if durations else 0,
            'count': len(durations)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < risk_free_rate]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        excess_return = returns.mean() * 252 - risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0
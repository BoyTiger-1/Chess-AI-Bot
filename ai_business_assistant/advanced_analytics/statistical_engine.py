"""
Advanced Statistical Analysis Engine
Provides comprehensive statistical testing, correlation analysis, and hypothesis testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, mannwhitneyu
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
import warnings

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Statistical test types."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    MANN_WHITNEY_U = "mann_whitney_u"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    COINTEGRATION = "cointegration"
    STATIONARITY = "stationarity"
    DURBIN_WATSON = "durbin_watson"


@dataclass
class HypothesisTest:
    """Represents a statistical hypothesis test result."""
    test_type: TestType
    null_hypothesis: str
    alternative_hypothesis: str
    p_value: float
    statistic: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.05
    is_significant: bool = field(init=False)
    effect_size: Optional[float] = None
    power: Optional[float] = None
    sample_size: int = 0
    degrees_freedom: Optional[int] = None
    test_details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate significance after initialization."""
        self.is_significant = self.p_value < self.confidence_level


@dataclass 
class CorrelationResult:
    """Correlation analysis result."""
    method: str
    correlation_matrix: pd.DataFrame
    p_values: pd.DataFrame
    confidence_interval: pd.DataFrame
    significant_pairs: List[Tuple[str, str]]
    effect_sizes: Dict[str, Dict[str, float]]


@dataclass
class RegressionResult:
    """Regression analysis result."""
    model_type: str
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_p_value: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    model_summary: Dict[str, Any]


@dataclass
class TimeSeriesResult:
    """Time series analysis result."""
    series_name: str
    is_stationary: bool
    stationarity_test: str
    stationarity_p_value: float
    trend: Optional[np.ndarray] = None
    seasonal_component: Optional[np.ndarray] = None
    residual_component: Optional[np.ndarray] = None
    decomposition_details: Dict[str, Any] = field(default_factory=dict)


class StatisticalAnalyzer:
    """Advanced statistical analysis engine."""
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def t_test(self, 
               sample1: Union[List[float], np.ndarray, pd.Series],
               sample2: Union[List[float], np.ndarray, pd.Series],
               test_type: str = "two_sample",
               equal_var: bool = False) -> HypothesisTest:
        """
        Perform t-test analysis.
        
        Args:
            sample1, sample2: Data samples
            test_type: "one_sample", "two_sample", "paired"
            equal_var: Whether to assume equal variances (Welch's test if False)
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        
        if test_type == "one_sample":
            # One-sample t-test (compare sample1 to mean=0)
            statistic, p_value = stats.ttest_1samp(sample1, 0)
            null_hypothesis = "The sample mean equals 0"
            alternative_hypothesis = "The sample mean does not equal 0"
            
        elif test_type == "two_sample":
            # Two-sample t-test
            statistic, p_value = ttest_ind(sample1, sample2, equal_var=equal_var)
            null_hypothesis = "The means of the two samples are equal"
            alternative_hypothesis = "The means of the two samples are different"
            
        elif test_type == "paired":
            # Paired t-test
            statistic, p_value = stats.ttest_rel(sample1, sample2)
            null_hypothesis = "The mean difference between paired samples is 0"
            alternative_hypothesis = "The mean difference between paired samples is not 0"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) + 
                             (len(sample2) - 1) * np.var(sample2, ddof=1)) / 
                            (len(sample1) + len(sample2) - 2))
        effect_size = (np.mean(sample1) - np.mean(sample2)) / pooled_std if pooled_std > 0 else 0
        
        return HypothesisTest(
            test_type=TestType.T_TEST,
            null_hypothesis=null_hypothesis,
            alternative_hypothesis=alternative_hypothesis,
            p_value=p_value,
            statistic=statistic,
            effect_size=effect_size,
            sample_size=len(sample1) + len(sample2),
            degrees_freedom=len(sample1) + len(sample2) - 2
        )
    
    def chi_square_test(self, 
                       contingency_table: np.ndarray) -> HypothesisTest:
        """
        Perform chi-square test for independence.
        
        Args:
            contingency_table: 2D array of observed frequencies
        """
        if contingency_table.shape != (2, 2):
            # For larger contingency tables, use general chi-square
            statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        else:
            # Use Fisher's exact test for 2x2 tables when appropriate
            if min(contingency_table.sum(axis=0)) < 5 or min(contingency_table.sum(axis=1)) < 5:
                from scipy.stats import fisher_exact
                odds_ratio, p_value = fisher_exact(contingency_table)
                statistic = np.log(odds_ratio) if odds_ratio > 0 else 0
            else:
                statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return HypothesisTest(
            test_type=TestType.CHI_SQUARE,
            null_hypothesis="The variables are independent",
            alternative_hypothesis="The variables are dependent",
            p_value=p_value,
            statistic=statistic,
            degrees_freedom=dof if 'dof' in locals() else (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1),
            test_details={
                "expected_frequencies": expected.tolist() if 'expected' in locals() else None,
                "contingency_table": contingency_table.tolist()
            }
        )
    
    def anova_test(self, 
                   groups: List[Union[List[float], np.ndarray, pd.Series]],
                   test_type: str = "one_way") -> HypothesisTest:
        """
        Perform ANOVA (Analysis of Variance).
        
        Args:
            groups: List of samples to compare
            test_type: "one_way", "two_way", "repeated_measures"
        """
        groups = [np.array(group) for group in groups]
        
        if test_type == "one_way":
            statistic, p_value = f_oneway(*groups)
            null_hypothesis = "All group means are equal"
            alternative_hypothesis = "At least one group mean is different"
        else:
            raise NotImplementedError(f"ANOVA test type '{test_type}' not yet implemented")
        
        # Calculate effect size (eta squared)
        grand_mean = np.mean(np.concatenate(groups))
        ss_total = sum(sum((group - grand_mean) ** 2) for group in groups)
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        
        return HypothesisTest(
            test_type=TestType.ANOVA,
            null_hypothesis=null_hypothesis,
            alternative_hypothesis=alternative_hypothesis,
            p_value=p_value,
            statistic=statistic,
            effect_size=effect_size,
            sample_size=sum(len(group) for group in groups),
            degrees_freedom=len(groups) - 1
        )
    
    def non_parametric_tests(self, 
                           sample1: Union[List[float], np.ndarray, pd.Series],
                           sample2: Union[List[float], np.ndarray, pd.Series],
                           test_type: str = "mann_whitney") -> HypothesisTest:
        """
        Perform non-parametric tests.
        
        Args:
            sample1, sample2: Data samples
            test_type: "mann_whitney_u", "kruskal_wallis", "wilcoxon"
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        
        if test_type == "mann_whitney_u":
            statistic, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
            null_hypothesis = "The distributions of the two samples are identical"
            alternative_hypothesis = "The distributions are different"
        elif test_type == "wilcoxon":
            if len(sample1) == len(sample2):
                statistic, p_value = stats.wilcoxon(sample1, sample2)
                null_hypothesis = "The median difference between paired samples is 0"
                alternative_hypothesis = "The median difference is not 0"
            else:
                raise ValueError("Wilcoxon test requires paired samples of equal length")
        else:
            raise ValueError(f"Non-parametric test type '{test_type}' not recognized")
        
        return HypothesisTest(
            test_type=TestType.MANN_WHITNEY_U,
            null_hypothesis=null_hypothesis,
            alternative_hypothesis=alternative_hypothesis,
            p_value=p_value,
            statistic=statistic,
            sample_size=len(sample1) + len(sample2)
        )
    
    def correlation_analysis(self, 
                           data: pd.DataFrame,
                           method: str = "pearson") -> CorrelationResult:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            data: DataFrame with numeric columns
            method: "pearson", "spearman", "kendall"
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if method == "pearson":
            correlation_matrix = numeric_data.corr(method="pearson")
            # Calculate p-values for correlations
            n = len(numeric_data)
            p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
            
            for i in correlation_matrix.index:
                for j in correlation_matrix.columns:
                    if i != j:
                        r = correlation_matrix.loc[i, j]
                        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.inf
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        p_values.loc[i, j] = p_val
                    else:
                        p_values.loc[i, j] = 0.0
            
        elif method == "spearman":
            correlation_matrix = numeric_data.corr(method="spearman")
            # For Spearman correlation p-values
            p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
            from scipy.stats import spearmanr
            corr_matrix, p_matrix = spearmanr(numeric_data.values, axis=0)
            p_values = pd.DataFrame(p_matrix, 
                                   index=correlation_matrix.index, 
                                   columns=correlation_matrix.columns)
        else:
            raise ValueError(f"Correlation method '{method}' not supported")
        
        # Find significant correlations
        significant_pairs = []
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i < j:  # Avoid duplicates
                    p_val = p_values.loc[i, j]
                    if p_val < self.alpha:
                        significant_pairs.append((i, j))
        
        # Calculate confidence intervals for correlations
        n = len(numeric_data)
        confidence_interval = pd.DataFrame(index=correlation_matrix.index, 
                                         columns=correlation_matrix.columns)
        
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i != j:
                    r = correlation_matrix.loc[i, j]
                    # Fisher's z-transformation for confidence interval
                    z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
                    se = 1 / np.sqrt(n - 3)
                    z_critical = stats.norm.ppf(1 - self.alpha / 2)
                    
                    z_lower = z - z_critical * se
                    z_upper = z + z_critical * se
                    
                    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                    
                    confidence_interval.loc[i, j] = (r_lower, r_upper)
                else:
                    confidence_interval.loc[i, j] = (1.0, 1.0)
        
        # Calculate effect sizes (Cohen's conventions for correlation)
        effect_sizes = {}
        for i in correlation_matrix.index:
            effect_sizes[i] = {}
            for j in correlation_matrix.columns:
                if i != j:
                    r = abs(correlation_matrix.loc[i, j])
                    if r >= 0.5:
                        effect_sizes[i][j] = "large"
                    elif r >= 0.3:
                        effect_sizes[i][j] = "medium"
                    elif r >= 0.1:
                        effect_sizes[i][j] = "small"
                    else:
                        effect_sizes[i][j] = "negligible"
                else:
                    effect_sizes[i][j] = "perfect"
        
        return CorrelationResult(
            method=method,
            correlation_matrix=correlation_matrix,
            p_values=p_values,
            confidence_interval=confidence_interval,
            significant_pairs=significant_pairs,
            effect_sizes=effect_sizes
        )
    
    def regression_analysis(self, 
                          dependent_var: str,
                          independent_vars: List[str],
                          data: pd.DataFrame,
                          regression_type: str = "linear") -> RegressionResult:
        """
        Perform comprehensive regression analysis.
        
        Args:
            dependent_var: Name of dependent variable column
            independent_vars: List of independent variable columns
            data: DataFrame containing the data
            regression_type: "linear", "logistic", "polynomial", "ridge", "lasso"
        """
        y = data[dependent_var].values
        X = data[independent_vars].values
        var_names = independent_vars
        
        if regression_type == "linear":
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate statistics
            r_squared = r2_score(y, y_pred)
            n = len(y)
            k = len(independent_vars)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
            
            # Calculate F-statistic
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            f_stat = (ss_tot - ss_res) / k / (ss_res / (n - k - 1))
            f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
            
            # Calculate p-values for coefficients
            from scipy.stats import t
            
            # Standard errors (simplified calculation)
            mse = ss_res / (n - k - 1)
            var_coeff = mse * np.linalg.inv(X.T @ X).diagonal()
            std_errors = np.sqrt(var_coeff)
            t_stats = model.coef_ / std_errors
            p_values = 2 * (1 - t.cdf(np.abs(t_stats), n - k - 1))
            
            coefficients = {var_names[i]: model.coef_[i] for i in range(len(var_names))}
            p_value_dict = {var_names[i]: p_values[i] for i in range(len(var_names))}
            
        else:
            raise NotImplementedError(f"Regression type '{regression_type}' not yet implemented")
        
        return RegressionResult(
            model_type=regression_type,
            coefficients=coefficients,
            p_values=p_value_dict,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            f_statistic=f_stat,
            f_p_value=f_p_value,
            residuals=y - y_pred,
            fitted_values=y_pred,
            model_summary={
                "sample_size": n,
                "variables": var_names,
                "intercept": model.intercept_,
                "standard_errors": std_errors.tolist(),
                "t_statistics": t_stats.tolist(),
                "mean_squared_error": mse
            }
        )
    
    def cointegration_test(self, 
                          series1: pd.Series, 
                          series2: pd.Series,
                          method: str = "engle_granger") -> HypothesisTest:
        """
        Test for cointegration between time series.
        
        Args:
            series1, series2: Time series to test
            method: "engle_granger", "johansen"
        """
        # Ensure series are aligned
        combined = pd.concat([series1, series2], axis=1).dropna()
        y1, y2 = combined.iloc[:, 0], combined.iloc[:, 1]
        
        if method == "engle_granger":
            # Engle-Granger test
            coint_t, p_value, critical_values = coint(y1.values, y2.values)
            
            return HypothesisTest(
                test_type=TestType.COINTEGRATION,
                null_hypothesis="The series are not cointegrated",
                alternative_hypothesis="The series are cointegrated",
                p_value=p_value,
                statistic=coint_t,
                critical_value=critical_values[1],  # 5% critical value
                test_details={
                    "method": "Engle-Granger",
                    "critical_values": critical_values.tolist(),
                    "combined_series_length": len(combined)
                }
            )
        else:
            raise NotImplementedError(f"Cointegration test method '{method}' not yet implemented")
    
    def stationarity_test(self, 
                         series: pd.Series,
                         method: str = "adf") -> HypothesisTest:
        """
        Test for stationarity in time series.
        
        Args:
            series: Time series to test
            method: "adf", "kpss"
        """
        if method == "adf":
            # Augmented Dickey-Fuller test
            result = adfuller(series.dropna())
            statistic, p_value, used_lags, n_obs, critical_values, icbest = result
            
            return HypothesisTest(
                test_type=TestType.STATIONARITY,
                null_hypothesis="The series has a unit root (non-stationary)",
                alternative_hypothesis="The series is stationary",
                p_value=p_value,
                statistic=statistic,
                critical_value=critical_values['5%'],
                test_details={
                    "method": "Augmented Dickey-Fuller",
                    "used_lags": used_lags,
                    "n_obs": n_obs,
                    "critical_values": critical_values,
                    "icbest": icbest
                }
            )
            
        elif method == "kpss":
            # KPSS test
            statistic, p_value, lags, critical_values = kpss(series.dropna())
            
            return HypothesisTest(
                test_type=TestType.STATIONARITY,
                null_hypothesis="The series is stationary",
                alternative_hypothesis="The series has a unit root (non-stationary)",
                p_value=p_value,
                statistic=statistic,
                critical_value=critical_values['5%'],
                test_details={
                    "method": "KPSS",
                    "lags": lags,
                    "critical_values": critical_values
                }
            )
        else:
            raise ValueError(f"Stationarity test method '{method}' not supported")
    
    def durbin_watson_test(self, residuals: np.ndarray) -> HypothesisTest:
        """
        Perform Durbin-Watson test for autocorrelation.
        
        Args:
            residuals: Array of regression residuals
        """
        dw_stat = durbin_watson(residuals)
        
        # Durbin-Watson test for autocorrelation
        # Value around 2.0 indicates no autocorrelation
        # Values < 2.0 indicate positive autocorrelation
        # Values > 2.0 indicate negative autocorrelation
        
        # Approximate p-value calculation (simplified)
        n = len(residuals)
        dw_mean = 2 * (n - 1) / (n - 1) if n > 1 else 2
        
        # Simple heuristic for significance
        if dw_stat < 1.5 or dw_stat > 2.5:
            is_significant = True
            p_value = 0.01  # Strong indication of autocorrelation
        elif dw_stat < 1.8 or dw_stat > 2.2:
            p_value = 0.05  # Moderate indication
        else:
            p_value = 0.10  # Weak or no indication
            is_significant = False
        
        return HypothesisTest(
            test_type=TestType.DURBIN_WATSON,
            null_hypothesis="No autocorrelation in residuals",
            alternative_hypothesis="Autocorrelation exists in residuals",
            p_value=p_value,
            statistic=dw_stat,
            test_details={
                "interpretation": "Values near 2.0 indicate no autocorrelation",
                "positive_autocorr": dw_stat < 2.0,
                "negative_autocorr": dw_stat > 2.0
            }
        )
    
    def granger_causality_test(self, 
                             series1: pd.Series, 
                             series2: pd.Series,
                             max_lags: int = 5) -> HypothesisTest:
        """
        Test for Granger causality between two time series.
        
        Args:
            series1, series2: Time series to test
            max_lags: Maximum number of lags to test
        """
        # Combine series
        data = pd.concat([series1, series2], axis=1).dropna()
        data.columns = ['series1', 'series2']
        
        # Granger causality test using statsmodels
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Test if series1 Granger-causes series2
            gc_result = grangercausalitytests(data[['series2', 'series1']], maxlag=max_lags, verbose=False)
            
            # Get F-statistic and p-value for lag 1
            f_stat = gc_result[1][0]['ssr_ftest'][0]
            p_value = gc_result[1][0]['ssr_ftest'][1]
            
            return HypothesisTest(
                test_type=TestType.CORRELATION,  # Using existing enum
                null_hypothesis="series1 does not Granger-cause series2",
                alternative_hypothesis="series1 Granger-causes series2",
                p_value=p_value,
                statistic=f_stat,
                test_details={
                    "method": "Granger Causality",
                    "direction": "series1 -> series2",
                    "max_lags": max_lags,
                    "full_results": {k: v[0]['ssr_ftest'] for k, v in gc_result.items()}
                }
            )
        except ImportError:
            raise ImportError("statsmodels required for Granger causality test")
    
    def multiple_comparison_correction(self, 
                                     p_values: List[float],
                                     method: str = "bonferroni") -> Dict[str, Any]:
        """
        Correct p-values for multiple comparisons.
        
        Args:
            p_values: List of raw p-values
            method: "bonferroni", "holm", "benjamini_hochberg"
        """
        p_array = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            corrected_p = np.minimum(p_array * n, 1.0)
        elif method == "holm":
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.zeros(n)
            for i in range(n):
                corrected_p[sorted_indices[i]] = min(1.0, sorted_p[i] * (n - i))
        elif method == "benjamini_hochberg":
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.zeros(n)
            for i in range(n - 1, -1, -1):
                if i == n - 1:
                    corrected_p[sorted_indices[i]] = sorted_p[i]
                else:
                    corrected_p[sorted_indices[i]] = min(
                        sorted_p[i] * n / (i + 1),
                        corrected_p[sorted_indices[i + 1]]
                    )
        else:
            raise ValueError(f"Multiple comparison correction method '{method}' not supported")
        
        return {
            "method": method,
            "original_p_values": p_values,
            "corrected_p_values": corrected_p.tolist(),
            "significant_before": np.sum(p_array < self.alpha),
            "significant_after": np.sum(corrected_p < self.alpha),
            "alpha": self.alpha
        }
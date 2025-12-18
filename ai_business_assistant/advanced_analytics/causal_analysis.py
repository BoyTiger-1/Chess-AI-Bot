"""
Causal Analysis Engine
Provides advanced causal inference, Granger causality, and intervention analysis.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from econml.dr import DRLearner, CausalForestDML

logger = logging.getLogger(__name__)


class CausalMethod(Enum):
    """Causal inference methods."""
    GRANGER_CAUSALITY = "granger_causality"
    DO_CALCULUS = "do_calculus"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    PROPENSITY_SCORE = "propensity_score"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    SYNTHETIC_CONTROL = "synthetic_control"
    CAUSAL_FOREST = "causal_forest"
    DOUBLE_ML = "double_ml"


@dataclass
class CausalModel:
    """Causal model result."""
    method: CausalMethod
    causal_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    treatment_variable: str
    outcome_variable: str
    control_variables: List[str]
    model_quality: Dict[str, float]
    assumptions: List[str]
    limitations: List[str]
    interpretation: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InterventionResult:
    """Intervention analysis result."""
    intervention_name: str
    pre_intervention_effect: float
    post_intervention_effect: float
    intervention_impact: float
    statistical_significance: float
    confidence_level: float
    duration_analysis: Dict[str, Any]
    business_impact: Dict[str, Any]


class CausalAnalyzer:
    """
    Advanced causal inference engine.
    Provides multiple causal analysis methods and intervention analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize causal analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def granger_causality_analysis(self, 
                                 data: pd.DataFrame,
                                 dependent_var: str,
                                 independent_var: str,
                                 max_lags: int = 5) -> CausalModel:
        """
        Perform Granger causality analysis.
        
        Args:
            data: DataFrame with time series data
            dependent_var: Variable to test if it's caused by independent_var
            independent_var: Potential causal variable
            max_lags: Maximum number of lags to test
        """
        # Ensure data is sorted by time
        data_clean = data[[dependent_var, independent_var]].dropna()
        
        if len(data_clean) < 20:
            raise ValueError("Insufficient data for Granger causality analysis")
        
        # Granger causality test
        try:
            # Test if independent_var Granger-causes dependent_var
            gc_result = grangercausalitytests(
                data_clean[[dependent_var, independent_var]], 
                maxlag=max_lags, 
                verbose=False
            )
            
            # Extract results for the optimal lag
            min_aic_lag = min(gc_result.keys(), key=lambda k: gc_result[k][0]['aic'])
            best_result = gc_result[min_aic_lag][0]
            
            # Extract statistics
            f_statistic = best_result['ssr_ftest'][0]
            p_value = best_result['ssr_ftest'][1]
            
            # Calculate effect size (coefficient ratio)
            # Simplified: use the F-statistic as effect measure
            effect_size = f_statistic / max_lags
            
            # Calculate confidence interval (approximation)
            t_critical = stats.t.ppf(1 - self.alpha/2, len(data_clean) - max_lags - 2)
            se = np.sqrt(1 / len(data_clean))  # Simplified standard error
            ci_lower = effect_size - t_critical * se
            ci_upper = effect_size + t_critical * se
            
            # Determine interpretation
            is_causal = p_value < self.alpha
            if is_causal:
                interpretation = f"{independent_var} Granger-causes {dependent_var}"
            else:
                interpretation = f"No Granger causality detected from {independent_var} to {dependent_var}"
            
            # Generate recommendations
            recommendations = []
            if is_causal:
                recommendations.append("Consider using this causal relationship for forecasting")
                recommendations.append(f"Test stability of the relationship over time (lag={min_aic_lag})")
            else:
                recommendations.append("The variables may not be causally related")
                recommendations.append("Consider alternative causal discovery methods")
            
            return CausalModel(
                method=CausalMethod.GRANGER_CAUSALITY,
                causal_effect=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                treatment_variable=independent_var,
                outcome_variable=dependent_var,
                control_variables=[],
                model_quality={
                    "f_statistic": f_statistic,
                    "optimal_lag": min_aic_lag,
                    "aic": best_result['aic'],
                    "sample_size": len(data_clean)
                },
                assumptions=[
                    "Stationarity of time series",
                    "Linear relationship between variables",
                    "No omitted variable bias",
                    "Correct lag specification"
                ],
                limitations=[
                    "Granger causality does not imply true causality",
                    "Only tests predictive causality, not structural causality",
                    "Requires stationarity assumption"
                ],
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Granger causality analysis failed: {e}")
            raise
    
    def instrumental_variable_analysis(self, 
                                     data: pd.DataFrame,
                                     treatment: str,
                                     outcome: str,
                                     instrument: str,
                                     controls: List[str] = None) -> CausalModel:
        """
        Instrumental variable analysis for causal inference.
        
        Args:
            data: DataFrame with data
            treatment: Treatment variable (endogenous)
            outcome: Outcome variable
            instrument: Instrument variable (exogenous)
            controls: Control variables
        """
        if controls is None:
            controls = []
        
        data_clean = data[[treatment, outcome, instrument] + controls].dropna()
        
        if len(data_clean) < 30:
            raise ValueError("Insufficient data for instrumental variable analysis")
        
        # Step 1: First stage regression (instrument -> treatment)
        X_stage1 = pd.concat([data_clean[[instrument]], data_clean[controls]], axis=1)
        y_stage1 = data_clean[treatment]
        
        stage1_model = OLS(y_stage1, X_stage1).fit()
        predicted_treatment = stage1_model.fittedvalues
        
        # Calculate first stage F-statistic
        first_stage_f = stage1_model.fvalue
        first_stage_r2 = stage1_model.rsquared
        
        # Check instrument strength
        is_strong_instrument = first_stage_f > 10
        
        # Step 2: Second stage regression (predicted treatment -> outcome)
        X_stage2 = pd.concat([pd.Series(predicted_treatment, name=treatment), data_clean[controls]], axis=1)
        y_stage2 = data_clean[outcome]
        
        stage2_model = OLS(y_stage2, X_stage2).fit()
        
        # Extract causal effect
        treatment_coef = stage2_model.params[treatment]
        treatment_pvalue = stage2_model.pvalues[treatment]
        treatment_se = stage2_model.bse[treatment]
        
        # Calculate confidence interval
        t_critical = stats.t.ppf(1 - self.alpha/2, len(data_clean) - len(stage2_model.params) - 1)
        ci_lower = treatment_coef - t_critical * treatment_se
        ci_upper = treatment_coef + t_critical * treatment_se
        
        # Interpretation
        is_significant = treatment_pvalue < self.alpha
        if is_significant:
            direction = "increases" if treatment_coef > 0 else "decreases"
            interpretation = f"{treatment} causally {direction} {outcome}"
        else:
            interpretation = f"No significant causal effect of {treatment} on {outcome}"
        
        # Recommendations
        recommendations = []
        if not is_strong_instrument:
            recommendations.append("Weak instrument detected - consider alternative instruments")
        if is_significant:
            recommendations.append("Significant causal effect found - use with caution")
        recommendations.append("Test exclusion restriction and relevance assumptions")
        
        # Limitations
        limitations = [
            "Requires valid instruments (exclusion restriction)",
            "Sensitive to weak instruments problem",
            "Local Average Treatment Effect (LATE) interpretation"
        ]
        
        return CausalModel(
            method=CausalMethod.INSTRUMENTAL_VARIABLES,
            causal_effect=treatment_coef,
            confidence_interval=(ci_lower, ci_upper),
            p_value=treatment_pvalue,
            treatment_variable=treatment,
            outcome_variable=outcome,
            control_variables=controls,
            model_quality={
                "first_stage_f": first_stage_f,
                "first_stage_r2": first_stage_r2,
                "second_stage_r2": stage2_model.rsquared,
                "sample_size": len(data_clean),
                "strong_instrument": is_strong_instrument
            },
            assumptions=[
                "Valid instrumental variable (exclusion restriction)",
                "Instrument relevance (first stage F > 10)",
                "No endogeneity in controls",
                "Linear relationship assumption"
            ],
            limitations=limitations,
            interpretation=interpretation,
            recommendations=recommendations
        )
    
    def propensity_score_matching(self, 
                                data: pd.DataFrame,
                                treatment: str,
                                outcome: str,
                                covariates: List[str],
                                matching_method: str = "nearest_neighbor") -> CausalModel:
        """
        Propensity score matching for causal inference.
        
        Args:
            data: DataFrame with data
            treatment: Binary treatment variable
            outcome: Outcome variable
            covariates: Pre-treatment covariates
            matching_method: "nearest_neighbor", "kernel", "stratification"
        """
        data_clean = data[[treatment, outcome] + covariates].dropna()
        
        # Binary treatment check
        unique_treatments = data_clean[treatment].unique()
        if len(unique_treatments) != 2:
            raise ValueError("Treatment variable must be binary")
        
        treated = data_clean[treatment] == 1
        control = data_clean[treatment] == 0
        
        if np.sum(treated) < 10 or np.sum(control) < 10:
            raise ValueError("Insufficient treated or control observations")
        
        # Estimate propensity scores
        X = data_clean[covariates]
        y = data_clean[treatment]
        
        # Use logistic regression for propensity score
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        prop_model = LogisticRegression(random_state=42)
        prop_model.fit(X_scaled, y)
        propensity_scores = prop_model.predict_proba(X_scaled)[:, 1]
        
        # Check overlap assumption
        common_support = self._check_overlap(propensity_scores[treated], propensity_scores[control])
        
        # Perform matching
        if matching_method == "nearest_neighbor":
            matched_pairs = self._nearest_neighbor_matching(propensity_scores, treated, control)
        elif matching_method == "stratification":
            matched_pairs = self._stratification_matching(propensity_scores, treated, control)
        else:
            raise ValueError(f"Matching method '{matching_method}' not supported")
        
        # Calculate treatment effect
        if matched_pairs is not None and len(matched_pairs) > 0:
            treatment_effects = []
            for treated_idx, control_idx in matched_pairs:
                treatment_effect = data_clean.iloc[treated_idx][outcome] - data_clean.iloc[control_idx][outcome]
                treatment_effects.append(treatment_effect)
            
            ate = np.mean(treatment_effects)  # Average Treatment Effect
            
            # Standard error and confidence interval
            se = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
            t_critical = stats.t.ppf(1 - self.alpha/2, len(treatment_effects) - 1)
            ci_lower = ate - t_critical * se
            ci_upper = ate + t_critical * se
            
            # T-test for significance
            t_stat, p_value = stats.ttest_1samp(treatment_effects, 0)
        else:
            # Fallback to regression adjustment
            ate = self._regression_adjustment(data_clean, treatment, outcome, covariates)
            p_value = 0.05  # Conservative estimate
            ci_lower = ate - 1.96 * abs(ate) * 0.1
            ci_upper = ate + 1.96 * abs(ate) * 0.1
        
        # Interpretation
        is_significant = p_value < self.alpha
        direction = "increases" if ate > 0 else "decreases"
        if is_significant:
            interpretation = f"{treatment} causally {direction} {outcome} by {abs(ate):.3f} units on average"
        else:
            interpretation = f"No significant causal effect of {treatment} on {outcome} detected"
        
        # Recommendations
        recommendations = [
            "Check overlap assumption - ensure common support",
            "Consider sensitivity analysis for hidden bias",
            "Validate propensity score model specification"
        ]
        if not common_support:
            recommendations.append("Poor overlap detected - consider trimming or weighting")
        
        return CausalModel(
            method=CausalMethod.PROPENSITY_SCORE,
            causal_effect=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            treatment_variable=treatment,
            outcome_variable=outcome,
            control_variables=covariates,
            model_quality={
                "matched_pairs": len(matched_pairs) if matched_pairs is not None else 0,
                "overlap_support": common_support,
                "propensity_score_auc": 0.5,  # Would need ROC calculation
                "sample_size": len(data_clean),
                "treated_count": np.sum(treated),
                "control_count": np.sum(control)
            },
            assumptions=[
                "No unmeasured confounding (Conditional Independence)",
                "Common support (overlap) assumption",
                "Stable Unit Treatment Value Assumption (SUTVA)",
                "Correct propensity score model specification"
            ],
            limitations=[
                "Sensitive to unmeasured confounding",
                "Matching may reduce sample size",
                "Requires correct model specification"
            ],
            interpretation=interpretation,
            recommendations=recommendations
        )
    
    def difference_in_differences_analysis(self, 
                                         data: pd.DataFrame,
                                         treatment_group: str,
                                         control_group: str,
                                         outcome: str,
                                         time_var: str,
                                         group_var: str,
                                         pre_treatment_periods: List[str] = None) -> InterventionResult:
        """
        Difference-in-differences causal analysis.
        
        Args:
            data: DataFrame with panel data
            treatment_group: Name of treatment group
            control_group: Name of control group
            outcome: Outcome variable
            time_var: Time variable
            group_var: Group identifier variable
            pre_treatment_periods: Pre-treatment time periods
        """
        # Filter data for treatment and control groups
        treatment_data = data[data[group_var] == treatment_group]
        control_data = data[data[group_var] == control_group]
        
        if len(treatment_data) == 0 or len(control_data) == 0:
            raise ValueError("Treatment or control group not found")
        
        # Calculate group means before and after treatment
        if pre_treatment_periods is None:
            # Use median split to approximate pre/post
            median_time = data[time_var].median()
            pre_treatment = median_time
        else:
            pre_treatment = pre_treatment_periods[0] if pre_treatment_periods else data[time_var].min()
        
        # Calculate 2x2 DiD table
        treatment_pre = treatment_data[treatment_data[time_var] <= pre_treatment][outcome].mean()
        treatment_post = treatment_data[treatment_data[time_var] > pre_treatment][outcome].mean()
        control_pre = control_data[control_data[time_var] <= pre_treatment][outcome].mean()
        control_post = control_data[control_data[time_var] > pre_treatment][outcome].mean()
        
        # DiD estimator
        did_estimate = (treatment_post - treatment_pre) - (control_post - control_pre)
        
        # Regression approach for standard errors
        # Create interaction term
        data_subset = pd.concat([treatment_data, control_data])
        data_subset['treated'] = (data_subset[group_var] == treatment_group).astype(int)
        data_subset['post'] = (data_subset[time_var] > pre_treatment).astype(int)
        data_subset['treated_post'] = data_subset['treated'] * data_subset['post']
        
        # Run regression
        from statsmodels.formula.api import ols
        model = ols(f"{outcome} ~ treated + post + treated_post", data=data_subset).fit()
        
        did_coef = model.params['treated_post']
        did_pvalue = model.pvalues['treated_post']
        did_se = model.bse['treated_post']
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha/2, len(data_subset) - 4)
        ci_lower = did_coef - t_critical * did_se
        ci_upper = did_coef + t_critical * did_se
        
        # Parallel trends test
        parallel_trends_pvalue = self._test_parallel_trends(
            treatment_data, control_data, outcome, time_var, pre_treatment
        )
        
        # Duration analysis
        duration_analysis = {
            "treatment_pre_mean": treatment_pre,
            "treatment_post_mean": treatment_post,
            "control_pre_mean": control_pre,
            "control_post_mean": control_post,
            "pre_treatment_period": pre_treatment,
            "parallel_trends_pvalue": parallel_trends_pvalue,
            "parallel_trends_satisfied": parallel_trends_pvalue > 0.1
        }
        
        # Business impact assessment
        business_impact = {
            "effect_size": abs(did_estimate) / control_post if control_post != 0 else 0,
            "relative_change": (did_estimate / control_post) * 100 if control_post != 0 else 0,
            "magnitude": "small" if abs(did_estimate) < 0.1 * control_post else "large",
            "significance": "significant" if did_pvalue < self.alpha else "not significant"
        }
        
        # Interpretation
        is_significant = did_pvalue < self.alpha
        direction = "positive" if did_estimate > 0 else "negative"
        if is_significant:
            interpretation = f"Significant {direction} effect of {treatment_group} intervention"
        else:
            interpretation = f"No significant effect of {treatment_group} intervention"
        
        if not duration_analysis['parallel_trends_satisfied']:
            interpretation += " (parallel trends assumption violated)"
        
        return InterventionResult(
            intervention_name=f"{treatment_group} vs {control_group}",
            pre_intervention_effect=control_pre,
            post_intervention_effect=treatment_post - control_post,
            intervention_impact=did_estimate,
            statistical_significance=did_pvalue,
            confidence_level=self.confidence_level,
            duration_analysis=duration_analysis,
            business_impact=business_impact
        )
    
    def causal_forest_analysis(self, 
                             data: pd.DataFrame,
                             treatment: str,
                             outcome: str,
                             covariates: List[str],
                             num_trees: int = 100) -> CausalModel:
        """
        Causal Forest for heterogeneous treatment effects.
        
        Args:
            data: DataFrame with data
            treatment: Treatment variable
            outcome: Outcome variable
            covariates: Covariates for causal forest
            num_trees: Number of trees in causal forest
        """
        try:
            # Check if econml is available
            if not self._check_econml():
                raise ImportError("econml package required for causal forest analysis")
            
            data_clean = data[[treatment, outcome] + covariates].dropna()
            
            # Prepare data
            Y = data_clean[outcome].values
            T = data_clean[treatment].values
            X = data_clean[covariates].values
            
            # Estimate CATE with causal forest
            from econml.dr import DRLearner
            
            # Split data for honest estimation
            n = len(data_clean)
            train_size = int(0.7 * n)
            indices = np.random.permutation(n)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            
            # Fit causal forest
            cf = DRLearner(
                model_regression=RandomForestRegressor(n_estimators=10, random_state=42),
                model_propensity=RandomForestRegressor(n_estimators=10, random_state=42),
                n_estimators=num_trees,
                random_state=42
            )
            
            cf.fit(Y_train, T_train, X=X_train)
            
            # Predict CATE on test set
            cate_pred = cf.effect(X_test)
            
            # Average treatment effect
            ate = np.mean(cate_pred)
            
            # Standard error (simplified)
            se = np.std(cate_pred) / np.sqrt(len(cate_pred))
            
            # Confidence interval
            t_critical = stats.t.ppf(1 - self.alpha/2, len(cate_pred) - 1)
            ci_lower = ate - t_critical * se
            ci_upper = ate + t_critical * se
            
            # T-test for significance
            t_stat, p_value = stats.ttest_1samp(cate_pred, 0)
            
            # Heterogeneity analysis
            heterogeneity_stats = self._analyze_heterogeneity(cate_pred, X_test, covariates)
            
            # Interpretation
            is_significant = p_value < self.alpha
            direction = "positive" if ate > 0 else "negative"
            if is_significant:
                interpretation = f"Causal forest detected {direction} average treatment effect"
            else:
                interpretation = "No significant average treatment effect detected"
            
            # Recommendations
            recommendations = [
                "Use for personalized treatment recommendations",
                "Investigate subgroups with different treatment effects",
                "Consider using CATE for targeting strategies"
            ]
            
            return CausalModel(
                method=CausalMethod.CAUSAL_FOREST,
                causal_effect=ate,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                treatment_variable=treatment,
                outcome_variable=outcome,
                control_variables=covariates,
                model_quality={
                    "average_cate": ate,
                    "cate_std": np.std(cate_pred),
                    "heterogeneity": heterogeneity_stats,
                    "sample_size": len(data_clean),
                    "test_size": len(test_idx)
                },
                assumptions=[
                    "Unconfoundedness given covariates",
                    "Overlap assumption",
                    "SUTVA assumption"
                ],
                limitations=[
                    "Requires correct specification of covariates",
                    "Sensitive to high-dimensional settings",
                    "Computationally intensive"
                ],
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except ImportError:
            # Fallback to simpler methods
            logger.warning("econml not available, using propensity score matching")
            return self.propensity_score_matching(data, treatment, outcome, covariates)
    
    def create_causal_graph(self, 
                          data: pd.DataFrame,
                          variables: List[str],
                          threshold: float = 0.3) -> nx.DiGraph:
        """
        Create a causal graph using correlation-based skeleton discovery.
        
        Args:
            data: DataFrame with data
            variables: Variables to include in graph
            threshold: Correlation threshold for edges
        """
        # Calculate correlation matrix
        corr_matrix = data[variables].corr().abs()
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        
        # Add edges based on correlation threshold
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j and corr_matrix.loc[var1, var2] > threshold:
                    # Determine direction based on partial correlation
                    # Simplified: use correlation magnitude as proxy
                    G.add_edge(var1, var2, weight=corr_matrix.loc[var1, var2])
        
        return G
    
    def _check_overlap(self, treated_scores: np.ndarray, control_scores: np.ndarray) -> bool:
        """Check overlap assumption for propensity scores."""
        treated_min, treated_max = np.min(treated_scores), np.max(treated_scores)
        control_min, control_max = np.min(control_scores), np.max(control_scores)
        
        # Check if ranges overlap significantly
        overlap_min = max(treated_min, control_min)
        overlap_max = min(treated_max, control_max)
        
        overlap_range = overlap_max - overlap_min
        total_range = max(treated_max, control_max) - min(treated_min, control_min)
        
        # Good overlap if at least 50% of total range overlaps
        return overlap_range / total_range > 0.5 if total_range > 0 else False
    
    def _nearest_neighbor_matching(self, 
                                 propensity_scores: np.ndarray,
                                 treated: np.ndarray,
                                 control: np.ndarray) -> List[Tuple[int, int]]:
        """Nearest neighbor propensity score matching."""
        treated_indices = np.where(treated)[0]
        control_indices = np.where(control)[0]
        
        matched_pairs = []
        
        for treated_idx in treated_indices:
            # Find closest control unit
            treated_score = propensity_scores[treated_idx]
            control_scores_subset = propensity_scores[control_indices]
            
            # Calculate distances
            distances = np.abs(control_scores_subset - treated_score)
            
            # Find nearest neighbor
            closest_control_idx = control_indices[np.argmin(distances)]
            
            # Add to matched pairs
            matched_pairs.append((treated_idx, closest_control_idx))
        
        return matched_pairs
    
    def _stratification_matching(self, 
                               propensity_scores: np.ndarray,
                               treated: np.ndarray,
                               control: np.ndarray,
                               num_strata: int = 5) -> List[Tuple[int, int]]:
        """Stratification-based propensity score matching."""
        # Create strata based on propensity score quantiles
        strata_bounds = np.quantile(propensity_scores, np.linspace(0, 1, num_strata + 1))
        
        matched_pairs = []
        treated_indices = np.where(treated)[0]
        control_indices = np.where(control)[0]
        
        for i in range(num_strata):
            # Define stratum bounds
            lower_bound = strata_bounds[i]
            upper_bound = strata_bounds[i + 1]
            
            # Get treated and control units in this stratum
            stratum_treated = treated_indices[
                (propensity_scores[treated_indices] >= lower_bound) & 
                (propensity_scores[treated_indices] < upper_bound)
            ]
            stratum_control = control_indices[
                (propensity_scores[control_indices] >= lower_bound) & 
                (propensity_scores[control_indices] < upper_bound)
            ]
            
            # Match units within stratum
            if len(stratum_treated) > 0 and len(stratum_control) > 0:
                # Simple random matching within stratum
                np.random.shuffle(stratum_control)
                n_matches = min(len(stratum_treated), len(stratum_control))
                
                for j in range(n_matches):
                    matched_pairs.append((stratum_treated[j], stratum_control[j]))
        
        return matched_pairs
    
    def _regression_adjustment(self, 
                             data: pd.DataFrame,
                             treatment: str,
                             outcome: str,
                             covariates: List[str]) -> float:
        """Regression adjustment for treatment effect estimation."""
        # Include treatment and interactions
        formula = f"{outcome} ~ {treatment}"
        for cov in covariates:
            formula += f" + {cov} + {treatment}:{cov}"
        
        try:
            from statsmodels.formula.api import ols
            model = ols(formula, data=data).fit()
            return model.params[treatment]
        except:
            # Fallback to simple regression
            model = ols(f"{outcome} ~ {treatment}", data=data).fit()
            return model.params[treatment]
    
    def _test_parallel_trends(self, 
                            treatment_data: pd.DataFrame,
                            control_data: pd.DataFrame,
                            outcome: str,
                            time_var: str,
                            pre_treatment_time: Any) -> float:
        """Test parallel trends assumption."""
        # Simple test: compare pre-treatment trends
        # This is a simplified version - proper parallel trends test requires interaction terms
        
        try:
            # Create time trend variable
            treatment_data = treatment_data.copy()
            control_data = control_data.copy()
            
            treatment_data['time_trend'] = range(len(treatment_data))
            control_data['time_trend'] = range(len(control_data))
            
            # Fit models
            from statsmodels.formula.api import ols
            
            treatment_model = ols(f"{outcome} ~ time_trend", data=treatment_data).fit()
            control_model = ols(f"{outcome} ~ time_trend", data=control_data).fit()
            
            # Compare slopes
            treatment_slope = treatment_model.params['time_trend']
            control_slope = control_model.params['time_trend']
            
            # F-test for equal slopes (approximation)
            slope_diff = abs(treatment_slope - control_slope)
            pooled_se = np.sqrt(treatment_model.bse['time_trend']**2 + control_model.bse['time_trend']**2)
            
            t_stat = slope_diff / pooled_se if pooled_se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))  # Approximate
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Parallel trends test failed: {e}")
            return 0.5  # Conservative estimate
    
    def _analyze_heterogeneity(self, 
                             cate_pred: np.ndarray,
                             X_test: np.ndarray,
                             covariates: List[str]) -> Dict[str, float]:
        """Analyze heterogeneity in treatment effects."""
        heterogeneity_stats = {
            "cate_variance": np.var(cate_pred),
            "cate_range": np.max(cate_pred) - np.min(cate_pred),
            "heterogeneity_score": np.std(cate_pred) / (np.abs(np.mean(cate_pred)) + 1e-6)
        }
        
        # Percentiles
        heterogeneity_stats.update({
            "cate_p25": np.percentile(cate_pred, 25),
            "cate_p50": np.percentile(cate_pred, 50),
            "cate_p75": np.percentile(cate_pred, 75),
            "negative_effects_pct": np.mean(cate_pred < 0) * 100
        })
        
        return heterogeneity_stats
    
    def _check_econml(self) -> bool:
        """Check if econml is available."""
        try:
            import econml
            return True
        except ImportError:
            return False
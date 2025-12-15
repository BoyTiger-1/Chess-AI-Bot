"""Model Evaluation Framework.

Provides metrics, benchmarking, and A/B testing capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    task_type: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


@dataclass
class ABTestResult:
    """Result of A/B test comparison."""
    model_a_metric: float
    model_b_metric: float
    improvement: float
    p_value: float
    is_significant: bool
    confidence_level: float
    recommendation: str


class ModelEvaluator:
    """Model evaluation framework with comprehensive metrics and A/B testing.
    
    Assumptions:
    - Test data is representative of production data
    - Metrics are appropriate for the task
    - A/B tests have sufficient sample size
    
    Limitations:
    - Statistical tests assume certain distributions
    - Metrics may not capture all business objectives
    - A/B tests require careful experimental design
    """
    
    def __init__(self, task_type: str = "classification"):
        """Initialize the evaluator.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            
        Returns:
            EvaluationMetrics object
        """
        if self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            return self._evaluate_regression(y_true, y_pred)
    
    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Evaluate classification model."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except:
                metrics["roc_auc"] = 0.5
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        
        return EvaluationMetrics(
            task_type="classification",
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
        )
    
    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationMetrics:
        """Evaluate regression model."""
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
        
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics["mape"] = float(mape)
        
        return EvaluationMetrics(
            task_type="regression",
            metrics=metrics,
        )
    
    def benchmark_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Benchmark multiple models.
        
        Args:
            models: Dictionary of model name to model object
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                
                y_pred_proba = None
                if self.task_type == "classification" and hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                
                eval_metrics = self.evaluate(y_test, y_pred, y_pred_proba)
                
                result = {"model": name}
                result.update(eval_metrics.metrics)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return pd.DataFrame(results)
    
    def ab_test(
        self,
        model_a_predictions: np.ndarray,
        model_b_predictions: np.ndarray,
        y_true: np.ndarray,
        metric: str = "accuracy",
        confidence_level: float = 0.95,
    ) -> ABTestResult:
        """Perform A/B test between two models.
        
        Args:
            model_a_predictions: Predictions from model A
            model_b_predictions: Predictions from model B
            y_true: True labels
            metric: Metric to compare
            confidence_level: Confidence level for statistical test
            
        Returns:
            ABTestResult object
        """
        if self.task_type == "classification":
            if metric == "accuracy":
                score_a = accuracy_score(y_true, model_a_predictions)
                score_b = accuracy_score(y_true, model_b_predictions)
            elif metric == "f1":
                score_a = f1_score(y_true, model_a_predictions, average="weighted", zero_division=0)
                score_b = f1_score(y_true, model_b_predictions, average="weighted", zero_division=0)
            else:
                raise ValueError(f"Unsupported metric for classification: {metric}")
        else:
            if metric == "mse":
                score_a = mean_squared_error(y_true, model_a_predictions)
                score_b = mean_squared_error(y_true, model_b_predictions)
            elif metric == "mae":
                score_a = mean_absolute_error(y_true, model_a_predictions)
                score_b = mean_absolute_error(y_true, model_b_predictions)
            else:
                raise ValueError(f"Unsupported metric for regression: {metric}")
        
        errors_a = np.abs(y_true - model_a_predictions)
        errors_b = np.abs(y_true - model_b_predictions)
        
        t_statistic, p_value = stats.ttest_rel(errors_a, errors_b)
        
        alpha = 1 - confidence_level
        is_significant = p_value < alpha
        
        improvement = ((score_b - score_a) / score_a) * 100
        
        if is_significant:
            if improvement > 0 and metric not in ["mse", "mae"]:
                recommendation = "Deploy Model B - significant improvement detected"
            elif improvement < 0 and metric in ["mse", "mae"]:
                recommendation = "Deploy Model B - significant improvement detected"
            else:
                recommendation = "Keep Model A - Model B performs worse"
        else:
            recommendation = "No significant difference - keep current model"
        
        return ABTestResult(
            model_a_metric=float(score_a),
            model_b_metric=float(score_b),
            improvement=float(improvement),
            p_value=float(p_value),
            is_significant=is_significant,
            confidence_level=confidence_level,
            recommendation=recommendation,
        )
    
    def cross_model_validation(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """Perform cross-validation across multiple models.
        
        Args:
            models: Dictionary of models
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            DataFrame with CV results
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "accuracy"
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "neg_mean_squared_error"
        
        results = []
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                results.append({
                    "model": name,
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "min_score": scores.min(),
                    "max_score": scores.max(),
                })
            except Exception as e:
                print(f"Error in CV for {name}: {e}")
        
        return pd.DataFrame(results).sort_values("mean_score", ascending=False)

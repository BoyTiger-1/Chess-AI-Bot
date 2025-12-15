"""Tests for Model Evaluation Framework."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ai_business_assistant.ml_pipeline.evaluation import (
    ModelEvaluator,
    EvaluationMetrics,
    ABTestResult,
)


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""
    
    @pytest.fixture
    def classification_setup(self):
        """Setup classification test data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X[:80], y[:80])
        y_pred = model.predict(X[80:])
        y_pred_proba = model.predict_proba(X[80:])
        return y[80:], y_pred, y_pred_proba
    
    @pytest.fixture
    def regression_setup(self):
        """Setup regression test data."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X[:80], y[:80])
        y_pred = model.predict(X[80:])
        return y[80:], y_pred
    
    def test_evaluate_classification(self, classification_setup):
        """Test classification evaluation."""
        y_true, y_pred, y_pred_proba = classification_setup
        evaluator = ModelEvaluator(task_type="classification")
        
        metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert "accuracy" in metrics.metrics
        assert "precision" in metrics.metrics
        assert "recall" in metrics.metrics
        assert "f1" in metrics.metrics
        assert 0 <= metrics.metrics["accuracy"] <= 1
    
    def test_evaluate_regression(self, regression_setup):
        """Test regression evaluation."""
        y_true, y_pred = regression_setup
        evaluator = ModelEvaluator(task_type="regression")
        
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert "mse" in metrics.metrics
        assert "rmse" in metrics.metrics
        assert "mae" in metrics.metrics
        assert "r2" in metrics.metrics
    
    def test_ab_test(self, classification_setup):
        """Test A/B testing."""
        y_true, y_pred_a, _ = classification_setup
        y_pred_b = y_pred_a.copy()
        y_pred_b[:5] = 1 - y_pred_b[:5]
        
        evaluator = ModelEvaluator(task_type="classification")
        result = evaluator.ab_test(y_pred_a, y_pred_b, y_true, metric="accuracy")
        
        assert isinstance(result, ABTestResult)
        assert result.p_value >= 0
        assert isinstance(result.is_significant, bool)
        assert result.recommendation is not None

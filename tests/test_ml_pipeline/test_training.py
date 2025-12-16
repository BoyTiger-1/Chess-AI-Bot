"""Tests for ML Training Pipeline."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from ai_business_assistant.ml_pipeline.training import (
    ModelTrainer,
    TrainingConfig,
    TrainingResult,
)


class TestModelTrainer:
    """Test suite for ModelTrainer."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    def test_classification_training(self, classification_data):
        """Test classification model training."""
        X, y = classification_data
        config = TrainingConfig(task_type="classification", cv_folds=3)
        trainer = ModelTrainer(config)
        
        result = trainer.train_with_cv(X, y, "random_forest", search_type="random", n_iter=2)
        
        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.mean_cv_score > 0
        assert len(result.cv_scores) == 3
    
    def test_regression_training(self, regression_data):
        """Test regression model training."""
        X, y = regression_data
        config = TrainingConfig(task_type="regression", cv_folds=3)
        trainer = ModelTrainer(config)
        
        result = trainer.train_with_cv(X, y, "random_forest", search_type="random", n_iter=2)
        
        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.feature_importance is not None
    
    def test_prepare_data(self, classification_data):
        """Test data preparation."""
        X, y = classification_data
        X_with_missing = X.copy()
        X_with_missing.iloc[0, 0] = np.nan
        
        config = TrainingConfig(task_type="classification", handle_missing=True)
        trainer = ModelTrainer(config)
        
        X_processed, y_processed = trainer.prepare_data(X_with_missing, y)
        
        assert not np.isnan(X_processed).any()
        assert len(X_processed) == len(X)
    
    def test_compare_models(self, classification_data):
        """Test model comparison."""
        X, y = classification_data
        config = TrainingConfig(task_type="classification", cv_folds=2)
        trainer = ModelTrainer(config)
        
        results = trainer.compare_models(X, y, model_names=["random_forest", "logistic"])
        
        assert len(results) >= 1
        assert all(isinstance(r, TrainingResult) for r in results.values())

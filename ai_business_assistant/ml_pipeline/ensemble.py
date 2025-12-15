"""Ensemble Methods.

Combines multiple model predictions using various strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    predictions: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    weights: Optional[Dict[str, float]]
    confidence: Optional[np.ndarray]


class EnsembleModel:
    """Ensemble model combining multiple predictors.
    
    Assumptions:
    - All base models are trained on same feature space
    - Models are compatible (all classification or all regression)
    - Predictions are on same scale (or will be normalized)
    
    Limitations:
    - Ensemble is only as good as base models
    - Can be computationally expensive
    - May overfit if base models are too similar
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        method: str = "voting",
        weights: Optional[List[float]] = None,
        task_type: str = "classification",
    ):
        """Initialize ensemble model.
        
        Args:
            models: Dictionary of model name to model object
            method: Ensemble method ('voting', 'averaging', 'stacking', 'weighted')
            weights: Optional weights for weighted methods
            task_type: 'classification' or 'regression'
        """
        self.models = models
        self.method = method
        self.weights = weights
        self.task_type = task_type
        self._ensemble_model: Optional[Any] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ensemble model.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.method == "voting":
            self._fit_voting(X, y)
        elif self.method == "stacking":
            self._fit_stacking(X, y)
        else:
            for name, model in self.models.items():
                if not hasattr(model, "predict"):
                    model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make ensemble predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            EnsembleResult object
        """
        individual_preds = {}
        
        for name, model in self.models.items():
            individual_preds[name] = model.predict(X)
        
        if self.method == "voting" and self._ensemble_model:
            predictions = self._ensemble_model.predict(X)
        elif self.method == "stacking" and self._ensemble_model:
            predictions = self._ensemble_model.predict(X)
        elif self.method == "averaging":
            predictions = self._average_predictions(individual_preds)
        elif self.method == "weighted":
            predictions = self._weighted_predictions(individual_preds)
        else:
            predictions = self._majority_vote(individual_preds)
        
        confidence = self._calculate_confidence(individual_preds, predictions)
        
        return EnsembleResult(
            predictions=predictions,
            individual_predictions=individual_preds,
            weights=self.weights,
            confidence=confidence,
        )
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (classification only).
        
        Args:
            X: Features to predict
            
        Returns:
            Probability array or None
        """
        if self.task_type != "classification":
            return None
        
        if self._ensemble_model and hasattr(self._ensemble_model, "predict_proba"):
            return self._ensemble_model.predict_proba(X)
        
        probas = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                probas.append(model.predict_proba(X))
        
        if not probas:
            return None
        
        return np.mean(probas, axis=0)
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit voting ensemble."""
        estimators = [(name, model) for name, model in self.models.items()]
        
        if self.task_type == "classification":
            self._ensemble_model = VotingClassifier(
                estimators=estimators,
                voting="soft" if self.weights else "hard",
                weights=self.weights,
            )
        else:
            self._ensemble_model = VotingRegressor(
                estimators=estimators,
                weights=self.weights,
            )
        
        self._ensemble_model.fit(X, y)
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit stacking ensemble."""
        estimators = [(name, model) for name, model in self.models.items()]
        
        if self.task_type == "classification":
            final_estimator = LogisticRegression()
            self._ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
            )
        else:
            final_estimator = Ridge()
            self._ensemble_model = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
            )
        
        self._ensemble_model.fit(X, y)
    
    def _average_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Average predictions from all models."""
        pred_array = np.array(list(predictions.values()))
        return np.mean(pred_array, axis=0)
    
    def _weighted_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of predictions."""
        if self.weights is None:
            return self._average_predictions(predictions)
        
        pred_array = np.array(list(predictions.values()))
        weights_array = np.array(self.weights).reshape(-1, 1)
        
        return np.sum(pred_array * weights_array, axis=0) / np.sum(weights_array)
    
    def _majority_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Majority vote for classification."""
        pred_array = np.array(list(predictions.values()))
        
        if self.task_type == "classification":
            from scipy import stats
            mode_result = stats.mode(pred_array, axis=0, keepdims=False)
            return mode_result.mode
        else:
            return np.median(pred_array, axis=0)
    
    def _calculate_confidence(
        self,
        individual_preds: Dict[str, np.ndarray],
        final_preds: np.ndarray,
    ) -> np.ndarray:
        """Calculate confidence based on agreement among models."""
        pred_array = np.array(list(individual_preds.values()))
        
        if self.task_type == "classification":
            agreement = np.mean(pred_array == final_preds, axis=0)
        else:
            std_dev = np.std(pred_array, axis=0)
            mean_val = np.mean(np.abs(pred_array), axis=0)
            coefficient_of_variation = std_dev / (mean_val + 1e-10)
            agreement = 1 / (1 + coefficient_of_variation)
        
        return agreement

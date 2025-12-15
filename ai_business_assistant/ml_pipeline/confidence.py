"""Confidence Scoring.

Provides confidence scoring for predictions and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ConfidenceScore:
    """Confidence score for a prediction."""
    prediction: Any
    confidence: float
    confidence_interval: Optional[tuple[float, float]]
    factors: Dict[str, float]
    interpretation: str


class ConfidenceScorer:
    """Calculates confidence scores for predictions and recommendations.
    
    Assumptions:
    - Models provide probability estimates or residual information
    - Calibration data is available for more accurate confidence
    - Prediction errors follow known distributions
    
    Limitations:
    - Confidence is estimated, not true probability
    - Depends on model calibration quality
    - May not generalize to distribution shifts
    """
    
    def __init__(
        self,
        calibrated: bool = False,
        confidence_level: float = 0.95,
    ):
        """Initialize confidence scorer.
        
        Args:
            calibrated: Whether the model is probability-calibrated
            confidence_level: Confidence level for intervals
        """
        self.calibrated = calibrated
        self.confidence_level = confidence_level
        self._calibration_params: Dict[str, Any] = {}
    
    def score_classification(
        self,
        prediction: Any,
        predicted_proba: np.ndarray,
        model_accuracy: Optional[float] = None,
        ensemble_agreement: Optional[float] = None,
    ) -> ConfidenceScore:
        """Score confidence for classification prediction.
        
        Args:
            prediction: Predicted class
            predicted_proba: Predicted probabilities
            model_accuracy: Overall model accuracy
            ensemble_agreement: Agreement among ensemble models
            
        Returns:
            ConfidenceScore object
        """
        max_proba = float(np.max(predicted_proba))
        
        entropy = float(-np.sum(predicted_proba * np.log(predicted_proba + 1e-10)))
        normalized_entropy = entropy / np.log(len(predicted_proba))
        entropy_confidence = 1 - normalized_entropy
        
        factors = {
            "probability": max_proba,
            "entropy_confidence": entropy_confidence,
        }
        
        if model_accuracy is not None:
            factors["model_accuracy"] = model_accuracy
        
        if ensemble_agreement is not None:
            factors["ensemble_agreement"] = ensemble_agreement
        
        weights = {
            "probability": 0.5,
            "entropy_confidence": 0.2,
            "model_accuracy": 0.15,
            "ensemble_agreement": 0.15,
        }
        
        confidence = 0.0
        total_weight = 0.0
        
        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 0)
            confidence += factor_value * weight
            total_weight += weight
        
        confidence = confidence / total_weight if total_weight > 0 else max_proba
        
        interpretation = self._interpret_confidence(confidence)
        
        return ConfidenceScore(
            prediction=prediction,
            confidence=float(confidence),
            confidence_interval=None,
            factors=factors,
            interpretation=interpretation,
        )
    
    def score_regression(
        self,
        prediction: float,
        residual_std: float,
        model_r2: Optional[float] = None,
        ensemble_std: Optional[float] = None,
    ) -> ConfidenceScore:
        """Score confidence for regression prediction.
        
        Args:
            prediction: Predicted value
            residual_std: Standard deviation of residuals
            model_r2: Model R² score
            ensemble_std: Standard deviation among ensemble predictions
            
        Returns:
            ConfidenceScore object
        """
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z_score * residual_std
        
        confidence_interval = (
            float(prediction - margin),
            float(prediction + margin),
        )
        
        residual_confidence = 1 / (1 + residual_std / (abs(prediction) + 1e-10))
        
        factors = {
            "residual_confidence": float(residual_confidence),
        }
        
        if model_r2 is not None:
            factors["model_r2"] = model_r2
        
        if ensemble_std is not None:
            ensemble_confidence = 1 / (1 + ensemble_std / (abs(prediction) + 1e-10))
            factors["ensemble_confidence"] = float(ensemble_confidence)
        
        weights = {
            "residual_confidence": 0.4,
            "model_r2": 0.3,
            "ensemble_confidence": 0.3,
        }
        
        confidence = 0.0
        total_weight = 0.0
        
        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 0)
            confidence += factor_value * weight
            total_weight += weight
        
        confidence = confidence / total_weight if total_weight > 0 else residual_confidence
        
        interpretation = self._interpret_confidence(confidence)
        
        return ConfidenceScore(
            prediction=prediction,
            confidence=float(confidence),
            confidence_interval=confidence_interval,
            factors=factors,
            interpretation=interpretation,
        )
    
    def score_batch(
        self,
        predictions: np.ndarray,
        predicted_probas: Optional[np.ndarray] = None,
        task_type: str = "classification",
        **kwargs,
    ) -> List[ConfidenceScore]:
        """Score confidence for batch of predictions.
        
        Args:
            predictions: Array of predictions
            predicted_probas: Array of predicted probabilities (classification)
            task_type: 'classification' or 'regression'
            **kwargs: Additional arguments for scoring
            
        Returns:
            List of ConfidenceScore objects
        """
        scores = []
        
        for i, pred in enumerate(predictions):
            if task_type == "classification":
                proba = predicted_probas[i] if predicted_probas is not None else np.array([0.5, 0.5])
                score = self.score_classification(pred, proba, **kwargs)
            else:
                residual_std = kwargs.get("residual_std", 1.0)
                score = self.score_regression(pred, residual_std, **kwargs)
            
            scores.append(score)
        
        return scores
    
    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        predicted_probas: Optional[np.ndarray] = None,
        task_type: str = "classification",
    ) -> Dict[str, float]:
        """Calibrate confidence scorer using validation data.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            predicted_probas: Predicted probabilities (classification)
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with calibration metrics
        """
        if task_type == "classification":
            from sklearn.calibration import calibration_curve
            
            if predicted_probas is None:
                return {"error": "Probabilities required for calibration"}
            
            if len(np.unique(y_true)) == 2:
                prob_true, prob_pred = calibration_curve(
                    y_true,
                    predicted_probas[:, 1],
                    n_bins=10,
                )
                
                calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
                
                self._calibration_params["calibration_error"] = calibration_error
                self.calibrated = True
                
                return {
                    "calibration_error": calibration_error,
                    "is_calibrated": calibration_error < 0.1,
                }
        else:
            residuals = y_true - y_pred
            residual_std = float(np.std(residuals))
            residual_mean = float(np.mean(residuals))
            
            self._calibration_params["residual_std"] = residual_std
            self._calibration_params["residual_mean"] = residual_mean
            self.calibrated = True
            
            return {
                "residual_std": residual_std,
                "residual_mean": residual_mean,
                "is_unbiased": abs(residual_mean) < 0.1 * residual_std,
            }
        
        return {}
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score as human-readable text."""
        if confidence >= 0.9:
            return "Very High - Prediction is highly reliable"
        elif confidence >= 0.75:
            return "High - Prediction is reliable"
        elif confidence >= 0.6:
            return "Moderate - Prediction has reasonable confidence"
        elif confidence >= 0.4:
            return "Low - Prediction should be used with caution"
        else:
            return "Very Low - Prediction is unreliable"
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration status and parameters.
        
        Returns:
            Dictionary with calibration information
        """
        return {
            "calibrated": self.calibrated,
            "parameters": self._calibration_params,
        }

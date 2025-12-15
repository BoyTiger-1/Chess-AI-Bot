"""Model Explainability Layer.

Provides SHAP values, feature importance, and decision explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class FeatureImportance:
    """Feature importance analysis result."""
    feature_names: List[str]
    importances: np.ndarray
    importance_type: str
    top_features: List[Tuple[str, float]]


@dataclass
class SHAPExplanation:
    """SHAP-based explanation."""
    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    top_contributors: List[Tuple[str, float]]


@dataclass
class DecisionExplanation:
    """Human-readable decision explanation."""
    prediction: Any
    confidence: float
    key_factors: List[Tuple[str, str, float]]
    explanation_text: str


class ExplainabilityAnalyzer:
    """Model explainability using SHAP and feature importance.
    
    Assumptions:
    - Models have feature_importances_ or are compatible with SHAP
    - Feature names are provided or can be inferred
    - SHAP library is available for advanced explanations
    
    Limitations:
    - SHAP computation can be slow for large datasets
    - Tree-based models have native feature importance
    - Explanations assume feature independence
    """
    
    def __init__(self, use_shap: bool = True):
        """Initialize the explainability analyzer.
        
        Args:
            use_shap: Whether to use SHAP for explanations
        """
        self.use_shap = use_shap and SHAP_AVAILABLE
        self._explainers: Dict[str, Any] = {}
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> FeatureImportance:
        """Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            importance_type: Type of importance ('gain', 'split', 'weight')
            
        Returns:
            FeatureImportance object
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature importance attributes")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [
            (feature_names[i], float(importances[i]))
            for i in sorted_idx[:10]
        ]
        
        return FeatureImportance(
            feature_names=feature_names,
            importances=importances,
            importance_type=importance_type,
            top_features=top_features,
        )
    
    def explain_with_shap(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_type: str = "tree",
    ) -> SHAPExplanation:
        """Generate SHAP-based explanations.
        
        Args:
            model: Trained model
            X: Input features
            feature_names: Names of features
            model_type: Type of model ('tree', 'linear', 'kernel')
            
        Returns:
            SHAPExplanation object
        """
        if not self.use_shap:
            raise RuntimeError("SHAP not available or disabled")
        
        model_key = id(model)
        
        if model_key not in self._explainers:
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X)
            elif model_type == "kernel":
                explainer = shap.KernelExplainer(model.predict, X[:100])
            else:
                explainer = shap.Explainer(model, X[:100])
            
            self._explainers[model_key] = explainer
        else:
            explainer = self._explainers[model_key]
        
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if hasattr(explainer, "expected_value"):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:10]
        top_contributors = [
            (feature_names[i], float(mean_abs_shap[i]))
            for i in top_idx
        ]
        
        return SHAPExplanation(
            shap_values=shap_values,
            base_value=float(base_value),
            feature_names=feature_names,
            top_contributors=top_contributors,
        )
    
    def explain_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        feature_values: Optional[Dict[str, Any]] = None,
        threshold: float = 0.05,
    ) -> DecisionExplanation:
        """Generate human-readable explanation for a single prediction.
        
        Args:
            model: Trained model
            instance: Single instance to explain
            feature_names: Names of features
            feature_values: Optional dict of feature values for context
            threshold: Threshold for including features in explanation
            
        Returns:
            DecisionExplanation object
        """
        instance_2d = instance.reshape(1, -1)
        prediction = model.predict(instance_2d)[0]
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(instance_2d)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 0.7
        
        key_factors = []
        
        if self.use_shap:
            try:
                shap_exp = self.explain_with_shap(model, instance_2d, feature_names)
                shap_values_instance = shap_exp.shap_values[0]
                
                for i, (feat_name, shap_val) in enumerate(zip(feature_names, shap_values_instance)):
                    if abs(shap_val) > threshold:
                        direction = "increases" if shap_val > 0 else "decreases"
                        feat_value = instance[i]
                        key_factors.append((feat_name, direction, float(abs(shap_val))))
            except:
                key_factors = self._fallback_explanation(model, instance, feature_names)
        else:
            key_factors = self._fallback_explanation(model, instance, feature_names)
        
        key_factors.sort(key=lambda x: x[2], reverse=True)
        key_factors = key_factors[:5]
        
        explanation_text = self._generate_explanation_text(
            prediction,
            confidence,
            key_factors,
            feature_values,
        )
        
        return DecisionExplanation(
            prediction=prediction,
            confidence=confidence,
            key_factors=key_factors,
            explanation_text=explanation_text,
        )
    
    def _fallback_explanation(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Fallback explanation using feature importance."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = np.ones(len(feature_names))
        
        contributions = instance * importances
        
        key_factors = []
        for i, (feat_name, contrib) in enumerate(zip(feature_names, contributions)):
            if abs(contrib) > 0:
                direction = "increases" if contrib > 0 else "decreases"
                key_factors.append((feat_name, direction, float(abs(contrib))))
        
        return key_factors
    
    def _generate_explanation_text(
        self,
        prediction: Any,
        confidence: float,
        key_factors: List[Tuple[str, str, float]],
        feature_values: Optional[Dict[str, Any]],
    ) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"Prediction: {prediction} (Confidence: {confidence:.2%})",
            "",
            "Key contributing factors:",
        ]
        
        for i, (feature, direction, importance) in enumerate(key_factors, 1):
            value_str = ""
            if feature_values and feature in feature_values:
                value_str = f" (value: {feature_values[feature]})"
            lines.append(f"  {i}. {feature}{value_str} {direction} the prediction (impact: {importance:.4f})")
        
        return "\n".join(lines)
    
    def summarize_model_behavior(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Summarize overall model behavior across dataset.
        
        Args:
            model: Trained model
            X: Dataset to analyze
            feature_names: Names of features
            
        Returns:
            Dictionary with behavior summary
        """
        summary = {}
        
        feature_imp = self.get_feature_importance(model, feature_names)
        summary["top_features"] = feature_imp.top_features
        
        if self.use_shap:
            try:
                shap_exp = self.explain_with_shap(model, X, feature_names)
                summary["shap_top_contributors"] = shap_exp.top_contributors
            except:
                summary["shap_top_contributors"] = None
        
        predictions = model.predict(X)
        summary["prediction_distribution"] = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
        }
        
        return summary

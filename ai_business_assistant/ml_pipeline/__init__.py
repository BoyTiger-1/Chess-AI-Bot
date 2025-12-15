"""ML Pipeline Infrastructure.

This package provides:
- Model training pipeline with hyperparameter tuning
- Model evaluation framework with metrics and benchmarking
- Explainability layer (SHAP, feature importance)
- Ensemble methods
- Batch and real-time inference
- Model versioning and rollback
- Confidence scoring
"""

from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .explainability import ExplainabilityAnalyzer
from .ensemble import EnsembleModel
from .inference import InferenceEngine
from .versioning import ModelVersionManager
from .confidence import ConfidenceScorer

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
    "ExplainabilityAnalyzer",
    "EnsembleModel",
    "InferenceEngine",
    "ModelVersionManager",
    "ConfidenceScorer",
]

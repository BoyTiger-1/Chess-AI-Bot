"""Model Caching and Pre-loading Module.

Provides a centralized cache for heavy ML models to eliminate cold-start latency.
Supports scikit-learn, Prophet, and Transformer models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ai_business_assistant.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelCache:
    """Singleton cache for ML models."""
    
    _instance: Optional[ModelCache] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._loaded = False
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Return whether models have been loaded."""
        return self._loaded

    async def load_all(self) -> None:
        """Load all configured models into memory."""
        if self._loaded:
            logger.info("Models already loaded in cache")
            return

        start_time = time.time()
        logger.info("Pre-caching ML models...")

        # Ensure model directories exist
        settings.ensure_dirs()

        # 1. Load Scikit-Learn models
        self._load_sklearn_models()

        # 2. Load Prophet models
        self._load_prophet_models()

        # 3. Load Transformer models
        self._load_transformer_models()

        self._loaded = True
        duration = time.time() - start_time
        logger.info(f"Model pre-caching completed in {duration:.2f} seconds")

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from the cache."""
        return self._models.get(model_name)

    def set_model(self, model_name: str, model: Any) -> None:
        """Add a model to the cache."""
        self._models[model_name] = model

    def _load_sklearn_models(self) -> None:
        """Load scikit-learn models from disk."""
        try:
            import joblib
            
            model_path = settings.abs_path(settings.MODEL_DIR)
            # Try to load any .pkl files in the models directory
            for pkl_file in model_path.glob("*.pkl"):
                try:
                    model_name = pkl_file.stem
                    logger.info(f"Loading scikit-learn model: {model_name}")
                    model = joblib.load(pkl_file)
                    self._models[model_name] = model
                except Exception as e:
                    logger.error(f"Failed to load scikit-learn model {pkl_file.name}: {e}")
            
            # Warm up scikit-learn by importing commonly used modules
            import sklearn.ensemble  # noqa: F401
            import sklearn.linear_model  # noqa: F401
            import sklearn.preprocessing  # noqa: F401
            import sklearn.cluster  # noqa: F401
            import sklearn.model_selection  # noqa: F401
            logger.info("Scikit-learn modules warmed up")
            
        except ImportError:
            logger.warning("joblib or scikit-learn not installed, skipping scikit-learn pre-caching")

    def _load_prophet_models(self) -> None:
        """Warm up Prophet by importing it and potentially loading saved models."""
        try:
            from prophet import Prophet  # noqa: F401
            logger.info("Prophet library warmed up")
            
            # Example: Load a default Prophet model if it exists
            # p_file = settings.abs_path(settings.MODEL_DIR) / "default_prophet_model.json"
            # if p_file.exists():
            #     from prophet.serialize import model_from_json
            #     with open(p_file, 'r') as f:
            #         self._models["default_prophet"] = model_from_json(f.read())
        except ImportError:
            logger.warning("prophet not installed, skipping Prophet pre-caching")

    def _load_transformer_models(self) -> None:
        """Load Transformer models/tokenizers."""
        try:
            import torch
            from transformers import pipeline
            
            logger.info("PyTorch and Transformers libraries warmed up")
            
            # Pre-load sentiment analysis pipeline (common heavy model)
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            cache_key = f"pipeline_sentiment_{model_name}"
            
            if cache_key not in self._models:
                logger.info(f"Pre-loading transformer pipeline: sentiment-analysis ({model_name})")
                
                device = -1
                if torch.cuda.is_available():
                    device = 0
                    
                # Note: This might take time and internet connection if model not cached locally
                sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=device,
                )
                self._models[cache_key] = sentiment_pipe
            
        except ImportError:
            logger.warning("torch or transformers not installed, skipping transformer pre-caching")
        except Exception as e:
            # We don't want to fail startup if a model fails to load (e.g. no internet)
            logger.error(f"Failed to pre-load transformer pipeline: {e}")


def get_model_cache() -> ModelCache:
    """Get the model cache singleton."""
    return ModelCache()

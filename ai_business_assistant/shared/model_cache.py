"""Model Caching and Pre-loading Module.

Provides a centralized cache for heavy ML models to eliminate cold-start latency.
Supports scikit-learn, Prophet, and Transformer models with async loading.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_business_assistant.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ModelMetadata:
    """Metadata for a cached model."""
    
    name: str
    model_type: str  # sklearn, prophet, transformer, arima
    loaded_at: float
    load_duration: float
    status: str  # loaded, failed, skipped
    error: Optional[str] = None
    version: Optional[str] = None


class ModelCache:
    """Singleton cache for ML models with async loading support."""
    
    _instance: Optional[ModelCache] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models: Dict[str, Any] = {}
            cls._instance._registry: Dict[str, ModelMetadata] = {}
            cls._instance._loaded = False
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Return whether models have been loaded."""
        return self._loaded

    @property
    def loaded_models(self) -> List[str]:
        """Return list of successfully loaded model names."""
        return [
            name for name, meta in self._registry.items()
            if meta.status == "loaded"
        ]

    @property
    def failed_models(self) -> List[str]:
        """Return list of models that failed to load."""
        return [
            name for name, meta in self._registry.items()
            if meta.status == "failed"
        ]

    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self._registry.get(model_name)

    def get_registry(self) -> Dict[str, ModelMetadata]:
        """Get complete model registry."""
        return self._registry.copy()

    async def load_all(self) -> None:
        """Load all configured models into memory asynchronously."""
        if self._loaded:
            logger.info("Models already loaded in cache")
            return

        start_time = time.time()
        logger.info("Pre-caching ML models...")

        # Ensure model directories exist
        settings.ensure_dirs()

        # Load all model types in parallel
        load_tasks = [
            self._load_sklearn_models_async(),
            self._load_prophet_models_async(),
            self._load_transformer_models_async(),
        ]

        # Execute all loading tasks
        await asyncio.gather(*load_tasks, return_exceptions=True)

        self._loaded = True
        duration = time.time() - start_time

        # Log summary
        loaded_count = len(self.loaded_models)
        failed_count = len(self.failed_models)
        
        logger.info(
            f"Model pre-caching completed in {duration:.2f} seconds - "
            f"{loaded_count} loaded, {failed_count} failed"
        )

        if loaded_count > 0:
            logger.info(f"Successfully loaded models: {', '.join(self.loaded_models)}")

        if failed_count > 0:
            logger.warning(f"Failed to load models: {', '.join(self.failed_models)}")

    async def unload_all(self) -> None:
        """Unload all models from memory and clear registry."""
        if not self._loaded:
            logger.info("Models not loaded, nothing to unload")
            return

        logger.info(f"Unloading {len(self._models)} models from cache...")

        # Clear models from memory
        unloaded_count = 0
        for name, model in self._models.items():
            try:
                # Force cleanup if possible
                del model
                unloaded_count += 1
                logger.debug(f"Unloaded model: {name}")
            except Exception as e:
                logger.warning(f"Error unloading model {name}: {e}")

        # Clear dictionaries
        self._models.clear()
        self._registry.clear()
        self._loaded = False

        logger.info(f"Successfully unloaded {unloaded_count} models")

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from the cache."""
        return self._models.get(model_name)

    def set_model(
        self,
        model_name: str,
        model: Any,
        model_type: str = "custom",
        version: Optional[str] = None,
    ) -> None:
        """Add a model to the cache with metadata."""
        self._models[model_name] = model
        self._registry[model_name] = ModelMetadata(
            name=model_name,
            model_type=model_type,
            loaded_at=time.time(),
            load_duration=0.0,
            status="loaded",
            version=version,
        )

    async def _load_sklearn_models_async(self) -> None:
        """Load scikit-learn models from disk asynchronously."""
        load_start = time.time()
        
        try:
            import joblib
            
            model_path = settings.abs_path(settings.MODEL_DIR)
            loaded_count = 0
            
            # Load any .pkl files in the models directory
            for pkl_file in model_path.glob("*.pkl"):
                try:
                    model_name = pkl_file.stem
                    logger.info(f"Loading scikit-learn model: {model_name}")
                    
                    # Run blocking load in thread pool
                    model = await asyncio.to_thread(joblib.load, pkl_file)
                    self._models[model_name] = model
                    
                    # Record metadata
                    self._registry[model_name] = ModelMetadata(
                        name=model_name,
                        model_type="sklearn",
                        loaded_at=time.time(),
                        load_duration=0.0,
                        status="loaded",
                    )
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load scikit-learn model {pkl_file.name}: {e}")
                    self._registry[pkl_file.stem] = ModelMetadata(
                        name=pkl_file.stem,
                        model_type="sklearn",
                        loaded_at=time.time(),
                        load_duration=0.0,
                        status="failed",
                        error=str(e),
                    )
            
            # Warm up scikit-learn by importing commonly used modules
            await asyncio.to_thread(self._warmup_sklearn_modules)
            
            duration = time.time() - load_start
            logger.info(f"Scikit-learn pre-caching completed in {duration:.2f}s ({loaded_count} models)")
            
        except ImportError:
            logger.warning("joblib or scikit-learn not installed, skipping scikit-learn pre-caching")
            self._registry["sklearn"] = ModelMetadata(
                name="sklearn",
                model_type="sklearn",
                loaded_at=time.time(),
                load_duration=time.time() - load_start,
                status="skipped",
                error="Module not installed",
            )

    def _warmup_sklearn_modules(self) -> None:
        """Warm up scikit-learn imports."""
        import sklearn.ensemble  # noqa: F401
        import sklearn.linear_model  # noqa: F401
        import sklearn.preprocessing  # noqa: F401
        import sklearn.cluster  # noqa: F401
        import sklearn.model_selection  # noqa: F401

    async def _load_prophet_models_async(self) -> None:
        """Warm up Prophet by importing it and potentially loading saved models."""
        load_start = time.time()
        
        try:
            # Run import in thread pool to avoid blocking
            await asyncio.to_thread(self._import_prophet)
            
            duration = time.time() - load_start
            logger.info(f"Prophet pre-caching completed in {duration:.2f}s")
            
            # Register prophet as warmed up
            self._registry["prophet"] = ModelMetadata(
                name="prophet",
                model_type="prophet",
                loaded_at=time.time(),
                load_duration=duration,
                status="loaded",
            )
            
        except ImportError:
            logger.warning("prophet not installed, skipping Prophet pre-caching")
            self._registry["prophet"] = ModelMetadata(
                name="prophet",
                model_type="prophet",
                loaded_at=time.time(),
                load_duration=time.time() - load_start,
                status="skipped",
                error="Module not installed",
            )

    def _import_prophet(self) -> None:
        """Import Prophet module."""
        from prophet import Prophet  # noqa: F401

    async def _load_transformer_models_async(self) -> None:
        """Load Transformer models/tokenizers asynchronously."""
        load_start = time.time()
        
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
                    logger.info(f"Using GPU device {device} for transformer models")
                    
                # Run blocking model load in thread pool
                sentiment_pipe = await asyncio.to_thread(
                    self._load_sentiment_pipeline,
                    model_name,
                    device,
                )
                self._models[cache_key] = sentiment_pipe
                
                # Record metadata
                duration = time.time() - load_start
                self._registry[cache_key] = ModelMetadata(
                    name=cache_key,
                    model_type="transformer",
                    loaded_at=time.time(),
                    load_duration=duration,
                    status="loaded",
                    version=model_name,
                )
                
                logger.info(f"Transformer pre-caching completed in {duration:.2f}s")
            else:
                logger.info(f"Transformer pipeline already loaded: {cache_key}")
            
        except ImportError:
            logger.warning("torch or transformers not installed, skipping transformer pre-caching")
            self._registry["transformers"] = ModelMetadata(
                name="transformers",
                model_type="transformer",
                loaded_at=time.time(),
                load_duration=time.time() - load_start,
                status="skipped",
                error="Module not installed",
            )
        except Exception as e:
            # We don't want to fail startup if a model fails to load (e.g. no internet)
            logger.error(f"Failed to pre-load transformer pipeline: {e}")
            self._registry["transformers"] = ModelMetadata(
                name="transformers",
                model_type="transformer",
                loaded_at=time.time(),
                load_duration=time.time() - load_start,
                status="failed",
                error=str(e),
            )

    def _load_sentiment_pipeline(self, model_name: str, device: int) -> Any:
        """Load sentiment analysis pipeline (blocking call)."""
        from transformers import pipeline
        
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
        )


def get_model_cache() -> ModelCache:
    """Get the model cache singleton."""
    return ModelCache()

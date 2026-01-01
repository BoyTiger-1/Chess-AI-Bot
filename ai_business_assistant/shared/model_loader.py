"""ML Model Loader Module.

Provides convenient interfaces for loading and accessing pre-cached ML models.
Supports lazy loading, async loading, and graceful fallbacks for optional dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ai_business_assistant.shared.model_cache import (
    ModelCache,
    ModelMetadata,
    get_model_cache,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Convenient interface for accessing pre-cached ML models."""

    def __init__(self, cache: Optional[ModelCache] = None):
        """Initialize model loader.
        
        Args:
            cache: ModelCache instance. If None, uses singleton.
        """
        self._cache = cache if cache is not None else get_model_cache()

    @property
    def cache(self) -> ModelCache:
        """Get the underlying model cache."""
        return self._cache

    def is_loaded(self) -> bool:
        """Check if models have been loaded."""
        return self._cache.is_loaded

    def get_loaded_models(self) -> List[str]:
        """Get list of successfully loaded model names."""
        return self._cache.loaded_models

    def get_failed_models(self) -> List[str]:
        """Get list of models that failed to load."""
        return self._cache.failed_models

    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self._cache.get_model_info(model_name)

    def get_registry(self) -> Dict[str, ModelMetadata]:
        """Get complete model registry."""
        return self._cache.get_registry()

    def get_sentiment_pipeline(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ) -> Optional[Any]:
        """Get pre-loaded sentiment analysis pipeline.
        
        Args:
            model_name: Name of the transformer model to use
            
        Returns:
            Sentiment pipeline if loaded, None otherwise
        """
        cache_key = f"pipeline_sentiment_{model_name}"
        pipeline = self._cache.get_model(cache_key)

        if pipeline is None:
            logger.warning(f"Sentiment pipeline not loaded: {cache_key}")
        else:
            logger.debug(f"Retrieved cached sentiment pipeline: {cache_key}")

        return pipeline

    def get_sklearn_model(self, model_name: str) -> Optional[Any]:
        """Get a pre-loaded scikit-learn model.
        
        Args:
            model_name: Name of the sklearn model (filename without .pkl)
            
        Returns:
            Scikit-learn model if loaded, None otherwise
        """
        model = self._cache.get_model(model_name)

        if model is None:
            logger.warning(f"Scikit-learn model not loaded: {model_name}")
        else:
            logger.debug(f"Retrieved cached scikit-learn model: {model_name}")

        return model

    def list_sklearn_models(self) -> List[str]:
        """List all available scikit-learn models."""
        return [
            name for name, meta in self._cache.get_registry().items()
            if meta.model_type == "sklearn" and meta.status == "loaded"
        ]

    def has_prophet(self) -> bool:
        """Check if Prophet library is available and loaded."""
        prophet_meta = self._cache.get_model_info("prophet")
        return prophet_meta is not None and prophet_meta.status == "loaded"

    def has_transformers(self) -> bool:
        """Check if Transformers library is available and loaded."""
        transformers_meta = self._cache.get_model_info("transformers")
        return transformers_meta is not None and transformers_meta.status == "loaded"

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get any model from the cache by name."""
        return self._cache.get_model(model_name)

    async def load_models_async(self) -> None:
        """Load all models asynchronously (delegates to cache)."""
        await self._cache.load_all()

    async def unload_models_async(self) -> None:
        """Unload all models asynchronously (delegates to cache)."""
        await self._cache.unload_all()


class LazyModelLoader:
    """Lazy loader that creates models on-demand with graceful fallbacks."""

    def __init__(self, cache: Optional[ModelCache] = None):
        """Initialize lazy model loader.
        
        Args:
            cache: ModelCache instance. If None, uses singleton.
        """
        self._cache = cache if cache is not None else get_model_cache()
        self._local_cache: Dict[str, Any] = {}

    def get_or_create_sentiment_pipeline(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ) -> Any:
        """Get or create sentiment pipeline with fallback.
        
        Tries to use pre-loaded model first, falls back to TextBlob
        if transformers not available.
        
        Args:
            model_name: Name of transformer model
            
        Returns:
            Sentiment pipeline or TextBlob-based fallback
        """
        cache_key = f"pipeline_sentiment_{model_name}"

        # Check cache first
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        # Try global cache
        pipeline = self._cache.get_model(cache_key)
        if pipeline is not None:
            self._local_cache[cache_key] = pipeline
            return pipeline

        # Lazy load if transformers available
        try:
            import torch
            from transformers import pipeline as transformers_pipeline

            logger.info(f"Lazily loading sentiment pipeline: {model_name}")
            device = 0 if torch.cuda.is_available() else -1

            pipeline = transformers_pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
            )
            self._local_cache[cache_key] = pipeline
            self._cache.set_model(cache_key, pipeline, model_type="transformer")
            return pipeline

        except ImportError:
            logger.warning("Transformers not available, using TextBlob fallback")
            return TextBlobSentimentFallback()

    def get_or_create_sklearn_model(
        self,
        model_name: str,
        model_factory: Optional[callable] = None,
    ) -> Optional[Any]:
        """Get or create scikit-learn model.
        
        Args:
            model_name: Name of the model
            model_factory: Optional factory function to create model if not cached
            
        Returns:
            Scikit-learn model or None
        """
        # Check local cache
        if model_name in self._local_cache:
            return self._local_cache[model_name]

        # Try global cache
        model = self._cache.get_model(model_name)
        if model is not None:
            self._local_cache[model_name] = model
            return model

        # Create if factory provided
        if model_factory is not None:
            try:
                logger.info(f"Creating sklearn model: {model_name}")
                model = model_factory()
                self._local_cache[model_name] = model
                self._cache.set_model(model_name, model, model_type="sklearn")
                return model
            except Exception as e:
                logger.error(f"Failed to create sklearn model {model_name}: {e}")
                return None

        return None

    def clear_local_cache(self) -> None:
        """Clear local lazy-loaded models."""
        self._local_cache.clear()


class TextBlobSentimentFallback:
    """Fallback sentiment analyzer using TextBlob.

    Used when transformers library is not available.
    """

    def __call__(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with label and score (compatible with transformers output)
        """
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)

        # Map TextBlob polarity to labels
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        score = float(abs(polarity))

        return {
            "label": label,
            "score": score,
        }

    def __call__(self, texts: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Analyze sentiment of one or multiple texts."""
        if isinstance(texts, str):
            return self._analyze_single(texts)
        return [self._analyze_single(text) for text in texts]

    def _analyze_single(self, text: str) -> Dict[str, Any]:
        """Analyze a single text."""
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)

        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        score = float(abs(polarity))

        return {"label": label, "score": score}


# Convenience functions
def get_model_loader() -> ModelLoader:
    """Get the singleton model loader."""
    return ModelLoader()


def get_lazy_loader() -> LazyModelLoader:
    """Get a lazy model loader instance."""
    return LazyModelLoader()


def get_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
) -> Optional[Any]:
    """Get pre-loaded sentiment analysis pipeline (convenience function)."""
    loader = get_model_loader()
    return loader.get_sentiment_pipeline(model_name)


def get_sklearn_model(model_name: str) -> Optional[Any]:
    """Get a pre-loaded scikit-learn model (convenience function)."""
    loader = get_model_loader()
    return loader.get_sklearn_model(model_name)


async def preload_models() -> None:
    """Preload all models (convenience function)."""
    cache = get_model_cache()
    await cache.load_all()


async def unload_models() -> None:
    """Unload all models (convenience function)."""
    cache = get_model_cache()
    await cache.unload_all()

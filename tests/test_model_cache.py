"""Tests for Model Cache and Model Loader.

Verifies model pre-caching, async loading, and lazy loading with graceful fallbacks.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from ai_business_assistant.shared.model_cache import ModelCache, ModelMetadata, get_model_cache
from ai_business_assistant.shared.model_loader import (
    LazyModelLoader,
    ModelLoader,
    TextBlobSentimentFallback,
    get_lazy_loader,
    get_model_loader,
)


class TestModelCache:
    """Test ModelCache singleton and async loading."""

    @pytest.fixture
    def cache(self):
        """Get a fresh model cache instance."""
        # Reset singleton for clean tests
        ModelCache._instance = None
        return get_model_cache()

    @pytest.mark.asyncio
    async def test_singleton(self, cache):
        """Test that ModelCache is a singleton."""
        cache2 = get_model_cache()
        assert cache is cache2
        assert id(cache) == id(cache2)

    @pytest.mark.asyncio
    async def test_load_all_async(self, cache):
        """Test async loading of all models."""
        assert not cache.is_loaded

        start_time = time.time()
        await cache.load_all()
        duration = time.time() - start_time

        assert cache.is_loaded
        logger.info(f"load_all() completed in {duration:.2f}s")

        # Should have loaded or skipped at least some models
        registry = cache.get_registry()
        assert len(registry) > 0

    @pytest.mark.asyncio
    async def test_double_load_is_idempotent(self, cache):
        """Test that calling load_all twice doesn't re-load models."""
        await cache.load_all()
        first_registry = cache.get_registry().copy()

        await cache.load_all()
        second_registry = cache.get_registry()

        assert first_registry == second_registry

    @pytest.mark.asyncio
    async def test_unload_all(self, cache):
        """Test unloading all models."""
        await cache.load_all()
        assert cache.is_loaded
        assert len(cache.get_registry()) > 0

        await cache.unload_all()

        assert not cache.is_loaded
        assert len(cache.get_registry()) == 0
        assert len(cache._models) == 0

    @pytest.mark.asyncio
    async def test_get_and_set_model(self, cache):
        """Test getting and setting models."""
        dummy_model = {"type": "test", "value": 42}
        cache.set_model("test_model", dummy_model, model_type="custom", version="1.0.0")

        retrieved = cache.get_model("test_model")
        assert retrieved == dummy_model

        # Check metadata
        meta = cache.get_model_info("test_model")
        assert meta is not None
        assert meta.name == "test_model"
        assert meta.model_type == "custom"
        assert meta.version == "1.0.0"
        assert meta.status == "loaded"

    @pytest.mark.asyncio
    async def test_loaded_models_property(self, cache):
        """Test loaded_models property filters successful loads."""
        await cache.load_all()

        loaded = cache.loaded_models
        assert isinstance(loaded, list)

        for name in loaded:
            meta = cache.get_model_info(name)
            assert meta is not None
            assert meta.status == "loaded"

    @pytest.mark.asyncio
    async def test_failed_models_property(self, cache):
        """Test failed_models property filters failed loads."""
        await cache.load_all()

        failed = cache.failed_models
        assert isinstance(failed, list)

        for name in failed:
            meta = cache.get_model_info(name)
            assert meta is not None
            assert meta.status == "failed"

    @pytest.mark.asyncio
    async def test_sklearn_warmup(self, cache):
        """Test scikit-learn warmup."""
        try:
            import joblib
            import sklearn.ensemble  # noqa: F401

            await cache.load_all()

            # sklearn should be registered
            meta = cache.get_model_info("sklearn")
            assert meta is not None
            assert meta.model_type == "sklearn"

            if meta.status == "loaded":
                logger.info("Sklearn warmed up successfully")

        except ImportError:
            pytest.skip("sklearn not installed")

    @pytest.mark.asyncio
    async def test_prophet_warmup(self, cache):
        """Test Prophet warmup."""
        try:
            from prophet import Prophet  # noqa: F401

            await cache.load_all()

            # prophet should be registered
            meta = cache.get_model_info("prophet")
            assert meta is not None
            assert meta.model_type == "prophet"

            if meta.status == "loaded":
                logger.info("Prophet warmed up successfully")

        except ImportError:
            pytest.skip("prophet not installed")

    @pytest.mark.asyncio
    async def test_transformer_warmup(self, cache):
        """Test transformer warmup."""
        try:
            import torch
            from transformers import pipeline

            await cache.load_all()

            # transformers should be registered
            meta = cache.get_model_info("transformers")
            assert meta is not None
            assert meta.model_type == "transformer"

            if meta.status == "loaded":
                logger.info("Transformers warmed up successfully")

                # Check for sentiment pipeline
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                cache_key = f"pipeline_sentiment_{model_name}"
                pipeline = cache.get_model(cache_key)
                assert pipeline is not None

        except ImportError:
            pytest.skip("transformers or torch not installed")


class TestModelLoader:
    """Test ModelLoader convenience interface."""

    @pytest.fixture
    def loader(self, cache):
        """Get a fresh model loader."""
        return ModelLoader(cache)

    @pytest.mark.asyncio
    async def test_get_sentiment_pipeline(self, loader):
        """Test getting sentiment pipeline."""
        pipeline = loader.get_sentiment_pipeline()

        # May be None if transformers not installed
        if pipeline is not None:
            # Should be callable
            assert callable(pipeline)

            # Should produce valid output
            result = pipeline("This is a test")
            assert isinstance(result, list) or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_sklearn_model(self, loader):
        """Test getting sklearn model."""
        model = loader.get_sklearn_model("nonexistent_model")
        assert model is None  # Should return None if not loaded

    @pytest.mark.asyncio
    async def test_list_sklearn_models(self, loader):
        """Test listing sklearn models."""
        models = loader.list_sklearn_models()
        assert isinstance(models, list)

    @pytest.mark.asyncio
    async def test_has_prophet(self, loader):
        """Test checking Prophet availability."""
        has_prophet = loader.has_prophet()
        assert isinstance(has_prophet, bool)

    @pytest.mark.asyncio
    async def test_has_transformers(self, loader):
        """Test checking Transformers availability."""
        has_transformers = loader.has_transformers()
        assert isinstance(has_transformers, bool)

    @pytest.mark.asyncio
    async def test_get_loaded_models(self, loader):
        """Test getting list of loaded models."""
        await loader.cache.load_all()

        loaded = loader.get_loaded_models()
        assert isinstance(loaded, list)

        for name in loaded:
            meta = loader.get_model_info(name)
            assert meta is not None
            assert meta.status == "loaded"

    @pytest.mark.asyncio
    async def test_get_failed_models(self, loader):
        """Test getting list of failed models."""
        await loader.cache.load_all()

        failed = loader.get_failed_models()
        assert isinstance(failed, list)

    @pytest.mark.asyncio
    async def test_get_model_info(self, loader):
        """Test getting model metadata."""
        await loader.cache.load_all()

        # Try to get info for a loaded model
        loaded = loader.get_loaded_models()
        if loaded:
            meta = loader.get_model_info(loaded[0])
            assert meta is not None
            assert isinstance(meta, ModelMetadata)


class TestLazyModelLoader:
    """Test LazyModelLoader for on-demand loading."""

    @pytest.fixture
    def lazy_loader(self):
        """Get a fresh lazy model loader."""
        ModelCache._instance = None
        return LazyModelLoader()

    def test_get_or_create_sentiment_pipeline_with_fallback(self, lazy_loader):
        """Test sentiment pipeline with TextBlob fallback."""
        pipeline = lazy_loader.get_or_create_sentiment_pipeline()

        # Should always return something (either transformers or TextBlob)
        assert pipeline is not None
        assert callable(pipeline)

        # Test it works
        result = pipeline("This is a great product!")
        assert "label" in result or isinstance(result, list)

    def test_sentiment_pipeline_caching(self, lazy_loader):
        """Test that lazy loader caches models."""
        pipeline1 = lazy_loader.get_or_create_sentiment_pipeline()
        pipeline2 = lazy_loader.get_or_create_sentiment_pipeline()

        # Should be the same instance (cached)
        if isinstance(pipeline1, TextBlobSentimentFallback):
            # TextBlob fallbacks are new instances, so skip this check
            return

        assert pipeline1 is pipeline2

    def test_get_or_create_sklearn_model_with_factory(self, lazy_loader):
        """Test creating sklearn model with factory."""
        from sklearn.linear_model import LinearRegression

        model = lazy_loader.get_or_create_sklearn_model(
            "test_lr",
            model_factory=lambda: LinearRegression(),
        )

        assert model is not None
        assert isinstance(model, LinearRegression)

        # Should be cached
        model2 = lazy_loader.get_or_create_sklearn_model("test_lr")
        assert model is model2

    def test_clear_local_cache(self, lazy_loader):
        """Test clearing local cache."""
        from sklearn.linear_model import LinearRegression

        # Create a model
        lazy_loader.get_or_create_sklearn_model(
            "test_lr",
            model_factory=lambda: LinearRegression(),
        )

        assert len(lazy_loader._local_cache) > 0

        # Clear cache
        lazy_loader.clear_local_cache()

        assert len(lazy_loader._local_cache) == 0


class TestTextBlobSentimentFallback:
    """Test TextBlob sentiment fallback."""

    def test_analyze_single_text(self):
        """Test analyzing a single text."""
        fallback = TextBlobSentimentFallback()

        result = fallback("This is amazing!")
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert 0.0 <= result["score"] <= 1.0

    def test_analyze_multiple_texts(self):
        """Test analyzing multiple texts."""
        fallback = TextBlobSentimentFallback()

        results = fallback([
            "This is great!",
            "This is terrible!",
            "This is okay.",
        ])

        assert isinstance(results, list)
        assert len(results) == 3

        for result in results:
            assert isinstance(result, dict)
            assert "label" in result
            assert "score" in result

    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        fallback = TextBlobSentimentFallback()
        result = fallback("I love this product! It's amazing!")
        assert result["label"] == "POSITIVE"

    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        fallback = TextBlobSentimentFallback()
        result = fallback("I hate this product! It's terrible!")
        assert result["label"] == "NEGATIVE"

    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        fallback = TextBlobSentimentFallback()
        result = fallback("The product is a product.")
        assert result["label"] == "NEUTRAL"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_model_loader(self):
        """Test get_model_loader convenience function."""
        loader = get_model_loader()
        assert isinstance(loader, ModelLoader)

        # Should return singleton
        loader2 = get_model_loader()
        assert loader is loader2

    def test_get_lazy_loader(self):
        """Test get_lazy_loader convenience function."""
        loader = get_lazy_loader()
        assert isinstance(loader, LazyModelLoader)

        # Should return new instance (not singleton)
        loader2 = get_lazy_loader()
        assert loader is not loader2

    def test_get_sentiment_pipeline(self):
        """Test get_sentiment_pipeline convenience function."""
        pipeline = get_sentiment_pipeline()
        # May be None or a pipeline

    def test_get_sklearn_model(self):
        """Test get_sklearn_model convenience function."""
        model = get_sklearn_model("nonexistent")
        assert model is None


@pytest.mark.asyncio
async def test_first_request_performance():
    """Test that first request completes in reasonable time (<2s)."""
    ModelCache._instance = None
    cache = get_model_cache()

    start_time = time.time()
    await cache.load_all()
    duration = time.time() - start_time

    logger.info(f"First request (model loading): {duration:.2f}s")
    assert duration < 2.0, "Model loading should complete in <2s"


@pytest.mark.asyncio
async def test_cached_request_performance():
    """Test that cached model access is fast (<100ms)."""
    ModelCache._instance = None
    cache = get_model_cache()

    # Load models first
    await cache.load_all()

    # Test cached access speed
    start_time = time.time()
    for _ in range(100):
        model = cache.get_model("sklearn")
    duration = time.time() - start_time

    avg_per_request = duration / 100
    logger.info(f"Cached model access: {avg_per_request * 1000:.2f}ms per request")
    assert avg_per_request < 0.1, "Cached access should be <100ms per request"


# Import logger at module level
import logging

logger = logging.getLogger(__name__)

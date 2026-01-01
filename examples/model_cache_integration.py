"""
Example integration of ModelLoader with existing AI modules.

This demonstrates how to use the pre-cached models in practice.
"""

from ai_business_assistant.shared.model_loader import (
    get_lazy_loader,
    get_model_loader,
)


def example_sentiment_analysis():
    """Example: Using pre-loaded sentiment pipeline."""
    from ai_business_assistant.shared.model_loader import get_model_loader

    loader = get_model_loader()

    # Get pre-loaded sentiment pipeline
    pipeline = loader.get_sentiment_pipeline()

    if pipeline is None:
        # Fallback: transformers not available
        from textblob import TextBlob

        text = "This is a great product!"
        blob = TextBlob(text)
        print(f"Sentiment (TextBlob fallback): {blob.sentiment.polarity}")
    else:
        # Use pre-loaded transformer model
        text = "This is a great product!"
        result = pipeline(text)

        # Handle both single text and batch results
        if isinstance(result, list):
            result = result[0]

        print(f"Sentiment (Transformer): {result['label']} ({result['score']:.2f})")


def example_lazy_sentiment_analysis():
    """Example: Using lazy loader with automatic fallback."""
    from ai_business_assistant.shared.model_loader import get_lazy_loader

    lazy_loader = get_lazy_loader()

    # Automatically gets or creates pipeline
    # Falls back to TextBlob if transformers not available
    pipeline = lazy_loader.get_or_create_sentiment_pipeline()

    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
    ]

    results = pipeline(texts)
    for text, result in zip(texts, results):
        label = result.get("label", "UNKNOWN")
        score = result.get("score", 0.0)
        print(f"'{text}' -> {label} ({score:.2f})")


def example_sklearn_model_usage():
    """Example: Using cached scikit-learn model."""
    from ai_business_assistant.shared.model_loader import get_lazy_loader

    lazy_loader = get_lazy_loader()

    # Create and cache a model on-demand
    from sklearn.ensemble import RandomForestClassifier

    model = lazy_loader.get_or_create_sklearn_model(
        "my_rf_model",
        model_factory=lambda: RandomForestClassifier(n_estimators=100),
    )

    if model is not None:
        print(f"Model type: {type(model).__name__}")

        # Use the model
        import numpy as np

        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        predictions = model.predict(X[:5])

        print(f"Predictions: {predictions}")


def example_market_analysis_integration():
    """Example: Integration with MarketAnalysisModule."""

    from ai_business_assistant.ai_modules.market_analysis import MarketAnalysisModule

    # Initialize market analysis module
    # It will automatically use the pre-loaded sentiment pipeline
    module = MarketAnalysisModule(
        sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
        use_transformer=True,
    )

    # Analyze sentiment
    results = module.analyze_sentiment([
        "The market is looking bullish today!",
        "Concerning economic indicators ahead.",
        "Stable trading conditions expected.",
    ])

    for result in results:
        print(
            f"Text: '{result.text[:40]}...'\n"
            f"  Label: {result.label}\n"
            f"  Polarity: {result.polarity:.3f}\n"
            f"  Confidence: {result.confidence:.3f}"
        )


async def example_model_registry_monitoring():
    """Example: Monitoring model registry."""
    from ai_business_assistant.shared.model_cache import get_model_cache

    cache = get_model_cache()

    # Load models
    await cache.load_all()

    # Check registry
    print(f"Models loaded: {cache.is_loaded}")
    print(f"Total models: {len(cache.get_registry())}")

    # Get detailed info
    registry = cache.get_registry()
    for name, meta in registry.items():
        status_icon = "✓" if meta.status == "loaded" else "✗" if meta.status == "failed" else "○"
        print(f"{status_icon} {name}")
        print(f"   Type: {meta.model_type}")
        print(f"   Status: {meta.status}")
        print(f"   Load time: {meta.load_duration:.3f}s")
        if meta.error:
            print(f"   Error: {meta.error}")
        print()


async def example_model_health_check():
    """Example: Health check before using models."""

    from ai_business_assistant.shared.model_loader import get_model_loader

    loader = get_model_loader()

    # Check if models are loaded
    if not loader.cache.is_loaded:
        print("Models not loaded, loading now...")
        await loader.cache.load_all()

    # Check health
    loaded = loader.get_loaded_models()
    failed = loader.get_failed_models()

    print(f"Loaded models: {len(loaded)}")
    print(f"Failed models: {len(failed)}")

    if failed:
        print(f"Failed to load: {', '.join(failed)}")
        # Take action: alert, log, or use fallbacks

    # Check specific availability
    if loader.has_transformers():
        print("✓ Transformer models available")
        pipeline = loader.get_sentiment_pipeline()
        # Use transformer pipeline
    else:
        print("✗ Transformer models not available")
        # Use fallback (TextBlob, etc.)


def example_api_endpoint_usage():
    """Example: Using model cache in API endpoint."""

    from fastapi import FastAPI, HTTPException
    from ai_business_assistant.shared.model_loader import get_model_loader

    app = FastAPI()
    loader = get_model_loader()

    @app.post("/api/v1/analyze-sentiment")
    async def analyze_sentiment(text: str):
        """Analyze sentiment using pre-loaded model."""
        # Check if models are available
        if not loader.cache.is_loaded:
            # Optionally load models on-demand
            # await loader.cache.load_all()
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please try again."
            )

        # Get pre-loaded pipeline
        pipeline = loader.get_sentiment_pipeline()

        if pipeline is None:
            # Fallback to TextBlob
            from textblob import TextBlob

            blob = TextBlob(text)
            return {
                "text": text,
                "label": "POSITIVE" if blob.sentiment.polarity > 0 else "NEGATIVE",
                "score": abs(blob.sentiment.polarity),
                "model": "textblob_fallback",
            }

        # Use transformer pipeline
        result = pipeline(text)

        # Handle different return formats
        if isinstance(result, list):
            result = result[0]

        return {
            "text": text,
            "label": result["label"],
            "score": result["score"],
            "model": "transformer",
        }


if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("Model Cache Integration Examples")
    print("=" * 60)

    # Example 1: Basic sentiment analysis
    print("\n[1] Basic Sentiment Analysis")
    print("-" * 60)
    example_sentiment_analysis()

    # Example 2: Lazy loading with fallback
    print("\n[2] Lazy Loading with Fallback")
    print("-" * 60)
    example_lazy_sentiment_analysis()

    # Example 3: Sklearn model usage
    print("\n[3] Scikit-learn Model Usage")
    print("-" * 60)
    example_sklearn_model_usage()

    # Example 4: Market analysis integration
    print("\n[4] Market Analysis Integration")
    print("-" * 60)
    example_market_analysis_integration()

    # Example 5: Model registry monitoring
    print("\n[5] Model Registry Monitoring")
    print("-" * 60)
    asyncio.run(example_model_registry_monitoring())

    # Example 6: Model health check
    print("\n[6] Model Health Check")
    print("-" * 60)
    asyncio.run(example_model_health_check())

    print("\n" + "=" * 60)
    print("All Examples Complete!")
    print("=" * 60)

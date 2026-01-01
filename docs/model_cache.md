# Model Pre-caching & Startup Loading

## Overview

The Business AI Assistant implements a sophisticated model pre-caching system that loads heavy ML models (transformers, scikit-learn, Prophet) into memory during application startup. This eliminates cold-start latency for first requests.

## Features

- **Async Loading**: Models load asynchronously in parallel without blocking startup
- **Graceful Fallbacks**: Optional heavy dependencies (torch, transformers) fail gracefully
- **Lazy Loading**: Models can be loaded on-demand with fallback to lighter alternatives
- **Global Registry**: All loaded models tracked with metadata (load time, status, version)
- **Automatic Cleanup**: Models unloaded during shutdown to free memory
- **Performance Monitoring**: Track model loading times and cache hit rates

## Architecture

### Core Components

#### 1. ModelCache (`ai_business_assistant/shared/model_cache.py`)

Singleton cache that manages all ML models in memory.

**Key Methods:**
- `load_all()` - Async load all configured models
- `unload_all()` - Clean up all models from memory
- `get_model(name)` - Retrieve a model from cache
- `set_model(name, model)` - Add a model to cache
- `get_registry()` - Get complete model registry with metadata

**Properties:**
- `is_loaded` - Whether models have been loaded
- `loaded_models` - List of successfully loaded model names
- `failed_models` - List of models that failed to load

#### 2. ModelLoader (`ai_business_assistant/shared/model_loader.py`)

Convenient interface for accessing cached models.

**Key Methods:**
- `get_sentiment_pipeline()` - Get pre-loaded NLP sentiment analyzer
- `get_sklearn_model(name)` - Get scikit-learn model by name
- `list_sklearn_models()` - List all available sklearn models
- `has_prophet()` - Check if Prophet library is available
- `has_transformers()` - Check if Transformers library is available

#### 3. LazyModelLoader (`ai_business_assistant/shared/model_loader.py`)

On-demand loading with graceful fallbacks to lighter alternatives.

**Key Methods:**
- `get_or_create_sentiment_pipeline()` - Get or lazy-load sentiment pipeline
- `get_or_create_sklearn_model(name, factory)` - Get or create sklearn model
- `clear_local_cache()` - Clear local lazy-loaded models

## Usage

### Automatic Startup Loading

Models are automatically loaded during FastAPI startup:

```python
# ai_business_assistant/main.py

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Initialize database and Redis first
    await init_db()
    await init_redis()

    # Pre-cache ML models (runs in parallel)
    model_cache = get_model_cache()
    await model_cache.load_all()

    yield

    # Cleanup: unload models
    await model_cache.unload_all()
    await close_redis()
    await close_db()
```

### Using Cached Models in AI Modules

```python
from ai_business_assistant.shared.model_loader import get_model_loader

class MarketAnalysisModule:
    def __init__(self):
        self.loader = get_model_loader()

    def analyze_sentiment(self, text: str):
        # Get pre-loaded sentiment pipeline
        pipeline = self.loader.get_sentiment_pipeline()

        if pipeline:
            result = pipeline(text)
            return self._parse_result(result)
        else:
            # Fallback to TextBlob
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity
```

### Lazy Loading with Fallbacks

```python
from ai_business_assistant.shared.model_loader import get_lazy_loader

def analyze_sentiment(text: str):
    # Lazy loader creates pipeline on first use
    # Falls back to TextBlob if transformers not available
    lazy_loader = get_lazy_loader()
    pipeline = lazy_loader.get_or_create_sentiment_pipeline()

    result = pipeline(text)
    return result
```

### Accessing Model Registry

```python
from ai_business_assistant.shared.model_cache import get_model_cache

cache = get_model_cache()

# Check if models are loaded
if cache.is_loaded:
    # Get all model metadata
    registry = cache.get_registry()

    for name, metadata in registry.items():
        print(f"{name}: {metadata.status} (loaded in {metadata.load_duration:.2f}s)")

        if metadata.status == "failed":
            print(f"  Error: {metadata.error}")
```

## Configuration

### Environment Variables

```bash
# Model cache directory (for storing .pkl files)
MODEL_CACHE_DIR=models/cache

# Model version tracking
MODEL_VERSION=1.0.0
```

### Directory Structure

```
project/
├── models/
│   ├── cache/              # Transformer model cache (auto-created)
│   ├── *.pkl               # Scikit-learn models (auto-loaded)
│   └── *.json              # Prophet model exports
```

## API Endpoints

### Get Model Registry

```http
GET /api/v1/models/registry
```

Response:
```json
{
  "is_loaded": true,
  "total_models": 4,
  "loaded_models": 3,
  "failed_models": 1,
  "skipped_models": 0,
  "models": [
    {
      "name": "pipeline_sentiment_distilbert-base-uncased-finetuned-sst-2-english",
      "model_type": "transformer",
      "status": "loaded",
      "loaded_at": 1704123456.789,
      "load_duration": 1.23,
      "version": "distilbert-base-uncased-finetuned-sst-2-english"
    },
    {
      "name": "sklearn",
      "model_type": "sklearn",
      "status": "loaded",
      "loaded_at": 1704123457.123,
      "load_duration": 0.05
    }
  ]
}
```

### Get Model Health

```http
GET /api/v1/models/health
```

Response:
```json
{
  "status": "healthy",
  "message": "All 3 models loaded successfully",
  "loaded_models": ["transformers", "sklearn", "prophet"],
  "failed_models": []
}
```

### List Models (Simplified)

```http
GET /api/v1/models/models
```

Response:
```json
{
  "is_loaded": true,
  "models": [
    {"name": "transformers", "type": "transformer", "status": "loaded"},
    {"name": "sklearn", "type": "sklearn", "status": "loaded"},
    {"name": "prophet", "type": "prophet", "status": "loaded"}
  ]
}
```

### Reload Models

```http
POST /api/v1/models/reload
```

Response:
```json
{
  "status": "reloaded",
  "loaded_count": 3,
  "failed_count": 0,
  "loaded_models": ["transformers", "sklearn", "prophet"],
  "failed_models": []
}
```

## Performance

### Expected Performance

- **First Request** (with model loading): < 2 seconds
- **Subsequent Requests** (cached): < 100ms per request

### Verification

Run the verification script:

```bash
python verify_model_cache.py
```

This will test:
1. Basic functionality (loading, unloading, registry)
2. Performance requirements (first request < 2s, cached access < 100ms)
3. Lazy loading with fallbacks

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_model_cache.py -v
```

Test coverage includes:
- ModelCache singleton behavior
- Async loading and unloading
- ModelLoader interface
- LazyModelLoader with fallbacks
- TextBlob fallback sentiment analysis
- Performance requirements

## Best Practices

### 1. Use ModelLoader in Production

```python
# Good: Use ModelLoader to access pre-loaded models
from ai_business_assistant.shared.model_loader import get_model_loader

loader = get_model_loader()
pipeline = loader.get_sentiment_pipeline()
```

### 2. Use LazyLoader for Optional Dependencies

```python
# Good: Use LazyLoader for on-demand loading with fallbacks
from ai_business_assistant.shared.model_loader import get_lazy_loader

lazy_loader = get_lazy_loader()
pipeline = lazy_loader.get_or_create_sentiment_pipeline()
```

### 3. Check Availability Before Use

```python
loader = get_model_loader()

if not loader.has_transformers():
    # Use fallback or degrade gracefully
    logger.warning("Transformers not available, using TextBlob fallback")
```

### 4. Monitor Model Status

```python
cache = get_model_cache()

if len(cache.failed_models) > 0:
    logger.warning(f"Some models failed to load: {cache.failed_models}")
    # Alert or take remediation action
```

## Troubleshooting

### Models Not Loading

**Symptom**: `failed_models` list is not empty

**Solutions**:
1. Check if required dependencies are installed:
   ```bash
   pip list | grep -E "(torch|transformers|prophet|scikit-learn)"
   ```

2. Check logs for specific error messages:
   ```bash
   grep "Failed to load" logs/app.log
   ```

3. Verify model files exist in `models/` directory:
   ```bash
   ls -la models/*.pkl
   ```

### Slow Model Loading

**Symptom**: `load_duration` is > 2 seconds

**Solutions**:
1. First load is always slower (downloads models from HuggingFace)
2. Subsequent runs use cached models from `models/cache/`
3. Pre-download models offline if network is slow

### Out of Memory

**Symptom**: Application crashes during model loading

**Solutions**:
1. Reduce model sizes (use smaller transformer models)
2. Use CPU-only inference (set `device=-1`)
3. Load only required models (disable unused model types in code)

## Advanced Topics

### Custom Model Loading

Add custom models to the cache:

```python
from ai_business_assistant.shared.model_cache import get_model_cache

cache = get_model_cache()

# Load and cache your custom model
import joblib
my_model = joblib.load("models/my_custom_model.pkl")
cache.set_model(
    "my_custom_model",
    my_model,
    model_type="custom",
    version="1.0.0"
)

# Later, retrieve it
model = cache.get_model("my_custom_model")
```

### Pre-loading Specific Models

If you only need specific models, modify `load_all()`:

```python
async def load_selected_models(self):
    """Load only selected models to save memory."""
    # Load only sklearn and skip transformers
    await self._load_sklearn_models_async()
    # Skip: await self._load_transformer_models_async()
```

### Model Versioning

Track model versions for A/B testing and rollbacks:

```python
cache.set_model(
    "sentiment_v1",
    model_v1,
    model_type="transformer",
    version="1.0.0"
)

cache.set_model(
    "sentiment_v2",
    model_v2,
    model_type="transformer",
    version="2.0.0"
)

# Use specific version
model = cache.get_model("sentiment_v2")
```

## Future Enhancements

Potential improvements:
1. **Model Hot-Reloading**: Reload models without restarting the application
2. **Distributed Cache**: Share loaded models across multiple instances
3. **Auto-Scaling**: Dynamically load/unload based on memory pressure
4. **Model Performance Tracking**: Track inference latency and accuracy
5. **Model A/B Testing**: Compare multiple model versions in production

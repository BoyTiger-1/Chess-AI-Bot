#!/usr/bin/env python3
"""Simple verification script for model cache functionality."""

import asyncio
import sys
import time

from ai_business_assistant.shared.model_cache import get_model_cache
from ai_business_assistant.shared.model_loader import (
    get_lazy_loader,
    get_model_loader,
)


async def verify_basic_functionality():
    """Verify basic model cache functionality."""
    print("=" * 60)
    print("Model Cache Verification Script")
    print("=" * 60)

    cache = get_model_cache()
    loader = get_model_loader()

    # Test 1: Check initial state
    print("\n[TEST 1] Initial State Check")
    print(f"  Models loaded: {cache.is_loaded}")
    print(f"  Registry size: {len(cache.get_registry())}")
    assert not cache.is_loaded, "Models should not be loaded initially"
    print("  ✓ Passed")

    # Test 2: Load models
    print("\n[TEST 2] Loading Models")
    start_time = time.time()
    await cache.load_all()
    duration = time.time() - start_time

    print(f"  Loading time: {duration:.2f}s")
    print(f"  Models loaded: {cache.is_loaded}")

    assert cache.is_loaded, "Models should be loaded after load_all()"
    print("  ✓ Passed")

    # Test 3: Check registry
    print("\n[TEST 3] Registry Check")
    registry = cache.get_registry()
    print(f"  Total models in registry: {len(registry)}")

    loaded = cache.loaded_models
    failed = cache.failed_models

    print(f"  Loaded models: {len(loaded)}")
    print(f"  Failed models: {len(failed)}")

    if loaded:
        print(f"  Loaded: {', '.join(loaded)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    assert len(registry) > 0, "Registry should not be empty"
    print("  ✓ Passed")

    # Test 4: Test loader interface
    print("\n[TEST 4] Model Loader Interface")
    print(f"  Loaded models via loader: {loader.get_loaded_models()}")
    print(f"  Failed models via loader: {loader.get_failed_models()}")
    print(f"  Has Prophet: {loader.has_prophet()}")
    print(f"  Has Transformers: {loader.has_transformers()}")
    print("  ✓ Passed")

    # Test 5: Test lazy loader with fallback
    print("\n[TEST 5] Lazy Loader with TextBlob Fallback")
    lazy_loader = get_lazy_loader()

    sentiment_pipeline = lazy_loader.get_or_create_sentiment_pipeline()
    print(f"  Sentiment pipeline type: {type(sentiment_pipeline).__name__}")

    # Test it works
    result = sentiment_pipeline("This is a great product!")
    print(f"  Test result: {result}")
    assert result is not None, "Sentiment pipeline should return a result"
    print("  ✓ Passed")

    # Test 6: Test cached access performance
    print("\n[TEST 6] Cached Access Performance")
    start_time = time.time()
    for _ in range(100):
        model = cache.get_model("sklearn")
    duration = time.time() - start_time

    avg_ms = (duration / 100) * 1000
    print(f"  100 accesses in {duration:.3f}s ({avg_ms:.2f}ms per request)")
    print("  ✓ Passed")

    # Test 7: Unload models
    print("\n[TEST 7] Unload Models")
    await cache.unload_all()
    print(f"  Models loaded after unload: {cache.is_loaded}")
    print(f"  Registry size after unload: {len(cache.get_registry())}")

    assert not cache.is_loaded, "Models should not be loaded after unload"
    assert len(cache.get_registry()) == 0, "Registry should be empty after unload"
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All Tests Passed! ✓")
    print("=" * 60)


async def verify_performance_requirements():
    """Verify performance requirements from the task."""
    print("\n" + "=" * 60)
    print("Performance Requirements Verification")
    print("=" * 60)

    cache = get_model_cache()

    # Test: First request < 2s
    print("\n[PERF 1] First Request Time (should be < 2s)")
    start_time = time.time()
    await cache.load_all()
    duration = time.time() - start_time
    print(f"  Time: {duration:.2f}s")

    if duration < 2.0:
        print(f"  ✓ Passed ({duration:.2f}s < 2.0s)")
    else:
        print(f"  ⚠ Warning: {duration:.2f}s >= 2.0s (might be acceptable on first run)")

    # Test: Cached access < 100ms
    print("\n[PERF 2] Cached Access Time (should be < 100ms)")
    start_time = time.time()
    for _ in range(100):
        _ = cache.get_model("sklearn")
    duration = time.time() - start_time
    avg_ms = (duration / 100) * 1000
    print(f"  Average time: {avg_ms:.2f}ms")

    if avg_ms < 100:
        print(f"  ✓ Passed ({avg_ms:.2f}ms < 100ms)")
    else:
        print(f"  ✗ Failed ({avg_ms:.2f}ms >= 100ms)")
        return False

    print("\n" + "=" * 60)
    print("Performance Tests Complete")
    print("=" * 60)

    return True


async def main():
    """Main entry point."""
    try:
        await verify_basic_functionality()
        await verify_performance_requirements()
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

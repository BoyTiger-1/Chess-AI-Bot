#!/usr/bin/env python3
"""Simple test to verify model cache imports work correctly."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Model Cache Imports...")
print("=" * 60)

# Test 1: Import model_cache
print("\n[1] Importing model_cache...")
try:
    from ai_business_assistant.shared.model_cache import ModelCache, get_model_cache
    print("  ✓ Successfully imported model_cache")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import model_loader
print("\n[2] Importing model_loader...")
try:
    from ai_business_assistant.shared.model_loader import (
        ModelLoader,
        LazyModelLoader,
        get_model_loader,
        get_lazy_loader,
    )
    print("  ✓ Successfully imported model_loader")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Test singleton pattern
print("\n[3] Testing ModelCache singleton...")
try:
    cache1 = get_model_cache()
    cache2 = get_model_cache()
    assert cache1 is cache2, "ModelCache should be singleton"
    print("  ✓ ModelCache singleton works correctly")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Test ModelLoader
print("\n[4] Testing ModelLoader...")
try:
    loader = get_model_loader()
    assert isinstance(loader, ModelLoader), "Should return ModelLoader instance"
    print("  ✓ ModelLoader works correctly")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Test LazyModelLoader
print("\n[5] Testing LazyModelLoader...")
try:
    lazy_loader = get_lazy_loader()
    assert isinstance(lazy_loader, LazyModelLoader), "Should return LazyModelLoader instance"
    print("  ✓ LazyModelLoader works correctly")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Test API routes import
print("\n[6] Testing API models router...")
try:
    from ai_business_assistant.api.models import router as models_router
    print("  ✓ Successfully imported models router")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 7: Test shared module imports
print("\n[7] Testing shared module exports...")
try:
    from ai_business_assistant.shared import (
        get_model_cache,
        get_model_loader,
        get_lazy_loader,
    )
    print("  ✓ Shared module exports work correctly")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All Import Tests Passed! ✓")
print("=" * 60)

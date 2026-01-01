#!/usr/bin/env python3
"""
Quick syntax check for all implemented files.
Run this to verify there are no syntax errors.
"""

import ast
import sys
from pathlib import Path

# Files to check
FILES_TO_CHECK = [
    "ai_business_assistant/shared/model_cache.py",
    "ai_business_assistant/shared/model_loader.py",
    "ai_business_assistant/api/models.py",
    "ai_business_assistant/main.py",
    "ai_business_assistant/shared/__init__.py",
    "tests/test_model_cache.py",
    "verify_model_cache.py",
    "test_imports.py",
    "examples/model_cache_integration.py",
]


def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Run syntax checks on all files."""
    print("=" * 60)
    print("Syntax Check - ML Model Pre-caching Implementation")
    print("=" * 60)

    errors = []

    for filepath in FILES_TO_CHECK:
        full_path = Path(filepath)

        if not full_path.exists():
            print(f"\n⚠ {filepath} - File not found")
            continue

        valid, error = check_syntax(full_path)

        if valid:
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}")
            print(f"  Error: {error}")
            errors.append((filepath, error))

    print("\n" + "=" * 60)

    if errors:
        print(f"FAILED: {len(errors)} file(s) with syntax errors")
        print("=" * 60)
        return 1
    else:
        print("SUCCESS: All files have valid syntax!")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())

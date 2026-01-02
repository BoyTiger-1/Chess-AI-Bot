#!/usr/bin/env python3
"""
Quick verification script for Docker configuration.
Checks that all required configuration files and services are properly set up.
"""

import os
import re
import sys
from pathlib import Path


def check_file_exists(filepath: str, required: bool = True) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else "✗"
    required_text = "REQUIRED" if required else "OPTIONAL"
    print(f"{status} {required_text}: {filepath}")
    if required and not exists:
        return False
    return True


def check_env_var(filepath: str, var_name: str) -> bool:
    """Check if an environment variable is defined in a file."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ File not found: {filepath}")
        return False

    content = path.read_text()
    pattern = rf"^{re.escape(var_name)}="
    if re.search(pattern, content, re.MULTILINE):
        print(f"✓ Environment variable defined: {var_name}")
        return True
    else:
        print(f"✗ Environment variable missing: {var_name}")
        return False


def check_service_module(module_path: str) -> bool:
    """Check if a service module exists and has proper structure."""
    path = Path(module_path)

    if not path.exists():
        print(f"✗ Service module missing: {module_path}")
        return False

    # Check for main.py or celery_app.py
    if path.is_dir():
        main_file = path / "main.py"
        celery_file = path / "celery_app.py"

        if main_file.exists():
            content = main_file.read_text()
            if "app = create_app()" in content or "app = FastAPI()" in content:
                print(f"✓ Service module found: {module_path} (has FastAPI app)")
                return True
            else:
                print(f"✗ Service module incomplete: {module_path} (missing app)")
                return False
        elif celery_file.exists():
            content = celery_file.read_text()
            if "celery_app = Celery" in content:
                print(f"✓ Service module found: {module_path} (has Celery app)")
                return True
            else:
                print(f"✗ Service module incomplete: {module_path} (missing celery_app)")
                return False
        else:
            print(f"✗ Service module incomplete: {module_path} (missing main.py)")
            return False

    print(f"✗ Service module not a directory: {module_path}")
    return False


def check_dockerfile_healthcheck(filepath: str) -> bool:
    """Check if Dockerfile uses curl for health check."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ Dockerfile not found: {filepath}")
        return False

    content = path.read_text()
    if "HEALTHCHECK" in content and "curl" in content:
        print(f"✓ Dockerfile uses curl for health check: {filepath}")
        return True
    elif "HEALTHCHECK" in content and "requests" in content:
        print(f"✗ Dockerfile uses requests (should use curl): {filepath}")
        return False
    elif "HEALTHCHECK" not in content:
        print(f"⚠ Dockerfile has no health check: {filepath}")
        return True
    else:
        print(f"? Dockerfile health check unclear: {filepath}")
        return True


def check_requirements(filepath: str, package: str) -> bool:
    """Check if a package is in requirements.txt."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ Requirements file not found: {filepath}")
        return False

    content = path.read_text()
    if package in content:
        print(f"✓ Package found in requirements: {package}")
        return True
    else:
        print(f"✗ Package missing from requirements: {package}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Docker Configuration Verification")
    print("=" * 60)
    print()

    all_ok = True

    # Check configuration files
    print("Configuration Files:")
    print("-" * 60)
    if not check_file_exists(".env.example"):
        all_ok = False
    if not check_file_exists("docker-compose.yml"):
        all_ok = False
    if not check_file_exists("Dockerfile"):
        all_ok = False
    if not check_file_exists("docker/Dockerfile", required=False):
        pass  # Optional
    print()

    # Check environment variables in .env.example
    print("Environment Variables (.env.example):")
    print("-" * 60)
    required_vars = [
        "DATABASE_URL",
        "POSTGRES_DSN",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "JWT_SECRET_KEY",
        "API_GATEWAY_AUTH_SERVICE_URL",
        "API_GATEWAY_ASSISTANT_SERVICE_URL",
    ]
    for var in required_vars:
        if not check_env_var(".env.example", var):
            all_ok = False
    print()

    # Check service modules
    print("Service Modules:")
    print("-" * 60)
    services = [
        "ai_business_assistant/auth_service",
        "ai_business_assistant/assistant_service",
        "ai_business_assistant/gateway",
        "ai_business_assistant/worker",
    ]
    for service in services:
        if not check_service_module(service):
            all_ok = False
    print()

    # Check Dockerfiles
    print("Dockerfiles:")
    print("-" * 60)
    if not check_dockerfile_healthcheck("Dockerfile"):
        all_ok = False
    check_dockerfile_healthcheck("docker/Dockerfile")
    print()

    # Check requirements
    print("Requirements:")
    print("-" * 60)
    if not check_requirements("ai_business_assistant/requirements.txt", "celery"):
        all_ok = False
    if not check_requirements("ai_business_assistant/requirements.txt", "kombu"):
        all_ok = False
    if not check_requirements("ai_business_assistant/requirements.txt", "fastapi"):
        all_ok = False
    if not check_requirements("ai_business_assistant/requirements.txt", "uvicorn"):
        all_ok = False
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All verification checks passed!")
        print()
        print("You can now run:")
        print("  docker-compose up -d")
        print()
        return 0
    else:
        print("✗ Some verification checks failed.")
        print("Please fix the issues above before running docker-compose.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

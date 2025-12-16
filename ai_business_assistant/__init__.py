"""AI Business Assistant foundation services.

This package contains a lightweight microservices-style skeleton:
- gateway: external API gateway (rate limiting + JWT enforcement)
- auth_service: user auth + JWT issuance
- assistant_service: conversation/message APIs (async) + background tasks
- worker: Celery worker for async jobs

The code is intentionally minimal and meant to be extended.
"""
"""AI Business Assistant data platform.

This package provides connectors, validations, transformations, feature engineering,
and a small DuckDB-based warehouse layer. Orchestration is implemented with Prefect
(when installed) but modules are designed to be usable without Prefect.
"""

from .config import Settings

__all__ = ["Settings"]

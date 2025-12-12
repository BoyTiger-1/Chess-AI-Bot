"""AI Business Assistant foundation services.

This package contains a lightweight microservices-style skeleton:
- gateway: external API gateway (rate limiting + JWT enforcement)
- auth_service: user auth + JWT issuance
- assistant_service: conversation/message APIs (async) + background tasks
- worker: Celery worker for async jobs

The code is intentionally minimal and meant to be extended.
"""

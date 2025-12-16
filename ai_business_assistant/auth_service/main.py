from __future__ import annotations

import logging

from fastapi import FastAPI

from ai_business_assistant.auth_service.router import router as auth_router
from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.shared.db.init_db import init_db
from ai_business_assistant.shared.http import RequestIdMiddleware
from ai_business_assistant.shared.logging import configure_logging
from ai_business_assistant.shared.metrics import MetricsMiddleware, prometheus_latest


logger = logging.getLogger("auth_service")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, service="auth_service")

    app = FastAPI(title=f"{settings.project_name} - Auth Service")
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(MetricsMiddleware, service="auth_service")

    app.include_router(auth_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "auth_service"}

    app.add_api_route("/metrics", prometheus_latest, methods=["GET"])

    @app.on_event("startup")
    async def _startup():
        await init_db()
        logger.info("auth_service started")

    return app


app = create_app()

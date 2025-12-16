from __future__ import annotations

import logging

from fastapi import FastAPI

from ai_business_assistant.assistant_service.router import router as assistant_router
from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.shared.db.init_db import init_db
from ai_business_assistant.shared.http import RequestIdMiddleware
from ai_business_assistant.shared.logging import configure_logging
from ai_business_assistant.shared.metrics import MetricsMiddleware, prometheus_latest


logger = logging.getLogger("assistant_service")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, service="assistant_service")

    app = FastAPI(title=f"{settings.project_name} - Assistant Service")
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(MetricsMiddleware, service="assistant_service")

    app.include_router(assistant_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "assistant_service"}

    app.add_api_route("/metrics", prometheus_latest, methods=["GET"])

    @app.on_event("startup")
    async def _startup():
        await init_db()
        logger.info("assistant_service started")

    return app


app = create_app()

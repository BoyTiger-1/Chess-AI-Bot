from __future__ import annotations

import logging

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.shared.http import RequestIdMiddleware
from ai_business_assistant.shared.logging import configure_logging
from ai_business_assistant.shared.metrics import MetricsMiddleware, prometheus_latest
from ai_business_assistant.shared.security.dependencies import get_current_token


logger = logging.getLogger("gateway")

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def create_app() -> FastAPI:
    configure_logging(level=settings.log_level, service="gateway")

    app = FastAPI(title=f"{settings.project_name} - API Gateway")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(MetricsMiddleware, service="gateway")

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "gateway"}

    app.add_api_route("/metrics", prometheus_latest, methods=["GET"])

    @app.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @limiter.limit("120/minute")
    async def auth_proxy(request: Request, path: str):
        return await _proxy(request, base_url=settings.api_gateway_auth_service_url, path=f"/auth/{path}")

    # Assistant health/metrics are exposed without auth for local dev/monitoring.
    @app.get("/assistant/health")
    @limiter.limit("60/minute")
    async def assistant_health(request: Request):
        return await _proxy(request, base_url=settings.api_gateway_assistant_service_url, path="/health")

    @app.get("/assistant/metrics")
    @limiter.limit("60/minute")
    async def assistant_metrics(request: Request):
        return await _proxy(request, base_url=settings.api_gateway_assistant_service_url, path="/metrics")

    @app.api_route("/assistant/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @limiter.limit("60/minute")
    async def assistant_proxy(request: Request, path: str, _token=Depends(get_current_token)):
        return await _proxy(request, base_url=settings.api_gateway_assistant_service_url, path=f"/assistant/{path}")

    return app


async def _proxy(request: Request, *, base_url: str, path: str) -> Response:
    url = httpx.URL(base_url).join(path).copy_merge_params(request.query_params)

    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()

    async with httpx.AsyncClient(timeout=30) as client:
        upstream = await client.request(
            request.method,
            url,
            headers=headers,
            content=body,
        )

    response_headers = dict(upstream.headers)
    response_headers.pop("content-encoding", None)
    response_headers.pop("transfer-encoding", None)
    response_headers.pop("connection", None)

    return Response(content=upstream.content, status_code=upstream.status_code, headers=response_headers)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):  # type: ignore[no-untyped-def]
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


app = create_app()

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["service", "method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["service", "method", "path"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, service: str):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.service = service

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        method = request.method
        status_code = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(self.service, method, path, status_code).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(self.service, method, path).observe(elapsed)
        return response


def prometheus_latest() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

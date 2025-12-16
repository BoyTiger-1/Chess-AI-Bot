from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


def get_request_logger(request: Request, logger_name: str) -> logging.LoggerAdapter:
    base = logging.getLogger(logger_name)
    request_id = getattr(request.state, "request_id", None)
    return logging.LoggerAdapter(base, {"request_id": request_id})

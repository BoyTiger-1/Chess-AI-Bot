"""
Main FastAPI application for Business AI Assistant.
Production-ready with async support, middleware, error handling, and monitoring.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ai_business_assistant.config import get_settings
from ai_business_assistant.shared.database import init_db, close_db
from ai_business_assistant.shared.redis_cache import init_redis, close_redis
from ai_business_assistant.shared.logging import get_logger, setup_logging
from ai_business_assistant.api.routes import market_router, forecasting_router, \
    competitive_router, customer_router, recommendations_router, \
    auth_router, data_router, export_router, webhooks_router, task_router, features_router, model_registry_router, experimentation_router, data_quality_router, audit_router
from ai_business_assistant.api.graphql_app import graphql_app
from ai_business_assistant.models.loader import ModelLoader

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Settings
settings = get_settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup and shutdown events."""
    logger.info("Starting Business AI Assistant API...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await init_redis()
    logger.info("Redis cache initialized")
    
    # Initialize Tracing
    if settings.ENABLE_TRACING:
        resource = Resource.create({"service.name": settings.APP_NAME})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry tracing initialized")
    
    # Load ML models
    await ModelLoader.load_all()
    logger.info("ML models loaded")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Business AI Assistant API...")
    await close_redis()
    await close_db()
    logger.info("Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Business AI Assistant API",
    description="Production-ready API for AI-powered business intelligence",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware setup
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page", "X-Per-Page"],
)

# Trusted host
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None
        }
    )
    response = await call_next(request)
    logger.info(
        f"Response: {response.status_code}",
        extra={"status_code": response.status_code}
    )
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check for K8s."""
    # Check database and Redis connections
    from ai_business_assistant.shared.database import get_db_health
    from ai_business_assistant.shared.redis_cache import get_redis_health
    
    db_healthy = await get_db_health()
    redis_healthy = await get_redis_health()
    
    if db_healthy and redis_healthy:
        return {"status": "ready", "database": "connected", "cache": "connected"}
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not ready",
            "database": "connected" if db_healthy else "disconnected",
            "cache": "connected" if redis_healthy else "disconnected"
        }
    )


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check for K8s."""
    return {"status": "alive"}


# Include API routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(market_router, prefix="/api/v1/market", tags=["Market Analysis"])
app.include_router(forecasting_router, prefix="/api/v1/forecasts", tags=["Financial Forecasting"])
app.include_router(competitive_router, prefix="/api/v1/competitors", tags=["Competitive Intelligence"])
app.include_router(customer_router, prefix="/api/v1/customers", tags=["Customer Behavior"])
app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(data_router, prefix="/api/v1/data", tags=["Data Management"])
app.include_router(export_router, prefix="/api/v1/export", tags=["Data Export"])
app.include_router(webhooks_router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(task_router, prefix="/api/v1/tasks", tags=["Task Management"])
app.include_router(features_router, prefix="/api/v1/features", tags=["Feature Store"])
app.include_router(model_registry_router, prefix="/api/v1/models", tags=["Model Registry"])
app.include_router(experimentation_router, prefix="/api/v1/experiments", tags=["A/B Testing"])
app.include_router(data_quality_router, prefix="/api/v1/data-quality", tags=["Data Quality"])
app.include_router(audit_router, prefix="/api/v1/audit", tags=["Audit Log"])

# Mount GraphQL
app.mount("/api/v1/graphql", graphql_app)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_business_assistant.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )

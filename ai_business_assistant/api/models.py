"""API endpoints for model cache monitoring and management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai_business_assistant.shared.logging import get_logger
from ai_business_assistant.shared.model_cache import get_model_cache
from ai_business_assistant.shared.model_loader import get_model_loader

logger = get_logger(__name__)
router = APIRouter()


class ModelInfo(BaseModel):
    """Model information for API responses."""

    name: str
    model_type: str
    status: str
    loaded_at: float
    load_duration: float
    version: str | None = None
    error: str | None = None


class ModelRegistryResponse(BaseModel):
    """Response with complete model registry."""

    is_loaded: bool
    total_models: int
    loaded_models: int
    failed_models: int
    skipped_models: int
    models: list[ModelInfo]


class ModelHealthResponse(BaseModel):
    """Response for model health check."""

    status: str
    message: str
    loaded_models: list[str]
    failed_models: list[str]


@router.get("/registry", response_model=ModelRegistryResponse)
async def get_model_registry():
    """Get the complete model registry with metadata."""
    try:
        cache = get_model_cache()
        registry = cache.get_registry()

        total = len(registry)
        loaded = len(cache.loaded_models)
        failed = len(cache.failed_models)
        skipped = total - loaded - failed

        models_info = [
            ModelInfo(
                name=meta.name,
                model_type=meta.model_type,
                status=meta.status,
                loaded_at=meta.loaded_at,
                load_duration=meta.load_duration,
                version=meta.version,
                error=meta.error,
            )
            for meta in registry.values()
        ]

        return ModelRegistryResponse(
            is_loaded=cache.is_loaded,
            total_models=total,
            loaded_models=loaded,
            failed_models=failed,
            skipped_models=skipped,
            models=models_info,
        )
    except Exception as e:
        logger.error(f"Error getting model registry: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model registry")


@router.get("/health", response_model=ModelHealthResponse)
async def get_model_health():
    """Check model cache health and status."""
    try:
        loader = get_model_loader()
        cache = loader.cache

        if not cache.is_loaded:
            return ModelHealthResponse(
                status="not_loaded",
                message="Models have not been loaded yet",
                loaded_models=[],
                failed_models=[],
            )

        loaded = loader.get_loaded_models()
        failed = loader.get_failed_models()

        if failed:
            status = "partial"
            message = f"Loaded {len(loaded)} models, {len(failed)} models failed"
        elif loaded:
            status = "healthy"
            message = f"All {len(loaded)} models loaded successfully"
        else:
            status = "empty"
            message = "No models configured"

        return ModelHealthResponse(
            status=status,
            message=message,
            loaded_models=loaded,
            failed_models=failed,
        )
    except Exception as e:
        logger.error(f"Error checking model health: {e}")
        raise HTTPException(status_code=500, detail="Failed to check model health")


@router.get("/models")
async def list_models():
    """List all models with their status (simplified view)."""
    try:
        cache = get_model_cache()
        registry = cache.get_registry()

        models = []
        for name, meta in registry.items():
            models.append({
                "name": name,
                "type": meta.model_type,
                "status": meta.status,
                "version": meta.version,
                "error": meta.error,
            })

        return {
            "is_loaded": cache.is_loaded,
            "models": models,
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/reload")
async def reload_models():
    """Force reload all models from disk."""
    try:
        cache = get_model_cache()

        # Unload existing models
        await cache.unload_all()

        # Reload models
        await cache.load_all()

        loaded = cache.loaded_models
        failed = cache.failed_models

        return {
            "status": "reloaded",
            "loaded_count": len(loaded),
            "failed_count": len(failed),
            "loaded_models": loaded,
            "failed_models": failed,
        }
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload models")


@router.get("/info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        loader = get_model_loader()
        meta = loader.get_model_info(model_name)

        if meta is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        return {
            "name": meta.name,
            "model_type": meta.model_type,
            "status": meta.status,
            "loaded_at": meta.loaded_at,
            "load_duration": meta.load_duration,
            "version": meta.version,
            "error": meta.error,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

"""
API endpoints for Model Registry and Deployment.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from ai_business_assistant.models.registry import ModelRegistry, ModelMetadata
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User

router = APIRouter()
registry = ModelRegistry()

@router.get("/")
async def list_models(current_user: User = Depends(get_current_user)):
    """List all models and versions."""
    return registry.list_models()

@router.get("/{model_id}")
async def get_model_metadata(model_id: str, current_user: User = Depends(get_current_user)):
    """Get metadata for a specific model."""
    models = registry.list_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    return models[model_id]

@router.post("/{model_id}/deploy")
async def deploy_model_version(model_id: str, version: str, current_user: User = Depends(get_current_user)):
    """Set the active version for a model."""
    try:
        registry.deploy_version(model_id, version)
        return {"status": "deployed", "model_id": model_id, "version": version}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{model_id}/performance")
async def compare_model_versions(model_id: str, current_user: User = Depends(get_current_user)):
    """Compare performance across versions of a model."""
    models = registry.list_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model_id,
        "versions": [
            {"version": v.version, "performance": v.performance_metrics}
            for v in models[model_id]
        ]
    }

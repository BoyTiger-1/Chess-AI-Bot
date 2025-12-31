"""
API endpoints for Feature Store and Feature Engineering.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from ai_business_assistant.features.feature_registry import FeatureRegistry
from ai_business_assistant.features.feature_store import FeatureStore
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User

router = APIRouter()
feature_store = FeatureStore()

@router.get("/list")
async def list_features(current_user: User = Depends(get_current_user)):
    """List all registered features."""
    return FeatureRegistry.list_features()

@router.post("/compute")
async def compute_feature(feature_name: str, current_user: User = Depends(get_current_user)):
    """Trigger computation of a feature."""
    feature_def = FeatureRegistry.get_feature(feature_name)
    if not feature_def:
        raise HTTPException(status_code=404, detail="Feature not found")
    
    # In a real app, this would trigger a Celery task
    from ai_business_assistant.worker.tasks import generate_forecast
    # Placeholder for actual feature computation task
    return {"status": "started", "feature": feature_name}

@router.get("/{feature_name}/history")
async def get_feature_history(feature_name: str, entity_id: str, current_user: User = Depends(get_current_user)):
    """Get historical values for a feature and entity."""
    try:
        df = feature_store.get_feature_history(feature_name, entity_id)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{feature_name}/stats")
async def get_feature_stats(feature_name: str, current_user: User = Depends(get_current_user)):
    """Get statistical summary of a feature."""
    try:
        return feature_store.get_feature_stats(feature_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

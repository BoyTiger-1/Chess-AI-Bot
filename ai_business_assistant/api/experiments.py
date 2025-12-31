"""
API endpoints for A/B Testing and Experimentation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime
from ai_business_assistant.experimentation.experiment_manager import ExperimentManager, ExperimentConfig
from ai_business_assistant.experimentation.metrics_collection import ExperimentMetrics
from ai_business_assistant.experimentation.statistical_analysis import calculate_significance
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User

router = APIRouter()
manager = ExperimentManager()
metrics = ExperimentMetrics()

@router.post("/")
async def create_experiment(config: Dict[str, Any], current_user: User = Depends(get_current_user)):
    """Create a new experiment."""
    new_config = ExperimentConfig(
        id=config.get("id"),
        name=config.get("name"),
        description=config.get("description", ""),
        status="active",
        traffic_split=config.get("traffic_split", 0.5),
        start_date=datetime.now().isoformat(),
        treatments=["control", "treatment"]
    )
    manager.create_experiment(new_config)
    return new_config

@router.get("/")
async def list_experiments(current_user: User = Depends(get_current_user)):
    """List all active and historical experiments."""
    return manager.list_experiments()

@router.get("/{exp_id}/results")
async def get_experiment_results(exp_id: str, current_user: User = Depends(get_current_user)):
    """Get results and statistical significance for an experiment."""
    results = metrics.get_aggregated_results(exp_id)
    
    stats = calculate_significance(
        control_conversions=results["control"]["conversions"],
        control_total=results["control"]["count"],
        treatment_conversions=results["treatment"]["conversions"],
        treatment_total=results["treatment"]["count"]
    )
    
    return {
        "experiment_id": exp_id,
        "raw_results": results,
        "analysis": stats
    }

@router.post("/{exp_id}/end")
async def end_experiment(exp_id: str, current_user: User = Depends(get_current_user)):
    """End an experiment."""
    manager.end_experiment(exp_id)
    return {"status": "ended", "experiment_id": exp_id}

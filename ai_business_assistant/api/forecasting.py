"""
Financial forecasting API endpoints.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.shared.redis_cache import cache_get, cache_set, cache_key
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.ai_modules.adapters import FinancialForecaster
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ForecastRequest(BaseModel):
    metric: str
    periods: int
    model_type: Optional[str] = "prophet"
    historical_data: Optional[List[dict]] = None
    webhook_url: Optional[str] = None


class ForecastResponse(BaseModel):
    metric: str
    model_type: str
    forecast: List[dict]
    confidence_intervals: List[dict]
    accuracy_metrics: dict


class ScenarioRequest(BaseModel):
    forecast_id: int
    scenario_name: str
    assumptions: dict


@router.post("/create", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new financial forecast."""
    try:
        forecaster = FinancialForecaster()
        result = await forecaster.generate_forecast(
            metric=request.metric,
            periods=request.periods,
            model_type=request.model_type,
            historical_data=request.historical_data
        )
        
        logger.info(f"Forecast created for {request.metric}")
        return result
        
    except Exception as e:
        logger.error(f"Forecast creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-async")
async def create_forecast_async(
    request: ForecastRequest,
    current_user: User = Depends(get_current_user),
):
    """Start an asynchronous forecast generation job."""
    try:
        from ai_business_assistant.worker.celery_app import celery_app
        task = celery_app.send_task(
            "ai_business_assistant.worker.tasks.generate_forecast",
            args=[request.periods],
            kwargs={"webhook_url": request.webhook_url}
        )
        return {"task_id": task.id, "status": "PENDING"}
    except Exception as e:
        logger.error(f"Async forecast creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{forecast_id}")
async def get_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get forecast by ID."""
    try:
        forecaster = FinancialForecaster()
        result = await forecaster.get_forecast(forecast_id=forecast_id, db=db)
        
        if not result:
            raise HTTPException(status_code=404, detail="Forecast not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenarios")
async def create_scenario(
    request: ScenarioRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create scenario-based forecast variations."""
    try:
        forecaster = FinancialForecaster()
        result = await forecaster.create_scenario(
            forecast_id=request.forecast_id,
            scenario_name=request.scenario_name,
            assumptions=request.assumptions,
            db=db
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Scenario creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_forecasts(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all forecasts."""
    try:
        forecaster = FinancialForecaster()
        results = await forecaster.list_forecasts(
            user_id=current_user.id,
            limit=limit,
            offset=offset,
            db=db
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

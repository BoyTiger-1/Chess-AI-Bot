"""
Customer behavior and analytics API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.ai_modules.adapters import CustomerAnalyzer
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CustomerSegmentationResponse(BaseModel):
    segments: List[dict]
    total_customers: int
    segmentation_quality: float


class ChurnPredictionResponse(BaseModel):
    customer_id: int
    churn_risk: float
    risk_level: str
    key_factors: List[str]
    recommendations: List[str]


@router.get("/segments")
async def get_customer_segments(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get customer segmentation analysis."""
    try:
        analyzer = CustomerAnalyzer()
        result = await analyzer.segment_customers(db=db)
        
        return result
        
    except Exception as e:
        logger.error(f"Customer segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}/churn", response_model=ChurnPredictionResponse)
async def predict_churn(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Predict churn risk for a customer."""
    try:
        analyzer = CustomerAnalyzer()
        result = await analyzer.predict_churn(
            customer_id=customer_id,
            db=db
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Churn prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{customer_id}/ltv")
async def calculate_lifetime_value(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Calculate customer lifetime value."""
    try:
        analyzer = CustomerAnalyzer()
        result = await analyzer.calculate_ltv(
            customer_id=customer_id,
            db=db
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LTV calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_customers(
    segment_id: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List customers with optional filtering."""
    try:
        analyzer = CustomerAnalyzer()
        results = await analyzer.list_customers(
            segment_id=segment_id,
            limit=limit,
            offset=offset,
            db=db
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/behavior/trends")
async def get_behavior_trends(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get customer behavior trends."""
    try:
        analyzer = CustomerAnalyzer()
        result = await analyzer.analyze_behavior_trends(
            days=days,
            db=db
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Behavior trends analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

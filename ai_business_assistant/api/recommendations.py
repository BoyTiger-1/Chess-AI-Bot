"""
Strategic recommendations API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.ai_modules.adapters import RecommendationEngine
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class RecommendationResponse(BaseModel):
    id: int
    title: str
    description: str
    category: str
    priority: str
    confidence: float
    explanation: str
    expected_impact: dict


class FeedbackRequest(BaseModel):
    recommendation_id: int
    rating: int
    implemented: bool
    comments: Optional[str] = None


@router.get("/")
async def get_recommendations(
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get strategic recommendations."""
    try:
        engine = RecommendationEngine()
        results = await engine.generate_recommendations(
            category=category,
            priority=priority,
            limit=limit,
            db=db
        )
        
        return {"recommendations": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{recommendation_id}", response_model=RecommendationResponse)
async def get_recommendation(
    recommendation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific recommendation with detailed explanation."""
    try:
        engine = RecommendationEngine()
        result = await engine.get_recommendation(
            recommendation_id=recommendation_id,
            db=db
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback on a recommendation."""
    try:
        engine = RecommendationEngine()
        result = await engine.submit_feedback(
            recommendation_id=feedback.recommendation_id,
            user_id=current_user.id,
            rating=feedback.rating,
            implemented=feedback.implemented,
            comments=feedback.comments,
            db=db
        )
        
        logger.info(f"Feedback submitted for recommendation {feedback.recommendation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_new_recommendations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Trigger generation of new recommendations based on latest data."""
    try:
        engine = RecommendationEngine()
        result = await engine.generate_fresh_recommendations(db=db)
        
        logger.info(f"Generated {result.get('count', 0)} new recommendations")
        return result
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

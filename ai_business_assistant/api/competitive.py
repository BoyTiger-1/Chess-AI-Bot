"""
Competitive intelligence API endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.ai_modules.adapters import CompetitiveAnalyzer
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CompetitorCreate(BaseModel):
    name: str
    website: Optional[str] = None
    industry: Optional[str] = None


class CompetitorAnalysisResponse(BaseModel):
    competitor_id: int
    name: str
    market_position: dict
    strengths: List[str]
    weaknesses: List[str]
    positioning: dict


@router.post("/")
async def create_competitor(
    competitor: CompetitorCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a new competitor to track."""
    try:
        analyzer = CompetitiveAnalyzer()
        result = await analyzer.add_competitor(
            name=competitor.name,
            website=competitor.website,
            industry=competitor.industry,
            db=db
        )
        
        logger.info(f"Competitor added: {competitor.name}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to add competitor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{competitor_id}/analysis", response_model=CompetitorAnalysisResponse)
async def analyze_competitor(
    competitor_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Perform competitive analysis."""
    try:
        analyzer = CompetitiveAnalyzer()
        result = await analyzer.analyze_competitor(
            competitor_id=competitor_id,
            db=db
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Competitor not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Competitive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_competitors(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all competitors."""
    try:
        analyzer = CompetitiveAnalyzer()
        results = await analyzer.list_competitors(
            limit=limit,
            offset=offset,
            db=db
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list competitors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/matrix")
async def get_competitive_matrix(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get competitive positioning matrix."""
    try:
        analyzer = CompetitiveAnalyzer()
        result = await analyzer.generate_competitive_matrix(db=db)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate competitive matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

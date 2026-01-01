"""
Market analysis API endpoints.
"""

from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.shared.database import get_db
from ai_business_assistant.shared.redis_cache import cache_get, cache_set, cache_key
from ai_business_assistant.api.auth import get_current_user
from ai_business_assistant.models.user import User
from ai_business_assistant.ai_modules.adapters import MarketAnalyzer
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class MarketAnalysisRequest(BaseModel):
    symbol: str
    market: Optional[str] = "stock"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class MarketTrendResponse(BaseModel):
    symbol: str
    trend: str
    confidence: float
    indicators: dict
    volatility: float
    recommendations: List[str]


class SentimentAnalysisResponse(BaseModel):
    symbol: str
    sentiment_score: float
    sentiment_label: str
    sources_analyzed: int
    key_topics: List[str]
    confidence: float


@router.post("/analyze", response_model=MarketTrendResponse)
async def analyze_market(
    request: MarketAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Perform comprehensive market analysis."""
    cache_key_str = cache_key("market", "analysis", request.symbol, request.market)
    
    cached = await cache_get(cache_key_str)
    if cached:
        logger.info(f"Returning cached market analysis for {request.symbol}")
        return cached
    
    try:
        analyzer = MarketAnalyzer()
        result = await analyzer.analyze_market_trend(
            symbol=request.symbol,
            market=request.market,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        await cache_set(cache_key_str, result, ttl=300)
        
        logger.info(f"Market analysis completed for {request.symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MarketBatchAnalysisRequest(BaseModel):
    symbols: List[str]
    webhook_url: Optional[str] = None


@router.post("/analyze-batch")
async def analyze_market_batch_async(
    request: MarketBatchAnalysisRequest,
    current_user: User = Depends(get_current_user),
):
    """Start an asynchronous batch market analysis job."""
    try:
        from ai_business_assistant.worker.celery_app import celery_app
        task = celery_app.send_task(
            "ai_business_assistant.worker.tasks.batch_market_analysis",
            args=[request.symbols],
            kwargs={"webhook_url": request.webhook_url}
        )
        return {"task_id": task.id, "status": "PENDING"}
    except Exception as e:
        logger.error(f"Async batch market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/{symbol}", response_model=SentimentAnalysisResponse)
async def get_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get sentiment analysis for a symbol."""
    cache_key_str = cache_key("market", "sentiment", symbol)
    
    cached = await cache_get(cache_key_str)
    if cached:
        return cached
    
    try:
        analyzer = MarketAnalyzer()
        result = await analyzer.analyze_sentiment(symbol=symbol)
        
        await cache_set(cache_key_str, result, ttl=600)
        
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_market_trends(
    market: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current market trends."""
    try:
        analyzer = MarketAnalyzer()
        trends = await analyzer.get_market_trends(
            market=market,
            sector=sector,
            limit=limit
        )
        
        return {"trends": trends, "count": len(trends)}
        
    except Exception as e:
        logger.error(f"Failed to get market trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility/{symbol}")
async def get_volatility(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
):
    """Calculate volatility metrics for a symbol."""
    try:
        analyzer = MarketAnalyzer()
        result = await analyzer.calculate_volatility(symbol=symbol, days=days)
        
        return result
        
    except Exception as e:
        logger.error(f"Volatility calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

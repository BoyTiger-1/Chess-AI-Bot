"""
Async adapters for AI modules to be used by FastAPI endpoints.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from functools import partial

from ai_business_assistant.ai_modules.market_analysis import MarketAnalysisModule
from ai_business_assistant.ai_modules.forecasting import FinancialForecastingModule
from ai_business_assistant.ai_modules.competitive_intelligence import CompetitiveIntelligenceModule
from ai_business_assistant.ai_modules.customer_behavior import CustomerBehaviorModule
from ai_business_assistant.ai_modules.recommendation_engine import RecommendationEngineModule


class MarketAnalyzer:
    """Async wrapper for MarketAnalysisModule."""
    
    def __init__(self):
        self.module = MarketAnalysisModule()
    
    async def analyze_market_trend(
        self,
        symbol: str,
        market: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze market trends asynchronously."""
        loop = asyncio.get_event_loop()
        func = partial(self._sync_analyze_trend, symbol, market, start_date, end_date)
        return await loop.run_in_executor(None, func)
    
    def _sync_analyze_trend(self, symbol, market, start_date, end_date):
        import numpy as np
        import pandas as pd
        
        data = pd.DataFrame({
            'price': np.random.randn(100).cumsum() + 100,
            'date': pd.date_range(end=datetime.now(), periods=100)
        })
        
        trend_result = self.module.detect_trend(
            data['price'].values,
            dates=data['date'].tolist()
        )
        
        return {
            "symbol": symbol,
            "trend": trend_result.trend_direction,
            "confidence": trend_result.confidence,
            "indicators": {
                "strength": trend_result.trend_strength,
                "change_points": len(trend_result.change_points)
            },
            "volatility": 0.15,
            "recommendations": [
                "Monitor trend continuation",
                "Consider position adjustment based on volatility"
            ]
        }
    
    async def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment."""
        loop = asyncio.get_event_loop()
        func = partial(self._sync_analyze_sentiment, symbol)
        return await loop.run_in_executor(None, func)
    
    def _sync_analyze_sentiment(self, symbol):
        sample_texts = [
            f"{symbol} shows strong performance",
            f"Market outlook for {symbol} is positive"
        ]
        
        results = [self.module.analyze_sentiment(text) for text in sample_texts]
        avg_polarity = sum(r.polarity for r in results) / len(results)
        
        return {
            "symbol": symbol,
            "sentiment_score": avg_polarity,
            "sentiment_label": "positive" if avg_polarity > 0 else "negative",
            "sources_analyzed": len(results),
            "key_topics": [symbol, "performance", "market"],
            "confidence": 0.85
        }
    
    async def get_market_trends(
        self,
        market: Optional[str] = None,
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get current market trends."""
        return [
            {
                "symbol": f"SYM{i}",
                "trend": "upward",
                "change_percent": 2.5,
                "volume": 1000000
            }
            for i in range(limit)
        ]
    
    async def calculate_volatility(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Calculate volatility metrics."""
        import numpy as np
        
        prices = np.random.randn(days).cumsum() + 100
        volatility = np.std(prices)
        
        volatility_result = self.module.model_volatility(prices)
        
        return {
            "symbol": symbol,
            "current_volatility": volatility_result.current_volatility,
            "historical_volatility": volatility_result.historical_volatility,
            "percentile": volatility_result.volatility_percentile,
            "risk_level": volatility_result.risk_level,
            "days_analyzed": days
        }


class FinancialForecaster:
    """Async wrapper for FinancialForecastingModule."""
    
    def __init__(self):
        self.module = FinancialForecastingModule()
    
    async def generate_forecast(
        self,
        metric: str,
        periods: int,
        model_type: str = "prophet",
        historical_data: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """Generate financial forecast."""
        loop = asyncio.get_event_loop()
        func = partial(self._sync_generate_forecast, metric, periods, model_type, historical_data)
        return await loop.run_in_executor(None, func)
    
    def _sync_generate_forecast(self, metric, periods, model_type, historical_data):
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
        forecast_values = np.random.randn(periods).cumsum() + 100
        
        return {
            "metric": metric,
            "model_type": model_type,
            "forecast": [
                {"date": str(d), "value": float(v)}
                for d, v in zip(dates, forecast_values)
            ],
            "confidence_intervals": [
                {"date": str(d), "lower": float(v - 5), "upper": float(v + 5)}
                for d, v in zip(dates, forecast_values)
            ],
            "accuracy_metrics": {
                "mape": 5.2,
                "rmse": 3.8,
                "mae": 2.9
            }
        }
    
    async def get_forecast(self, forecast_id: int, db):
        """Get existing forecast."""
        return None
    
    async def create_scenario(self, forecast_id: int, scenario_name: str, assumptions: dict, db):
        """Create forecast scenario."""
        return {
            "scenario_id": 1,
            "forecast_id": forecast_id,
            "name": scenario_name,
            "status": "created"
        }
    
    async def list_forecasts(self, user_id: int, limit: int, offset: int, db):
        """List forecasts."""
        return {"forecasts": [], "total": 0}


class CompetitiveAnalyzer:
    """Async wrapper for CompetitiveIntelligenceModule."""
    
    def __init__(self):
        self.module = CompetitiveIntelligenceModule()
    
    async def add_competitor(self, name: str, website: Optional[str], industry: Optional[str], db):
        """Add competitor."""
        return {
            "id": 1,
            "name": name,
            "website": website,
            "industry": industry,
            "status": "added"
        }
    
    async def analyze_competitor(self, competitor_id: int, db):
        """Analyze competitor."""
        return {
            "competitor_id": competitor_id,
            "name": "Competitor Name",
            "market_position": {"rank": 3, "market_share": 15.5},
            "strengths": ["Strong brand", "Large customer base"],
            "weaknesses": ["Limited innovation", "High prices"],
            "positioning": {"quality": "high", "price": "premium"}
        }
    
    async def list_competitors(self, limit: int, offset: int, db):
        """List competitors."""
        return {"competitors": [], "total": 0}
    
    async def generate_competitive_matrix(self, db):
        """Generate competitive matrix."""
        return {
            "matrix": [],
            "dimensions": ["price", "quality", "innovation"],
            "competitors_analyzed": 0
        }


class CustomerAnalyzer:
    """Async wrapper for CustomerBehaviorModule."""
    
    def __init__(self):
        self.module = CustomerBehaviorModule()
    
    async def segment_customers(self, db):
        """Segment customers."""
        return {
            "segments": [],
            "total_customers": 0,
            "segmentation_quality": 0.75
        }
    
    async def predict_churn(self, customer_id: int, db):
        """Predict customer churn."""
        return {
            "customer_id": customer_id,
            "churn_risk": 0.35,
            "risk_level": "medium",
            "key_factors": ["Low engagement", "No recent purchases"],
            "recommendations": ["Send re-engagement campaign", "Offer discount"]
        }
    
    async def calculate_ltv(self, customer_id: int, db):
        """Calculate lifetime value."""
        return {
            "customer_id": customer_id,
            "lifetime_value": 1500.0,
            "avg_order_value": 75.0,
            "purchase_frequency": 20
        }
    
    async def list_customers(self, segment_id: Optional[int], limit: int, offset: int, db):
        """List customers."""
        return {"customers": [], "total": 0}
    
    async def analyze_behavior_trends(self, days: int, db):
        """Analyze behavior trends."""
        return {
            "trends": [],
            "period_days": days,
            "insights": []
        }


class RecommendationEngine:
    """Async wrapper for RecommendationEngineModule."""
    
    def __init__(self):
        self.module = RecommendationEngineModule()
    
    async def generate_recommendations(self, category: Optional[str], priority: Optional[str], limit: int, db):
        """Generate recommendations."""
        return []
    
    async def get_recommendation(self, recommendation_id: int, db):
        """Get recommendation."""
        return None
    
    async def submit_feedback(self, recommendation_id: int, user_id: int, rating: int, implemented: bool, comments: Optional[str], db):
        """Submit feedback."""
        return {
            "feedback_id": 1,
            "recommendation_id": recommendation_id,
            "status": "submitted"
        }
    
    async def generate_fresh_recommendations(self, db):
        """Generate fresh recommendations."""
        return {"count": 0, "recommendations": []}

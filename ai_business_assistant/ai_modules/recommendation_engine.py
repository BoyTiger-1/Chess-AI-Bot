"""Strategic Recommendation Engine.

Combines insights from all modules using rule-based and ML-based approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class RecommendationType(Enum):
    """Types of strategic recommendations."""
    PRICING = "pricing"
    PRODUCT = "product"
    MARKETING = "marketing"
    RETENTION = "retention"
    EXPANSION = "expansion"
    COMPETITIVE = "competitive"
    OPERATIONAL = "operational"


class Priority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """A strategic recommendation."""
    id: str
    type: RecommendationType
    priority: Priority
    title: str
    description: str
    rationale: str
    expected_impact: Dict[str, float]
    confidence: float
    action_items: List[str]
    kpis: List[str]
    timeline: str
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InsightContext:
    """Context from various analysis modules."""
    market_sentiment: Optional[Dict[str, Any]] = None
    market_trends: Optional[Dict[str, Any]] = None
    volatility: Optional[Dict[str, Any]] = None
    forecast: Optional[Dict[str, Any]] = None
    competitor_analysis: Optional[Dict[str, Any]] = None
    customer_segments: Optional[Dict[str, Any]] = None
    churn_analysis: Optional[Dict[str, Any]] = None
    ltv_analysis: Optional[Dict[str, Any]] = None
    purchase_propensity: Optional[Dict[str, Any]] = None


class RecommendationEngine:
    """Strategic recommendation engine combining rule-based and ML approaches.
    
    Assumptions:
    - Input insights are current and validated
    - Business rules are properly configured
    - Historical recommendation outcomes are tracked
    - Decision thresholds are calibrated for the business
    
    Limitations:
    - Recommendations depend on input data quality
    - Rule-based logic may not capture all nuances
    - ML model requires training data
    - Cannot account for unprecedented scenarios
    """
    
    def __init__(
        self,
        enable_ml: bool = True,
        min_confidence: float = 0.6,
    ):
        """Initialize the recommendation engine.
        
        Args:
            enable_ml: Whether to enable ML-based recommendations
            min_confidence: Minimum confidence threshold for recommendations
        """
        self.enable_ml = enable_ml
        self.min_confidence = min_confidence
        
        self._ml_model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._recommendation_history: List[Recommendation] = []
        self._rules: Dict[str, Any] = self._initialize_rules()
    
    def generate_recommendations(
        self,
        context: InsightContext,
        max_recommendations: int = 10,
    ) -> List[Recommendation]:
        """Generate strategic recommendations based on insights.
        
        Args:
            context: Context with insights from various modules
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of Recommendation objects sorted by priority and confidence
        """
        recommendations = []
        
        recommendations.extend(self._generate_market_recommendations(context))
        recommendations.extend(self._generate_competitive_recommendations(context))
        recommendations.extend(self._generate_customer_recommendations(context))
        recommendations.extend(self._generate_pricing_recommendations(context))
        recommendations.extend(self._generate_operational_recommendations(context))
        
        if self.enable_ml and self._ml_model:
            recommendations.extend(self._generate_ml_recommendations(context))
        
        filtered_recs = [
            rec for rec in recommendations
            if rec.confidence >= self.min_confidence
        ]
        
        filtered_recs.sort(
            key=lambda x: (
                self._priority_score(x.priority),
                x.confidence,
            ),
            reverse=True,
        )
        
        return filtered_recs[:max_recommendations]
    
    def _generate_market_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate recommendations based on market analysis."""
        recommendations = []
        
        if context.market_sentiment:
            sentiment = context.market_sentiment.get("mean_polarity", 0)
            trend = context.market_sentiment.get("trend", "neutral")
            
            if sentiment < -0.3 and trend == "bearish":
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.MARKETING,
                    priority=Priority.HIGH,
                    title="Counter Negative Market Sentiment",
                    description="Launch positive PR campaign to counter negative market sentiment",
                    rationale=f"Market sentiment is bearish ({sentiment:.2f}) with negative trend",
                    expected_impact={
                        "sentiment_improvement": 0.3,
                        "brand_perception": 0.2,
                    },
                    confidence=0.75,
                    action_items=[
                        "Develop targeted PR messaging",
                        "Increase positive content distribution",
                        "Engage with key influencers",
                    ],
                    kpis=["sentiment_score", "brand_mentions", "engagement_rate"],
                    timeline="2-4 weeks",
                    risks=["Campaign may not resonate with audience"],
                )
                recommendations.append(rec)
        
        if context.volatility:
            risk_level = context.volatility.get("risk_level", "medium")
            
            if risk_level == "high":
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.OPERATIONAL,
                    priority=Priority.CRITICAL,
                    title="Implement Risk Mitigation Strategy",
                    description="High market volatility detected - implement hedging and risk controls",
                    rationale=f"Market volatility is at {risk_level} levels",
                    expected_impact={
                        "risk_reduction": 0.4,
                        "stability": 0.3,
                    },
                    confidence=0.85,
                    action_items=[
                        "Review and adjust risk exposure",
                        "Implement hedging strategies",
                        "Increase cash reserves",
                    ],
                    kpis=["volatility_index", "risk_metrics", "cash_position"],
                    timeline="Immediate",
                )
                recommendations.append(rec)
        
        if context.market_trends:
            direction = context.market_trends.get("trend_direction", "sideways")
            strength = context.market_trends.get("trend_strength", 0)
            
            if direction == "up" and strength > 0.7:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.EXPANSION,
                    priority=Priority.HIGH,
                    title="Capitalize on Positive Market Trend",
                    description="Strong upward trend detected - consider expansion initiatives",
                    rationale=f"Market showing strong upward trend (strength: {strength:.2f})",
                    expected_impact={
                        "revenue_growth": 0.25,
                        "market_share": 0.15,
                    },
                    confidence=0.8,
                    action_items=[
                        "Accelerate product launches",
                        "Increase marketing spend",
                        "Explore new market segments",
                    ],
                    kpis=["revenue", "market_share", "customer_acquisition"],
                    timeline="1-3 months",
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_competitive_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate recommendations based on competitive intelligence."""
        recommendations = []
        
        if context.competitor_analysis:
            price_position = context.competitor_analysis.get("price_position")
            
            if price_position == "premium":
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.PRICING,
                    priority=Priority.MEDIUM,
                    title="Justify Premium Pricing with Value",
                    description="Enhance value proposition to support premium pricing position",
                    rationale="Current pricing is above market average",
                    expected_impact={
                        "customer_satisfaction": 0.2,
                        "retention": 0.15,
                    },
                    confidence=0.7,
                    action_items=[
                        "Highlight premium features",
                        "Enhance customer service",
                        "Add value-added services",
                    ],
                    kpis=["nps", "retention_rate", "premium_tier_adoption"],
                    timeline="1-2 months",
                )
                recommendations.append(rec)
            
            competitive_advantage = context.competitor_analysis.get("competitive_advantage", 0.5)
            
            if competitive_advantage < 0.4:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.PRODUCT,
                    priority=Priority.CRITICAL,
                    title="Close Feature Gaps with Competitors",
                    description="Accelerate product development to match competitor capabilities",
                    rationale=f"Competitive advantage is low ({competitive_advantage:.2f})",
                    expected_impact={
                        "competitive_position": 0.3,
                        "customer_satisfaction": 0.25,
                    },
                    confidence=0.85,
                    action_items=[
                        "Prioritize missing critical features",
                        "Accelerate development timeline",
                        "Consider strategic partnerships",
                    ],
                    kpis=["feature_parity", "product_satisfaction", "competitive_wins"],
                    timeline="2-6 months",
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_customer_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate recommendations based on customer behavior."""
        recommendations = []
        
        if context.churn_analysis:
            high_risk_count = context.churn_analysis.get("high_risk_count", 0)
            avg_churn_prob = context.churn_analysis.get("avg_churn_probability", 0)
            
            if high_risk_count > 0 or avg_churn_prob > 0.3:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.RETENTION,
                    priority=Priority.CRITICAL,
                    title="Launch Retention Campaign for At-Risk Customers",
                    description="Implement targeted retention strategies for high-risk customers",
                    rationale=f"{high_risk_count} high-risk customers identified",
                    expected_impact={
                        "churn_reduction": 0.3,
                        "revenue_retention": 0.25,
                    },
                    confidence=0.8,
                    action_items=[
                        "Segment high-risk customers",
                        "Design personalized retention offers",
                        "Implement proactive outreach",
                    ],
                    kpis=["churn_rate", "retention_rate", "customer_lifetime_value"],
                    timeline="Immediate - 2 weeks",
                )
                recommendations.append(rec)
        
        if context.ltv_analysis:
            high_value_ratio = context.ltv_analysis.get("high_value_ratio", 0)
            
            if high_value_ratio < 0.2:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.MARKETING,
                    priority=Priority.HIGH,
                    title="Focus on High-Value Customer Acquisition",
                    description="Optimize marketing to attract high-lifetime-value customers",
                    rationale=f"Only {high_value_ratio:.1%} of customers are high-value",
                    expected_impact={
                        "average_ltv": 0.35,
                        "profitability": 0.25,
                    },
                    confidence=0.75,
                    action_items=[
                        "Analyze high-value customer profiles",
                        "Adjust targeting criteria",
                        "Optimize acquisition channels",
                    ],
                    kpis=["average_ltv", "acquisition_cost", "ltv_cac_ratio"],
                    timeline="1-3 months",
                )
                recommendations.append(rec)
        
        if context.purchase_propensity:
            high_propensity_count = context.purchase_propensity.get("high_propensity_count", 0)
            
            if high_propensity_count > 10:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.MARKETING,
                    priority=Priority.HIGH,
                    title="Target High-Propensity Customers",
                    description="Launch targeted campaign for customers with high purchase propensity",
                    rationale=f"{high_propensity_count} customers showing high purchase intent",
                    expected_impact={
                        "conversion_rate": 0.4,
                        "revenue": 0.2,
                    },
                    confidence=0.85,
                    action_items=[
                        "Create personalized offers",
                        "Optimize contact timing",
                        "Implement automated nurture sequences",
                    ],
                    kpis=["conversion_rate", "revenue", "campaign_roi"],
                    timeline="1-2 weeks",
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_pricing_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate pricing-related recommendations."""
        recommendations = []
        
        if context.forecast and context.competitor_analysis:
            forecast_trend = context.forecast.get("trend", "stable")
            price_competitiveness = context.competitor_analysis.get("price_competitiveness", 0.5)
            
            if forecast_trend == "growth" and price_competitiveness > 0.7:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.PRICING,
                    priority=Priority.MEDIUM,
                    title="Test Price Optimization",
                    description="Conduct A/B test for price optimization given positive forecast",
                    rationale="Growth forecast + competitive pricing = opportunity for optimization",
                    expected_impact={
                        "revenue": 0.15,
                        "margin": 0.1,
                    },
                    confidence=0.7,
                    action_items=[
                        "Design price test scenarios",
                        "Implement A/B testing framework",
                        "Monitor customer response",
                    ],
                    kpis=["revenue", "margin", "price_elasticity"],
                    timeline="1-2 months",
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_operational_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate operational recommendations."""
        recommendations = []
        
        if context.forecast:
            confidence = context.forecast.get("confidence", 0.5)
            
            if confidence < 0.5:
                rec = Recommendation(
                    id=self._generate_id(),
                    type=RecommendationType.OPERATIONAL,
                    priority=Priority.MEDIUM,
                    title="Improve Data Collection for Better Forecasting",
                    description="Enhance data infrastructure to improve forecast accuracy",
                    rationale=f"Forecast confidence is low ({confidence:.2f})",
                    expected_impact={
                        "forecast_accuracy": 0.3,
                        "decision_quality": 0.2,
                    },
                    confidence=0.65,
                    action_items=[
                        "Audit current data collection",
                        "Identify data gaps",
                        "Implement enhanced tracking",
                    ],
                    kpis=["forecast_accuracy", "data_completeness"],
                    timeline="2-3 months",
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_ml_recommendations(
        self,
        context: InsightContext,
    ) -> List[Recommendation]:
        """Generate ML-based recommendations (placeholder for trained model)."""
        recommendations = []
        
        return recommendations
    
    def train_ml_model(
        self,
        training_data: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, float]:
        """Train ML model for recommendation generation.
        
        Args:
            training_data: Feature matrix from historical contexts
            labels: Historical recommendation success labels
            
        Returns:
            Dictionary with training metrics
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(training_data)
        self._scaler = scaler
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        model.fit(X_scaled, labels)
        self._ml_model = model
        
        accuracy = model.score(X_scaled, labels)
        
        return {
            "accuracy": float(accuracy),
            "n_features": training_data.shape[1],
            "n_samples": len(training_data),
        }
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize rule-based logic."""
        return {
            "sentiment_threshold": -0.3,
            "volatility_threshold": 0.7,
            "churn_risk_threshold": 0.3,
            "competitive_advantage_threshold": 0.4,
            "ltv_threshold": 5000,
        }
    
    def _generate_id(self) -> str:
        """Generate unique recommendation ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        return f"REC_{timestamp}_{random_suffix}"
    
    def _priority_score(self, priority: Priority) -> int:
        """Convert priority to numeric score."""
        priority_map = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1,
        }
        return priority_map.get(priority, 0)

"""Competitive Intelligence Module.

Provides competitor tracking, pricing analysis, feature/capability comparison, and market positioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


@dataclass
class Competitor:
    """Represents a competitor entity."""
    id: str
    name: str
    market_share: float
    pricing: Dict[str, float]
    features: Set[str]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PricingAnalysisResult:
    """Result of pricing analysis."""
    competitor_prices: Dict[str, float]
    mean_price: float
    median_price: float
    price_position: str
    price_competitiveness: float
    recommendations: List[str]


@dataclass
class FeatureComparisonResult:
    """Result of feature comparison."""
    our_features: Set[str]
    competitor_features: Dict[str, Set[str]]
    unique_features: Set[str]
    missing_features: Set[str]
    feature_coverage: float
    competitive_advantage: float


@dataclass
class MarketPositioningResult:
    """Result of market positioning analysis."""
    position: Tuple[float, float]
    quadrant: str
    competitors_positions: Dict[str, Tuple[float, float]]
    distance_to_leader: float
    strategic_recommendations: List[str]


class CompetitiveIntelligenceModule:
    """Competitive intelligence analysis for strategic decision-making.
    
    Assumptions:
    - Competitor data is up-to-date and accurate
    - Market share data sums to <= 1.0
    - Pricing data is in consistent currency
    - Feature comparisons use standardized feature names
    
    Limitations:
    - Analysis quality depends on data completeness
    - Market positioning is 2D simplification of complex space
    - Competitor strategies may change rapidly
    - Missing data is imputed with median values
    """
    
    def __init__(self, our_company_id: str = "us"):
        """Initialize the competitive intelligence module.
        
        Args:
            our_company_id: Identifier for our company
        """
        self.our_company_id = our_company_id
        self._competitors: Dict[str, Competitor] = {}
        self._market_data: Dict[str, Any] = {}
    
    def add_competitor(self, competitor: Competitor) -> None:
        """Add or update competitor information.
        
        Args:
            competitor: Competitor object to add
        """
        self._competitors[competitor.id] = competitor
    
    def analyze_pricing(
        self,
        product: str,
        our_price: float,
        include_recommendations: bool = True,
    ) -> PricingAnalysisResult:
        """Analyze pricing relative to competitors.
        
        Args:
            product: Product identifier
            our_price: Our current price for the product
            include_recommendations: Whether to generate recommendations
            
        Returns:
            PricingAnalysisResult object
        """
        competitor_prices = {}
        
        for comp_id, comp in self._competitors.items():
            if product in comp.pricing:
                competitor_prices[comp.name] = comp.pricing[product]
        
        if not competitor_prices:
            return PricingAnalysisResult(
                competitor_prices={},
                mean_price=our_price,
                median_price=our_price,
                price_position="unknown",
                price_competitiveness=0.5,
                recommendations=["Insufficient competitor pricing data"],
            )
        
        prices = list(competitor_prices.values())
        mean_price = float(np.mean(prices))
        median_price = float(np.median(prices))
        
        price_diff = (our_price - mean_price) / mean_price
        
        if price_diff < -0.15:
            position = "low"
            competitiveness = 0.8
        elif price_diff < -0.05:
            position = "competitive_low"
            competitiveness = 0.7
        elif price_diff < 0.05:
            position = "competitive"
            competitiveness = 0.9
        elif price_diff < 0.15:
            position = "competitive_high"
            competitiveness = 0.7
        else:
            position = "premium"
            competitiveness = 0.6
        
        recommendations = []
        if include_recommendations:
            if position == "low":
                recommendations.append("Consider raising prices to improve margins")
                recommendations.append("Ensure value proposition justifies low pricing")
            elif position == "premium":
                recommendations.append("Highlight premium features to justify higher pricing")
                recommendations.append("Monitor for price sensitivity among customers")
            elif position == "competitive":
                recommendations.append("Maintain current pricing strategy")
                recommendations.append("Focus on value-added services for differentiation")
        
        return PricingAnalysisResult(
            competitor_prices=competitor_prices,
            mean_price=mean_price,
            median_price=median_price,
            price_position=position,
            price_competitiveness=competitiveness,
            recommendations=recommendations,
        )
    
    def compare_features(
        self,
        our_features: Set[str],
        focus_competitors: Optional[List[str]] = None,
    ) -> FeatureComparisonResult:
        """Compare feature sets with competitors.
        
        Args:
            our_features: Set of our product features
            focus_competitors: Optional list of competitor IDs to focus on
            
        Returns:
            FeatureComparisonResult object
        """
        if focus_competitors:
            competitors = {
                cid: comp for cid, comp in self._competitors.items()
                if cid in focus_competitors
            }
        else:
            competitors = self._competitors
        
        competitor_features = {
            comp.name: comp.features
            for comp in competitors.values()
        }
        
        all_competitor_features = set()
        for features in competitor_features.values():
            all_competitor_features.update(features)
        
        unique_features = our_features - all_competitor_features
        missing_features = all_competitor_features - our_features
        
        if all_competitor_features:
            coverage = len(our_features & all_competitor_features) / len(all_competitor_features)
        else:
            coverage = 1.0
        
        advantage = (len(unique_features) - len(missing_features)) / (len(our_features) + 1)
        advantage = float(np.clip((advantage + 1) / 2, 0, 1))
        
        return FeatureComparisonResult(
            our_features=our_features,
            competitor_features=competitor_features,
            unique_features=unique_features,
            missing_features=missing_features,
            feature_coverage=float(coverage),
            competitive_advantage=advantage,
        )
    
    def analyze_market_positioning(
        self,
        our_metrics: Dict[str, float],
        dimensions: Tuple[str, str] = ("quality", "price"),
    ) -> MarketPositioningResult:
        """Analyze market positioning using 2D projection.
        
        Args:
            our_metrics: Dictionary of metrics for our company
            dimensions: Tuple of two dimension names to analyze
            
        Returns:
            MarketPositioningResult object
        """
        all_metrics = {self.our_company_id: our_metrics}
        
        for comp_id, comp in self._competitors.items():
            comp_metrics = {
                "quality": comp.market_share * 100,
                "price": np.mean(list(comp.pricing.values())) if comp.pricing else 50,
                "features": len(comp.features),
                "market_share": comp.market_share * 100,
            }
            all_metrics[comp_id] = comp_metrics
        
        dim1, dim2 = dimensions
        
        positions = {}
        for company_id, metrics in all_metrics.items():
            x = metrics.get(dim1, 50)
            y = metrics.get(dim2, 50)
            positions[company_id] = (float(x), float(y))
        
        our_position = positions[self.our_company_id]
        x, y = our_position
        
        if x >= 50 and y >= 50:
            quadrant = "Leader"
        elif x >= 50 and y < 50:
            quadrant = "Challenger"
        elif x < 50 and y >= 50:
            quadrant = "Niche"
        else:
            quadrant = "Follower"
        
        competitor_positions = {
            self._competitors[cid].name: pos
            for cid, pos in positions.items()
            if cid != self.our_company_id and cid in self._competitors
        }
        
        leader_position = max(positions.values(), key=lambda p: p[0] + p[1])
        distance = float(np.sqrt(
            (our_position[0] - leader_position[0])**2 +
            (our_position[1] - leader_position[1])**2
        ))
        
        recommendations = self._generate_positioning_recommendations(
            quadrant,
            our_position,
            positions,
        )
        
        return MarketPositioningResult(
            position=our_position,
            quadrant=quadrant,
            competitors_positions=competitor_positions,
            distance_to_leader=distance,
            strategic_recommendations=recommendations,
        )
    
    def track_competitor_changes(
        self,
        competitor_id: str,
        new_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Track and analyze changes in competitor data.
        
        Args:
            competitor_id: Competitor identifier
            new_data: New competitor data
            
        Returns:
            Dictionary of detected changes and insights
        """
        if competitor_id not in self._competitors:
            return {"error": "Competitor not found"}
        
        old_comp = self._competitors[competitor_id]
        changes = {}
        
        if "pricing" in new_data:
            for product, new_price in new_data["pricing"].items():
                if product in old_comp.pricing:
                    old_price = old_comp.pricing[product]
                    price_change = (new_price - old_price) / old_price * 100
                    if abs(price_change) > 1:
                        changes.setdefault("pricing", {})[product] = {
                            "old": old_price,
                            "new": new_price,
                            "change_pct": price_change,
                        }
        
        if "features" in new_data:
            new_features = set(new_data["features"])
            added_features = new_features - old_comp.features
            removed_features = old_comp.features - new_features
            
            if added_features:
                changes["features_added"] = list(added_features)
            if removed_features:
                changes["features_removed"] = list(removed_features)
        
        if "market_share" in new_data:
            old_share = old_comp.market_share
            new_share = new_data["market_share"]
            share_change = (new_share - old_share) * 100
            
            if abs(share_change) > 0.5:
                changes["market_share"] = {
                    "old": old_share,
                    "new": new_share,
                    "change_pts": share_change,
                }
        
        return {
            "competitor": old_comp.name,
            "changes": changes,
            "timestamp": datetime.now(),
            "alert_level": self._calculate_alert_level(changes),
        }
    
    def identify_market_gaps(
        self,
        market_segments: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify underserved market segments and opportunities.
        
        Args:
            market_segments: Dictionary of market segments with metrics
            
        Returns:
            List of market gap opportunities
        """
        gaps = []
        
        for segment_name, segment_data in market_segments.items():
            segment_value = segment_data.get("value", 0)
            competition_level = segment_data.get("competition", 0)
            growth_rate = segment_data.get("growth_rate", 0)
            
            opportunity_score = (
                segment_value * 0.4 +
                (1 - competition_level) * 0.3 +
                growth_rate * 0.3
            )
            
            competitors_in_segment = segment_data.get("competitors", [])
            
            if opportunity_score > 0.6 or (competition_level < 0.3 and segment_value > 0.5):
                gaps.append({
                    "segment": segment_name,
                    "opportunity_score": float(opportunity_score),
                    "market_value": segment_value,
                    "competition_level": competition_level,
                    "growth_rate": growth_rate,
                    "competitors": competitors_in_segment,
                    "recommendation": self._generate_gap_recommendation(
                        segment_name,
                        opportunity_score,
                        competition_level,
                    ),
                })
        
        gaps.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return gaps
    
    def _generate_positioning_recommendations(
        self,
        quadrant: str,
        our_position: Tuple[float, float],
        all_positions: Dict[str, Tuple[float, float]],
    ) -> List[str]:
        """Generate strategic recommendations based on market positioning."""
        recommendations = []
        
        if quadrant == "Leader":
            recommendations.append("Maintain market leadership through innovation")
            recommendations.append("Defend position against challengers")
            recommendations.append("Consider premium pricing strategy")
        elif quadrant == "Challenger":
            recommendations.append("Invest in quality/features to move towards leadership")
            recommendations.append("Maintain competitive pricing")
            recommendations.append("Focus on differentiation")
        elif quadrant == "Niche":
            recommendations.append("Leverage specialized positioning")
            recommendations.append("Consider expanding feature set")
            recommendations.append("Evaluate pricing optimization opportunities")
        else:
            recommendations.append("Urgent: Improve both quality and pricing competitiveness")
            recommendations.append("Identify quick wins to move towards challenger position")
            recommendations.append("Consider strategic partnerships")
        
        return recommendations
    
    def _calculate_alert_level(self, changes: Dict[str, Any]) -> str:
        """Calculate alert level based on competitor changes."""
        if not changes:
            return "none"
        
        score = 0
        
        if "pricing" in changes:
            max_price_change = max(
                abs(v["change_pct"]) for v in changes["pricing"].values()
            )
            if max_price_change > 20:
                score += 3
            elif max_price_change > 10:
                score += 2
            else:
                score += 1
        
        if "features_added" in changes and len(changes["features_added"]) > 3:
            score += 2
        
        if "market_share" in changes:
            if abs(changes["market_share"]["change_pts"]) > 2:
                score += 3
            elif abs(changes["market_share"]["change_pts"]) > 1:
                score += 2
        
        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_gap_recommendation(
        self,
        segment: str,
        opportunity_score: float,
        competition_level: float,
    ) -> str:
        """Generate recommendation for market gap."""
        if opportunity_score > 0.8:
            return f"High priority: Enter {segment} segment immediately with aggressive strategy"
        elif opportunity_score > 0.6:
            if competition_level < 0.3:
                return f"Consider entering {segment} - low competition, good opportunity"
            else:
                return f"Evaluate {segment} carefully - moderate opportunity with competition"
        else:
            return f"Monitor {segment} for future opportunities"

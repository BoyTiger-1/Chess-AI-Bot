"""Customer Behavior Modeling Module.

Provides clustering, segmentation, churn prediction, lifetime value modeling, and purchase propensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score


@dataclass
class CustomerSegment:
    """Represents a customer segment."""
    id: int
    name: str
    size: int
    characteristics: Dict[str, Any]
    avg_ltv: float
    churn_risk: float


@dataclass
class ChurnPrediction:
    """Result of churn prediction."""
    customer_id: str
    churn_probability: float
    churn_risk: str
    key_factors: List[Tuple[str, float]]
    recommendations: List[str]


@dataclass
class LTVPrediction:
    """Result of lifetime value prediction."""
    customer_id: str
    predicted_ltv: float
    confidence: float
    ltv_segment: str
    contributing_factors: Dict[str, float]


@dataclass
class PurchasePropensity:
    """Result of purchase propensity analysis."""
    customer_id: str
    propensity_score: float
    product_recommendations: List[Tuple[str, float]]
    best_contact_time: str
    predicted_purchase_date: Optional[datetime]


@dataclass
class SegmentationResult:
    """Simplified segmentation output used by the public/test API."""

    segments: List[Dict[str, Any]]
    labels: List[int]
    silhouette_score: float


@dataclass
class ChurnPredictionResult:
    """Simplified churn output used by the public/test API."""

    churn_probabilities: List[float]


class CustomerBehaviorModule:
    """Customer behavior modeling for retention and revenue optimization.
    
    Assumptions:
    - Customer data includes transaction history
    - Feature engineering is performed on input data
    - Time-based features use consistent timezone
    - Customer IDs are unique and stable
    
    Limitations:
    - Models require minimum training data (>100 samples recommended)
    - Predictions assume historical patterns continue
    - Segment characteristics may overlap
    - LTV predictions have longer-term uncertainty
    """
    
    def __init__(
        self,
        n_segments: int = 5,
        churn_threshold_days: int = 90,
        random_state: int = 42,
    ):
        """Initialize the customer behavior module.
        
        Args:
            n_segments: Number of customer segments to create
            churn_threshold_days: Days of inactivity to consider as churned
            random_state: Random state for reproducibility
        """
        self.n_segments = n_segments
        self.churn_threshold_days = churn_threshold_days
        self.random_state = random_state
        
        self._segmentation_model: Optional[KMeans] = None
        self._churn_model: Optional[Any] = None
        self._ltv_model: Optional[Any] = None
        self._propensity_model: Optional[Any] = None
        
        self._scalers: Dict[str, StandardScaler] = {}
        self._segments: Dict[int, CustomerSegment] = {}
    
    def segment_customers(
        self,
        customer_features: pd.DataFrame,
        *,
        n_segments: Optional[int] = None,
    ) -> SegmentationResult:
        """Segment customers (simplified interface).

        This method is intentionally lightweight and deterministic, and is what the
        public API/tests use.

        Args:
            customer_features: DataFrame containing customer features.
            n_segments: Number of segments to create.

        Returns:
            SegmentationResult with a list of segments and a silhouette score.
        """

        features = customer_features.select_dtypes(include=[np.number]).fillna(0)
        if features.empty:
            return SegmentationResult(segments=[], labels=[], silhouette_score=0.0)

        k = int(n_segments or self.n_segments)
        k = max(1, min(k, len(features)))

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self._scalers["segmentation"] = scaler

        model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(features_scaled)
        self._segmentation_model = model

        score = 0.0
        if k > 1 and len(features) > k:
            try:
                score = float(silhouette_score(features_scaled, labels))
            except Exception:  # noqa: BLE001
                score = 0.0

        segments: list[dict[str, Any]] = []
        for seg_id in range(k):
            mask = labels == seg_id
            seg_features = features[mask]
            centroid = model.cluster_centers_[seg_id].tolist()
            segments.append(
                {
                    "id": int(seg_id),
                    "size": int(mask.sum()),
                    "centroid": centroid,
                    "feature_means": {c: float(seg_features[c].mean()) for c in seg_features.columns},
                }
            )

        return SegmentationResult(segments=segments, labels=[int(x) for x in labels.tolist()], silhouette_score=score)

    def segment_customers_detailed(
        self,
        customer_features: pd.DataFrame,
        method: str = "kmeans",
    ) -> Tuple[pd.Series, Dict[int, CustomerSegment]]:
        """Detailed segmentation returning segment objects.

        This retains the richer output format for internal/advanced use.
        """

        features = customer_features.select_dtypes(include=[np.number]).fillna(0)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self._scalers["segmentation"] = scaler

        if method == "kmeans":
            model = KMeans(
                n_clusters=self.n_segments,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(features_scaled)
            self._segmentation_model = model
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(features_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")

        segment_labels = pd.Series(labels, index=customer_features.index)

        segments = {}
        for segment_id in np.unique(labels):
            if segment_id == -1:
                continue

            mask = labels == segment_id
            segment_features = features[mask]

            characteristics: dict[str, Any] = {}
            for col in segment_features.columns:
                characteristics[col] = {
                    "mean": float(segment_features[col].mean()),
                    "median": float(segment_features[col].median()),
                    "std": float(segment_features[col].std()),
                }

            avg_ltv = characteristics.get("ltv", {}).get("mean", 0)
            churn_risk = characteristics.get("churn_score", {}).get("mean", 0.5)

            segments[int(segment_id)] = CustomerSegment(
                id=int(segment_id),
                name=self._generate_segment_name(int(segment_id), characteristics),
                size=int(mask.sum()),
                characteristics=characteristics,
                avg_ltv=float(avg_ltv),
                churn_risk=float(churn_risk),
            )

        self._segments = segments

        return segment_labels, segments
    
    def predict_churn(self, customer_features: pd.DataFrame) -> ChurnPredictionResult:
        """Predict churn probabilities (simplified, no training required).

        The more detailed, model-based implementation is available via
        `predict_churn_detailed()`.
        """

        if customer_features.empty:
            return ChurnPredictionResult(churn_probabilities=[])

        df = customer_features.copy()

        def _series(col: str, default: float) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return pd.Series([default] * len(df), index=df.index, dtype=float)

        recency = _series("recency_days", float(self.churn_threshold_days))
        frequency = _series("frequency", 1.0)
        spend = _series("total_spend", 0.0)
        engagement = _series("engagement_score", 0.5)

        def _minmax(s: pd.Series) -> pd.Series:
            lo = float(s.min())
            hi = float(s.max())
            if hi - lo < 1e-9:
                return pd.Series([0.5] * len(s), index=s.index, dtype=float)
            return (s - lo) / (hi - lo)

        recency_n = _minmax(recency)
        freq_n = _minmax(frequency)
        spend_n = _minmax(spend)
        engagement_n = _minmax(engagement)

        prob = (
            0.55 * recency_n
            + 0.20 * (1.0 - freq_n)
            + 0.15 * (1.0 - spend_n)
            + 0.10 * (1.0 - engagement_n)
        )

        prob = prob.clip(lower=0.0, upper=1.0)

        return ChurnPredictionResult(churn_probabilities=[float(x) for x in prob.tolist()])

    def predict_churn_detailed(
        self,
        customer_features: pd.DataFrame,
        train: bool = False,
        labels: Optional[pd.Series] = None,
    ) -> List[ChurnPrediction]:
        """Predict customer churn probability.

        Args:
            customer_features: DataFrame with customer features
            train: Whether to train the model
            labels: True labels for training (required if train=True)

        Returns:
            List of ChurnPrediction objects
        """
        feature_cols = customer_features.select_dtypes(include=[np.number]).columns
        X = customer_features[feature_cols].fillna(0)
        
        if train:
            if labels is None:
                raise ValueError("Labels required for training")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._scalers["churn"] = scaler
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                labels,
                test_size=0.2,
                random_state=self.random_state,
            )
            
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
            self._churn_model = model
        
        if self._churn_model is None:
            raise ValueError("Model not trained. Set train=True first.")
        
        scaler = self._scalers["churn"]
        X_scaled = scaler.transform(X)
        
        probabilities = self._churn_model.predict_proba(X_scaled)[:, 1]
        
        feature_importance = self._churn_model.feature_importances_
        important_features = sorted(
            zip(feature_cols, feature_importance),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        predictions = []
        
        for idx, (customer_id, prob) in enumerate(zip(customer_features.index, probabilities)):
            if prob > 0.7:
                risk = "high"
            elif prob > 0.4:
                risk = "medium"
            else:
                risk = "low"
            
            customer_features_vals = X.iloc[idx]
            key_factors = []
            for feat_name, importance in important_features[:3]:
                feat_value = customer_features_vals[feat_name]
                key_factors.append((feat_name, float(importance)))
            
            recommendations = self._generate_churn_recommendations(risk, key_factors)
            
            predictions.append(ChurnPrediction(
                customer_id=str(customer_id),
                churn_probability=float(prob),
                churn_risk=risk,
                key_factors=key_factors,
                recommendations=recommendations,
            ))
        
        return predictions
    
    def predict_ltv(
        self,
        customer_features: pd.DataFrame,
        train: bool = False,
        ltv_values: Optional[pd.Series] = None,
    ) -> List[LTVPrediction]:
        """Predict customer lifetime value.
        
        Args:
            customer_features: DataFrame with customer features
            train: Whether to train the model
            ltv_values: True LTV values for training (required if train=True)
            
        Returns:
            List of LTVPrediction objects
        """
        feature_cols = customer_features.select_dtypes(include=[np.number]).columns
        X = customer_features[feature_cols].fillna(0)
        
        if train:
            if ltv_values is None:
                raise ValueError("LTV values required for training")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._scalers["ltv"] = scaler
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
            )
            model.fit(X_scaled, ltv_values)
            self._ltv_model = model
        
        if self._ltv_model is None:
            raise ValueError("Model not trained. Set train=True first.")
        
        scaler = self._scalers["ltv"]
        X_scaled = scaler.transform(X)
        
        predictions = self._ltv_model.predict(X_scaled)
        
        feature_importance = self._ltv_model.feature_importances_
        
        results = []
        
        for idx, (customer_id, pred_ltv) in enumerate(zip(customer_features.index, predictions)):
            customer_vals = X.iloc[idx]
            
            contributing_factors = {}
            for feat_name, importance in zip(feature_cols, feature_importance):
                if importance > 0.05:
                    contributing_factors[feat_name] = float(importance)
            
            if pred_ltv > 10000:
                segment = "high_value"
            elif pred_ltv > 5000:
                segment = "medium_value"
            else:
                segment = "low_value"
            
            confidence = 0.7
            
            results.append(LTVPrediction(
                customer_id=str(customer_id),
                predicted_ltv=float(pred_ltv),
                confidence=confidence,
                ltv_segment=segment,
                contributing_factors=contributing_factors,
            ))
        
        return results
    
    def predict_purchase_propensity(
        self,
        customer_features: pd.DataFrame,
        product_interactions: Optional[pd.DataFrame] = None,
        train: bool = False,
        labels: Optional[pd.Series] = None,
    ) -> List[PurchasePropensity]:
        """Predict purchase propensity for customers.
        
        Args:
            customer_features: DataFrame with customer features
            product_interactions: Optional product interaction data
            train: Whether to train the model
            labels: Purchase labels for training (required if train=True)
            
        Returns:
            List of PurchasePropensity objects
        """
        feature_cols = customer_features.select_dtypes(include=[np.number]).columns
        X = customer_features[feature_cols].fillna(0)
        
        if train:
            if labels is None:
                raise ValueError("Labels required for training")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._scalers["propensity"] = scaler
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
            )
            model.fit(X_scaled, labels)
            self._propensity_model = model
        
        if self._propensity_model is None:
            raise ValueError("Model not trained. Set train=True first.")
        
        scaler = self._scalers["propensity"]
        X_scaled = scaler.transform(X)
        
        propensity_scores = self._propensity_model.predict_proba(X_scaled)[:, 1]
        
        results = []
        
        for idx, (customer_id, score) in enumerate(zip(customer_features.index, propensity_scores)):
            product_recs = self._generate_product_recommendations(
                customer_features.iloc[idx],
                score,
            )
            
            best_time = self._determine_best_contact_time(
                customer_features.iloc[idx],
            )
            
            predicted_date = None
            if score > 0.5:
                days_until = int(30 * (1 - score))
                predicted_date = datetime.now() + timedelta(days=days_until)
            
            results.append(PurchasePropensity(
                customer_id=str(customer_id),
                propensity_score=float(score),
                product_recommendations=product_recs,
                best_contact_time=best_time,
                predicted_purchase_date=predicted_date,
            ))
        
        return results
    
    def _generate_segment_name(
        self,
        segment_id: int,
        characteristics: Dict[str, Any],
    ) -> str:
        """Generate a descriptive name for a segment."""
        avg_ltv = characteristics.get("ltv", {}).get("mean", 0)
        avg_recency = characteristics.get("recency_days", {}).get("mean", 100)
        avg_frequency = characteristics.get("purchase_frequency", {}).get("mean", 1)
        
        if avg_ltv > 10000 and avg_frequency > 5:
            return f"Premium_Frequent_{segment_id}"
        elif avg_ltv > 5000:
            return f"High_Value_{segment_id}"
        elif avg_recency < 30 and avg_frequency > 3:
            return f"Active_Loyal_{segment_id}"
        elif avg_recency > 180:
            return f"At_Risk_{segment_id}"
        else:
            return f"Standard_{segment_id}"
    
    def _generate_churn_recommendations(
        self,
        risk: str,
        key_factors: List[Tuple[str, float]],
    ) -> List[str]:
        """Generate recommendations based on churn risk."""
        recommendations = []
        
        if risk == "high":
            recommendations.append("Immediate intervention required")
            recommendations.append("Offer personalized retention incentive")
            recommendations.append("Schedule direct outreach from account manager")
        elif risk == "medium":
            recommendations.append("Monitor closely and engage proactively")
            recommendations.append("Send targeted re-engagement campaign")
            recommendations.append("Provide value-add content or offers")
        else:
            recommendations.append("Maintain regular engagement")
            recommendations.append("Continue delivering value")
        
        return recommendations
    
    def _generate_product_recommendations(
        self,
        customer_row: pd.Series,
        propensity: float,
    ) -> List[Tuple[str, float]]:
        """Generate product recommendations for a customer."""
        recommendations = [
            ("Product_A", propensity * 0.9),
            ("Product_B", propensity * 0.7),
            ("Product_C", propensity * 0.5),
        ]
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    def _determine_best_contact_time(self, customer_row: pd.Series) -> str:
        """Determine best time to contact customer."""
        hour = hash(str(customer_row.name)) % 24
        
        if 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "morning"

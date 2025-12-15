"""Real-time Market Analysis Module.

Provides sentiment analysis on news/social media, trend detection, and volatility modeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from textblob import TextBlob
from transformers import pipeline

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    polarity: float
    subjectivity: float
    label: str
    confidence: float
    timestamp: datetime


@dataclass
class TrendResult:
    """Result of trend detection."""
    trend_direction: str
    trend_strength: float
    change_points: List[int]
    confidence: float
    period_start: datetime
    period_end: datetime


@dataclass
class VolatilityResult:
    """Result of volatility modeling."""
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    risk_level: str
    confidence: float


class MarketAnalysisModule:
    """Real-time market analysis with sentiment, trends, and volatility.
    
    Assumptions:
    - Text data is in English
    - Time series data is regularly sampled
    - Volatility is calculated using rolling windows
    
    Limitations:
    - Sentiment models may not capture domain-specific language
    - Trend detection is sensitive to noise in short time series
    - Volatility models assume stationary variance within windows
    """
    
    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        use_transformer: bool = True,
        volatility_window: int = 20,
        trend_sensitivity: float = 0.05,
    ):
        """Initialize the market analysis module.
        
        Args:
            sentiment_model: HuggingFace model for sentiment analysis
            use_transformer: Whether to use transformer models (slower but more accurate)
            volatility_window: Window size for volatility calculations
            trend_sensitivity: Sensitivity for trend detection (0-1)
        """
        self.sentiment_model_name = sentiment_model
        self.use_transformer = use_transformer and TORCH_AVAILABLE
        self.volatility_window = volatility_window
        self.trend_sensitivity = trend_sensitivity
        
        if self.use_transformer:
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                device=0 if TORCH_AVAILABLE else -1,
            )
        else:
            self._sentiment_pipeline = None
    
    def analyze_sentiment(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[SentimentResult]:
        """Analyze sentiment of text data.
        
        Args:
            texts: List of text strings to analyze
            timestamps: Optional timestamps for each text
            
        Returns:
            List of SentimentResult objects
        """
        if timestamps is None:
            timestamps = [datetime.now()] * len(texts)
        
        results = []
        
        for text, timestamp in zip(texts, timestamps):
            cleaned_text = self._preprocess_text(text)
            
            if self.use_transformer and self._sentiment_pipeline:
                transformer_result = self._sentiment_pipeline(cleaned_text[:512])[0]
                polarity = transformer_result["score"] if transformer_result["label"] == "POSITIVE" else -transformer_result["score"]
                confidence = transformer_result["score"]
                label = transformer_result["label"]
            else:
                blob = TextBlob(cleaned_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                if polarity > 0.1:
                    label = "POSITIVE"
                elif polarity < -0.1:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                confidence = abs(polarity)
            
            blob = TextBlob(cleaned_text)
            subjectivity = blob.sentiment.subjectivity
            
            results.append(SentimentResult(
                text=text,
                polarity=float(polarity),
                subjectivity=float(subjectivity),
                label=label,
                confidence=float(confidence),
                timestamp=timestamp,
            ))
        
        return results
    
    def aggregate_sentiment(
        self,
        sentiment_results: List[SentimentResult],
        window: timedelta = timedelta(hours=24),
    ) -> Dict[str, Any]:
        """Aggregate sentiment results over a time window.
        
        Args:
            sentiment_results: List of sentiment results
            window: Time window for aggregation
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not sentiment_results:
            return {
                "mean_polarity": 0.0,
                "mean_confidence": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "volume": 0,
            }
        
        df = pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "polarity": r.polarity,
                "confidence": r.confidence,
                "label": r.label,
            }
            for r in sentiment_results
        ])
        
        cutoff = datetime.now() - window
        recent_df = df[df["timestamp"] >= cutoff]
        
        if len(recent_df) == 0:
            recent_df = df
        
        label_counts = recent_df["label"].value_counts()
        total = len(recent_df)
        
        return {
            "mean_polarity": float(recent_df["polarity"].mean()),
            "std_polarity": float(recent_df["polarity"].std()),
            "mean_confidence": float(recent_df["confidence"].mean()),
            "positive_ratio": float(label_counts.get("POSITIVE", 0) / total),
            "negative_ratio": float(label_counts.get("NEGATIVE", 0) / total),
            "neutral_ratio": float(label_counts.get("NEUTRAL", 0) / total),
            "volume": int(total),
            "trend": "bullish" if recent_df["polarity"].mean() > 0.1 else (
                "bearish" if recent_df["polarity"].mean() < -0.1 else "neutral"
            ),
        }
    
    def detect_trends(
        self,
        time_series: pd.Series,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> TrendResult:
        """Detect trends in time series data.
        
        Args:
            time_series: Time series data
            timestamps: Timestamps for the data
            
        Returns:
            TrendResult object
        """
        if timestamps is None:
            timestamps = pd.date_range(
                end=datetime.now(),
                periods=len(time_series),
                freq="D",
            )
        
        values = time_series.values
        x = np.arange(len(values))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        normalized_slope = slope / (values.mean() + 1e-10)
        
        if abs(normalized_slope) < self.trend_sensitivity:
            direction = "sideways"
            strength = 0.0
        elif normalized_slope > 0:
            direction = "up"
            strength = min(abs(normalized_slope) * 100, 1.0)
        else:
            direction = "down"
            strength = min(abs(normalized_slope) * 100, 1.0)
        
        change_points = self._detect_change_points(values)
        
        confidence = abs(r_value)
        
        return TrendResult(
            trend_direction=direction,
            trend_strength=float(strength),
            change_points=change_points.tolist() if isinstance(change_points, np.ndarray) else change_points,
            confidence=float(confidence),
            period_start=timestamps[0],
            period_end=timestamps[-1],
        )
    
    def calculate_volatility(
        self,
        time_series: pd.Series,
        returns: bool = True,
    ) -> VolatilityResult:
        """Calculate volatility metrics for time series data.
        
        Args:
            time_series: Time series data (prices or returns)
            returns: If False, will calculate returns from prices
            
        Returns:
            VolatilityResult object
        """
        if not returns:
            returns_series = time_series.pct_change().dropna()
        else:
            returns_series = time_series
        
        if len(returns_series) < self.volatility_window:
            window = len(returns_series)
        else:
            window = self.volatility_window
        
        current_vol = float(returns_series.tail(window).std() * np.sqrt(252))
        
        historical_vol = float(returns_series.std() * np.sqrt(252))
        
        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252)
        percentile = float(stats.percentileofscore(rolling_vol.dropna(), current_vol))
        
        if percentile >= 80:
            risk_level = "high"
        elif percentile >= 50:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        confidence = min(len(returns_series) / (self.volatility_window * 10), 1.0)
        
        return VolatilityResult(
            current_volatility=current_vol,
            historical_volatility=historical_vol,
            volatility_percentile=percentile,
            risk_level=risk_level,
            confidence=float(confidence),
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """Detect change points in time series using peak detection."""
        if len(values) < 3:
            return []
        
        diff = np.diff(values)
        diff_smooth = pd.Series(diff).rolling(window=3, center=True).mean().fillna(0).values
        
        peaks, _ = find_peaks(np.abs(diff_smooth), height=np.std(diff_smooth))
        
        return peaks.tolist()

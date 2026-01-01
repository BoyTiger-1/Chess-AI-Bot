"""Real-time Market Analysis Module.

Provides sentiment analysis, trend detection, and volatility modeling.

Notes on dependencies:
- Transformer-based sentiment analysis is optional and is only enabled when both
  `transformers` and `torch` are available.
- The module exposes a backwards-compatible API (`detect_trend`, `model_volatility`)
  for legacy callers while keeping the newer `detect_trends`/`calculate_volatility`
  APIs used by the more detailed test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, overload

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from textblob import TextBlob

from ai_business_assistant.shared.model_cache import get_model_cache

try:  # Optional dependency
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:  # Optional dependency
    from transformers import pipeline

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # noqa: BLE001
    pipeline = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    text: str
    polarity: float
    subjectivity: float
    label: str  # positive|negative|neutral
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
    """Market analysis with sentiment, trends, and volatility."""

    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        use_transformer: bool = True,
        volatility_window: int = 20,
        trend_sensitivity: float = 0.05,
    ):
        self.sentiment_model_name = sentiment_model
        self.volatility_window = volatility_window
        self.trend_sensitivity = trend_sensitivity

        self._transformer_enabled = bool(use_transformer and _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE)
        self._sentiment_pipeline = None

        if self._transformer_enabled:
            cache = get_model_cache()
            cache_key = f"pipeline_sentiment_{sentiment_model}"
            self._sentiment_pipeline = cache.get_model(cache_key)

            if self._sentiment_pipeline is None:
                device = -1
                if torch is not None and torch.cuda.is_available():
                    device = 0
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model,
                    device=device,
                )
                cache.set_model(cache_key, self._sentiment_pipeline)

    @overload
    def analyze_sentiment(self, texts: str, timestamps: None = None) -> SentimentResult: ...

    @overload
    def analyze_sentiment(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[SentimentResult]: ...

    def analyze_sentiment(
        self,
        texts: str | List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> SentimentResult | List[SentimentResult]:
        """Analyze sentiment.

        Backwards-compatible behavior:
        - If a single string is provided, returns a single SentimentResult.
        - If a list is provided, returns a list of SentimentResult.
        """

        if isinstance(texts, str):
            results = self.analyze_sentiment([texts], timestamps=None)
            return results[0]

        if not texts:
            return []

        if timestamps is None:
            timestamps = [datetime.now()] * len(texts)

        results: list[SentimentResult] = []

        for text, ts in zip(texts, timestamps):
            cleaned = self._preprocess_text(text)

            if self._transformer_enabled and self._sentiment_pipeline is not None:
                out = self._sentiment_pipeline(cleaned[:512])[0]
                score = float(out.get("score", 0.0))
                raw_label = str(out.get("label", "NEUTRAL")).upper()
                polarity = score if raw_label == "POSITIVE" else -score if raw_label == "NEGATIVE" else 0.0
                confidence = score

                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                blob = TextBlob(cleaned)
                subjectivity = float(blob.sentiment.subjectivity)
            else:
                blob = TextBlob(cleaned)
                polarity = float(blob.sentiment.polarity)
                subjectivity = float(blob.sentiment.subjectivity)

                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                confidence = float(min(abs(polarity), 1.0))

            results.append(
                SentimentResult(
                    text=text,
                    polarity=float(np.clip(polarity, -1.0, 1.0)),
                    subjectivity=float(np.clip(subjectivity, 0.0, 1.0)),
                    label=label,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    timestamp=ts,
                )
            )

        return results

    def aggregate_sentiment(
        self,
        sentiment_results: List[SentimentResult],
        window: timedelta = timedelta(hours=24),
    ) -> Dict[str, Any]:
        if not sentiment_results:
            return {
                "mean_polarity": 0.0,
                "mean_confidence": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "volume": 0,
            }

        df = pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "polarity": r.polarity,
                    "confidence": r.confidence,
                    "label": r.label,
                }
                for r in sentiment_results
            ]
        )

        cutoff = datetime.now() - window
        recent = df[df["timestamp"] >= cutoff]
        if len(recent) == 0:
            recent = df

        label_counts = recent["label"].value_counts()
        total = int(len(recent))

        mean_polarity = float(recent["polarity"].mean())

        return {
            "mean_polarity": mean_polarity,
            "std_polarity": float(recent["polarity"].std()),
            "mean_confidence": float(recent["confidence"].mean()),
            "positive_ratio": float(label_counts.get("positive", 0) / total),
            "negative_ratio": float(label_counts.get("negative", 0) / total),
            "neutral_ratio": float(label_counts.get("neutral", 0) / total),
            "volume": total,
            "trend": "bullish" if mean_polarity > 0.1 else ("bearish" if mean_polarity < -0.1 else "neutral"),
        }

    def detect_trends(
        self,
        time_series: pd.Series,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> TrendResult:
        if timestamps is None:
            timestamps = pd.date_range(end=datetime.now(), periods=len(time_series), freq="D")

        values = time_series.values
        x = np.arange(len(values))

        slope, _, r_value, _, _ = stats.linregress(x, values)
        normalized_slope = float(slope / (float(np.mean(values)) + 1e-10))

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

        return TrendResult(
            trend_direction=direction,
            trend_strength=float(strength),
            change_points=change_points,
            confidence=float(np.clip(abs(r_value), 0.0, 1.0)),
            period_start=timestamps[0].to_pydatetime() if hasattr(timestamps[0], "to_pydatetime") else timestamps[0],
            period_end=timestamps[-1].to_pydatetime() if hasattr(timestamps[-1], "to_pydatetime") else timestamps[-1],
        )

    def detect_trend(
        self,
        prices: Sequence[float] | np.ndarray,
        *,
        dates: Optional[List[datetime]] = None,
    ) -> TrendResult:
        """Legacy API used by older callers/tests.

        Returns a TrendResult whose `trend_direction` is one of:
        - upward
        - downward
        - sideways
        """

        series = pd.Series(list(prices))
        if dates is not None:
            idx = pd.to_datetime(dates)
            series.index = idx
            result = self.detect_trends(series, timestamps=idx)
        else:
            result = self.detect_trends(series)

        direction_map = {"up": "upward", "down": "downward", "sideways": "sideways"}
        result.trend_direction = direction_map.get(result.trend_direction, result.trend_direction)
        return result

    def calculate_volatility(self, time_series: pd.Series, *, returns: bool = True) -> VolatilityResult:
        if not returns:
            returns_series = time_series.pct_change().dropna()
        else:
            returns_series = time_series.dropna()

        if len(returns_series) == 0:
            return VolatilityResult(
                current_volatility=0.0,
                historical_volatility=0.0,
                volatility_percentile=0.0,
                risk_level="low",
                confidence=0.0,
            )

        window = min(self.volatility_window, len(returns_series))

        current_vol = float(returns_series.tail(window).std() * np.sqrt(252))
        historical_vol = float(returns_series.std() * np.sqrt(252))

        rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(252)
        try:
            percentile = float(stats.percentileofscore(rolling_vol.dropna(), current_vol))
        except Exception:  # noqa: BLE001
            percentile = 0.0

        if percentile >= 80:
            risk_level = "high"
        elif percentile >= 50:
            risk_level = "medium"
        else:
            risk_level = "low"

        confidence = float(min(len(returns_series) / (self.volatility_window * 10), 1.0))

        return VolatilityResult(
            current_volatility=max(current_vol, 0.0),
            historical_volatility=max(historical_vol, 0.0),
            volatility_percentile=float(np.clip(percentile, 0.0, 100.0)),
            risk_level=risk_level,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def model_volatility(self, prices: Sequence[float] | np.ndarray) -> VolatilityResult:
        """Legacy API used by older callers/tests."""

        series = pd.Series(list(prices))
        return self.calculate_volatility(series, returns=False)

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        if len(values) < 3:
            return []

        diff = np.diff(values)
        diff_smooth = pd.Series(diff).rolling(window=3, center=True).mean().fillna(0).values

        peaks, _ = find_peaks(np.abs(diff_smooth), height=float(np.std(diff_smooth) or 0.0))
        return peaks.tolist()

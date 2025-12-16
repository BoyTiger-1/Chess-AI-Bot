# AI Modules Documentation

## Overview

The AI Business Assistant provides five core AI intelligence modules that enable comprehensive business analysis and strategic decision-making.

## Modules

### 1. Market Analysis Module

**Purpose:** Real-time analysis of market conditions through sentiment, trends, and volatility.

**Key Features:**
- Sentiment analysis using TextBlob and transformer models (DistilBERT)
- Trend detection with linear regression and change-point detection
- Volatility modeling with rolling windows and percentile analysis

**Usage Example:**
```python
from ai_business_assistant.ai_modules import MarketAnalysisModule

module = MarketAnalysisModule(use_transformer=False)

# Analyze sentiment
texts = ["Great product!", "Not happy with service"]
results = module.analyze_sentiment(texts)

# Detect trends
import pandas as pd
prices = pd.Series([100, 102, 105, 103, 108, 110])
trend = module.detect_trends(prices)
print(f"Trend: {trend.trend_direction}, Strength: {trend.trend_strength}")

# Calculate volatility
volatility = module.calculate_volatility(prices, returns=False)
print(f"Risk Level: {volatility.risk_level}")
```

**Assumptions:**
- Text data is in English
- Time series data is regularly sampled
- Volatility is calculated using rolling windows (default: 20 periods)

**Limitations:**
- Sentiment models may not capture domain-specific language
- Trend detection is sensitive to noise in short time series
- Transformer models require significant computational resources

---

### 2. Financial Forecasting Module

**Purpose:** Time-series forecasting using multiple statistical and ML models.

**Key Features:**
- ARIMA/SARIMA forecasting with auto-parameter selection
- Facebook Prophet for seasonal patterns
- Regression models (Linear, Ridge, Random Forest, Gradient Boosting)
- Scenario modeling for multiple futures

**Usage Example:**
```python
from ai_business_assistant.ai_modules import ForecastingModule

module = ForecastingModule(confidence_level=0.95)

# ARIMA forecast
forecast = module.forecast_arima(time_series, periods=30)
print(f"Confidence: {forecast.confidence}")

# Prophet forecast
forecast = module.forecast_prophet(time_series, periods=30)

# Scenario modeling
scenarios = {
    "optimistic": {"growth_rate": 0.1, "volatility": 0.05, "probability": 0.3},
    "base": {"growth_rate": 0.0, "volatility": 0.1, "probability": 0.5},
    "pessimistic": {"growth_rate": -0.1, "volatility": 0.15, "probability": 0.2},
}
results = module.scenario_modeling(time_series, periods=30, scenarios=scenarios)
```

**Assumptions:**
- Time series data is regularly sampled
- Missing values are handled via interpolation
- Stationarity is tested and differencing applied as needed

**Limitations:**
- Prophet requires at least 2 periods of historical data
- ARIMA is sensitive to outliers
- Long-term forecasts have decreasing accuracy

---

### 3. Competitive Intelligence Module

**Purpose:** Track competitors, analyze pricing, compare features, and assess market position.

**Key Features:**
- Competitor tracking with change detection
- Pricing analysis and positioning
- Feature/capability comparison
- Market gap identification
- 2D market positioning analysis

**Usage Example:**
```python
from ai_business_assistant.ai_modules import CompetitiveIntelligenceModule, Competitor

module = CompetitiveIntelligenceModule(our_company_id="us")

# Add competitors
competitor = Competitor(
    id="comp1",
    name="Competitor A",
    market_share=0.15,
    pricing={"product_x": 99.99},
    features={"feature1", "feature2", "feature3"},
)
module.add_competitor(competitor)

# Analyze pricing
pricing_analysis = module.analyze_pricing("product_x", our_price=89.99)
print(pricing_analysis.price_position)
print(pricing_analysis.recommendations)

# Compare features
our_features = {"feature1", "feature2", "feature4"}
comparison = module.compare_features(our_features)
print(f"Unique features: {comparison.unique_features}")
print(f"Missing features: {comparison.missing_features}")

# Market positioning
our_metrics = {"quality": 85, "price": 75}
positioning = module.analyze_market_positioning(our_metrics)
print(f"Quadrant: {positioning.quadrant}")
```

**Assumptions:**
- Competitor data is current and accurate
- Pricing data is in consistent currency
- Feature names are standardized

**Limitations:**
- Analysis quality depends on data completeness
- Market positioning is 2D simplification
- Competitor strategies may change rapidly

---

### 4. Customer Behavior Module

**Purpose:** Model customer behavior for retention and revenue optimization.

**Key Features:**
- Customer segmentation (K-Means, DBSCAN)
- Churn prediction with Gradient Boosting
- Lifetime Value (LTV) prediction
- Purchase propensity modeling
- Actionable recommendations

**Usage Example:**
```python
from ai_business_assistant.ai_modules import CustomerBehaviorModule

module = CustomerBehaviorModule(n_segments=5)

# Segment customers
labels, segments = module.segment_customers(customer_features)
for seg_id, segment in segments.items():
    print(f"{segment.name}: {segment.size} customers, Avg LTV: ${segment.avg_ltv:.2f}")

# Predict churn
churn_predictions = module.predict_churn(
    customer_features,
    train=True,
    labels=churn_labels,
)
high_risk = [p for p in churn_predictions if p.churn_risk == "high"]
print(f"High risk customers: {len(high_risk)}")

# Predict LTV
ltv_predictions = module.predict_ltv(
    customer_features,
    train=True,
    ltv_values=actual_ltv,
)

# Purchase propensity
propensity_predictions = module.predict_purchase_propensity(
    customer_features,
    train=True,
    labels=purchase_labels,
)
```

**Assumptions:**
- Customer data includes transaction history
- Customer IDs are unique and stable
- Features are properly engineered

**Limitations:**
- Models require minimum 100+ samples for training
- Predictions assume historical patterns continue
- Segment characteristics may overlap

---

### 5. Strategic Recommendation Engine

**Purpose:** Generate strategic recommendations by combining insights from all modules.

**Key Features:**
- Rule-based recommendation logic
- ML-based recommendation (when trained)
- Priority-based ranking
- Confidence scoring
- Expected impact estimation

**Usage Example:**
```python
from ai_business_assistant.ai_modules import RecommendationEngine, InsightContext

engine = RecommendationEngine(min_confidence=0.6)

# Prepare context
context = InsightContext(
    market_sentiment={"mean_polarity": -0.4, "trend": "bearish"},
    volatility={"risk_level": "high"},
    churn_analysis={"high_risk_count": 50},
    competitor_analysis={"competitive_advantage": 0.3},
)

# Generate recommendations
recommendations = engine.generate_recommendations(context, max_recommendations=10)

for rec in recommendations:
    print(f"\n{rec.priority.value.upper()}: {rec.title}")
    print(f"Type: {rec.type.value}")
    print(f"Confidence: {rec.confidence:.2%}")
    print(f"Description: {rec.description}")
    print(f"Action Items:")
    for item in rec.action_items:
        print(f"  - {item}")
```

**Assumptions:**
- Input insights are current and validated
- Business rules are properly configured
- Decision thresholds are calibrated

**Limitations:**
- Recommendations depend on input data quality
- Rule-based logic may not capture all nuances
- Cannot account for unprecedented scenarios

---

## Integration Workflow

### Example: Complete Analysis Pipeline

```python
from ai_business_assistant.ai_modules import (
    MarketAnalysisModule,
    ForecastingModule,
    CompetitiveIntelligenceModule,
    CustomerBehaviorModule,
    RecommendationEngine,
    InsightContext,
)

# 1. Market Analysis
market_module = MarketAnalysisModule()
sentiment_results = market_module.analyze_sentiment(social_media_texts)
sentiment_agg = market_module.aggregate_sentiment(sentiment_results)
volatility = market_module.calculate_volatility(stock_prices, returns=False)

# 2. Forecasting
forecast_module = ForecastingModule()
revenue_forecast = forecast_module.forecast_prophet(historical_revenue, periods=90)

# 3. Competitive Intelligence
comp_module = CompetitiveIntelligenceModule()
pricing_analysis = comp_module.analyze_pricing("main_product", our_price=99.99)
feature_comparison = comp_module.compare_features(our_features)

# 4. Customer Behavior
customer_module = CustomerBehaviorModule()
churn_predictions = customer_module.predict_churn(customer_data, train=True, labels=churn_labels)
high_risk_count = sum(1 for p in churn_predictions if p.churn_risk == "high")

# 5. Generate Recommendations
context = InsightContext(
    market_sentiment=sentiment_agg,
    volatility={"risk_level": volatility.risk_level},
    forecast={"confidence": revenue_forecast.confidence},
    competitor_analysis={
        "price_position": pricing_analysis.price_position,
        "competitive_advantage": feature_comparison.competitive_advantage,
    },
    churn_analysis={"high_risk_count": high_risk_count},
)

engine = RecommendationEngine(min_confidence=0.6)
recommendations = engine.generate_recommendations(context)

# Display top recommendations
for rec in recommendations[:5]:
    print(f"\n{'='*60}")
    print(f"{rec.priority.value.upper()}: {rec.title}")
    print(f"Confidence: {rec.confidence:.2%}")
    print(f"\n{rec.description}")
    print(f"\nExpected Impact:")
    for metric, impact in rec.expected_impact.items():
        print(f"  {metric}: {impact:+.1%}")
```

## Algorithm Details

### Sentiment Analysis
- **TextBlob:** Rule-based sentiment with polarity (-1 to +1) and subjectivity (0 to 1)
- **Transformer:** Fine-tuned DistilBERT for more accurate sentiment classification
- **Aggregation:** Weighted average over time windows with trend detection

### Trend Detection
- **Linear Regression:** Fits trend line to time series
- **Change Points:** Uses peak detection on differenced series
- **Strength:** Normalized slope relative to mean value

### Volatility Modeling
- **Annualized Volatility:** Standard deviation × √252 (trading days)
- **Rolling Windows:** Default 20-period window
- **Risk Levels:** Percentile-based classification (low <50%, medium 50-80%, high >80%)

### ARIMA Forecasting
- **Auto-Selection:** Grid search over (p, d, q) with AIC criterion
- **Stationarity:** Augmented Dickey-Fuller test determines differencing
- **Confidence Intervals:** Based on forecast standard errors

### Customer Segmentation
- **K-Means:** Unsupervised clustering with standardized features
- **Optimal K:** Silhouette score for cluster quality
- **Segment Naming:** Rule-based based on LTV, recency, frequency

### Churn Prediction
- **Gradient Boosting:** Ensemble of decision trees
- **Feature Importance:** Intrinsic importance from model
- **Calibration:** Probability estimates can be calibrated on validation set

## Performance Considerations

- **Sentiment Analysis:** ~10-50 texts/second (TextBlob), ~1-5 texts/second (Transformer with CPU)
- **Forecasting:** ARIMA ~1-5 seconds for 100 points, Prophet ~5-30 seconds
- **Customer Models:** Training ~5-60 seconds for 1000 samples, prediction <1ms per sample
- **Memory:** Transformer models require ~500MB-1GB RAM

## Best Practices

1. **Data Quality:** Ensure clean, consistent input data
2. **Regular Updates:** Retrain models periodically with new data
3. **Validation:** Use holdout sets to validate predictions
4. **Interpretation:** Always review model assumptions and limitations
5. **Monitoring:** Track model performance metrics over time
6. **Ensemble:** Combine multiple models for robust predictions

# Example Workflows

This document provides complete end-to-end examples showing how each module produces insights and how they work together.

## Workflow 1: Market Opportunity Analysis

**Objective:** Identify market opportunities by analyzing sentiment, trends, and competitive position.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_business_assistant.ai_modules import (
    MarketAnalysisModule,
    CompetitiveIntelligenceModule,
    RecommendationEngine,
    InsightContext,
    Competitor,
)

# Step 1: Analyze Market Sentiment
print("=== MARKET SENTIMENT ANALYSIS ===")
market_module = MarketAnalysisModule(use_transformer=False)

# Collect social media/news data
market_texts = [
    "Industry seeing strong growth in Q4",
    "New regulations may impact market",
    "Consumer confidence at all-time high",
    "Exciting innovations in the space",
    # ... more texts
]

timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(market_texts))]

sentiment_results = market_module.analyze_sentiment(market_texts, timestamps)
sentiment_agg = market_module.aggregate_sentiment(sentiment_results)

print(f"Mean Sentiment: {sentiment_agg['mean_polarity']:.3f}")
print(f"Trend: {sentiment_agg['trend']}")
print(f"Positive Ratio: {sentiment_agg['positive_ratio']:.2%}")

# Step 2: Analyze Market Trends
print("\n=== MARKET TREND ANALYSIS ===")
market_data = pd.Series(
    100 + np.cumsum(np.random.randn(180) * 2),
    index=pd.date_range(start="2023-06-01", periods=180, freq="D")
)

trend_result = market_module.detect_trends(market_data)
print(f"Trend Direction: {trend_result.trend_direction}")
print(f"Trend Strength: {trend_result.trend_strength:.3f}")
print(f"Confidence: {trend_result.confidence:.2%}")

# Step 3: Competitive Analysis
print("\n=== COMPETITIVE ANALYSIS ===")
comp_module = CompetitiveIntelligenceModule(our_company_id="us")

# Add competitors
competitors = [
    Competitor(
        id="comp1",
        name="Market Leader",
        market_share=0.35,
        pricing={"product_a": 129.99},
        features={"advanced_ai", "mobile_app", "api", "analytics", "integrations"},
    ),
    Competitor(
        id="comp2",
        name="Fast Follower",
        market_share=0.22,
        pricing={"product_a": 99.99},
        features={"mobile_app", "analytics", "integrations"},
    ),
]

for comp in competitors:
    comp_module.add_competitor(comp)

# Pricing analysis
our_price = 89.99
our_features = {"mobile_app", "analytics", "api", "custom_reports"}

pricing_analysis = comp_module.analyze_pricing("product_a", our_price)
print(f"Price Position: {pricing_analysis.price_position}")
print(f"Price Competitiveness: {pricing_analysis.price_competitiveness:.2%}")

# Feature comparison
feature_comparison = comp_module.compare_features(our_features)
print(f"\nUnique Features: {feature_comparison.unique_features}")
print(f"Missing Features: {feature_comparison.missing_features}")
print(f"Competitive Advantage: {feature_comparison.competitive_advantage:.2%}")

# Step 4: Generate Strategic Recommendations
print("\n=== STRATEGIC RECOMMENDATIONS ===")
context = InsightContext(
    market_sentiment=sentiment_agg,
    market_trends={
        "trend_direction": trend_result.trend_direction,
        "trend_strength": trend_result.trend_strength,
    },
    competitor_analysis={
        "price_position": pricing_analysis.price_position,
        "price_competitiveness": pricing_analysis.price_competitiveness,
        "competitive_advantage": feature_comparison.competitive_advantage,
    },
)

engine = RecommendationEngine(min_confidence=0.6)
recommendations = engine.generate_recommendations(context, max_recommendations=5)

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. [{rec.priority.value.upper()}] {rec.title}")
    print(f"   Type: {rec.type.value}")
    print(f"   Confidence: {rec.confidence:.2%}")
    print(f"   {rec.description}")
    print(f"   Expected Impact:")
    for metric, impact in rec.expected_impact.items():
        print(f"     - {metric}: {impact:+.1%}")
```

**Output Example:**
```
=== MARKET SENTIMENT ANALYSIS ===
Mean Sentiment: 0.342
Trend: bullish
Positive Ratio: 68%

=== MARKET TREND ANALYSIS ===
Trend Direction: up
Trend Strength: 0.658
Confidence: 87%

=== COMPETITIVE ANALYSIS ===
Price Position: competitive_low
Price Competitiveness: 70%

Unique Features: {'custom_reports'}
Missing Features: {'advanced_ai', 'integrations'}
Competitive Advantage: 45%

=== STRATEGIC RECOMMENDATIONS ===

1. [HIGH] Capitalize on Positive Market Trend
   Type: expansion
   Confidence: 80%
   Strong upward trend detected - consider expansion initiatives
   Expected Impact:
     - revenue_growth: +25.0%
     - market_share: +15.0%

2. [CRITICAL] Close Feature Gaps with Competitors
   Type: product
   Confidence: 85%
   Accelerate product development to match competitor capabilities
   Expected Impact:
     - competitive_position: +30.0%
     - customer_satisfaction: +25.0%
```

---

## Workflow 2: Customer Retention Campaign

**Objective:** Identify at-risk customers and create targeted retention strategies.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from ai_business_assistant.ai_modules import CustomerBehaviorModule
from ai_business_assistant.ml_pipeline import (
    ModelTrainer,
    ModelEvaluator,
    ExplainabilityAnalyzer,
    TrainingConfig,
)

# Step 1: Prepare Customer Data
print("=== PREPARING CUSTOMER DATA ===")

# Create synthetic customer features
n_customers = 1000
customer_features = pd.DataFrame({
    "recency_days": np.random.randint(1, 365, n_customers),
    "frequency": np.random.randint(1, 50, n_customers),
    "monetary_value": np.random.uniform(10, 10000, n_customers),
    "avg_order_value": np.random.uniform(50, 500, n_customers),
    "days_since_signup": np.random.randint(30, 1800, n_customers),
    "support_tickets": np.random.randint(0, 20, n_customers),
    "nps_score": np.random.randint(-100, 100, n_customers),
    "engagement_score": np.random.uniform(0, 100, n_customers),
})

# Create synthetic churn labels (1 = churned)
churn_labels = pd.Series(
    (customer_features["recency_days"] > 180) &
    (customer_features["frequency"] < 5) |
    (customer_features["nps_score"] < 0)
).astype(int)

print(f"Total customers: {len(customer_features)}")
print(f"Churn rate: {churn_labels.mean():.2%}")

# Step 2: Segment Customers
print("\n=== CUSTOMER SEGMENTATION ===")
behavior_module = CustomerBehaviorModule(n_segments=4)

customer_features["ltv"] = customer_features["monetary_value"]
customer_features["churn_score"] = churn_labels

segment_labels, segments = behavior_module.segment_customers(customer_features)

for seg_id, segment in segments.items():
    print(f"\n{segment.name}:")
    print(f"  Size: {segment.size} customers ({segment.size/len(customer_features):.1%})")
    print(f"  Avg LTV: ${segment.avg_ltv:.2f}")
    print(f"  Churn Risk: {segment.churn_risk:.2%}")

# Step 3: Train Churn Prediction Model
print("\n=== CHURN PREDICTION MODEL ===")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    customer_features.drop(columns=["ltv", "churn_score"]),
    churn_labels,
    test_size=0.2,
    stratify=churn_labels,
    random_state=42,
)

# Train model
config = TrainingConfig(task_type="classification", cv_folds=5)
trainer = ModelTrainer(config)

result = trainer.train_with_cv(
    X_train,
    y_train,
    model_name="xgboost",
    search_type="random",
    n_iter=10,
)

print(f"Model: XGBoost")
print(f"CV Score: {result.mean_cv_score:.4f} ± {result.std_cv_score:.4f}")

# Evaluate on test set
evaluator = ModelEvaluator(task_type="classification")
y_pred = result.model.predict(X_test)
y_pred_proba = result.model.predict_proba(X_test)

metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)
print(f"\nTest Set Performance:")
print(f"  Accuracy: {metrics.metrics['accuracy']:.4f}")
print(f"  Precision: {metrics.metrics['precision']:.4f}")
print(f"  Recall: {metrics.metrics['recall']:.4f}")
print(f"  F1-Score: {metrics.metrics['f1']:.4f}")
print(f"  ROC-AUC: {metrics.metrics.get('roc_auc', 0):.4f}")

# Step 4: Explain Model Predictions
print("\n=== MODEL EXPLAINABILITY ===")
analyzer = ExplainabilityAnalyzer(use_shap=False)

importance = analyzer.get_feature_importance(
    result.model,
    feature_names=X_train.columns.tolist(),
)

print("Top 5 Churn Predictors:")
for feature, imp in importance.top_features[:5]:
    print(f"  {feature}: {imp:.4f}")

# Step 5: Predict Churn on Full Dataset
print("\n=== CHURN PREDICTIONS ===")
churn_predictions = behavior_module.predict_churn(
    customer_features.drop(columns=["ltv", "churn_score"]),
    train=True,
    labels=churn_labels,
)

# Analyze high-risk customers
high_risk = [p for p in churn_predictions if p.churn_risk == "high"]
medium_risk = [p for p in churn_predictions if p.churn_risk == "medium"]

print(f"High Risk Customers: {len(high_risk)} ({len(high_risk)/len(churn_predictions):.1%})")
print(f"Medium Risk Customers: {len(medium_risk)} ({len(medium_risk)/len(churn_predictions):.1%})")

# Sample high-risk customer
if high_risk:
    sample = high_risk[0]
    print(f"\nSample High-Risk Customer (ID: {sample.customer_id}):")
    print(f"  Churn Probability: {sample.churn_probability:.2%}")
    print(f"  Key Factors:")
    for factor, importance in sample.key_factors[:3]:
        print(f"    - {factor}: {importance:.4f}")
    print(f"  Recommendations:")
    for rec in sample.recommendations:
        print(f"    - {rec}")

# Step 6: Calculate Customer Lifetime Value
print("\n=== LIFETIME VALUE PREDICTION ===")
ltv_predictions = behavior_module.predict_ltv(
    customer_features.drop(columns=["ltv", "churn_score"]),
    train=True,
    ltv_values=customer_features["monetary_value"],
)

high_value_customers = [p for p in ltv_predictions if p.ltv_segment == "high_value"]
print(f"High-Value Customers: {len(high_value_customers)} ({len(high_value_customers)/len(ltv_predictions):.1%})")
print(f"Avg Predicted LTV (High-Value): ${np.mean([p.predicted_ltv for p in high_value_customers]):.2f}")

# Step 7: Prioritize Retention Efforts
print("\n=== RETENTION CAMPAIGN PRIORITIZATION ===")

# Create priority matrix
retention_targets = []
for churn_pred, ltv_pred in zip(churn_predictions, ltv_predictions):
    if churn_pred.churn_risk in ["high", "medium"] and ltv_pred.ltv_segment == "high_value":
        retention_targets.append({
            "customer_id": churn_pred.customer_id,
            "churn_probability": churn_pred.churn_probability,
            "predicted_ltv": ltv_pred.predicted_ltv,
            "priority_score": churn_pred.churn_probability * ltv_pred.predicted_ltv,
        })

retention_targets = sorted(retention_targets, key=lambda x: x["priority_score"], reverse=True)

print(f"Total Retention Targets: {len(retention_targets)}")
print(f"\nTop 5 Priority Customers:")
for i, target in enumerate(retention_targets[:5], 1):
    print(f"{i}. Customer {target['customer_id']}")
    print(f"   Churn Prob: {target['churn_probability']:.2%}")
    print(f"   Predicted LTV: ${target['predicted_ltv']:.2f}")
    print(f"   Priority Score: {target['priority_score']:.2f}")
```

**Output Example:**
```
=== CUSTOMER SEGMENTATION ===

Premium_Frequent_0:
  Size: 187 customers (18.7%)
  Avg LTV: 7,532.42
  Churn Risk: 12%

High_Value_1:
  Size: 243 customers (24.3%)
  Avg LTV: 5,891.22
  Churn Risk: 23%

At_Risk_2:
  Size: 156 customers (15.6%)
  Avg LTV: 2,341.78
  Churn Risk: 67%

Standard_3:
  Size: 414 customers (41.4%)
  Avg LTV: 3,214.56
  Churn Risk: 34%

=== CHURN PREDICTION MODEL ===
Model: XGBoost
CV Score: 0.8542 ± 0.0231

Test Set Performance:
  Accuracy: 0.8650
  Precision: 0.8123
  Recall: 0.7845
  F1-Score: 0.7982
  ROC-AUC: 0.9123

=== MODEL EXPLAINABILITY ===
Top 5 Churn Predictors:
  recency_days: 0.2845
  nps_score: 0.2134
  engagement_score: 0.1923
  frequency: 0.1567
  support_tickets: 0.0891

=== RETENTION CAMPAIGN PRIORITIZATION ===
Total Retention Targets: 89

Top 5 Priority Customers:
1. Customer 42
   Churn Prob: 89%
   Predicted LTV: 9,234.56
   Priority Score: 8,218.76

2. Customer 127
   Churn Prob: 84%
   Predicted LTV: 8,756.23
   Priority Score: 7,355.23
...
```

---

## Workflow 3: Revenue Forecasting with Scenarios

**Objective:** Forecast revenue and model multiple business scenarios.

```python
import pandas as pd
import numpy as np

from ai_business_assistant.ai_modules import ForecastingModule
from ai_business_assistant.ml_pipeline import ConfidenceScorer

# Step 1: Prepare Historical Revenue Data
print("=== REVENUE FORECASTING ===")

# Generate synthetic historical revenue
dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
base_revenue = 100000
trend = np.linspace(0, 50000, len(dates))
seasonality = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
noise = np.random.normal(0, 5000, len(dates))

revenue = pd.Series(
    base_revenue + trend + seasonality + noise,
    index=dates,
)

print(f"Historical Period: {revenue.index[0]} to {revenue.index[-1]}")
print(f"Mean Daily Revenue: ${revenue.mean():,.2f}")
print(f"Revenue Trend: {(revenue.iloc[-30:].mean() / revenue.iloc[:30].mean() - 1) * 100:+.1f}%")

# Step 2: Prophet Forecast (Baseline)
print("\n=== BASELINE FORECAST (Prophet) ===")
forecast_module = ForecastingModule(confidence_level=0.95)

prophet_forecast = forecast_module.forecast_prophet(revenue, periods=90)

print(f"Model: {prophet_forecast.model_type}")
print(f"Forecast Confidence: {prophet_forecast.confidence:.2%}")
print(f"RMSE: ${prophet_forecast.metrics['rmse']:,.2f}")

# Show sample predictions
print(f"\nSample Forecasts (next 7 days):")
for date, pred, lower, upper in zip(
    prophet_forecast.forecast.index[:7],
    prophet_forecast.forecast.values[:7],
    prophet_forecast.lower_bound.values[:7],
    prophet_forecast.upper_bound.values[:7],
):
    print(f"  {date.date()}: ${pred:,.2f} [${lower:,.2f} - ${upper:,.2f}]")

# Step 3: Scenario Modeling
print("\n=== SCENARIO FORECASTS ===")

scenarios = {
    "best_case": {
        "growth_rate": 0.15,  # 15% growth
        "volatility": 0.05,
        "probability": 0.20,
        "description": "New product launch successful, strong market conditions",
    },
    "base_case": {
        "growth_rate": 0.05,  # 5% growth
        "volatility": 0.10,
        "probability": 0.50,
        "description": "Steady growth, normal market conditions",
    },
    "conservative": {
        "growth_rate": 0.0,  # No growth
        "volatility": 0.15,
        "probability": 0.20,
        "description": "Flat growth, increased competition",
    },
    "worst_case": {
        "growth_rate": -0.10,  # -10% decline
        "volatility": 0.20,
        "probability": 0.10,
        "description": "Market downturn, significant challenges",
    },
}

scenario_results = forecast_module.scenario_modeling(revenue, periods=90, scenarios=scenarios)

for scenario in scenario_results:
    forecast_sum = scenario.forecast.sum()
    print(f"\n{scenario.scenario_name.upper()} (prob: {scenario.probability:.1%}):")
    print(f"  Description: {scenario.assumptions.get('description', 'N/A')}")
    print(f"  90-day Revenue: ${forecast_sum:,.2f}")
    print(f"  vs Baseline: {(forecast_sum / prophet_forecast.forecast.sum() - 1) * 100:+.1f}%")

# Step 4: Calculate Expected Value
print("\n=== EXPECTED VALUE ANALYSIS ===")

expected_revenue = sum(
    scenario.forecast.sum() * scenario.probability
    for scenario in scenario_results
)

print(f"Expected 90-day Revenue: ${expected_revenue:,.2f}")
print(f"vs Current Daily Avg: ${expected_revenue / 90:,.2f}/day (currently ${revenue.tail(30).mean():,.2f}/day)")

# Calculate risk metrics
scenario_revenues = [s.forecast.sum() for s in scenario_results]
revenue_std = np.std(scenario_revenues)
revenue_var_95 = np.percentile(scenario_revenues, 5)  # 95% Value at Risk

print(f"\nRisk Metrics:")
print(f"  Standard Deviation: ${revenue_std:,.2f}")
print(f"  95% VaR (worst 5%): ${revenue_var_95:,.2f}")
print(f"  Best Case: ${max(scenario_revenues):,.2f}")
print(f"  Worst Case: ${min(scenario_revenues):,.2f}")

# Step 5: Confidence Scoring
print("\n=== FORECAST CONFIDENCE ===")
scorer = ConfidenceScorer()

for scenario in scenario_results[:2]:  # Show top 2
    avg_prediction = scenario.forecast.mean()
    residual_std = prophet_forecast.metrics['rmse']
    
    confidence = scorer.score_regression(
        prediction=avg_prediction,
        residual_std=residual_std,
        model_r2=0.85,  # Example R²
    )
    
    print(f"\n{scenario.scenario_name}:")
    print(f"  Confidence: {confidence.confidence:.2%}")
    print(f"  {confidence.interpretation}")
    print(f"  Confidence Interval: ${confidence.confidence_interval[0]:,.2f} - ${confidence.confidence_interval[1]:,.2f}")
```

**Output Example:**
```
=== REVENUE FORECASTING ===
Historical Period: 2022-01-01 to 2023-12-31
Mean Daily Revenue: $125,432.18
Revenue Trend: +38.5%

=== BASELINE FORECAST (Prophet) ===
Model: Prophet
Forecast Confidence: 74%
RMSE: $6,234.56

Sample Forecasts (next 7 days):
  2024-01-01: $148,532.12 [$136,234.87 - $160,829.37]
  2024-01-02: $149,123.45 [$136,891.23 - $161,355.67]
  ...

=== SCENARIO FORECASTS ===

BEST_CASE (prob: 20.0%):
  Description: New product launch successful, strong market conditions
  90-day Revenue: $15,234,567.89
  vs Baseline: +15.2%

BASE_CASE (prob: 50.0%):
  Description: Steady growth, normal market conditions
  90-day Revenue: $13,891,234.56
  vs Baseline: +5.1%

=== EXPECTED VALUE ANALYSIS ===
Expected 90-day Revenue: $13,456,789.12
vs Current Daily Avg: $149,519.88/day (currently $145,231.45/day)

Risk Metrics:
  Standard Deviation: $1,234,567.89
  95% VaR (worst 5%): $11,234,567.89
  Best Case: $15,234,567.89
  Worst Case: $11,892,345.67
```

---

## Key Takeaways

1. **Market Analysis** provides real-time insights into market conditions and sentiment
2. **Forecasting** enables data-driven planning with uncertainty quantification
3. **Competitive Intelligence** tracks the competitive landscape systematically
4. **Customer Behavior** models enable targeted retention and growth strategies
5. **Recommendation Engine** synthesizes insights into actionable strategies
6. **ML Pipeline** provides enterprise-grade model development and deployment
7. **Confidence Scoring** quantifies prediction reliability for better decision-making

These workflows can be combined and customized for specific business needs and integrated into dashboards, automated reports, or decision support systems.

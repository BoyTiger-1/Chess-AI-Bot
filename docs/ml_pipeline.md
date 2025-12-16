# ML Pipeline Documentation

## Overview

The ML Pipeline provides comprehensive infrastructure for training, evaluating, deploying, and managing machine learning models with enterprise-grade features.

## Components

### 1. Model Training Pipeline

**Purpose:** Automated model training with preprocessing, hyperparameter tuning, and cross-validation.

**Key Features:**
- Automated data preprocessing (scaling, imputation)
- Grid search and randomized search for hyperparameter tuning
- Cross-validation with stratified folds
- Multiple model types (Random Forest, XGBoost, LightGBM, etc.)
- Feature importance extraction

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import ModelTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    task_type="classification",
    cv_folds=5,
    scale_features=True,
    scaler_type="standard",
)

trainer = ModelTrainer(config)

# Train single model
result = trainer.train_with_cv(
    X=features_df,
    y=labels,
    model_name="xgboost",
    search_type="random",
    n_iter=20,
)

print(f"Best params: {result.best_params}")
print(f"CV Score: {result.mean_cv_score:.4f} ± {result.std_cv_score:.4f}")
print(f"Training time: {result.training_time:.2f}s")

# Compare multiple models
results = trainer.compare_models(
    X=features_df,
    y=labels,
    model_names=["random_forest", "xgboost", "lightgbm"],
)

for model_name, result in results.items():
    print(f"{model_name}: {result.mean_cv_score:.4f}")
```

**Supported Models:**
- **Classification:** Random Forest, Gradient Boosting, XGBoost, LightGBM, Logistic Regression, SVM
- **Regression:** Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, Lasso, SVR

**Hyperparameter Grids:**
- Automatically configured for each model type
- Customizable via param_grid parameter
- Random search recommended for large parameter spaces

---

### 2. Model Evaluation Framework

**Purpose:** Comprehensive model evaluation with metrics, benchmarking, and A/B testing.

**Key Features:**
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Regression metrics (MSE, RMSE, MAE, R², MAPE)
- Model benchmarking
- Statistical A/B testing
- Cross-model validation

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import ModelEvaluator

evaluator = ModelEvaluator(task_type="classification")

# Evaluate single model
metrics = evaluator.evaluate(
    y_true=test_labels,
    y_pred=predictions,
    y_pred_proba=predicted_probabilities,
)

print(f"Accuracy: {metrics.metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics.metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics.metrics['roc_auc']:.4f}")

# Benchmark multiple models
models = {
    "model_v1": model_v1,
    "model_v2": model_v2,
    "model_v3": model_v3,
}

benchmark_df = evaluator.benchmark_models(models, X_test, y_test)
print(benchmark_df.sort_values("accuracy", ascending=False))

# A/B test two models
ab_result = evaluator.ab_test(
    model_a_predictions=pred_a,
    model_b_predictions=pred_b,
    y_true=y_test,
    metric="f1",
    confidence_level=0.95,
)

print(f"Model A: {ab_result.model_a_metric:.4f}")
print(f"Model B: {ab_result.model_b_metric:.4f}")
print(f"Improvement: {ab_result.improvement:.2f}%")
print(f"P-value: {ab_result.p_value:.4f}")
print(f"Significant: {ab_result.is_significant}")
print(f"Recommendation: {ab_result.recommendation}")
```

**Evaluation Metrics:**

*Classification:*
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (binary classification)
- Confusion Matrix
- Classification Report

*Regression:*
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

---

### 3. Explainability Layer

**Purpose:** Explain model predictions using SHAP values and feature importance.

**Key Features:**
- SHAP (SHapley Additive exPlanations) values
- Feature importance extraction
- Individual prediction explanations
- Model behavior summarization

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(use_shap=True)

# Get feature importance
importance = analyzer.get_feature_importance(
    model=trained_model,
    feature_names=feature_names,
)

print("Top 5 features:")
for feature, imp in importance.top_features[:5]:
    print(f"  {feature}: {imp:.4f}")

# SHAP-based explanation
shap_explanation = analyzer.explain_with_shap(
    model=trained_model,
    X=test_data,
    feature_names=feature_names,
    model_type="tree",
)

print("Top SHAP contributors:")
for feature, contribution in shap_explanation.top_contributors[:5]:
    print(f"  {feature}: {contribution:.4f}")

# Explain single prediction
explanation = analyzer.explain_prediction(
    model=trained_model,
    instance=single_instance,
    feature_names=feature_names,
)

print(explanation.explanation_text)
```

**SHAP Model Types:**
- **Tree Models:** TreeExplainer (fast, exact for tree-based models)
- **Linear Models:** LinearExplainer
- **Any Model:** KernelExplainer (slower, model-agnostic)

**Interpretation:**
- Positive SHAP values increase prediction
- Negative SHAP values decrease prediction
- Magnitude indicates strength of contribution

---

### 4. Ensemble Methods

**Purpose:** Combine multiple models for improved predictions.

**Key Features:**
- Voting ensembles (hard/soft voting)
- Weighted averaging
- Stacking with meta-learner
- Confidence estimation from model agreement

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import EnsembleModel

# Create ensemble
models = {
    "rf": random_forest_model,
    "xgb": xgboost_model,
    "lgbm": lightgbm_model,
}

ensemble = EnsembleModel(
    models=models,
    method="stacking",
    task_type="classification",
)

# Fit ensemble
ensemble.fit(X_train, y_train)

# Predict
result = ensemble.predict(X_test)

print(f"Ensemble predictions: {result.predictions}")
print(f"Individual predictions: {result.individual_predictions}")
print(f"Confidence: {result.confidence}")

# Get probabilities (classification)
probabilities = ensemble.predict_proba(X_test)
```

**Ensemble Methods:**
- **Voting:** Simple or weighted voting across models
- **Averaging:** Average predictions (regression) or probabilities (classification)
- **Stacking:** Train meta-model on base model predictions
- **Weighted:** Custom weights for each model

**Confidence Calculation:**
- Classification: Agreement ratio among models
- Regression: Inverse of coefficient of variation

---

### 5. Inference Engine

**Purpose:** Deploy models for batch and real-time inference.

**Key Features:**
- Real-time single-instance prediction
- Batch processing with memory management
- Parallel processing for large batches
- Performance monitoring

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import InferenceEngine

# Initialize engine
engine = InferenceEngine(
    model=trained_model,
    preprocessing_pipeline=preprocessing,
    model_version="1.2.0",
    batch_size=1000,
    n_workers=4,
)

# Real-time prediction
result = engine.predict_realtime(single_instance, return_proba=True)
print(f"Prediction: {result.predictions}")
print(f"Latency: {result.prediction_time*1000:.2f}ms")

# Batch prediction
result = engine.predict_batch(large_dataset, parallel=True)
print(f"Predictions: {len(result.predictions)}")
print(f"Throughput: {result.metadata['throughput']:.1f} samples/sec")

# Get performance stats
stats = engine.get_performance_stats()
print(f"Total inferences: {stats['total_inferences']}")
print(f"Avg latency: {stats['avg_latency']*1000:.2f}ms")
print(f"Throughput: {stats['throughput']:.1f} samples/sec")
```

**Performance Optimization:**
- Batch processing for efficiency
- Parallel workers for CPU-bound models
- Memory-aware chunking
- Preprocessing caching

---

### 6. Model Versioning

**Purpose:** Manage model versions with storage and rollback capabilities.

**Key Features:**
- Model serialization and storage
- Metadata tracking (metrics, hyperparameters, timestamps)
- Production model management
- Version comparison
- Easy rollback

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import ModelVersionManager, ModelMetadata
from datetime import datetime

manager = ModelVersionManager(storage_path="models/")

# Save model version
metadata = ModelMetadata(
    version="1.2.0",
    created_at=datetime.now(),
    model_type="XGBoost",
    metrics={"accuracy": 0.92, "f1": 0.90},
    hyperparameters={"n_estimators": 100, "max_depth": 5},
    description="Improved feature engineering",
    tags=["production-candidate", "high-accuracy"],
)

manager.save_model(
    model=trained_model,
    version="1.2.0",
    metadata=metadata,
    preprocessing_pipeline=preprocessing,
    set_as_production=True,
)

# Load production model
model_version = manager.load_model(use_production=True)
print(f"Loaded: {model_version.metadata.version}")
print(f"Metrics: {model_version.metadata.metrics}")

# List versions
versions = manager.list_versions(limit=10)
for v in versions:
    print(f"{v['version']} - {v['created_at']} - Production: {v['is_production']}")

# Compare versions
comparison = manager.compare_versions("1.1.0", "1.2.0")
print(comparison["metrics_comparison"]["improvements"])

# Rollback if needed
manager.rollback("1.1.0")
```

**Storage Structure:**
```
models/
├── registry.json
├── 1.0.0/
│   ├── model.joblib
│   ├── preprocessing.joblib
│   └── metadata.json
├── 1.1.0/
│   └── ...
└── 1.2.0/
    └── ...
```

---

### 7. Confidence Scoring

**Purpose:** Provide confidence scores for predictions.

**Key Features:**
- Probability-based confidence (classification)
- Residual-based confidence (regression)
- Ensemble agreement scoring
- Calibration support
- Human-readable interpretation

**Usage Example:**
```python
from ai_business_assistant.ml_pipeline import ConfidenceScorer

scorer = ConfidenceScorer(calibrated=False, confidence_level=0.95)

# Score classification prediction
confidence = scorer.score_classification(
    prediction=1,
    predicted_proba=np.array([0.15, 0.85]),
    model_accuracy=0.90,
    ensemble_agreement=0.88,
)

print(f"Prediction: {confidence.prediction}")
print(f"Confidence: {confidence.confidence:.2%}")
print(f"Interpretation: {confidence.interpretation}")
print(f"Factors: {confidence.factors}")

# Score regression prediction
confidence = scorer.score_regression(
    prediction=105.7,
    residual_std=8.5,
    model_r2=0.85,
    ensemble_std=2.3,
)

print(f"Confidence Interval: {confidence.confidence_interval}")

# Calibrate on validation data
calibration_metrics = scorer.calibrate(
    y_true=y_val,
    y_pred=y_pred_val,
    predicted_probas=y_proba_val,
    task_type="classification",
)

print(f"Calibration error: {calibration_metrics['calibration_error']:.4f}")
```

**Confidence Factors:**
- **Classification:**
  - Maximum probability
  - Entropy (certainty)
  - Model accuracy
  - Ensemble agreement
  
- **Regression:**
  - Residual magnitude
  - Model R² score
  - Ensemble standard deviation

**Interpretation Levels:**
- Very High (≥0.9): Highly reliable
- High (≥0.75): Reliable
- Moderate (≥0.6): Reasonable confidence
- Low (≥0.4): Use with caution
- Very Low (<0.4): Unreliable

---

## Complete Workflow Example

```python
from ai_business_assistant.ml_pipeline import (
    ModelTrainer,
    ModelEvaluator,
    ExplainabilityAnalyzer,
    EnsembleModel,
    InferenceEngine,
    ModelVersionManager,
    ConfidenceScorer,
    TrainingConfig,
    ModelMetadata,
)
from datetime import datetime
import pandas as pd

# 1. Train multiple models
config = TrainingConfig(task_type="classification", cv_folds=5)
trainer = ModelTrainer(config)

results = trainer.compare_models(X_train, y_train)
best_model_name = max(results, key=lambda k: results[k].mean_cv_score)
best_result = results[best_model_name]

print(f"Best model: {best_model_name}")
print(f"CV Score: {best_result.mean_cv_score:.4f}")

# 2. Evaluate on test set
evaluator = ModelEvaluator(task_type="classification")
metrics = evaluator.evaluate(
    y_test,
    best_result.model.predict(X_test),
    best_result.model.predict_proba(X_test),
)

print(f"Test Accuracy: {metrics.metrics['accuracy']:.4f}")

# 3. Explain model
analyzer = ExplainabilityAnalyzer(use_shap=True)
importance = analyzer.get_feature_importance(best_result.model, feature_names)
print("Top features:", importance.top_features[:5])

# 4. Save versioned model
manager = ModelVersionManager()
metadata = ModelMetadata(
    version="2.0.0",
    created_at=datetime.now(),
    model_type=best_model_name,
    metrics=metrics.metrics,
    hyperparameters=best_result.best_params,
)

manager.save_model(
    best_result.model,
    "2.0.0",
    metadata,
    trainer._preprocessing_pipeline,
    set_as_production=True,
)

# 5. Deploy inference engine
engine = InferenceEngine(
    model=best_result.model,
    preprocessing_pipeline=trainer._preprocessing_pipeline,
    model_version="2.0.0",
)

# 6. Score confidence
scorer = ConfidenceScorer()
result = engine.predict_realtime(new_data, return_proba=True)
confidence = scorer.score_classification(
    result.predictions[0],
    result.predictions[0] if len(result.predictions.shape) > 1 else np.array([0.5, 0.5]),
    model_accuracy=metrics.metrics['accuracy'],
)

print(f"Prediction: {result.predictions[0]}")
print(f"Confidence: {confidence.confidence:.2%}")
```

## Best Practices

1. **Data Splitting:** Use stratified splits for imbalanced classification
2. **Feature Scaling:** Always scale features for distance-based models
3. **Hyperparameter Tuning:** Use random search for large parameter spaces
4. **Cross-Validation:** Use 5-10 folds depending on dataset size
5. **Model Selection:** Compare multiple model types
6. **Explainability:** Always explain important predictions
7. **Versioning:** Track all model versions and metadata
8. **Monitoring:** Monitor model performance in production
9. **Retraining:** Retrain models periodically with new data
10. **A/B Testing:** Always A/B test before full deployment

## Performance Considerations

- **Training:** Can take minutes to hours depending on data size and model complexity
- **Inference:** <1ms per sample for most models (excluding preprocessing)
- **SHAP:** Can be slow for large datasets; use sampling
- **Ensemble:** 2-5x slower than single model inference
- **Memory:** ~100MB-1GB per model depending on complexity

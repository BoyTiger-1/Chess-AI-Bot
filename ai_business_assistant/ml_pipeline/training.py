"""ML Model Training Pipeline.

Provides data preparation, hyperparameter tuning, cross-validation, and model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
import xgboost as xgb
import lightgbm as lgb


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    task_type: str  # 'classification' or 'regression'
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    test_size: float = 0.2
    scale_features: bool = True
    scaler_type: str = "standard"  # 'standard', 'minmax', 'robust'
    handle_missing: bool = True
    imputation_strategy: str = "median"


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    best_params: Dict[str, Any]
    cv_scores: np.ndarray
    mean_cv_score: float
    std_cv_score: float
    test_score: float
    feature_importance: Optional[Dict[str, float]]
    training_time: float


class ModelTrainer:
    """ML model training pipeline with automated preprocessing and tuning.
    
    Assumptions:
    - Input data is a pandas DataFrame
    - Target variable is provided separately
    - Features are numeric or will be encoded
    - Missing values are handled during preprocessing
    
    Limitations:
    - Grid search can be computationally expensive for large parameter spaces
    - Cross-validation assumes data is i.i.d.
    - Feature engineering must be done beforehand
    - Time series data requires special handling
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the model trainer.
        
        Args:
            config: Training configuration object
        """
        self.config = config or TrainingConfig(task_type="classification")
        self._preprocessing_pipeline: Optional[Pipeline] = None
        self._model_registry = self._initialize_model_registry()
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_preprocessing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target variable
            fit_preprocessing: Whether to fit preprocessing pipeline
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        if fit_preprocessing:
            steps = []
            
            if self.config.handle_missing:
                imputer = SimpleImputer(
                    strategy=self.config.imputation_strategy,
                    add_indicator=True,
                )
                steps.append(("imputer", imputer))
            
            if self.config.scale_features:
                if self.config.scaler_type == "standard":
                    scaler = StandardScaler()
                elif self.config.scaler_type == "minmax":
                    scaler = MinMaxScaler()
                elif self.config.scaler_type == "robust":
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
                
                steps.append(("scaler", scaler))
            
            self._preprocessing_pipeline = Pipeline(steps)
            X_processed = self._preprocessing_pipeline.fit_transform(X)
        else:
            if self._preprocessing_pipeline is None:
                raise ValueError("Preprocessing pipeline not fitted")
            X_processed = self._preprocessing_pipeline.transform(X)
        
        y_processed = y.values if isinstance(y, pd.Series) else y
        
        return X_processed, y_processed
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        search_type: str = "grid",
        n_iter: int = 20,
    ) -> TrainingResult:
        """Train model with cross-validation and hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to train
            param_grid: Parameter grid for hyperparameter search
            search_type: 'grid' or 'random' search
            n_iter: Number of iterations for random search
            
        Returns:
            TrainingResult object
        """
        import time
        start_time = time.time()
        
        X_processed, y_processed = self.prepare_data(X, y, fit_preprocessing=True)
        
        base_model = self._get_model(model_name)
        
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_name)
        
        if self.config.task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        
        if search_type == "grid":
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=self._get_scoring_metric(),
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=self._get_scoring_metric(),
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        search.fit(X_processed, y_processed)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        cv_scores = cross_val_score(
            best_model,
            X_processed,
            y_processed,
            cv=cv,
            scoring=self._get_scoring_metric(),
            n_jobs=self.config.n_jobs,
        )
        
        test_score = best_model.score(X_processed, y_processed)
        
        feature_importance = self._extract_feature_importance(
            best_model,
            X.columns if hasattr(X, "columns") else None,
        )
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model=best_model,
            best_params=best_params,
            cv_scores=cv_scores,
            mean_cv_score=float(cv_scores.mean()),
            std_cv_score=float(cv_scores.std()),
            test_score=float(test_score),
            feature_importance=feature_importance,
            training_time=training_time,
        )
    
    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, TrainingResult]:
        """Compare multiple models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_names: List of model names to compare
            
        Returns:
            Dictionary mapping model names to TrainingResult objects
        """
        if model_names is None:
            if self.config.task_type == "classification":
                model_names = ["random_forest", "gradient_boosting", "xgboost", "logistic"]
            else:
                model_names = ["random_forest", "gradient_boosting", "xgboost", "ridge"]
        
        results = {}
        
        for model_name in model_names:
            print(f"\nTraining {model_name}...")
            try:
                result = self.train_with_cv(
                    X,
                    y,
                    model_name,
                    param_grid=None,
                    search_type="random",
                    n_iter=10,
                )
                results[model_name] = result
                print(f"{model_name}: CV Score = {result.mean_cv_score:.4f} (+/- {result.std_cv_score:.4f})")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        return results
    
    def _initialize_model_registry(self) -> Dict[str, Any]:
        """Initialize registry of available models."""
        registry = {}
        
        if self.config.task_type == "classification":
            registry.update({
                "random_forest": RandomForestClassifier(random_state=self.config.random_state),
                "gradient_boosting": GradientBoostingClassifier(random_state=self.config.random_state),
                "xgboost": xgb.XGBClassifier(random_state=self.config.random_state, eval_metric="logloss"),
                "lightgbm": lgb.LGBMClassifier(random_state=self.config.random_state, verbose=-1),
                "logistic": LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                "svm": SVC(random_state=self.config.random_state),
            })
        else:
            registry.update({
                "random_forest": RandomForestRegressor(random_state=self.config.random_state),
                "gradient_boosting": GradientBoostingRegressor(random_state=self.config.random_state),
                "xgboost": xgb.XGBRegressor(random_state=self.config.random_state),
                "lightgbm": lgb.LGBMRegressor(random_state=self.config.random_state, verbose=-1),
                "ridge": Ridge(random_state=self.config.random_state),
                "lasso": Lasso(random_state=self.config.random_state),
                "svr": SVR(),
            })
        
        return registry
    
    def _get_model(self, model_name: str) -> Any:
        """Get model instance by name."""
        if model_name not in self._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        return self._model_registry[model_name]
    
    def _get_default_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """Get default parameter grid for model."""
        if model_name == "random_forest":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif model_name == "gradient_boosting":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            }
        elif model_name == "xgboost":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }
        elif model_name == "lightgbm":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 70],
            }
        elif model_name in ["logistic", "ridge", "lasso"]:
            return {
                "C": [0.001, 0.01, 0.1, 1, 10, 100] if model_name == "logistic" else [0.1, 1, 10, 100],
                "alpha": [0.001, 0.01, 0.1, 1, 10] if model_name in ["ridge", "lasso"] else [1],
            }
        else:
            return {}
    
    def _get_scoring_metric(self) -> str:
        """Get scoring metric based on task type."""
        if self.config.task_type == "classification":
            return "roc_auc"
        else:
            return "neg_mean_squared_error"
    
    def _extract_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]],
    ) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        if not hasattr(model, "feature_importances_"):
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))

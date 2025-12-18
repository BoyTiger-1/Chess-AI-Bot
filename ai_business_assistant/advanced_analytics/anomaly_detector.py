"""
Advanced Anomaly Detection Engine
Provides multiple algorithms for detecting anomalies in business data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class AnomalyMethod(Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    STATISTICAL_ZSCORE = "statistical_zscore"
    STATISTICAL_IQR = "statistical_iqr"
    STATISTICAL_MAD = "statistical_mad"
    COVARIANCE_ELLIPTIC = "covariance_elliptic"
    AUTOENCODER = "autoencoder"
    DBSCAN_CLUSTERING = "dbscan_clustering"
    MULTIVARIATE_ZSCORE = "multivariate_zscore"
    GRAPH_BASED = "graph_based"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    method: AnomalyMethod
    is_anomaly: Union[bool, np.ndarray]
    anomaly_score: Union[float, np.ndarray]
    confidence: float
    threshold: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    indices: Optional[np.ndarray] = None


@dataclass
class TimeAnomalyResult:
    """Time series anomaly detection result."""
    is_anomaly: np.ndarray
    anomaly_score: np.ndarray
    trend_break_indices: List[int]
    change_point_indices: List[int]
    seasonal_anomalies: Dict[str, List[int]]
    trend_anomalies: List[int]


class AnomalyDetector:
    """
    Advanced anomaly detection engine with multiple algorithms.
    Supports univariate, multivariate, and time series anomaly detection.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0-1)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.thresholds = {}
        
    def detect_isolation_forest(self, 
                               data: Union[pd.DataFrame, np.ndarray],
                               features: Optional[List[str]] = None) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: Input data
            features: Column names to use (for DataFrame)
        """
        # Prepare data
        if isinstance(data, pd.DataFrame):
            if features:
                data = data[features]
            data = data.select_dtypes(include=[np.number])
        
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(data_array)
        scores = iso_forest.decision_function(data_array)
        
        # Convert predictions to boolean (1 for normal, -1 for anomaly)
        is_anomaly = predictions == -1
        confidence = np.abs(scores) / np.max(np.abs(scores)) if np.max(np.abs(scores)) > 0 else np.zeros_like(scores)
        
        # Get threshold
        threshold = np.percentile(scores, self.contamination * 100)
        
        return AnomalyResult(
            method=AnomalyMethod.ISOLATION_FOREST,
            is_anomaly=is_anomaly,
            anomaly_score=-scores,  # Higher values indicate anomalies
            confidence=np.mean(confidence),
            threshold=-threshold,
            details={
                "n_estimators": 100,
                "contamination": self.contamination,
                "n_anomalies": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly),
                "feature_importance": iso_forest.feature_importances_.tolist() if hasattr(iso_forest, 'feature_importances_') else None
            }
        )
    
    def detect_one_class_svm(self, 
                           data: Union[pd.DataFrame, np.ndarray],
                           kernel: str = 'rbf',
                           nu: Optional[float] = None) -> AnomalyResult:
        """
        Detect anomalies using One-Class SVM.
        
        Args:
            data: Input data
            kernel: SVM kernel ('linear', 'poly', 'rbf', 'sigmoid')
            nu: Upper bound fraction of outliers (if None, uses contamination)
        """
        if nu is None:
            nu = self.contamination
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number])
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_array)
        
        # Fit One-Class SVM
        svm = OneClassSVM(kernel=kernel, nu=nu, gamma='scale')
        predictions = svm.fit_predict(data_scaled)
        scores = svm.decision_function(data_scaled)
        
        # Convert predictions to boolean (1 for normal, -1 for anomaly)
        is_anomaly = predictions == -1
        confidence = np.abs(scores) / np.max(np.abs(scores)) if np.max(np.abs(scores)) > 0 else np.zeros_like(scores)
        
        return AnomalyResult(
            method=AnomalyMethod.ONE_CLASS_SVM,
            is_anomaly=is_anomaly,
            anomaly_score=-scores,  # Higher values indicate anomalies
            confidence=np.mean(confidence),
            threshold=np.percentile(scores, nu * 100),
            details={
                "kernel": kernel,
                "nu": nu,
                "gamma": "scale",
                "n_anomalies": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly)
            }
        )
    
    def detect_local_outlier_factor(self, 
                                  data: Union[pd.DataFrame, np.ndarray],
                                  n_neighbors: int = 20) -> AnomalyResult:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            data: Input data
            n_neighbors: Number of neighbors to use
        """
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number])
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination
        )
        predictions = lof.fit_predict(data_array)
        scores = lof.negative_outlier_factor_
        
        # Convert predictions to boolean
        is_anomaly = predictions == -1
        confidence = np.abs(scores) / np.max(np.abs(scores)) if np.max(np.abs(scores)) > 0 else np.zeros_like(scores)
        
        return AnomalyResult(
            method=AnomalyMethod.LOCAL_OUTLIER_FACTOR,
            is_anomaly=is_anomaly,
            anomaly_score=-scores,  # Higher values indicate anomalies
            confidence=np.mean(confidence),
            threshold=np.percentile(scores, self.contamination * 100),
            details={
                "n_neighbors": n_neighbors,
                "contamination": self.contamination,
                "n_anomalies": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly),
                "lof_scores": scores.tolist()
            }
        )
    
    def detect_statistical_anomalies(self, 
                                   data: pd.Series,
                                   method: str = "zscore",
                                   threshold: float = 3.0) -> AnomalyResult:
        """
        Detect anomalies using statistical methods.
        
        Args:
            data: Univariate time series
            method: "zscore", "iqr", "mad"
            threshold: Z-score or IQR multiplier threshold
        """
        if method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(data.dropna()))
            is_anomaly = z_scores > threshold
            anomaly_scores = z_scores
            threshold_val = threshold
            
        elif method == "iqr":
            # Interquartile Range method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            is_anomaly = (data < lower_bound) | (data > upper_bound)
            anomaly_scores = np.where(data < lower_bound, 
                                    (lower_bound - data) / IQR,
                                    (data - upper_bound) / IQR)
            anomaly_scores = np.maximum(anomaly_scores, 0)
            threshold_val = threshold
            
        elif method == "mad":
            # Median Absolute Deviation method
            median = data.median()
            mad = np.median(np.abs(data - median))
            
            # Modified z-score
            modified_z_scores = 0.6745 * (data - median) / mad
            is_anomaly = np.abs(modified_z_scores) > threshold
            anomaly_scores = np.abs(modified_z_scores)
            threshold_val = threshold
            
        else:
            raise ValueError(f"Statistical method '{method}' not supported")
        
        confidence = np.minimum(anomaly_scores / threshold, 1.0)
        
        return AnomalyResult(
            method=AnomalyMethod.STATISTICAL_ZSCORE if method == "zscore" else 
                   AnomalyMethod.STATISTICAL_IQR if method == "iqr" else 
                   AnomalyMethod.STATISTICAL_MAD,
            is_anomaly=is_anomaly.values if hasattr(is_anomaly, 'values') else is_anomaly,
            anomaly_score=anomaly_scores if hasattr(anomaly_scores, 'values') else anomaly_scores,
            confidence=np.mean(confidence),
            threshold=threshold_val,
            details={
                "method": method,
                "threshold": threshold,
                "n_anomalies": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly)
            }
        )
    
    def detect_multivariate_anomalies(self, 
                                    data: pd.DataFrame,
                                    method: str = "mahalanobis") -> AnomalyResult:
        """
        Detect multivariate anomalies.
        
        Args:
            data: Multivariate data
            method: "mahalanobis", "robust_mahalanobis", "pca_reconstruction"
        """
        numeric_data = data.select_dtypes(include=[np.number])
        data_array = numeric_data.values
        
        if method == "mahalanobis":
            # Classical Mahalanobis distance
            mean = np.mean(data_array, axis=0)
            cov = np.cov(data_array.T)
            
            # Handle singular covariance matrix
            if np.linalg.det(cov) == 0:
                cov = np.eye(cov.shape[0]) * 1e-6
                cov += np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
            
            inv_cov = np.linalg.pinv(cov)
            distances = np.array([mahalanobis(row, mean, inv_cov) for row in data_array])
            
            threshold = np.percentile(distances, (1 - self.contamination) * 100)
            is_anomaly = distances > threshold
            
        elif method == "robust_mahalanobis":
            # Robust Mahalanobis using Minimum Covariance Determinant
            robust_cov = EllipticEnvelope(contamination=self.contamination, random_state=self.random_state)
            robust_cov.fit(data_array)
            
            distances = robust_cov.mahalanobise(data_array)
            threshold = np.percentile(distances, (1 - self.contamination) * 100)
            is_anomaly = distances > threshold
            
        elif method == "pca_reconstruction":
            # PCA-based reconstruction error
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Use fewer components to capture main variance
            n_components = min(data_scaled.shape[1] - 1, int(0.8 * data_scaled.shape[1]))
            pca = PCA(n_components=n_components)
            
            pca.fit(data_scaled)
            reconstructed = pca.inverse_transform(pca.transform(data_scaled))
            
            # Reconstruction error as anomaly score
            reconstruction_errors = np.sum((data_scaled - reconstructed) ** 2, axis=1)
            threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
            is_anomaly = reconstruction_errors > threshold
            distances = reconstruction_errors
            
        else:
            raise ValueError(f"Multivariate method '{method}' not supported")
        
        confidence = np.minimum(distances / threshold, 1.0)
        
        return AnomalyResult(
            method=AnomalyMethod.MULTIVARIATE_ZSCORE,
            is_anomaly=is_anomaly,
            anomaly_score=distances,
            confidence=np.mean(confidence),
            threshold=threshold,
            details={
                "method": method,
                "n_features": numeric_data.shape[1],
                "n_anomalies": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly),
                "threshold": threshold
            }
        )
    
    def detect_dbscan_anomalies(self, 
                              data: Union[pd.DataFrame, np.ndarray],
                              eps: float = 0.5,
                              min_samples: int = 5) -> AnomalyResult:
        """
        Detect anomalies using DBSCAN clustering.
        
        Args:
            data: Input data
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
        """
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number])
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data_array)
        
        # Points with label -1 are considered anomalies
        is_anomaly = cluster_labels == -1
        anomaly_scores = np.zeros(len(data_array))
        
        # Calculate distance to nearest cluster for anomaly scoring
        from sklearn.metrics import pairwise_distances_argmin_min
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        
        if len(unique_clusters) > 0:
            cluster_centers = []
            for cluster_id in unique_clusters:
                cluster_points = data_array[cluster_labels == cluster_id]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
            
            if cluster_centers:
                cluster_centers = np.array(cluster_centers)
                # Find distance to nearest cluster center
                _, distances = pairwise_distances_argmin_min(data_array, cluster_centers)
                anomaly_scores = distances
        
        threshold = np.percentile(anomaly_scores, (1 - self.contamination) * 100) if len(anomaly_scores) > 0 else 0
        confidence = np.minimum(anomaly_scores / threshold, 1.0) if threshold > 0 else np.zeros_like(anomaly_scores)
        
        return AnomalyResult(
            method=AnomalyMethod.DBSCAN_CLUSTERING,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_scores,
            confidence=np.mean(confidence),
            threshold=threshold,
            details={
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": len(unique_clusters),
                "n_noise_points": np.sum(is_anomaly),
                "anomaly_rate": np.mean(is_anomaly),
                "cluster_labels": cluster_labels.tolist()
            }
        )
    
    def detect_time_series_anomalies(self, 
                                   time_series: pd.Series,
                                   window_size: int = 10,
                                   method: str = "rolling_stats") -> TimeAnomalyResult:
        """
        Detect anomalies in time series data.
        
        Args:
            time_series: Time series data
            window_size: Size of rolling window
            method: "rolling_stats", "change_point", "seasonal", "trend"
        """
        timestamps = time_series.index
        values = time_series.values
        n = len(values)
        
        is_anomaly = np.zeros(n, dtype=bool)
        anomaly_scores = np.zeros(n)
        
        trend_break_indices = []
        change_point_indices = []
        seasonal_anomalies = {"daily": [], "weekly": [], "monthly": []}
        trend_anomalies = []
        
        if method == "rolling_stats":
            # Rolling statistics approach
            rolling_mean = pd.Series(values).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(values).rolling(window=window_size, center=True).std()
            
            # Detect anomalies using rolling z-score
            z_scores = np.abs((values - rolling_mean) / rolling_std)
            threshold = 2.5
            
            is_anomaly = z_scores > threshold
            anomaly_scores = z_scores
            
        elif method == "change_point":
            # Change point detection using cumulative sum
            values_centered = values - np.mean(values)
            cusum = np.cumsum(values_centered)
            
            # Detect significant changes in cumulative sum
            cusum_diff = np.diff(cusum)
            threshold = 2 * np.std(cusum_diff)
            
            change_points = np.where(np.abs(cusum_diff) > threshold)[0]
            is_anomaly[change_points] = True
            change_point_indices = change_points.tolist()
            anomaly_scores = np.abs(cusum_diff)
            
        elif method == "seasonal":
            # Seasonal anomaly detection
            if isinstance(timestamps, pd.DatetimeIndex):
                # Extract temporal features
                hour = timestamps.hour if hasattr(timestamps, 'hour') else pd.Series(timestamps).dt.hour
                day_of_week = timestamps.dayofweek if hasattr(timestamps, 'dayofweek') else pd.Series(timestamps).dt.dayofweek
                month = timestamps.month if hasattr(timestamps, 'month') else pd.Series(timestamps).dt.month
                
                # Detect daily anomalies
                for h in hour.unique():
                    daily_data = values[hour == h]
                    if len(daily_data) > 1:
                        daily_mean = np.mean(daily_data)
                        daily_std = np.std(daily_data)
                        if daily_std > 0:
                            z_scores = np.abs((daily_data - daily_mean) / daily_std)
                            anomaly_indices = np.where(hour == h)[0][z_scores > 2.5]
                            is_anomaly[anomaly_indices] = True
                            seasonal_anomalies["daily"].extend(anomaly_indices.tolist())
                
                # Weekly anomalies
                for day in day_of_week.unique():
                    weekly_data = values[day_of_week == day]
                    if len(weekly_data) > 1:
                        weekly_mean = np.mean(weekly_data)
                        weekly_std = np.std(weekly_data)
                        if weekly_std > 0:
                            z_scores = np.abs((weekly_data - weekly_mean) / weekly_std)
                            anomaly_indices = np.where(day_of_week == day)[0][z_scores > 2.5]
                            is_anomaly[anomaly_indices] = True
                            seasonal_anomalies["weekly"].extend(anomaly_indices.tolist())
                
                anomaly_scores = np.abs((values - np.mean(values)) / np.std(values))
        
        elif method == "trend":
            # Trend anomaly detection
            from sklearn.linear_model import LinearRegression
            
            # Fit linear trend
            X = np.arange(n).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, values)
            trend = lr.predict(X)
            
            # Detrend the series
            detrended = values - trend
            
            # Detect anomalies in detrended series
            z_scores = np.abs(stats.zscore(detrended))
            is_anomaly = z_scores > 2.5
            anomaly_scores = z_scores
            
            # Detect trend breaks
            trend_diff = np.diff(trend)
            trend_breaks = np.where(np.abs(np.diff(trend_diff)) > np.std(trend_diff) * 2)[0]
            trend_break_indices = trend_breaks.tolist()
            trend_anomalies = trend_breaks.tolist()
        
        else:
            raise ValueError(f"Time series method '{method}' not supported")
        
        return TimeAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_scores,
            trend_break_indices=trend_break_indices,
            change_point_indices=change_point_indices,
            seasonal_anomalies=seasonal_anomalies,
            trend_anomalies=trend_anomalies
        )
    
    def ensemble_anomaly_detection(self, 
                                 data: Union[pd.DataFrame, pd.Series],
                                 methods: List[str] = None,
                                 voting: str = "majority") -> AnomalyResult:
        """
        Ensemble anomaly detection combining multiple methods.
        
        Args:
            data: Input data
            methods: List of methods to combine
            voting: "majority", "unanimous", "weighted"
        """
        if methods is None:
            methods = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        
        results = []
        weights = []
        
        for method in methods:
            try:
                if method == "isolation_forest":
                    result = self.detect_isolation_forest(data)
                elif method == "one_class_svm":
                    result = self.detect_one_class_svm(data)
                elif method == "local_outlier_factor":
                    result = self.detect_local_outlier_factor(data)
                elif method == "statistical_zscore":
                    if isinstance(data, pd.Series):
                        result = self.detect_statistical_anomalies(data, "zscore")
                    else:
                        # Convert to series for univariate method
                        series = data.iloc[:, 0] if len(data.columns) > 0 else pd.Series(data.flatten())
                        result = self.detect_statistical_anomalies(series, "zscore")
                else:
                    logger.warning(f"Method '{method}' not supported in ensemble")
                    continue
                
                results.append(result)
                weights.append(result.confidence)
                
            except Exception as e:
                logger.error(f"Failed to run {method}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid methods found for ensemble detection")
        
        # Combine results
        if voting == "majority":
            # Majority voting
            anomaly_matrix = np.array([r.is_anomaly for r in results])
            combined_anomaly = np.sum(anomaly_matrix, axis=0) > len(results) / 2
            
        elif voting == "unanimous":
            # Unanimous voting
            anomaly_matrix = np.array([r.is_anomaly for r in results])
            combined_anomaly = np.all(anomaly_matrix, axis=0)
            
        elif voting == "weighted":
            # Weighted voting based on confidence
            weights = np.array(weights)
            anomaly_matrix = np.array([r.is_anomaly for r in results]).astype(float)
            
            # Calculate weighted average anomaly scores
            weighted_scores = np.average(anomaly_matrix, axis=0, weights=weights)
            threshold = 0.5
            combined_anomaly = weighted_scores > threshold
            
        else:
            raise ValueError(f"Voting method '{voting}' not supported")
        
        # Calculate combined confidence
        combined_confidence = np.mean([r.confidence for r in results])
        
        return AnomalyResult(
            method=AnomalyMethod.AUTOENCODER,  # Use as ensemble identifier
            is_anomaly=combined_anomaly,
            anomaly_score=np.mean([r.anomaly_score for r in results], axis=0),
            confidence=combined_confidence,
            threshold=0.5,
            details={
                "ensemble_methods": methods,
                "voting": voting,
                "individual_results": [r.details for r in results],
                "weights": weights,
                "n_anomalies": np.sum(combined_anomaly),
                "anomaly_rate": np.mean(combined_anomaly)
            }
        )
    
    def get_anomaly_interpretation(self, 
                                 result: AnomalyResult,
                                 data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Provide interpretation of anomaly detection results.
        
        Args:
            result: Anomaly detection result
            data: Original data for interpretation
        """
        interpretation = {
            "method_used": result.method.value,
            "summary": {
                "total_anomalies": np.sum(result.is_anomaly) if hasattr(result.is_anomaly, '__iter__') else int(result.is_anomaly),
                "anomaly_rate": np.mean(result.is_anomaly) if hasattr(result.is_anomaly, '__iter__') else float(result.is_anomaly),
                "confidence_level": result.confidence,
                "severity_distribution": {}
            },
            "recommendations": [],
            "business_impact": {},
            "next_steps": []
        }
        
        # Analyze anomaly patterns
        if hasattr(result.is_anomaly, '__iter__') and len(result.is_anomaly) > 0:
            anomaly_indices = np.where(result.is_anomaly)[0]
            
            if len(anomaly_indices) > 0:
                # Pattern analysis
                if len(anomaly_indices) > 1:
                    gaps = np.diff(anomaly_indices)
                    if np.max(gaps) > 10:
                        interpretation["recommendations"].append("Consider investigating potential system changes or external factors")
                    
                    if np.min(gaps) == 1:
                        interpretation["recommendations"].append("Anomalies appear consecutively - check for system malfunctions")
                
                # Severity analysis
                scores = result.anomaly_score[anomaly_indices] if hasattr(result.anomaly_score, '__getitem__') else result.anomaly_score
                if isinstance(scores, np.ndarray):
                    high_severity = np.sum(scores > np.percentile(scores, 75))
                    medium_severity = np.sum((scores > np.percentile(scores, 25)) & (scores <= np.percentile(scores, 75)))
                    low_severity = np.sum(scores <= np.percentile(scores, 25))
                    
                    interpretation["summary"]["severity_distribution"] = {
                        "high": int(high_severity),
                        "medium": int(medium_severity),
                        "low": int(low_severity)
                    }
        
        # Business impact assessment
        if result.method == AnomalyMethod.STATISTICAL_ZSCORE:
            interpretation["business_impact"] = {
                "type": "Statistical deviation",
                "risk_level": "Medium" if result.confidence > 0.7 else "Low",
                "affected_areas": ["Data quality", "Process monitoring"]
            }
        elif result.method == AnomalyMethod.AUTOENCODER:
            interpretation["business_impact"] = {
                "type": "Complex pattern deviation",
                "risk_level": "High" if result.confidence > 0.8 else "Medium",
                "affected_areas": ["Business processes", "Customer behavior", "Operational efficiency"]
            }
        
        # Next steps recommendations
        interpretation["next_steps"] = [
            "Investigate root cause of detected anomalies",
            "Validate results with domain experts",
            "Monitor for similar patterns in future data",
            "Consider adjusting detection thresholds based on business impact"
        ]
        
        return interpretation
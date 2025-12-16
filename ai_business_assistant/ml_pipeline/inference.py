"""Inference Engine.

Provides batch and real-time inference capabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


@dataclass
class InferenceResult:
    """Result of inference operation."""
    predictions: np.ndarray
    prediction_time: float
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]


class InferenceEngine:
    """Inference engine for batch and real-time predictions.
    
    Assumptions:
    - Models are pre-trained and loaded
    - Input data matches training schema
    - Preprocessing pipeline is available
    
    Limitations:
    - Real-time inference latency depends on model complexity
    - Batch inference memory-constrained by batch size
    - No automatic model reloading on updates
    """
    
    def __init__(
        self,
        model: Any,
        preprocessing_pipeline: Optional[Any] = None,
        model_version: str = "1.0.0",
        batch_size: int = 1000,
        n_workers: int = 4,
    ):
        """Initialize inference engine.
        
        Args:
            model: Trained model
            preprocessing_pipeline: Optional preprocessing pipeline
            model_version: Version identifier for the model
            batch_size: Batch size for batch inference
            n_workers: Number of parallel workers
        """
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model_version = model_version
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        self._inference_count = 0
        self._total_inference_time = 0.0
    
    def predict_realtime(
        self,
        X: np.ndarray,
        return_proba: bool = False,
    ) -> InferenceResult:
        """Perform real-time prediction on single instance or small batch.
        
        Args:
            X: Input features
            return_proba: Whether to return probabilities (classification)
            
        Returns:
            InferenceResult object
        """
        start_time = time.time()
        
        if self.preprocessing_pipeline:
            X_processed = self.preprocessing_pipeline.transform(X)
        else:
            X_processed = X
        
        if return_proba and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X_processed)
        else:
            predictions = self.model.predict(X_processed)
        
        inference_time = time.time() - start_time
        
        self._inference_count += len(X) if len(X.shape) > 1 else 1
        self._total_inference_time += inference_time
        
        return InferenceResult(
            predictions=predictions,
            prediction_time=inference_time,
            model_version=self.model_version,
            timestamp=datetime.now(),
            metadata={
                "n_samples": len(X) if len(X.shape) > 1 else 1,
                "latency_per_sample": inference_time / (len(X) if len(X.shape) > 1 else 1),
            },
        )
    
    def predict_batch(
        self,
        X: np.ndarray,
        return_proba: bool = False,
        parallel: bool = False,
    ) -> InferenceResult:
        """Perform batch prediction on large dataset.
        
        Args:
            X: Input features
            return_proba: Whether to return probabilities
            parallel: Whether to use parallel processing
            
        Returns:
            InferenceResult object
        """
        start_time = time.time()
        
        if self.preprocessing_pipeline:
            X_processed = self.preprocessing_pipeline.transform(X)
        else:
            X_processed = X
        
        if parallel and len(X) > self.batch_size:
            predictions = self._predict_parallel(X_processed, return_proba)
        else:
            predictions = self._predict_batched(X_processed, return_proba)
        
        inference_time = time.time() - start_time
        
        self._inference_count += len(X)
        self._total_inference_time += inference_time
        
        return InferenceResult(
            predictions=predictions,
            prediction_time=inference_time,
            model_version=self.model_version,
            timestamp=datetime.now(),
            metadata={
                "n_samples": len(X),
                "throughput": len(X) / inference_time,
                "avg_latency": inference_time / len(X),
            },
        )
    
    def predict_streaming(
        self,
        X_stream: List[np.ndarray],
        return_proba: bool = False,
    ) -> List[InferenceResult]:
        """Perform streaming predictions on incoming data.
        
        Args:
            X_stream: List of feature arrays
            return_proba: Whether to return probabilities
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for X_batch in X_stream:
            result = self.predict_realtime(X_batch, return_proba)
            results.append(result)
        
        return results
    
    def _predict_batched(
        self,
        X: np.ndarray,
        return_proba: bool,
    ) -> np.ndarray:
        """Predict in batches to manage memory."""
        n_samples = len(X)
        predictions_list = []
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            
            if return_proba and hasattr(self.model, "predict_proba"):
                batch_preds = self.model.predict_proba(X_batch)
            else:
                batch_preds = self.model.predict(X_batch)
            
            predictions_list.append(batch_preds)
        
        return np.concatenate(predictions_list, axis=0)
    
    def _predict_parallel(
        self,
        X: np.ndarray,
        return_proba: bool,
    ) -> np.ndarray:
        """Predict using parallel workers."""
        n_samples = len(X)
        chunk_size = max(self.batch_size, n_samples // self.n_workers)
        
        chunks = [
            X[i:i+chunk_size]
            for i in range(0, n_samples, chunk_size)
        ]
        
        predictions_list = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_idx = {
                executor.submit(self._predict_chunk, chunk, return_proba): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                predictions_list[idx] = future.result()
        
        return np.concatenate(predictions_list, axis=0)
    
    def _predict_chunk(
        self,
        X_chunk: np.ndarray,
        return_proba: bool,
    ) -> np.ndarray:
        """Predict on a single chunk."""
        if return_proba and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_chunk)
        else:
            return self.model.predict(X_chunk)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self._inference_count == 0:
            return {
                "total_inferences": 0,
                "total_time": 0.0,
                "avg_latency": 0.0,
                "throughput": 0.0,
            }
        
        return {
            "total_inferences": self._inference_count,
            "total_time": self._total_inference_time,
            "avg_latency": self._total_inference_time / self._inference_count,
            "throughput": self._inference_count / self._total_inference_time,
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._inference_count = 0
        self._total_inference_time = 0.0

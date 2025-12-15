"""Model Versioning and Rollback.

Provides model versioning, storage, and rollback mechanisms.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


@dataclass
class ModelMetadata:
    """Metadata for a versioned model."""
    version: str
    created_at: datetime
    model_type: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class ModelVersion:
    """Container for a versioned model."""
    model: Any
    metadata: ModelMetadata
    preprocessing_pipeline: Optional[Any] = None


class ModelVersionManager:
    """Manages model versions with storage and rollback capabilities.
    
    Assumptions:
    - Models are serializable with joblib or pickle
    - Storage directory is writable
    - Version strings follow semantic versioning
    
    Limitations:
    - Large models require significant storage
    - Serialization may not work for all model types
    - No automatic cleanup of old versions
    """
    
    def __init__(self, storage_path: Path = Path("models")):
        """Initialize version manager.
        
        Args:
            storage_path: Path to model storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._registry_path = self.storage_path / "registry.json"
        self._registry = self._load_registry()
    
    def save_model(
        self,
        model: Any,
        version: str,
        metadata: ModelMetadata,
        preprocessing_pipeline: Optional[Any] = None,
        set_as_production: bool = False,
    ) -> Path:
        """Save a model version.
        
        Args:
            model: Model to save
            version: Version identifier
            metadata: Model metadata
            preprocessing_pipeline: Optional preprocessing pipeline
            set_as_production: Whether to set as production model
            
        Returns:
            Path to saved model
        """
        version_dir = self.storage_path / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        if preprocessing_pipeline:
            pipeline_path = version_dir / "preprocessing.joblib"
            joblib.dump(preprocessing_pipeline, pipeline_path)
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self._registry[version] = {
            "version": version,
            "created_at": metadata.created_at.isoformat(),
            "model_type": metadata.model_type,
            "path": str(version_dir),
            "is_production": set_as_production,
        }
        
        if set_as_production:
            for v in self._registry:
                if v != version:
                    self._registry[v]["is_production"] = False
        
        self._save_registry()
        
        return model_path
    
    def load_model(
        self,
        version: Optional[str] = None,
        use_production: bool = False,
    ) -> ModelVersion:
        """Load a model version.
        
        Args:
            version: Version to load (if None and use_production=False, loads latest)
            use_production: Whether to load production model
            
        Returns:
            ModelVersion object
        """
        if use_production:
            version = self._get_production_version()
            if version is None:
                raise ValueError("No production model set")
        elif version is None:
            version = self._get_latest_version()
            if version is None:
                raise ValueError("No models available")
        
        version_dir = self.storage_path / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")
        
        model_path = version_dir / "model.joblib"
        model = joblib.load(model_path)
        
        preprocessing_pipeline = None
        pipeline_path = version_dir / "preprocessing.joblib"
        if pipeline_path.exists():
            preprocessing_pipeline = joblib.load(pipeline_path)
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        return ModelVersion(
            model=model,
            metadata=metadata,
            preprocessing_pipeline=preprocessing_pipeline,
        )
    
    def set_production(self, version: str) -> None:
        """Set a version as production.
        
        Args:
            version: Version to set as production
        """
        if version not in self._registry:
            raise ValueError(f"Version {version} not found")
        
        for v in self._registry:
            self._registry[v]["is_production"] = (v == version)
        
        self._save_registry()
    
    def rollback(self, version: str) -> ModelVersion:
        """Rollback to a previous version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            ModelVersion object
        """
        model_version = self.load_model(version)
        self.set_production(version)
        return model_version
    
    def list_versions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all available versions.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of version information dictionaries
        """
        versions = sorted(
            self._registry.values(),
            key=lambda x: x["created_at"],
            reverse=True,
        )
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def delete_version(self, version: str, force: bool = False) -> None:
        """Delete a model version.
        
        Args:
            version: Version to delete
            force: Force deletion even if it's production
        """
        if version not in self._registry:
            raise ValueError(f"Version {version} not found")
        
        if self._registry[version]["is_production"] and not force:
            raise ValueError("Cannot delete production model without force=True")
        
        version_dir = Path(self._registry[version]["path"])
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        del self._registry[version]
        self._save_registry()
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        mv1 = self.load_model(version1)
        mv2 = self.load_model(version2)
        
        return {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {
                "version1": mv1.metadata.metrics,
                "version2": mv2.metadata.metrics,
                "improvements": {
                    metric: mv2.metadata.metrics.get(metric, 0) - mv1.metadata.metrics.get(metric, 0)
                    for metric in set(list(mv1.metadata.metrics.keys()) + list(mv2.metadata.metrics.keys()))
                },
            },
            "model_types": {
                "version1": mv1.metadata.model_type,
                "version2": mv2.metadata.model_type,
            },
        }
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk."""
        if self._registry_path.exists():
            with open(self._registry_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
    
    def _get_production_version(self) -> Optional[str]:
        """Get current production version."""
        for version, info in self._registry.items():
            if info.get("is_production", False):
                return version
        return None
    
    def _get_latest_version(self) -> Optional[str]:
        """Get latest version."""
        if not self._registry:
            return None
        
        latest = max(
            self._registry.items(),
            key=lambda x: x[1]["created_at"],
        )
        return latest[0]

"""
Model Registry for tracking ML model versions and artifacts.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from ai_business_assistant.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class ModelMetadata:
    id: str
    version: str
    creation_date: str
    performance_metrics: Dict[str, float]
    features_used: List[str]
    source_commit: Optional[str] = None
    is_active: bool = False

class ModelRegistry:
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or settings.abs_path(settings.MODEL_DIR) / "registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, List[ModelMetadata]] = self._load_registry()

    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                registry = {}
                for model_id, versions in data.items():
                    registry[model_id] = [ModelMetadata(**v) for v in versions]
                return registry
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {}

    def _save_registry(self):
        try:
            data = {model_id: [asdict(v) for v in versions] for model_id, versions in self.models.items()}
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def register_model(self, metadata: ModelMetadata):
        if metadata.id not in self.models:
            self.models[metadata.id] = []
        
        # Deactivate other versions if this one is active
        if metadata.is_active:
            for v in self.models[metadata.id]:
                v.is_active = False
        
        self.models[metadata.id].append(metadata)
        self._save_registry()

    def get_active_version(self, model_id: str) -> Optional[ModelMetadata]:
        if model_id not in self.models:
            return None
        for v in self.models[model_id]:
            if v.is_active:
                return v
        return self.models[model_id][-1] if self.models[model_id] else None

    def list_models(self) -> Dict[str, List[ModelMetadata]]:
        return self.models

    def deploy_version(self, model_id: str, version: str):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        found = False
        for v in self.models[model_id]:
            if v.version == version:
                v.is_active = True
                found = True
            else:
                v.is_active = False
        
        if not found:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        self._save_registry()

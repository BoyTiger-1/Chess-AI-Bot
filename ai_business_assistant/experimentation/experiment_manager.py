"""
Manage active experiments and configurations.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from ai_business_assistant.config import get_settings

settings = get_settings()

@dataclass
class ExperimentConfig:
    id: str
    name: str
    description: str
    status: str  # active, paused, ended
    traffic_split: float
    start_date: str
    end_date: Optional[str] = None
    treatments: List[str] = None

class ExperimentManager:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.abs_path(settings.DATA_DIR) / "experiments.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentConfig] = self._load_experiments()

    def _load_experiments(self) -> Dict[str, ExperimentConfig]:
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {k: ExperimentConfig(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_experiments(self):
        with open(self.storage_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.experiments.items()}, f, indent=2)

    def create_experiment(self, config: ExperimentConfig):
        self.experiments[config.id] = config
        self._save_experiments()

    def get_experiment(self, exp_id: str) -> Optional[ExperimentConfig]:
        return self.experiments.get(exp_id)

    def list_experiments(self) -> List[ExperimentConfig]:
        return list(self.experiments.values())

    def end_experiment(self, exp_id: str):
        if exp_id in self.experiments:
            self.experiments[exp_id].status = "ended"
            self.experiments[exp_id].end_date = datetime.now().isoformat()
            self._save_experiments()

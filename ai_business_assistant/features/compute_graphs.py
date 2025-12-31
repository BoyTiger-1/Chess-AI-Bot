"""
DAG-based feature computation graphs.
"""

from typing import Dict, List, Callable, Any
import logging

logger = logging.getLogger(__name__)

class FeatureComputeGraph:
    def __init__(self):
        self.nodes = {}
        self.dependencies = {}

    def add_feature(self, name: str, compute_func: Callable, deps: List[str] = None):
        self.nodes[name] = compute_func
        self.dependencies[name] = deps or []

    def compute(self, target_feature: str, inputs: Dict[str, Any]) -> Any:
        if target_feature in inputs:
            return inputs[target_feature]

        if target_feature not in self.nodes:
            raise ValueError(f"Feature {target_feature} not found in compute graph")

        # Resolve dependencies
        dep_values = {}
        for dep in self.dependencies[target_feature]:
            dep_values[dep] = self.compute(dep, inputs)

        logger.info(f"Computing feature: {target_feature}")
        return self.nodes[target_feature](**dep_values)

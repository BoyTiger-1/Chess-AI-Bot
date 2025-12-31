"""
Registry of all available features in the Business AI Assistant.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class FeatureDefinition:
    name: str
    description: str
    data_type: str
    source: str
    version: str

class FeatureRegistry:
    _features: Dict[str, FeatureDefinition] = {}

    @classmethod
    def register(cls, feature: FeatureDefinition):
        cls._features[feature.name] = feature

    @classmethod
    def get_feature(cls, name: str) -> Optional[FeatureDefinition]:
        return cls._features.get(name)

    @classmethod
    def list_features(cls) -> List[FeatureDefinition]:
        return list(cls._features.values())

# Register some initial features
FeatureRegistry.register(FeatureDefinition(
    name="market_volatility_30d",
    description="30-day rolling market volatility",
    data_type="float",
    source="market_data",
    version="1.0.0"
))

FeatureRegistry.register(FeatureDefinition(
    name="customer_churn_risk",
    description="Predicted churn risk score",
    data_type="float",
    source="customer_analysis",
    version="1.1.0"
))

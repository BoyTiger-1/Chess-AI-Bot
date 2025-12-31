"""
Data quality rules and metrics.
"""

import pandas as pd
from typing import Dict, Any

def get_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate completeness, uniqueness, and validity metrics."""
    metrics = {
        "completeness": (1 - df.isnull().mean()).to_dict(),
        "uniqueness": {col: df[col].nunique() / len(df) for col in df.columns},
        "total_rows": len(df),
        "duplicate_rows": df.duplicated().sum()
    }
    return metrics

def check_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, float]:
    """Identify columns with high missing value ratios."""
    missing_ratios = df.isnull().mean()
    return missing_ratios[missing_ratios > threshold].to_dict()

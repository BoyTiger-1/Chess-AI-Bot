"""
Detect distribution changes and data drift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

def detect_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, columns: list) -> Dict[str, Any]:
    """
    Detect statistical drift using Kolmogorov-Smirnov test.
    """
    drift_report = {}
    for col in columns:
        if col in baseline_df.columns and col in current_df.columns:
            ks_stat, p_value = stats.ks_2samp(baseline_df[col].dropna(), current_df[col].dropna())
            drift_report[col] = {
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": p_value < 0.05
            }
    return drift_report

"""
Identify suspicious data points and outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """Identify outliers using Z-score."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df.index[z_scores > threshold]

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Identify outliers using Interquartile Range."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.index[(df[column] < lower_bound) | (df[column] > upper_bound)]

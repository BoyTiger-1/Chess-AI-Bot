"""
Feature transformation pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureTransformer:
    def __init__(self, method: str = "standard"):
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        return df_copy

    def transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[columns] = self.scaler.transform(df_copy[columns])
        return df_copy

def rolling_window_mean(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return df[column].rolling(window=window).mean()

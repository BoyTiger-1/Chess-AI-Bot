"""
Schema and data validation for Business AI Assistant.
"""

import pandas as pd
from typing import Dict, List, Any, Optional

def validate_schema(df: pd.DataFrame, expected_columns: Dict[str, Any]) -> List[str]:
    """Validate DataFrame schema (column names and types)."""
    errors = []
    for col, dtype in expected_columns.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        elif not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
            # Relaxed type checking
            try:
                df[col].astype(dtype)
            except Exception:
                errors.append(f"Invalid type for column {col}: expected {dtype}, got {df[col].dtype}")
    return errors

def validate_ranges(df: pd.DataFrame, ranges: Dict[str, tuple]) -> List[str]:
    """Validate data ranges for numerical columns."""
    errors = []
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            if df[col].min() < min_val:
                errors.append(f"Column {col} has values below {min_val}")
            if df[col].max() > max_val:
                errors.append(f"Column {col} has values above {max_val}")
    return errors

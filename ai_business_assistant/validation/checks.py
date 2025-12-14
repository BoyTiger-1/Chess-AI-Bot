from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    check: str
    message: str
    severity: str = "error"  # error|warn


def run_quality_checks(df: pd.DataFrame) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    if df.empty:
        issues.append(QualityIssue(check="non_empty", message="DataFrame is empty"))

    dupes = df.duplicated().sum() if not df.empty else 0
    if dupes:
        issues.append(QualityIssue(check="duplicates", message=f"Found {dupes} duplicate rows"))

    missing = df.isna().sum().to_dict() if not df.empty else {}
    missing_cols = {k: int(v) for k, v in missing.items() if v}
    if missing_cols:
        issues.append(QualityIssue(check="missing_values", message=str(missing_cols), severity="warn"))

    return issues

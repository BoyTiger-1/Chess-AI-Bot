from __future__ import annotations

import pandas as pd


def normalize_customer_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "customer_id": "customer_key",
        "id": "customer_key",
        "created": "created_at",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    required = ["customer_key"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out["customer_key"] = out["customer_key"].astype(str)

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")

    if "email" in out.columns:
        out["email"] = out["email"].astype(str).str.lower().str.strip()

    return out

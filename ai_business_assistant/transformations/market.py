from __future__ import annotations

from datetime import timezone

import pandas as pd

from ai_business_assistant.validation.schemas import validate_market_prices


def normalize_market_prices(df: pd.DataFrame, *, default_source: str = "unknown") -> pd.DataFrame:
    """Normalize market price payloads into a canonical schema.

    Expected input formats vary widely; this function standardizes column names,
    types, and fills the required fields.
    """

    out = df.copy()

    rename_map = {
        "timestamp": "ts",
        "time": "ts",
        "datetime": "ts",
        "ticker": "symbol",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    if "source" not in out.columns:
        out["source"] = default_source

    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="raise")
    else:
        raise ValueError("market prices missing timestamp column")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["symbol"] = out["symbol"].astype(str).str.upper()

    out = validate_market_prices(out)

    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)

    out["ts"] = out["ts"].dt.tz_convert(timezone.utc)

    return out

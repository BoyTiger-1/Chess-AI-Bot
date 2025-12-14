from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


try:  # optional dependency
    import pandera as pa
    from pandera import Column, DataFrameSchema
except Exception:  # noqa: BLE001
    pa = None
    Column = None
    DataFrameSchema = None


@dataclass(frozen=True)
class MarketPriceSchema:
    """Logical schema for market prices."""

    required_columns: tuple[str, ...] = (
        "symbol",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    )


def _pandera_schema() -> "DataFrameSchema":
    assert pa is not None
    ts_dtype = getattr(pa, "DateTimeTZ", getattr(pa, "DateTime"))
    return DataFrameSchema(
        {
            "symbol": Column(str, nullable=False),
            "ts": Column(ts_dtype, nullable=False),
            "open": Column(float, nullable=True),
            "high": Column(float, nullable=True),
            "low": Column(float, nullable=True),
            "close": Column(float, nullable=False),
            "volume": Column(float, nullable=True),
            "source": Column(str, nullable=False),
        },
        coerce=True,
        strict=False,
    )


def validate_market_prices(df: pd.DataFrame, schema: MarketPriceSchema | None = None) -> pd.DataFrame:
    schema = schema or MarketPriceSchema()
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if pa is not None:
        return _pandera_schema().validate(df, lazy=True)

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="raise")
    out["symbol"] = out["symbol"].astype(str)
    out["source"] = out["source"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="raise")
    for col in ["open", "high", "low", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["symbol"].isna().any() or (out["symbol"].str.len() == 0).any():
        raise ValueError("Invalid symbol")

    if out["source"].isna().any() or (out["source"].str.len() == 0).any():
        raise ValueError("Invalid source")

    if out["close"].isna().any():
        raise ValueError("Close contains nulls")

    return out

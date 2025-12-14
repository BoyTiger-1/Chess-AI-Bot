from __future__ import annotations

import pandas as pd


def build_market_features(
    prices: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 20),
) -> pd.DataFrame:
    """Feature engineering for downstream ML.

    Expects normalized market prices with columns: symbol, ts, close.
    """

    df = prices.sort_values(["symbol", "ts"]).copy()
    df["return_1"] = df.groupby("symbol")["close"].pct_change()

    for w in windows:
        df[f"sma_{w}"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(w).mean())
        df[f"vol_{w}"] = df.groupby("symbol")["return_1"].transform(lambda s: s.rolling(w).std())

    return df

from pathlib import Path

import pandas as pd

from ai_business_assistant.warehouse.duckdb_warehouse import DuckDbWarehouse
from ai_business_assistant.transformations.market import normalize_market_prices


def test_duckdb_warehouse_upsert_market_prices(tmp_path: Path):
    wh = DuckDbWarehouse(tmp_path / "warehouse.duckdb")
    wh.init_schema()

    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "ts": "2025-01-01T00:00:00Z",
                "close": 100,
                "open": 99,
                "high": 101,
                "low": 98,
                "volume": 1000,
                "source": "unit",
            }
        ]
    )
    prices = normalize_market_prices(df)

    wh.upsert_market_prices(prices)

    with wh.connect() as con:
        got = con.execute("select count(*) from fact_market_prices").fetchone()[0]
        assert got == 1

    # Update close and ensure idempotent upsert
    df.loc[0, "close"] = 101
    prices2 = normalize_market_prices(df)
    wh.upsert_market_prices(prices2)
    with wh.connect() as con:
        close = con.execute(
            """
            select close from fact_market_prices
            """
        ).fetchone()[0]
        assert close == 101

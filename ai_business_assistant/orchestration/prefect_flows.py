from __future__ import annotations

import pandas as pd

from ai_business_assistant.config import Settings
from ai_business_assistant.connectors.api import HttpApiConnector
from ai_business_assistant.connectors.realtime import PollingIngestor
from ai_business_assistant.features.market_features import build_market_features
from ai_business_assistant.transformations.customer import normalize_customer_records
from ai_business_assistant.transformations.market import normalize_market_prices
from ai_business_assistant.warehouse.duckdb_warehouse import DuckDbWarehouse


def _optional_prefect():
    try:
        from prefect import flow

        return flow
    except Exception:  # noqa: BLE001
        return None


def build_flows(settings: Settings) -> dict[str, callable]:
    """Build callables for flows.

    If Prefect is installed, flows/tasks are Prefect-native.
    Otherwise, returns plain Python callables.
    """

    flow = _optional_prefect()

    warehouse = DuckDbWarehouse(settings.abs_warehouse_path)

    def init_warehouse() -> None:
        warehouse.init_schema()

    def historical_market_data() -> None:
        # Example loader using a free, no-auth JSON endpoint. This is intentionally
        # decoupled; for production, configure a proper market data provider.
        api = HttpApiConnector(base_url="https://example.com")
        _ = api  # placeholder to show how an API connector is wired

        df = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "ts": "2025-01-01T00:00:00Z",
                    "open": 100,
                    "high": 110,
                    "low": 95,
                    "close": 105,
                    "volume": 1000,
                    "source": "demo",
                }
            ]
        )
        prices = normalize_market_prices(df, default_source="demo")
        warehouse.upsert_market_prices(prices)
        _ = build_market_features(prices)

    def customer_load() -> None:
        df = pd.DataFrame(
            [
                {"customer_key": "cust_1", "email": "A@EXAMPLE.COM", "created_at": "2024-01-01"},
                {"customer_key": "cust_2", "email": "b@example.com", "created_at": "2024-02-01"},
            ]
        )
        cust = normalize_customer_records(df)
        warehouse.load_customers(cust)

    def competitive_intel() -> None:
        # Placeholder example: in practice connect to RSS/news APIs.
        df = pd.DataFrame(
            [
                {
                    "ts": "2025-01-01T00:00:00Z",
                    "source": "demo_news",
                    "topic": "earnings",
                    "headline": "Competitor reports earnings",
                    "url": "https://example.com/article",
                }
            ]
        )
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        with warehouse.connect() as con:
            warehouse.init_schema()
            warehouse._ensure_dim_date(con, df["ts"])
            warehouse._ensure_dim_values(con, "dim_source", "source_key", "source", df["source"])

            load = df.copy()
            load["date_key"] = load["ts"].dt.year * 10000 + load["ts"].dt.month * 100 + load["ts"].dt.day

            con.register("_intel", load)
            con.execute(
                """
                insert into fact_competitive_intel
                select
                    s.source_key,
                    i.date_key,
                    i.ts,
                    i.topic,
                    i.headline,
                    i.url,
                    now() as ingested_at
                from _intel i
                join dim_source s on s.source = i.source
                """
            )
            con.unregister("_intel")

    def realtime_market_polling() -> None:
        chk = settings.abs_path(settings.checkpoint_dir) / "market_polling.json"
        chk.parent.mkdir(parents=True, exist_ok=True)

        def ingest_fn(state: dict) -> pd.DataFrame:
            # In real deployments, use state['last_run'] to fetch incremental data.
            _ = state
            return pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "ts": pd.Timestamp.utcnow().isoformat(),
                        "close": 100.0,
                        "open": 99.0,
                        "high": 101.0,
                        "low": 98.5,
                        "volume": 100.0,
                        "source": "demo_realtime",
                    }
                ]
            )

        def sink_fn(df: pd.DataFrame) -> None:
            prices = normalize_market_prices(df, default_source="demo_realtime")
            warehouse.upsert_market_prices(prices)

        ing = PollingIngestor(checkpoint_path=chk, ingest_fn=ingest_fn, sink_fn=sink_fn)
        ing.run_forever(interval_s=5.0)

    if flow:
        init_warehouse_f = flow(name="init_warehouse")(init_warehouse)
        historical_market_data_f = flow(name="historical_market_data")(historical_market_data)
        realtime_market_polling_f = flow(name="realtime_market_polling")(realtime_market_polling)
        customer_load_f = flow(name="customer_load")(customer_load)
        competitive_intel_f = flow(name="competitive_intel")(competitive_intel)
        return {
            "init_warehouse": init_warehouse_f,
            "historical_market_data": historical_market_data_f,
            "realtime_market_polling": realtime_market_polling_f,
            "customer_load": customer_load_f,
            "competitive_intel": competitive_intel_f,
        }

    return {
        "init_warehouse": init_warehouse,
        "historical_market_data": historical_market_data,
        "realtime_market_polling": realtime_market_polling,
        "customer_load": customer_load,
        "competitive_intel": competitive_intel,
    }

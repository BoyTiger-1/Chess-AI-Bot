from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


def _stable_int63(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


@dataclass
class DuckDbWarehouse:
    path: Path

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path))

    def init_schema(self, con: duckdb.DuckDBPyConnection | None = None) -> None:
        schema_sql = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
        if con is None:
            with self.connect() as c:
                c.execute(schema_sql)
            return
        con.execute(schema_sql)

    def _ensure_dim_date(self, con: duckdb.DuckDBPyConnection, ts: pd.Series) -> None:
        dates = pd.to_datetime(ts, utc=True).dt.date
        dim = pd.DataFrame({"date": dates.unique()})
        if dim.empty:
            return
        dim["year"] = pd.to_datetime(dim["date"]).dt.year
        dim["month"] = pd.to_datetime(dim["date"]).dt.month
        dim["day"] = pd.to_datetime(dim["date"]).dt.day
        dim["date_key"] = dim["year"] * 10000 + dim["month"] * 100 + dim["day"]
        con.register("_dim_date", dim)
        con.execute(
            """
            insert into dim_date
            select d.date_key, d.date, d.year, d.month, d.day
            from _dim_date d
            left join dim_date existing on existing.date_key = d.date_key
            where existing.date_key is null
            """
        )
        con.unregister("_dim_date")

    def _ensure_dim_values(
        self,
        con: duckdb.DuckDBPyConnection,
        table: str,
        key: str,
        col: str,
        values: pd.Series,
    ) -> None:
        uniq = pd.DataFrame({col: values.dropna().astype(str).unique()})
        if uniq.empty:
            return
        uniq[key] = uniq[col].map(_stable_int63)
        con.register("_dim", uniq)
        con.execute(
            f"""
            insert into {table}
            select d.{key}, d.{col}
            from _dim d
            left join {table} existing on existing.{col} = d.{col}
            where existing.{col} is null
            """
        )
        con.unregister("_dim")

    def _upsert_market_prices_with_con(self, con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
        required = {"symbol", "source", "ts", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Market prices missing required columns: {sorted(missing)}")

        self._ensure_dim_date(con, df["ts"])
        self._ensure_dim_values(con, "dim_symbol", "symbol_key", "symbol", df["symbol"])
        self._ensure_dim_values(con, "dim_source", "source_key", "source", df["source"])

        load = df.copy()
        load["symbol"] = load["symbol"].astype(str)
        load["source"] = load["source"].astype(str)
        load["ts"] = pd.to_datetime(load["ts"], utc=True)
        load["date_key"] = load["ts"].dt.year * 10000 + load["ts"].dt.month * 100 + load["ts"].dt.day

        con.register("_prices", load)
        con.execute(
            """
            insert into fact_market_prices
            select
                s.symbol_key,
                src.source_key,
                p.date_key,
                p.ts,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                ? as ingested_at
            from _prices p
            join dim_symbol s on s.symbol = p.symbol
            join dim_source src on src.source = p.source
            on conflict(symbol_key, source_key, ts) do update set
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                ingested_at = excluded.ingested_at
            """,
            [datetime.utcnow()],
        )
        con.unregister("_prices")

    def upsert_market_prices(self, df: pd.DataFrame) -> None:
        with self.connect() as con:
            self.init_schema(con)
            self._upsert_market_prices_with_con(con, df)

    def upsert_market_prices_iter(self, chunks: Iterable[pd.DataFrame]) -> int:
        total = 0
        with self.connect() as con:
            self.init_schema(con)
            for df in chunks:
                if df is None or df.empty:  # type: ignore[truthy-bool]
                    continue
                self._upsert_market_prices_with_con(con, df)
                total += int(df.shape[0])
        return total

    def load_customers(self, df: pd.DataFrame) -> None:
        if "customer_key" not in df.columns:
            raise ValueError("customer_key missing")
        with self.connect() as con:
            self.init_schema(con)
            con.register("_cust", df)
            con.execute(
                """
                insert into dim_customer
                select customer_key, email, created_at
                from _cust
                on conflict(customer_key) do update set
                    email = excluded.email,
                    created_at = excluded.created_at
                """
            )
            con.unregister("_cust")

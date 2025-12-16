from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class SqlAlchemyDatabase:
    """Database connector for extraction/loading via SQLAlchemy."""

    url: str

    def engine(self) -> Engine:
        return create_engine(self.url, future=True)

    def read_sql(self, query: str, *, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.engine().connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def write_df(self, df: pd.DataFrame, table: str, *, if_exists: str = "append") -> None:
        with self.engine().begin() as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=False)

    def execute(self, sql: str, *, params: dict[str, Any] | None = None) -> None:
        with self.engine().begin() as conn:
            conn.execute(text(sql), params or {})

    def executemany(self, sql: str, rows: Iterable[dict[str, Any]]) -> None:
        with self.engine().begin() as conn:
            conn.execute(text(sql), list(rows))

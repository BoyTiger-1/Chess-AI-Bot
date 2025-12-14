from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_business_assistant.connectors.files import CsvConnector
from ai_business_assistant.monitoring.alerts import AlertSink, StdoutAlertSink
from ai_business_assistant.transformations.market import normalize_market_prices
from ai_business_assistant.validation.checks import run_quality_checks
from ai_business_assistant.warehouse.duckdb_warehouse import DuckDbWarehouse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketPricesLoadResult:
    rows_loaded: int
    batches: int


def load_market_prices_from_csv(
    *,
    csv_path: Path,
    warehouse_path: Path,
    source: str,
    chunksize: int = 250_000,
    alert_sink: AlertSink = StdoutAlertSink(),
) -> MarketPricesLoadResult:
    """High-throughput historical loader.

    Uses chunked CSV reads to handle millions of rows with bounded memory usage.
    """

    connector = CsvConnector(csv_path)
    warehouse = DuckDbWarehouse(warehouse_path)

    total = 0
    batches = 0

    with warehouse.connect() as con:
        warehouse.init_schema(con)
        for chunk in connector.iter_chunks(chunksize=chunksize):
            try:
                batches += 1
                norm = normalize_market_prices(chunk, default_source=source)
                issues = run_quality_checks(norm)
                errors = [i for i in issues if i.severity == "error"]
                if errors:
                    raise ValueError(f"Quality checks failed: {errors}")
                warehouse._upsert_market_prices_with_con(con, norm)
                total += int(norm.shape[0])
            except Exception as e:  # noqa: BLE001
                alert_sink.send(
                    title="Market price batch failed",
                    message=str(e),
                    context={"csv_path": str(csv_path), "batch": batches},
                )
                raise

    logger.info("Loaded %s rows across %s batches", total, batches)
    return MarketPricesLoadResult(rows_loaded=total, batches=batches)

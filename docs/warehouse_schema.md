# Data warehouse schema

The warehouse is stored in DuckDB (`data/warehouse.duckdb`) and follows a simple star schema.

## Dimensions

- `dim_date(date_key, date, year, month, day)`
- `dim_symbol(symbol_key, symbol)`
- `dim_source(source_key, source)`
- `dim_customer(customer_key, email, created_at)`

## Facts

- `fact_market_prices(symbol_key, source_key, date_key, ts, open, high, low, close, volume, ingested_at)`
  - PK: `(symbol_key, source_key, ts)` (idempotent upserts)

- `fact_competitive_intel(source_key, date_key, ts, topic, headline, url, ingested_at)`

See `ai_business_assistant/warehouse/schema.sql`.

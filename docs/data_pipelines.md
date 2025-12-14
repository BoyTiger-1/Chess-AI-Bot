# AI Business Assistant – Data Pipelines & ETL

This repository contains a lightweight ETL framework under `ai_business_assistant/`.

## Goals

- Ingest business data from **APIs**, **databases**, and **CSV/JSON uploads**
- Perform **validation & quality checks** at each stage
- Apply **transformations/normalization** and **feature engineering**
- Load into a **DuckDB** warehouse with a **dimensional (star) schema**
- Support **historical backfills** and **real-time (polling) ingestion**
- Provide **monitoring/alerting** hooks, retries, and checkpoint-based recovery

## Architecture

### Layers

1. **Connectors** (`ai_business_assistant/connectors/`)
   - `HttpApiConnector` (requests + retries)
   - `SqlAlchemyDatabase` (DB extract/load)
   - `CsvConnector` / `JsonLinesConnector`
   - `PollingIngestor` for near-real-time polling feeds with checkpoints

2. **Validation** (`ai_business_assistant/validation/`)
   - Schema validation (Pandera when available; otherwise strong dtype checks)
   - Generic quality checks (duplicates, missing values, non-empty)

3. **Transformations** (`ai_business_assistant/transformations/`)
   - Normalizes market price payloads into a canonical schema

4. **Feature Engineering** (`ai_business_assistant/features/`)
   - Example features: returns, moving averages, rolling volatility

5. **Warehouse** (`ai_business_assistant/warehouse/`)
   - DuckDB-backed star schema with fact/dimension tables

## Running locally

Install core deps:

```bash
pip install -r requirements.txt
```

Optional (Prefect orchestration + Pandera dataframe schemas):

```bash
pip install -r requirements-etl.txt
```

Initialize the warehouse:

```bash
python -m ai_business_assistant.cli init-warehouse
```

Run an example flow:

```bash
python -m ai_business_assistant.cli run-flow historical_market_data
```

## Historical loading & archiving

Raw/staging outputs should be written into date-partitioned folders under `data/raw/...`.
The file-based retention policy can archive and remove old partitions.
See `ai_business_assistant/governance/retention.py`.

## Governance & retention

- Store raw copies for traceability
- Validate at each stage (raw -> staging -> curated)
- Archive raw partitions beyond the retention window (gzip)
- Purge when required (PII/contractual constraints)

## Monitoring & alerting

- All connectors include retry/error handling.
- Real-time ingestion uses checkpointing for recovery.
- Alert sinks support stdout logging and a generic webhook.

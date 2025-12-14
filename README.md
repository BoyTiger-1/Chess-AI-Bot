# Chess-AI-Bot

This repo started as a simple Flask chess bot.

It now also includes a lightweight **AI Business Assistant – Data Pipelines & ETL** framework under `ai_business_assistant/`.

- Orchestration: Prefect (optional import) with plain-Python fallback
- Connectors: APIs, databases (SQLAlchemy), CSV/JSON uploads
- Real-time ingestion: polling ingestor with checkpointing and recovery
- Validation: schema + quality checks
- Transformations/normalization and feature engineering
- Warehouse: DuckDB star schema (fact/dimension tables)

Docs:
- `docs/data_pipelines.md`
- `docs/warehouse_schema.md`

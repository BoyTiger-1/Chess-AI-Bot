# Business AI Assistant

A lightweight **AI Business Assistant – Data Pipelines & ETL** framework.

- Orchestration: Prefect (optional import) with plain-Python fallback
- Connectors: APIs, databases (SQLAlchemy), CSV/JSON uploads
- Real-time ingestion: polling ingestor with checkpointing and recovery
- Validation: schema + quality checks
- Transformations/normalization and feature engineering
- Warehouse: DuckDB star schema (fact/dimension tables)

Docs:
- `docs/data_pipelines.md`
- `docs/warehouse_schema.md`

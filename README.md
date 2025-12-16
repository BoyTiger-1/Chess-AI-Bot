# Project

This repository currently contains:

- `app.py` — an existing Flask-based chess demo application
- `ai_business_assistant/` — **AI Business Assistant foundation** (microservices skeleton)

## AI Business Assistant (Foundation)

Start here:

- System design (microservices + data flow diagrams): `docs/system-design.md`
- Local development environment guide: `docs/dev-setup.md`

Quick start:

```bash
docker compose up --build
```
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

## Frontend (UI/UX)

A premium React + TypeScript dashboard application is available under `frontend/`.

### Run locally

```bash
cd frontend
npm install
npm run dev
```

### Key features (scaffold)

- Responsive navigation + layout (mobile/tablet/desktop)
- Customizable dashboards with drag-and-drop widgets (`react-grid-layout`)
- Data visualization via Plotly (trends, heatmaps, confidence intervals)
- WebSocket client scaffold for live updates (`VITE_WS_URL`)
- Demo authentication + UI role-based access control
- Settings (dark mode, language, notification toggles)
- PDF export of dashboards
- UI tests (React Testing Library + Vitest) and Storybook component docs

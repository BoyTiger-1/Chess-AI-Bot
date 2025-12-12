# Development Environment Setup (Local)

This repository contains an **AI Business Assistant foundation** (microservices skeleton) alongside an existing Flask demo app.

## Prerequisites

- Docker + Docker Compose
- (Optional) Python 3.12 if you want to run services without Docker

## Quick start (Docker Compose)

From the repository root:

```bash
docker compose up --build
```

Services:

- API Gateway: http://localhost:8000
- Auth Service (internal): http://localhost:8001
- Assistant Service (internal): http://localhost:8002
- RabbitMQ UI: http://localhost:15672 (guest/guest)
- Flower (Celery UI): http://localhost:5555
- Qdrant: http://localhost:6333
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## API usage walkthrough

### 1) Register + login

```bash
curl -sS -X POST http://localhost:8000/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"demo@example.com","password":"changeme123"}'

TOKEN=$(curl -sS -X POST http://localhost:8000/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"demo@example.com","password":"changeme123"}' | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')

echo "$TOKEN"
```

### 2) Create a conversation + send a message

```bash
CONV=$(curl -sS -X POST http://localhost:8000/assistant/conversations \
  -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"title":"My first conversation"}' | python -c 'import sys, json; print(json.load(sys.stdin)["id"])')

curl -sS -X POST http://localhost:8000/assistant/conversations/$CONV/messages \
  -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"role":"user","content":"Summarize Q4 revenue drivers"}'
```

A Celery task will be enqueued to generate and store a (placeholder) embedding for the message in Qdrant.

## Running services without Docker (optional)

Install assistant stack dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ai_business_assistant/requirements.txt
```

Then run (in separate shells):

```bash
uvicorn ai_business_assistant.auth_service.main:app --port 8001 --reload
uvicorn ai_business_assistant.assistant_service.main:app --port 8002 --reload
uvicorn ai_business_assistant.gateway.main:app --port 8000 --reload
celery -A ai_business_assistant.worker.celery_app worker --loglevel=INFO
```

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

## Repository docs

- `docs/system-design.md` — architecture + diagrams

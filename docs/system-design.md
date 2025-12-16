# AI Business Assistant — System Design (Foundation)

## 1) Goals

- Provide a production-ready foundation for an end-to-end AI business assistant.
- Prefer **loosely-coupled microservices** connected via HTTP and async jobs.
- Support core platform needs: **auth/RBAC, async processing, caching, embeddings storage, observability, and secure config**.

## 2) High-level microservices architecture

**Services in this repository (skeleton):**

- **API Gateway** (`ai_business_assistant.gateway`)
  - External entry point
  - Rate limiting and request correlation IDs
  - JWT enforcement before forwarding traffic to internal services
- **Auth Service** (`ai_business_assistant.auth_service`)
  - User registration/login
  - JWT issuance
  - RBAC role management foundation
- **Assistant Service** (`ai_business_assistant.assistant_service`)
  - Conversation/message APIs
  - Persists structured data in Postgres
  - Enqueues background jobs (e.g., embeddings)
- **Worker** (`ai_business_assistant.worker`)
  - Celery workers for long-running/background tasks
  - Example task: `embed_message` storing embeddings in Qdrant

**Data infrastructure:**

- **PostgreSQL** (structured data)
- **Redis** (caching + task results)
- **RabbitMQ** (Celery broker)
- **Vector DB: Qdrant** (embeddings)

### Component diagram

```mermaid
flowchart LR
  user[Client / Web / Mobile]

  subgraph edge[Edge]
    gw[API Gateway]
  end

  subgraph svc[Internal Services]
    auth[Auth Service]
    asst[Assistant Service]
    worker[Celery Worker]
  end

  subgraph data[Data Layer]
    pg[(PostgreSQL)]
    redis[(Redis)]
    rmq[(RabbitMQ)]
    vect[(Qdrant Vector DB)]
  end

  user -->|HTTPS| gw
  gw -->|HTTP| auth
  gw -->|HTTP| asst

  auth --> pg
  asst --> pg
  asst --> redis

  asst -->|publish job| rmq
  worker -->|consume job| rmq
  worker --> redis
  worker --> vect
```

## 3) Data flow diagrams

### Login (JWT issuance)

```mermaid
sequenceDiagram
  participant C as Client
  participant G as Gateway
  participant A as Auth Service
  participant P as Postgres

  C->>G: POST /auth/login
  G->>A: Forward request
  A->>P: Verify user + password hash
  P-->>A: user + roles
  A-->>G: JWT access_token
  G-->>C: JWT access_token
```

### Send message + async embeddings

```mermaid
sequenceDiagram
  participant C as Client
  participant G as Gateway
  participant S as Assistant Service
  participant P as Postgres
  participant Q as RabbitMQ
  participant W as Worker
  participant V as Qdrant

  C->>G: POST /assistant/conversations/:id/messages (JWT)
  G->>G: Verify JWT + rate limit
  G->>S: Forward request
  S->>P: Persist message
  S->>Q: Publish embed_message job
  S-->>G: Message created
  G-->>C: Message created

  W->>Q: Consume embed_message
  W->>V: Upsert embedding vector
  W-->>Q: Ack
```

## 4) Database schema (foundation)

Structured data is in Postgres using async SQLAlchemy models.

### ER diagram

```mermaid
erDiagram
  USERS ||--o{ USER_ROLES : has
  ROLES ||--o{ USER_ROLES : has
  USERS ||--o{ CONVERSATIONS : owns
  CONVERSATIONS ||--o{ MESSAGES : contains

  USERS {
    uuid id PK
    string email
    string password_hash
    datetime created_at
  }

  ROLES {
    uuid id PK
    string name
  }

  USER_ROLES {
    uuid user_id FK
    uuid role_id FK
  }

  CONVERSATIONS {
    uuid id PK
    uuid user_id FK
    string title
    datetime created_at
  }

  MESSAGES {
    uuid id PK
    uuid conversation_id FK
    string role
    text content
    datetime created_at
  }
```

## 5) Async processing (Celery/RabbitMQ)

- **RabbitMQ** is the broker.
- **Redis** stores results (and can also be used for caching).
- Celery tasks live in `ai_business_assistant/worker/tasks.py`.

## 6) Configuration management

- All services use a shared `Settings` object (`ai_business_assistant/shared/config.py`).
- Configuration is via environment variables (optionally loaded from `.env`).
- A sample environment file is provided as `.env.example`.

Recommended next steps for production:

- Use a secret manager (AWS Secrets Manager, Vault, GCP Secret Manager).
- Rotate JWT secrets and database credentials.

## 7) Logging, monitoring, observability

Implemented foundation:

- JSON logging via `ai_business_assistant/shared/logging.py`.
- Request correlation IDs via `RequestIdMiddleware` (`X-Request-Id`).
- Prometheus metrics via `/metrics` on each service.
- Docker Compose includes Prometheus + Grafana.

Recommended next steps:

- OpenTelemetry tracing (OTLP exporter) for distributed traces.
- Central log aggregation (Loki/ELK).

## 8) Security foundation

- JWT authentication with access tokens.
- RBAC primitive: `roles` on JWT and DB.
- Rate limiting at the gateway.

Encryption at rest / transit:

- **Transit:** terminate TLS at the edge (gateway or external load balancer).
- **At rest:** use disk encryption (cloud-managed), Postgres encryption features where applicable, and encrypt application secrets (e.g., tool/API keys) before persisting.

## 9) API gateway

In this foundation the **gateway** is a FastAPI service that:

- Enforces rate limits
- Verifies JWT on protected routes
- Proxies traffic to internal services

This can be replaced with dedicated gateways (Kong/Envoy/Traefik) later without changing internal services.

## 10) Local development

See `docs/dev-setup.md` for end-to-end local setup using Docker Compose.

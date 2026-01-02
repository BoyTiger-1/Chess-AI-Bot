# Docker Setup Guide for Business AI Assistant

This guide explains the Docker configuration and how to set up and run the Business AI Assistant using Docker Compose.

## Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Services Overview

The docker-compose.yml file defines the following services:

### Infrastructure Services

1. **PostgreSQL** (`postgres:5432`)
   - Database: `aba`
   - User: `aba`
   - Password: `aba`
   - Image: `pgvector/pgvector:pg16`

2. **Redis** (`redis:6379`)
   - Used for caching and Celery broker
   - Image: `redis:7-alpine`

3. **RabbitMQ** (`rabbitmq:5672`, management: `15672`)
   - Alternative message broker for Celery
   - Management UI: http://localhost:15672
   - Default credentials: `guest`/`guest`

4. **Qdrant** (`qdrant:6333`, gRPC: `6334`)
   - Vector database for RAG features
   - Image: `qdrant/qdrant:latest`

### Application Services

5. **Auth Service** (`auth:8001`)
   - Entry point: `ai_business_assistant.auth_service.main:app`
   - Health: http://localhost:8001/health
   - Metrics: http://localhost:8001/metrics

6. **Assistant Service** (`assistant:8002`)
   - Entry point: `ai_business_assistant.assistant_service.main:app`
   - Health: http://localhost:8002/health
   - Metrics: http://localhost:8002/metrics

7. **Worker** (Celery)
   - Entry point: `ai_business_assistant.worker.celery_app`
   - Processes async tasks

8. **Gateway** (`gateway:8000`)
   - Entry point: `ai_business_assistant.gateway.main:app`
   - Health: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics
   - Routes to auth and assistant services

### Monitoring Services

9. **Flower** (`flower:5555`)
   - Celery task monitoring UI
   - URL: http://localhost:5555

10. **Prometheus** (`prometheus:9090`)
    - Metrics collection and storage
    - URL: http://localhost:9090

11. **Grafana** (`grafana:3000`)
    - Metrics visualization dashboards
    - URL: http://localhost:3000
    - Default credentials: `admin`/`admin`

## Environment Configuration

### Database Credentials

The application uses consistent credentials across all services:

- **Database**: `aba`
- **User**: `aba`
- **Password**: `aba`

These credentials are defined in:
- `docker-compose.yml` (PostgreSQL service environment)
- `.env.example` (DATABASE_URL and POSTGRES_DSN)
- `ai_business_assistant/shared/config.py` (default values)

### Service Communication

Services communicate using Docker network DNS names:

- `postgres` - PostgreSQL database
- `redis` - Redis cache
- `rabbitmq` - RabbitMQ message broker
- `qdrant` - Vector database
- `auth` - Auth service
- `assistant` - Assistant service
- `gateway` - API Gateway

### Environment Variables

Each application service receives environment variables from docker-compose.yml:

```yaml
environment:
  LOG_LEVEL: INFO
  POSTGRES_DSN: postgresql+asyncpg://aba:aba@postgres:5432/aba
  REDIS_URL: redis://redis:6379/0
  RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672//
  CELERY_RESULT_BACKEND: redis://redis:6379/1
  QDRANT_URL: http://qdrant:6333
  JWT_SECRET_KEY: CHANGE_ME_CHANGE_ME_CHANGE_ME_CHANGE_ME
```

**Important**: Change `JWT_SECRET_KEY` to a secure random value in production!

## Local Development vs Docker

### Running with Docker

Use the `.env.example` values as-is (they're pre-configured for Docker):

```bash
cp .env.example .env
docker-compose up -d
```

### Running Locally (without Docker)

Modify `.env` to use localhost instead of Docker service names:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://aba:aba@localhost:5432/aba
POSTGRES_DSN=postgresql+asyncpg://aba:aba@localhost:5432/aba

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672//

# Qdrant
QDRANT_URL=http://localhost:6333

# API Gateway
API_GATEWAY_AUTH_SERVICE_URL=http://localhost:8001
API_GATEWAY_ASSISTANT_SERVICE_URL=http://localhost:8002
```

You'll need to run infrastructure services locally (PostgreSQL, Redis, etc.) before starting the application.

## Health Checks

All application services implement health checks:

```bash
# Gateway
curl http://localhost:8000/health

# Auth Service
curl http://localhost:8001/health

# Assistant Service
curl http://localhost:8002/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "gateway"  # or "auth_service" or "assistant_service"
}
```

## Dockerfile Details

### Root Dockerfile

The root `Dockerfile` is a multi-stage build:

1. **Builder Stage**
   - Installs build dependencies (gcc, g++, make, libpq-dev)
   - Installs Python packages from requirements.txt

2. **Runtime Stage**
   - Installs runtime dependencies (libpq5, curl)
   - Copies Python packages from builder
   - Creates non-root user for security
   - Sets up health check using curl

Health check command:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Docker Directory Dockerfile

The `docker/Dockerfile` is a simplified version used by docker-compose:

- Single-stage build using python:3.12-slim
- Installs only runtime dependencies
- Suitable for development and testing

## Troubleshooting

### Database Connection Errors

If you see "authentication failed" errors:

1. Verify credentials match between docker-compose.yml and .env
2. Check PostgreSQL is running: `docker-compose ps postgres`
3. View PostgreSQL logs: `docker-compose logs postgres`

### Services Not Starting

1. Check service logs: `docker-compose logs <service-name>`
2. Verify dependencies are healthy: `docker-compose ps`
3. Check environment variables are correctly set

### Health Check Failures

1. Ensure all services have `/health` endpoints
2. Check service is listening on correct port
3. Verify service dependencies are running

### Memory Issues

If services are consuming too much memory:

1. Reduce Docker resource limits in docker-compose.yml
2. Adjust database connection pool sizes in .env
3. Limit Celery worker concurrency

## Production Deployment

For production deployment, use `docker-compose.prod.yml` instead:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Key differences in production:
- Uses PostgreSQL 15-alpine (without pgvector)
- Requires setting `POSTGRES_PASSWORD` environment variable
- Enables health checks for dependencies
- Uses production-ready configurations

**Before deploying to production:**

1. Change all default passwords and secrets
2. Set `ENVIRONMENT=production`
3. Configure proper CORS origins
4. Enable HTTPS/TLS
5. Set up proper logging and monitoring
6. Configure backup strategies
7. Review and adjust resource limits

## Volumes

The following Docker volumes persist data:

- `postgres_data` - PostgreSQL database files
- `qdrant_data` - Qdrant vector database storage

To remove all data (including volumes):
```bash
docker-compose down -v
```

To remove only containers and networks (keeping data):
```bash
docker-compose down
```

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Celery with Docker](https://docs.celeryq.dev/en/stable/userguide/configuration.html#rabbitmq-examples)

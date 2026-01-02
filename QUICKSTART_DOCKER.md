# Quick Start Guide - Docker Setup

This guide will help you quickly set up and run the Business AI Assistant using Docker Compose.

## Prerequisites

- Docker 20.10 or higher
- Docker Compose 2.0 or higher
- At least 4GB RAM available
- 10GB free disk space

## Installation Steps

### 1. Clone and Navigate

```bash
cd /path/to/business-ai-assistant
```

### 2. Verify Configuration

Run the verification script to ensure everything is set up correctly:

```bash
python3 verify_docker_config.py
```

You should see "✓ All verification checks passed!" at the end.

### 3. Start All Services

```bash
# Start all services in detached mode
docker-compose up -d

# View startup logs
docker-compose logs -f
```

Wait for all services to start (approximately 1-2 minutes). You'll see logs showing each service starting up.

### 4. Verify Services are Running

Check that all services are healthy:

```bash
docker-compose ps
```

You should see all services with "Up" status.

### 5. Test Health Endpoints

```bash
# Gateway (port 8000)
curl http://localhost:8000/health

# Auth Service (port 8001)
curl http://localhost:8001/health

# Assistant Service (port 8002)
curl http://localhost:8002/health
```

Each should return: `{"status": "ok", "service": "<service_name>"}`

### 6. Access Services

Open your browser and access:

- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Flower (Celery Monitor)**: http://localhost:5555
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Common Tasks

### View Logs for a Specific Service

```bash
docker-compose logs -f <service-name>

# Examples:
docker-compose logs -f gateway
docker-compose logs -f auth
docker-compose logs -f assistant
docker-compose logs -f worker
```

### Restart a Service

```bash
docker-compose restart <service-name>
```

### Stop All Services

```bash
docker-compose stop
```

### Stop and Remove Containers

```bash
docker-compose down
```

### Stop and Remove Everything (Including Volumes)

```bash
docker-compose down -v
```

### Rebuild a Service

```bash
docker-compose build --no-cache <service-name>
docker-compose up -d <service-name>
```

### Scale Services

```bash
# Scale assistant service to 3 instances
docker-compose up -d --scale assistant=3
```

## Configuration

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to customize your configuration. See `.env.example` for all available options.

**Important**: Before deploying to production, change the default secrets:

```bash
# Generate a secure JWT secret key
openssl rand -hex 32

# Update .env with:
JWT_SECRET_KEY=<your-generated-secret>
SECRET_KEY=<your-generated-secret>
```

### Database Configuration

Default credentials (change in production):
- Database: `aba`
- Username: `aba`
- Password: `aba`

To connect directly to PostgreSQL:

```bash
docker-compose exec postgres psql -U aba -d aba
```

### Redis Configuration

To connect directly to Redis:

```bash
docker-compose exec redis redis-cli
```

## Troubleshooting

### Services Won't Start

1. Check if ports are already in use:
   ```bash
   sudo lsof -i :8000
   sudo lsof -i :5432
   sudo lsof -i :6379
   ```

2. Check service logs:
   ```bash
   docker-compose logs <service-name>
   ```

3. Verify Docker is running:
   ```bash
   docker ps
   docker-compose ps
   ```

### Database Connection Errors

1. Check PostgreSQL is healthy:
   ```bash
   docker-compose ps postgres
   ```

2. Verify environment variables:
   ```bash
   docker-compose config | grep POSTGRES_DSN
   ```

3. Check PostgreSQL logs:
   ```bash
   docker-compose logs postgres
   ```

### Out of Memory Errors

If services are crashing due to memory constraints:

1. Reduce the number of workers in docker-compose.yml
2. Adjust memory limits in docker-compose.yml
3. Close other applications to free up memory

### Health Check Failures

1. Verify the service is running:
   ```bash
   docker-compose ps <service-name>
   ```

2. Check service logs for errors:
   ```bash
   docker-compose logs <service-name>
   ```

3. Test the health endpoint manually:
   ```bash
   curl http://localhost:<port>/health
   ```

### Permission Errors

If you encounter permission issues:

1. Ensure you have proper Docker permissions:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in for changes to take effect
   ```

2. Fix file permissions for volumes:
   ```bash
   sudo chown -R $USER:$USER .
   ```

## Development Workflow

### Running Services Locally (Without Docker)

If you prefer to run services outside of Docker:

1. Install dependencies:
   ```bash
   pip install -r ai_business_assistant/requirements.txt
   ```

2. Set up local infrastructure (PostgreSQL, Redis, etc.)

3. Update `.env` to use localhost:
   ```bash
   DATABASE_URL=postgresql+asyncpg://aba:aba@localhost:5432/aba
   REDIS_URL=redis://localhost:6379/0
   ```

4. Run services:
   ```bash
   uvicorn ai_business_assistant.auth_service.main:app --host 0.0.0.0 --port 8001
   uvicorn ai_business_assistant.assistant_service.main:app --host 0.0.0.0 --port 8002
   uvicorn ai_business_assistant.gateway.main:app --host 0.0.0.0 --port 8000
   celery -A ai_business_assistant.worker.celery_app worker --loglevel=INFO
   ```

### Hot Reloading in Development

For development with hot reload:

```bash
# Install with dev dependencies
pip install -r ai_business_assistant/requirements.txt[dev]

# Run with auto-reload
uvicorn ai_business_assistant.gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

## Production Deployment

For production deployment, use the production compose file:

```bash
# Build and start with production configuration
docker-compose -f docker-compose.prod.yml up -d
```

**Important**: Before deploying to production:

1. Change all default passwords and secrets
2. Configure proper CORS origins
3. Enable HTTPS/TLS
4. Set up proper logging and monitoring
5. Configure backup strategies
6. Review and adjust resource limits
7. Use environment variables or secrets management for sensitive data

See `DOCKER_SETUP_GUIDE.md` for detailed production deployment instructions.

## Next Steps

- Read `DOCKER_SETUP_GUIDE.md` for detailed configuration
- Read `CONFIG_FIXES_SUMMARY.md` for information about recent fixes
- Explore the API documentation at http://localhost:8000/docs
- Set up monitoring dashboards in Grafana
- Configure backup strategies for data persistence

## Support

For issues or questions:
- Check logs: `docker-compose logs <service-name>`
- Verify configuration: `docker-compose config`
- Run verification: `python3 verify_docker_config.py`
- Read troubleshooting section above

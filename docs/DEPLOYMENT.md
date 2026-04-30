# PY-BRAIN API Deployment Guide

## Overview

This guide covers deploying the PY-BRAIN API service using Docker Compose. The service includes:
- FastAPI application
- Celery worker for async job processing
- PostgreSQL database
- Redis (Celery broker and backend)
- Orthanc PACS (optional)

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker runtime for GPU support

## Quick Start

### 1. Clone and Setup

```bash
cd /path/to/PY-BRAIN
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://pybrain:pybrain@postgres:5432/pybrain

# Redis
REDIS_URL=redis://redis:6379/0

# Storage
STORAGE_BACKEND=local
STORAGE_PATH=/data/pybrain

# Authentication
JWT_SECRET=change-me-in-production-random-string
API_KEYS=dev-api-key-123

# Application
ENVIRONMENT=development
DEBUG=true

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Pipeline
PIPELINE_DEVICE=cuda
PIPELINE_ANALYSIS_MODE=auto
```

### 3. Start Services

```bash
docker compose up -d
```

This will start:
- `api`: FastAPI application on port 8000
- `worker`: Celery worker
- `postgres`: PostgreSQL on port 5432
- `redis`: Redis on port 6379
- `orthanc`: Orthanc PACS on ports 4242 (DICOM) and 8042 (HTTP)
- `flower`: Celery monitoring on port 5555

### 4. Run Database Migrations

```bash
docker compose exec api alembic upgrade head
```

### 5. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# OpenAPI docs
open http://localhost:8000/docs
```

## Service Scaling

### Scale Workers

For higher throughput, scale the Celery workers:

```bash
docker compose up -d --scale worker=3
```

This will run 3 worker containers.

### GPU Support

Uncomment the GPU configuration in `docker-compose.yml`:

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Ensure NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Storage Configuration

### Local Storage (Default)

Data is stored in `./data/pybrain` on the host:

```yaml
volumes:
  - ./data/pybrain:/data/pybrain
```

### S3 Storage

For production, use S3-compatible storage:

```bash
STORAGE_BACKEND=s3
S3_BUCKET=pybrain-data
S3_REGION=us-east-1
S3_ENDPOINT_URL=https://s3.amazonaws.com
```

Or for MinIO (self-hosted):
```bash
STORAGE_BACKEND=s3
S3_BUCKET=pybrain-data
S3_REGION=us-east-1
S3_ENDPOINT_URL=http://minio:9000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL (async) | postgresql+asyncpg://pybrain:pybrain@localhost:5432/pybrain |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379/0 |
| `STORAGE_BACKEND` | Storage backend (local/s3) | local |
| `STORAGE_PATH` | Local storage path | /data/pybrain |
| `S3_BUCKET` | S3 bucket name | None |
| `S3_REGION` | S3 region | us-east-1 |
| `S3_ENDPOINT_URL` | Custom S3 endpoint | None |
| `JWT_SECRET` | JWT signing secret | change-me-in-production |
| `JWT_ALGORITHM` | JWT algorithm | HS256 |
| `JWT_EXPIRATION_HOURS` | JWT token expiration | 24 |
| `API_KEYS` | Comma-separated API keys | None |
| `ALLOWED_ORIGINS` | CORS allowed origins | http://localhost:3000,http://localhost:8080 |
| `MAX_UPLOAD_SIZE_MB` | Max upload size in MB | 500 |
| `MAX_CONCURRENT_JOBS` | Max concurrent segmentation jobs | 10 |
| `JOB_TIMEOUT_MINUTES` | Job timeout in minutes | 30 |
| `PIPELINE_DEVICE` | Pipeline inference device | cuda |
| `PIPELINE_ANALYSIS_MODE` | Default analysis mode | auto |

## Monitoring

### Flower (Celery Monitoring)

Flower is available at `http://localhost:5555`

Monitor:
- Active tasks
- Task success/failure rates
- Worker status
- Task execution time

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Readiness (checks Redis + DB)
curl http://localhost:8000/ready
```

### Logs

```bash
# API logs
docker compose logs -f api

# Worker logs
docker compose logs -f worker

# All logs
docker compose logs -f
```

## Production Considerations

### Security

1. **Change secrets**: Update `JWT_SECRET` and `API_KEYS`
2. **HTTPS**: Use Traefik or Nginx reverse proxy for TLS
3. **Network isolation**: Use Docker networks
4. **Volume encryption**: Encrypt data volumes at rest
5. **Audit logging**: Ensure audit logs are backed up

### Performance

1. **Worker scaling**: Scale workers based on load
2. **Connection pooling**: Configure database connection pool size
3. **Caching**: Add Redis caching for frequently accessed data
4. **CDN**: Use CDN for static file downloads

### High Availability

1. **Database**: Use managed PostgreSQL (RDS, Cloud SQL) with read replicas
2. **Redis**: Use managed Redis (ElastiCache) with clustering
3. **Load balancer**: Use multiple API instances behind a load balancer
4. **Storage**: Use S3 with cross-region replication

### Backup

```bash
# Backup database
docker compose exec postgres pg_dump -U pybrain pybrain > backup.sql

# Restore database
docker compose exec -T postgres psql -U pybrain pybrain < backup.sql

# Backup data volume
docker run --rm -v pybrain_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz /data
```

## Troubleshooting

### Database Connection Failed

```bash
# Check postgres is running
docker compose ps postgres

# Check postgres logs
docker compose logs postgres

# Restart postgres
docker compose restart postgres
```

### Redis Connection Failed

```bash
# Check redis is running
docker compose ps redis

# Check redis logs
docker compose logs redis

# Test redis connection
docker compose exec redis redis-cli ping
```

### Worker Not Processing Jobs

```bash
# Check worker logs
docker compose logs worker

# Check flower for worker status
open http://localhost:5555

# Restart worker
docker compose restart worker
```

### Out of Memory

```bash
# Check container resource usage
docker stats

# Increase memory limit in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Kubernetes Deployment

For production Kubernetes deployments, convert the Docker Compose to Kubernetes manifests using `kompose`:

```bash
pip install kompose
kompose convert -f docker-compose.yml -o k8s/
```

Or use Helm charts for more complex deployments.

## CI/CD

Example GitHub Actions workflow:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          docker compose -f docker-compose.prod.yml up -d
          docker compose exec api alembic upgrade head
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/PY-BRAIN/issues
- Documentation: https://github.com/your-org/PY-BRAIN/tree/main/docs

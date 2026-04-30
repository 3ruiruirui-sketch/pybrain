# Multi-stage Dockerfile for PY-BRAIN API
# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY pybrain /app/pybrain
COPY alembic /app/alembic
COPY alembic.ini /app/
COPY pybrain/config /app/pybrain/config

# Create non-root user
RUN useradd -m -u 1000 pybrain && \
    mkdir -p /data/pybrain && \
    chown -R pybrain:pybrain /app /data/pybrain

# Switch to non-root user
USER pybrain

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "pybrain.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

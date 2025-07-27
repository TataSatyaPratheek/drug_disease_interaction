# docker/Dockerfile - MULTI-STAGE BUILD

# --- Builder Stage ---
FROM python:3.11-slim AS builder
WORKDIR /app

# Install uv for fast, reliable dependency management
RUN pip install --no-cache-dir uv

# Copy only the files needed to install dependencies for better caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies into the system (not user) environment
RUN uv pip install --no-cache --system .

# --- Final Stage ---
FROM python:3.11-slim
WORKDIR /app

# Install only the system dependencies needed for runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check to ensure the API is fully responsive before accepting traffic
HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

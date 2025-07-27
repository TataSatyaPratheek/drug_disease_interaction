# docker/Dockerfile - CORRECTED ORDER

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Copy project configuration files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY README.md ./

# Now install dependencies with the source code available
RUN uv sync --frozen --no-dev

# Copy any remaining files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Use uv to run the application
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

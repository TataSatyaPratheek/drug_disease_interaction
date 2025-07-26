# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install pip build dependencies
RUN pip install --upgrade pip

# Copy dependency files and README for setuptools
COPY pyproject.toml README.md ./
COPY src/ ./src

# Install project dependencies using pip and setuptools
RUN pip install .

# Copy the rest of the application code
COPY src/ ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

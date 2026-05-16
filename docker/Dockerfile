# Use a lightweight Python base image
FROM python:3.12-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures that Python output is sent straight to terminal (useful for logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1 is required for XGBoost to run on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and necessary artifacts
# We copy specific directories to keep the image clean
COPY api/ api/
COPY models/ models/
COPY artifacts/ artifacts/
COPY src/ src/

# Create logs directory and ensure it's writable
RUN mkdir -p logs && chmod 777 logs

# Use a non-root user for better security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck to ensure the container is running correctly
# Uses the /health endpoint we defined in the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using a production-ready Uvicorn configuration
# 0.0.0.0 allows connections from outside the container
# Workers can be tuned based on the number of CPU cores
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

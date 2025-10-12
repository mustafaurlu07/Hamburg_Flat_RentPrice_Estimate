# Base image
FROM python:3.11-slim as base

# Metadata
LABEL maintainer="MLOps Team"
LABEL description="ML Rent Estimate Pipeline"
LABEL version="1.0.0"

# Sistem bağımlılıkları ve security update
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY run_pipeline.py .
COPY src/ ./src/
COPY data/ ./data/

# Model ve result klasörlerini oluştur
RUN mkdir -p models results

# Environment
ENV PYTHONPATH=/app

# Pipeline çalıştır
CMD ["python", "run_pipeline.py"]

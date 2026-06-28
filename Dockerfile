# syntax=docker/dockerfile:1
# Multi-stage build for a slim runtime image.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QRDP_DATA_DIR=/app/data

WORKDIR /app

# System deps occasionally needed by scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Copy the rest of the project.
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data

# Create runtime output dirs.
RUN mkdir -p /app/reports/figures /app/experiments

ENTRYPOINT ["quant-platform"]
CMD ["--help"]

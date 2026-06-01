# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for fast package installation
RUN pip install uv

# Copy requirements files
COPY requirements.txt .
COPY requirements-ml.txt .

# Install core dependencies (fast, always needed)
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Install ML dependencies (slower, uses CPU-only torch)
# Set to false to skip ML deps for faster builds
ARG INSTALL_ML_DEPS=true
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_ML_DEPS" = "true" ]; then \
        uv pip install --system -r requirements-ml.txt; \
    fi

# Copy application code
COPY . .

EXPOSE 8001

# Set default environment variables
ENV PORT=8001
ENV AGENTFIELD_SERVER=http://control-plane:8080

CMD ["python", "main.py"]

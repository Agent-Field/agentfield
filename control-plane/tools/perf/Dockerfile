# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY nested_workflow_stress.py ./
COPY entrypoint.sh ./

ENTRYPOINT ["/app/entrypoint.sh"]

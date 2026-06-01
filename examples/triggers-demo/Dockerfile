# AgentField triggers-demo agent.
#
# Build context expectation: the repo root (so we can copy the local Python
# SDK and install it in the image — the demo deliberately uses the in-tree
# SDK so changes to triggers DX flow through immediately, without waiting on
# a PyPI release).
#
# Build:
#   docker build -f examples/triggers-demo/Dockerfile -t agentfield/triggers-demo-agent .
# Or via docker-compose from this directory.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the in-tree SDK (with trigger DX from Phase 5).
COPY sdk/python /tmp/python-sdk
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/python-sdk && \
    rm -rf /tmp/python-sdk

COPY examples/triggers-demo/agent.py /app/agent.py

EXPOSE 8001
CMD ["python", "agent.py"]

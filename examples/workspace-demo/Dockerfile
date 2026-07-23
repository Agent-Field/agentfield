# Workspace Demo Agent -- container image
#
# Build context MUST be the repo root (not this directory) so the LOCAL
# sdk/python tree can be installed instead of a published package. Run from
# the repo root:
#
#   docker build -f examples/workspace-demo/Dockerfile -t workspace-demo .
#
# See deploy.md in this directory for the full build/deploy/register flow.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the local AgentField Python SDK (from this repo tree), not a
# published package -- keeps the demo in lockstep with in-repo SDK changes.
COPY sdk/python /tmp/python-sdk
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/python-sdk && \
    rm -rf /tmp/python-sdk

# Copy the demo agent source.
COPY examples/workspace-demo/main.py /app/main.py

EXPOSE 8001
CMD ["python", "main.py"]

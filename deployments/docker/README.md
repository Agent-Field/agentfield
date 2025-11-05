# AgentField Docker Deployments

This directory contains reference Dockerfiles and a Compose stack for local development.

## Images

- `Dockerfile.control-plane` – builds the Go control plane and embeds the web UI.
- `Dockerfile.python-agent` – base image for Python agents that bundles the SDK.
- `Dockerfile.go-agent` – base image for Go agents with the Go SDK pre-fetched.

## Local Stack

```bash
cd deployments/docker
docker compose up --build
```

The stack exposes the control plane on `http://localhost:8080` and provisions PostgreSQL + Redis.

Override configuration by editing `docker-compose.yml` or passing environment variables when running Compose.

# Development Guide

This document provides instructions for working on the AgentField monorepo locally.

## Prerequisites

- **Go** ≥ 1.23
- **Node.js** ≥ 20 (for the control plane web UI)
- **Python** ≥ 3.10
- **Docker** (optional, for running the full stack)

## Initial Setup

```bash
git clone https://github.com/Agent-Field/agentfield.git
cd agentfield
./scripts/install.sh
```

The install script performs:

- `go install` of required tooling (e.g., `golangci-lint`, `goose`).
- `pip install -e .` for the Python SDK and development dependencies.
- `npm install` inside `control-plane/web`.

## Directory Conventions

- `control-plane/` — Go services, migrations, and web UI.
- `sdk/go/` — Distributed as its own Go module (`go get` friendly).
- `sdk/python/` — Packaged with `pyproject.toml` for PyPI.
- `deployments/docker/` — Container builds to orchestrate the stack.
- `scripts/` — Automation entry points, used by CI and developers.

## Useful Commands

| Action                | Command                                                      |
| --------------------- | ------------------------------------------------------------ |
| Build everything      | `./scripts/build-all.sh`                                     |
| Run tests             | `./scripts/test-all.sh`                                      |
| Format Go code        | `make fmt`                                                   |
| Tidy Go modules       | `make tidy`                                                  |
| Run the control plane | `cd control-plane && go run cmd/server/main.go`              |
| Run UI in development | `cd control-plane/web && npm run dev`                        |
| Start local stack     | `docker compose -f deployments/docker/docker-compose.yml up` |

## Environment Variables

Copy `control-plane/config/.env.example` to `.env` (if available) and adjust:

- `AGENTFIELD_DATABASE_URL` — PostgreSQL connection string.
- `AGENTFIELD_JWT_SECRET` — Authentication secret (development only).

## Database Migrations

```bash
cd control-plane
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" status
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" up
```

## Frontend Development

The UI lives in `control-plane/web`. It is built with React + TypeScript.

```bash
cd control-plane/web
npm install
npm run dev
```

During development, run the Go server (`go run cmd/server/main.go`) for API endpoints. The UI uses environment variables in `.env.local`.

## Testing

```bash
# Control plane
cd control-plane
go test ./...

# Go SDK
cd ../sdk/go
go test ./...

# Python SDK
cd ../python
pytest
```

## Troubleshooting

- Ensure Docker resources are sufficient (4 CPU, 8 GB RAM recommended).
- Run `make tidy` if Go modules drift.
- Delete `.venv` and rerun `./scripts/install.sh` if Python deps conflict.
- Clear `control-plane/web/node_modules` if UI builds fail after dependency upgrades.

## Conventions

- Follow Go, Python (PEP 8), and TypeScript style guides.
- Keep environment-specific secrets out of the repository.
- Use feature flags or configuration flags for experimental features.

## Publishing Releases

See `docs/RELEASING.md` for end-to-end release steps, required secrets, and how to run dry-run builds via GitHub Actions.

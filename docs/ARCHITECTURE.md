# Haxen Architecture

Haxen provides a modular platform for orchestrating AI agents. The system is composed of a Go-based control plane, SDKs for client languages, and optional runtime services.

## High-Level Overview

```
┌───────────────────────────────────────────────────────────────┐
│                           Clients                             │
│   - Web UI (`control-plane/web`)                              │
│   - Python SDK (`sdk/python`)                                 │
│   - Go SDK (`sdk/go`)                                         │
└───────────────────────────────────────────────────────────────┘
                      │            │
                      ▼            ▼
┌───────────────────────────────────────────────────────────────┐
│                           Control Plane                       │
│   - REST + gRPC API (`cmd`, `internal/handlers`)              │
│   - Workflow engine (`internal/workflows`)                    │
│   - Credential management (`internal/services`)               │
│   - Persistence + migrations (`migrations`, `pkg/db`)         │
│   - Configuration (`config`)                                  │
└───────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────┐
│                      External Dependencies                    │
│   - PostgreSQL, Redis, vector stores (configurable)           │
│   - Observability stack (OpenTelemetry, Prometheus, etc.)     │
│   - Pluggable LLM providers (via SDKs)                        │
└───────────────────────────────────────────────────────────────┘
```

## Control Plane

The control plane orchestrates agent workflows, provides API endpoints, manages credentials, and serves a React-based administration UI.

- **`cmd/`** – entry points for binaries (HTTP server, background workers).
- **`internal/`** – business logic, services, repositories, and use-cases.
- **`pkg/`** – reusable packages (database, telemetry, utility helpers).
- **`proto/`** – Protocol Buffers and generated gRPC service definitions.
- **`web/`** – Control plane UI built with TypeScript + React.
- **`migrations/`** – Database schema migrations (Goose).
- **`config/`** – Default configuration, environment templates, sample secrets.

### Data Flow

1. SDKs or the web UI call into the REST/gRPC API.
2. The API handler invokes services in `internal/services`.
3. Services interact with repositories (`internal/repositories`) and utilities.
4. Background jobs and workflows are queued via `internal/workflows`.
5. Persistent state lives in PostgreSQL (default) and is versioned by `migrations`.

## SDKs

### Python (`sdk/python`)

- Thin client for the control plane REST API.
- Async and sync helpers for agent execution.
- Type hints for common primitives.
- PyPI-ready project built with `pyproject.toml`.

### Go (`sdk/go`)

- Idiomatic Go client with `agent`, `client`, `types`, and `ai` packages.
- Implements interfaces shared by the control plane.
- Ready for consumption via `go get`.

## Deployment

- `deployments/docker/Dockerfile.control-plane` builds the Go binary and bundles the web UI.
- `deployments/docker/Dockerfile.python-agent` and `Dockerfile.go-agent` provide reference runtime images.
- `deployments/docker/docker-compose.yml` orchestrates a local stack (control plane + dependencies).

## Extensibility

- Replace the default persistence layer by implementing interfaces in `pkg/db`.
- Add new transports (e.g., GraphQL, WebSockets) via `cmd/` entry points.
- Extend SDKs by adding new modules in their respective directories.

For operational details, see `docs/DEVELOPMENT.md` and `docs/SECURITY.md`.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentField is a Kubernetes-style control plane for AI agents. It provides production infrastructure for deploying, orchestrating, and observing multi-agent systems with cryptographic identity and audit trails.

**Architecture:** Three-tier monorepo
- **Control Plane** (Go): Orchestration server providing REST/gRPC APIs, workflow execution, observability, and cryptographic identity
- **SDKs** (Python & Go): Libraries for building agents that communicate with the control plane
- **Web UI** (React/TypeScript): Embedded admin interface for monitoring workflows and managing agents

## Development Setup

### Prerequisites
- Go 1.23+
- Python 3.10+
- Node.js 20+
- PostgreSQL 15+ (for cloud mode)

### Initial Setup
```bash
# Install all dependencies
make install

# Or install components individually:
./scripts/install.sh

# Build everything
make build
```

### Running the Control Plane

**Local mode** (uses SQLite + BoltDB, no external dependencies):
```bash
cd control-plane
go run ./cmd/af dev
# Or: go run ./cmd/agentfield-server
```

**Cloud mode** (requires PostgreSQL):
```bash
# Run migrations first
cd control-plane
export AGENTFIELD_DATABASE_URL="postgres://agentfield:agentfield@localhost:5432/agentfield?sslmode=disable"
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" up

# Start server
AGENTFIELD_STORAGE_MODE=postgresql \
AGENTFIELD_DATABASE_URL="postgres://agentfield:agentfield@localhost:5432/agentfield?sslmode=disable" \
go run ./cmd/agentfield-server
```

**Docker Compose** (includes PostgreSQL):
```bash
cd deployments/docker
docker compose up
```

Control plane runs at `http://localhost:8080`
Web UI accessible at `http://localhost:8080/ui/`

## Common Commands

### Building
```bash
make build                 # Build all components
make control-plane         # Build control plane only
make sdk-go               # Build Go SDK
make sdk-python           # Build Python SDK
```

### Testing
```bash
make test                 # Run all tests

# Component-specific tests:
cd control-plane && go test ./...
cd sdk/go && go test ./...
cd sdk/python && pytest

# Python tests with coverage:
cd sdk/python && pytest --cov=agentfield --cov-report=term-missing

# Web UI linting:
cd control-plane/web/client && npm run lint
```

### Linting & Formatting
```bash
make lint                 # Lint all code
make fmt                  # Format all code
make tidy                 # Tidy Go modules

# Component-specific:
cd control-plane && golangci-lint run
cd sdk/python && ruff check
cd sdk/python && ruff format .
```

### Database Migrations
```bash
cd control-plane
export AGENTFIELD_DATABASE_URL="postgres://agentfield:agentfield@localhost:5432/agentfield?sslmode=disable"

# Check migration status
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" status

# Apply all pending migrations
goose -dir ./migrations postgres "$AGENTFIELD_DATABASE_URL" up

# Create new migration
goose -dir ./migrations create <migration_name> sql
```

### Web UI Development
```bash
cd control-plane/web/client
npm install
npm run dev    # Runs on http://localhost:5173

# In parallel, run the control plane server to handle API calls
cd control-plane
go run ./cmd/agentfield-server
```

The UI dev server proxies API requests to the control plane. In production, the UI is embedded via Go's `embed` package.

## Architecture Deep Dive

### Control Plane Structure (`control-plane/`)

**Entry Points:**
- `cmd/agentfield/` - Unified CLI with server + dev/init commands
- `cmd/agentfield-server/` - Standalone server binary

**Core Packages (`internal/`):**
- `cli/` - CLI command definitions and routing
- `server/` - HTTP server setup (Gin framework), middleware, routing
- `handlers/` - HTTP request handlers for REST/gRPC endpoints
- `services/` - Business logic layer (workflow execution, agent registry, DID/VC generation)
- `storage/` - Data persistence layer with multiple backends (local SQLite/BoltDB, PostgreSQL, cloud)
- `events/` - Event bus for workflow notifications and SSE streaming
- `core/` - Domain models and interfaces
- `application/` - Application service orchestration
- `infrastructure/` - Infrastructure utilities (database connection pooling, etc.)
- `mcp/` - Model Context Protocol integration
- `logger/` - Structured logging (zerolog)
- `config/` - Configuration management (Viper)
- `templates/` - Code generation templates for `af init`
- `utils/` - Shared utilities
- `encryption/` - Cryptographic primitives for DID/VC
- `packages/` - Shared internal packages
- `embedded/` - Embedded assets (web UI dist)

**Configuration:**
- Environment variables take precedence over `config/agentfield.yaml`
- See `control-plane/.env.example` for all options
- Key modes: `AGENTFIELD_MODE=local` (SQLite/BoltDB) vs `AGENTFIELD_STORAGE_MODE=postgresql` (cloud)

**Database Schema:**
- `migrations/` - SQL migrations managed by Goose
- Always run migrations before starting the server in PostgreSQL mode

### SDK Structure

**Python SDK (`sdk/python/agentfield/`):**
- Built on FastAPI/Uvicorn for agent HTTP servers
- Key modules: `Agent`, `agent_field_handler`, `client`, `execution_context`, `memory`, `ai`
- Agents register "reasoners" (decorated functions) that become REST endpoints
- Test with: `pytest` (see `pyproject.toml` for test markers: unit, functional, integration)
- Install locally: `pip install -e .[dev]`

**Go SDK (`sdk/go/`):**
- Modules: `agent/` (agent builder), `client/` (HTTP client), `types/` (shared types), `ai/` (LLM helpers)
- Agents register "skills" (functions) similar to Python SDK
- Test with: `go test ./...`

### Web UI (`control-plane/web/client/`)
- React + TypeScript + Vite
- Tailwind CSS + Radix UI components
- Build: `npm run build` → outputs to `dist/` → embedded in Go binary
- Dev mode: `npm run dev` (separate Vite server)

## Key Workflows

### Creating a New Agent (Python)
```bash
# Generate agent scaffold (run from repo root or any directory)
af init my-agent
cd my-agent

# Edit agent code (auto-generated template)
# Run agent locally (connects to control plane at AGENTFIELD_SERVER env var or --server flag)
af run
```

### Creating a New Agent (Go)
```go
import agentfieldagent "github.com/Agent-Field/agentfield/sdk/go/agent"

agent, _ := agentfieldagent.New(agentfieldagent.Config{
    NodeID:   "my-agent",
    AgentFieldURL: "http://localhost:8080",
})
agent.RegisterSkill("greet", func(ctx context.Context, input map[string]any) (any, error) {
    return map[string]any{"message": "hello"}, nil
})
agent.Run(context.Background())
```

### Adding a New Control Plane Endpoint
1. Define handler in `control-plane/internal/handlers/<domain>/`
2. Add route in `control-plane/internal/server/routes.go`
3. Add business logic in `control-plane/internal/services/<domain>/`
4. Add storage methods in `control-plane/internal/storage/<domain>/`
5. If adding new DB tables, create migration: `goose -dir ./migrations create <name> sql`

### Storage Modes
- **Local mode:** SQLite (relational) + BoltDB (key-value). No external dependencies. Good for dev/testing.
- **PostgreSQL mode:** Full PostgreSQL backend. Requires running migrations. Production-ready.
- **Cloud mode:** PostgreSQL backend. Used in distributed deployments.

Storage interface is unified—services call storage layer methods, storage layer switches backends based on config.

## Testing Strategy

**Control Plane:**
- Unit tests: `go test ./...` (mock storage/services)
- Integration tests: Spin up test database, run migrations, test full stack

**Python SDK:**
- Markers: `@pytest.mark.unit`, `@pytest.mark.functional`, `@pytest.mark.integration`, `@pytest.mark.mcp`
- Default: `pytest` runs all except MCP tests (use `-m mcp` to include)
- Coverage tracked for core modules (see `pyproject.toml`)

**Go SDK:**
- Standard `go test ./...`
- Table-driven tests preferred

## Important Patterns

### Error Handling
- Control plane: Return structured JSON errors with HTTP status codes
- SDKs: Raise/return typed exceptions/errors with context
- Log errors before returning (use zerolog in Go, standard logging in Python)

### Configuration Precedence
1. Environment variables (highest priority)
2. Config file (`config/agentfield.yaml` or `AGENTFIELD_CONFIG_FILE` path)
3. Defaults in code

### Agent-to-Agent Communication
- Agents call each other via control plane: `await agent.call("other-agent.function", input={...})`
- Control plane routes requests, tracks workflow DAG, injects metrics
- Never direct agent-to-agent HTTP—always through control plane

### Memory Scopes
- **Global:** Shared across all agents/sessions
- **Agent:** Scoped to one agent, all sessions
- **Session:** Scoped to one session (multi-turn conversation)
- **Run:** Scoped to single execution/workflow run

Automatically synced by control plane. Agents access via SDK methods: `agent.memory.get/set(scope, key, value)`

### DID/VC (Cryptographic Identity)
- Opt-in per agent: Set `app.vc_generator.set_enabled(True)` in Python or equivalent in Go
- Control plane generates W3C Verifiable Credentials for each execution
- Export audit trails: `GET /api/v1/did/workflow/{workflow_id}/vc-chain`
- Verify offline: `af verify audit.json`

## Module Naming

**Control Plane (Go):**
- Use `github.com/Agent-Field/agentfield/control-plane` as module path
- Internal packages: `github.com/Agent-Field/agentfield/control-plane/internal/<package>`

**SDKs:**
- Python: `agentfield` (PyPI package)
- Go: `github.com/Agent-Field/agentfield/sdk/go` (import path)

## Release Process

Releases are automated via `.github/workflows/release.yml` and `.goreleaser.yml`:
- Tag a commit: `git tag v0.1.0 && git push origin v0.1.0`
- GitHub Actions builds binaries for multiple platforms
- `control-plane/build-single-binary.sh` creates unified binary (embeds web UI)

## Debugging Tips

- **Control plane not starting:** Check `AGENTFIELD_DATABASE_URL` is set correctly (PostgreSQL mode) or ensure SQLite file path is writable (local mode)
- **Migrations failing:** Ensure PostgreSQL is running and connection string is correct. Check migration status with `goose status`
- **Agent can't connect:** Verify `AGENTFIELD_SERVER` env var points to control plane (default: `http://localhost:8080`)
- **UI not loading:** In dev, ensure both Vite dev server (`npm run dev`) and control plane server are running. In prod, ensure `make build` was run to embed UI in binary
- **Agent execution stuck:** Check workflow DAG in UI (`/ui/workflows`) for errors. Check agent logs for exceptions.
- **Database connection pool exhausted:** Increase `AGENTFIELD_STORAGE_POSTGRES_MAX_CONNECTIONS` in config

## Environment Variables Reference

See `control-plane/.env.example` for comprehensive list. Key vars:
- `AGENTFIELD_PORT` - HTTP server port (default: 8080)
- `AGENTFIELD_MODE` - `local` or `cloud`
- `AGENTFIELD_STORAGE_MODE` - `local`, `postgresql`, or `cloud`
- `AGENTFIELD_DATABASE_URL` - PostgreSQL connection string
- `AGENTFIELD_UI_ENABLED` - Enable/disable web UI
- `AGENTFIELD_UI_MODE` - `embedded` (production) or `development` (Vite proxy)
- `AGENTFIELD_CONFIG_FILE` - Path to config YAML
- `GIN_MODE` - `debug` or `release`
- `LOG_LEVEL` - `debug`, `info`, `warn`, `error`

## Code Style

**Go:**
- Use `gofmt` for formatting (enforced by `make fmt`)
- Follow [Effective Go](https://go.dev/doc/effective_go) conventions
- Use zerolog for structured logging: `logger.Logger.Info().Msg("message")`

**Python:**
- Use Ruff for linting and formatting (`make fmt` runs `ruff format`)
- Type hints required for public APIs
- Async/await for I/O operations
- Follow PEP 8

**TypeScript/React:**
- Use ESLint config in `control-plane/web/client/.eslintrc.json`
- Functional components with hooks
- Tailwind for styling (no CSS-in-JS)

# AgentField Architecture

AgentField provides a modular platform for orchestrating AI agents. The system is composed of a Go-based control plane, SDKs for client languages, and optional runtime services.

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
│   - PostgreSQL, vector stores (configurable)                  │
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

### Observability Webhook

The control plane provides a one-way data push mechanism for external observability systems. This allows you to build custom dashboards, alerting, and analytics by ingesting real-time lifecycle events from the control plane.

**Purpose:**
- Stream lifecycle events about agents, reasoners, and executions
- Monitor what's online/offline across your agent fleet
- Track workflow execution progress and outcomes
- Build custom observability dashboards with your preferred tools

**Event Types:**

| Source | Events |
|--------|--------|
| Node | `node_registered`, `node_unregistered`, `node_status_changed` |
| Reasoner | `reasoner_registered`, `reasoner_unregistered`, `reasoner_status_changed` |
| Execution | `execution_created`, `execution_started`, `execution_completed`, `execution_failed` |

**Configuration:**

Configure via the REST API or Web UI:

```bash
# Enable webhook forwarding
curl -X POST http://localhost:8080/api/v1/ui/observability/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-endpoint.example.com/events",
    "secret": "your-hmac-secret",
    "headers": {"Authorization": "Bearer your-token"}
  }'
```

**Payload Format:**

Events are delivered in batches as JSON:

```json
{
  "batch_id": "uuid",
  "timestamp": "2025-01-15T10:30:00Z",
  "events": [
    {
      "event_type": "execution_completed",
      "event_source": "execution",
      "timestamp": "2025-01-15T10:29:59Z",
      "data": {
        "execution_id": "exec-123",
        "agent_id": "my-agent",
        "status": "completed"
      }
    }
  ]
}
```

**Security:**
- HMAC-SHA256 signature in `X-AgentField-Signature` header when secret is configured
- Custom headers for authentication (API keys, bearer tokens)

**Reliability:**
- Automatic retries with exponential backoff on delivery failures
- Dead letter queue (DLQ) for events that fail after max retries
- DLQ management via API: list failed events, redrive, or clear

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

# Environment Variables

This repo supports running AgentField in multiple modes (local binary, Docker, Kubernetes). Most configuration is loaded via a YAML config file and can be overridden via environment variables.

AgentField uses Viper with the prefix `AGENTFIELD` and maps nested config keys using `_` (for example `storage.mode` → `AGENTFIELD_STORAGE_MODE`).

## Control Plane (Server)

### Core

- `AGENTFIELD_PORT` (optional): HTTP port for the control plane (default: `8080`).
- `AGENTFIELD_CONFIG_FILE` (optional): Path to `agentfield.yaml` (in containers this is typically `/etc/agentfield/config/agentfield.yaml`).
- `AGENTFIELD_HOME` (recommended in containers): Base directory where AgentField stores local state (SQLite DB, Bolt DB, keys, logs). In Kubernetes, mount a PVC and set `AGENTFIELD_HOME=/data`.

### Storage

AgentField supports:
- **local** (SQLite + BoltDB, stored under `AGENTFIELD_HOME`)
- **postgres** (PostgreSQL + pgvector)

Common:
- `AGENTFIELD_STORAGE_MODE`: `local` (default) or `postgres`.

Local storage (usually not needed if `AGENTFIELD_HOME` is set):
- `AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH`: SQLite path.
- `AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH`: BoltDB path.

PostgreSQL storage:
- `AGENTFIELD_POSTGRES_URL` (preferred) or `AGENTFIELD_STORAGE_POSTGRES_URL`: PostgreSQL DSN/URL (examples below).
- Alternatively, individual fields:
  - `AGENTFIELD_STORAGE_POSTGRES_HOST`
  - `AGENTFIELD_STORAGE_POSTGRES_PORT`
  - `AGENTFIELD_STORAGE_POSTGRES_DATABASE`
  - `AGENTFIELD_STORAGE_POSTGRES_USER`
  - `AGENTFIELD_STORAGE_POSTGRES_PASSWORD`
  - `AGENTFIELD_STORAGE_POSTGRES_SSLMODE`

Example DSNs:
- `postgres://agentfield:agentfield@postgres:5432/agentfield?sslmode=disable`
- `postgresql://agentfield:agentfield@postgres:5432/agentfield?sslmode=disable`

### API Authentication (optional)

If set, the control plane requires an API key for most endpoints.

- `AGENTFIELD_API_KEY` or `AGENTFIELD_API_AUTH_API_KEY`: API key checked by the control plane.

### UI

- `AGENTFIELD_UI_ENABLED` (default: `true`)
- `AGENTFIELD_UI_MODE` (default: `embedded`)

### CORS (HTTP API)

These map to `api.cors.*` in config. When set via env, use comma-separated values.

- `AGENTFIELD_API_CORS_ALLOWED_ORIGINS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOWED_METHODS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOWED_HEADERS` (comma-separated)
- `AGENTFIELD_API_CORS_EXPOSED_HEADERS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOW_CREDENTIALS` (`true`/`false`)

### Authorization (VC-Based Permissions)

When enabled, the control plane issues DID identities to agents and enforces tag-based access policies on agent-to-agent calls.

- `AGENTFIELD_AUTHORIZATION_ENABLED` (default: `false`): Enable VC-based authorization.
- `AGENTFIELD_AUTHORIZATION_MASTER_SEED` (required when enabled): Master seed for deriving Ed25519 keypairs for agent DIDs. Keep this secret and consistent across restarts — changing it invalidates all existing DID signatures.
- `AGENTFIELD_AUTHORIZATION_TAG_APPROVAL_MODE` (default: `auto`): `auto` (tags approved immediately) or `admin` (tags require admin approval before the agent becomes ready).

### Connector (External Management API)

The connector API provides token-authenticated management endpoints for external systems (CI/CD, orchestration platforms, dashboards).

- `AGENTFIELD_CONNECTOR_TOKEN` (optional): Bearer token required for all `/connector/*` endpoints.
- `AGENTFIELD_CONNECTOR_CAPABILITIES` (optional, default: all): Comma-separated list of granted capabilities. Available capabilities: `reasoners:read`, `reasoners:write`, `versions:read`, `versions:write`, `restart`.

Example:
```
AGENTFIELD_CONNECTOR_TOKEN=my-secret-token
AGENTFIELD_CONNECTOR_CAPABILITIES=reasoners:read,versions:read,versions:write,restart
```

## Agent Nodes

Agent nodes run as separate processes/pods and register with the control plane. The most important Kubernetes-specific concept is:

- The **control plane must be able to reach the agent** at the URL the agent registers (its callback/public URL).
- In Kubernetes, this should usually be a `Service` DNS name (for example `http://my-agent.default.svc.cluster.local:8001`).

The same concept applies to **Docker**:

- If the control plane runs in a container and the agent runs on your host, set the agent’s callback/public URL to `host.docker.internal` (or the Docker host gateway on Linux).
- If both run in the same Docker network/Compose project, set the callback/public URL to the agent service name (for example `http://demo-go-agent:8001`).

### Go SDK agents (example: `examples/go_agent_nodes`)

- `AGENTFIELD_URL` (optional): Control plane base URL (example: `http://agentfield:8080`).
- `AGENTFIELD_TOKEN` (optional): Bearer token (use this if you enable `AGENTFIELD_API_KEY` on the control plane).
- `AGENT_NODE_ID` (optional): Node id (default varies by example).
- `AGENT_LISTEN_ADDR` (optional): Listen address (default: `:8001`).
- `AGENT_PUBLIC_URL` (recommended in Docker/Kubernetes): Public URL the control plane will call back to (example: `http://my-agent:8001`).

### Python SDK agents

- `AGENTFIELD_URL` (recommended): Control plane base URL.
- `AGENT_NODE_ID` (optional): Node id.
- `AGENT_CALLBACK_URL` (recommended in Docker/Kubernetes): URL the control plane will call back to (examples: `http://my-agent:8001`, or for host-run agents with Dockerized control plane: `http://host.docker.internal:8001`).

Many Python examples also require model provider credentials (for example `OPENAI_API_KEY`), depending on the `AIConfig` you choose.

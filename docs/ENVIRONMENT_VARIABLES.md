# Environment Variables

This repo supports running AgentField in multiple modes (local binary, Docker, Kubernetes). Most configuration is loaded via a YAML config file and can be overridden via environment variables.

AgentField uses Viper with the prefix `AGENTFIELD` and maps nested config keys using `_` (for example `storage.mode` → `AGENTFIELD_STORAGE_MODE`).

## Control Plane (Server)

### Core

- `AGENTFIELD_PORT` (optional): HTTP port for the control plane (default: `8080`).
- `AGENTFIELD_CONFIG_FILE` (optional): Path to `agentfield.yaml` (in containers this is typically `/etc/agentfield/config/agentfield.yaml`).
- `AGENTFIELD_HOME` (recommended in containers): Base directory where AgentField stores local state (SQLite DB, Bolt DB, keys, logs). In Kubernetes, mount a PVC and set `AGENTFIELD_HOME=/data`.

### Coding harness (aforge)

`af` distributes one harness CLI itself: `aforge`. Every install surface (the curl
installer, `af skill install --all`, the desktop app on launch, and the
`python-agent` / `go-agent` / cloud control-plane images) provisions the pinned
build into `$AGENTFIELD_HOME/bin/aforge` (`~/.agentfield/bin` by default), verified
against the published sha256 of the uncompressed binary.

- `AGENTFIELD_AFORGE_BASE_URL` (optional): Whole-base override for the download host,
  e.g. an internal mirror. The pinned version is **not** appended — the URL must
  already point at a directory holding `aforge-<os>-<arch>[.exe].gz` and
  `checksums.txt`. Default: `https://agentfield.ai/downloads/aforge/<pinned-version>`.
- `AGENTFIELD_SKIP_AFORGE` (optional): Set to `1` to make every aforge provisioning
  step a no-op — air-gapped hosts, or images that vendor their own harness.

Shell-installer equivalents: `--no-aforge` / `AFORGE_MODE=none` (`scripts/install.sh`),
`-NoAforge` / `$env:AFORGE_MODE='none'` (`scripts/install.ps1`).

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

It is optional only for a control plane used from the machine it runs on. The
endpoints that install packages or read and write credentials — package
install/update/uninstall, the secret store, agent `env` and agent `config` —
additionally require the caller to be on the local host while no key is set, and
refuse everything else with `401`. That covers the default single-user setup
without a key, and means any other topology needs one:

- another machine on the network, including a browser on your laptop pointed at
  a control plane running on a server
- a container, where a client on the host reaches the server through a bridge
  network rather than loopback
- anything behind a reverse proxy or tunnel

The reverse-proxy case needs a key for a different reason than the others: the
check looks at the connection's real peer address, so if the proxy runs on the
same host as the control plane, every forwarded request already looks local and
the restriction protects nothing. Forwarded headers such as `X-Forwarded-For`
are deliberately ignored here — trusting them would let any caller claim to be
local — so a proxied deployment must set a key.

Clients send the key as the `X-API-Key` header (or `Authorization: Bearer`). For
the CLI, `af auth login` stores it and every later command sends it
automatically.

### UI

- `AGENTFIELD_UI_ENABLED` (default: `true`)
- `AGENTFIELD_UI_MODE` (default: `embedded`)

### Anonymous Telemetry

Anonymous usage telemetry is enabled by default to help us improve AgentField. It records coarse product signals such as startup, agent registration, SDK language, runtime type, storage mode, and execution status buckets. Events use a pseudonymous, installation-scoped identifier; it represents an AgentField installation, not a person or account.

The telemetry payload does not include prompts, inputs, outputs, logs, secrets, API keys, IP addresses, hostnames, user IDs, DIDs, or raw error text. Sending is best-effort and does not affect control-plane or execution behavior.

- `AGENTFIELD_TELEMETRY_ENABLED` (default: `true`): Set to `false` to disable anonymous usage telemetry.
- `AGENTFIELD_TELEMETRY_ENDPOINT` (default: `https://agentfield.ai/api/oss/telemetry`): Hosted anonymous telemetry endpoint.
- `AGENTFIELD_TELEMETRY_INSTALL_ID` (optional): Stable externally managed installation ID. Use a random, opaque value—not an email, account name, hostname, or other identifying value. The control plane hashes it before sending.
- `AGENTFIELD_TELEMETRY_INSTALL_ID_PATH` (optional): Path for the persisted local install ID.
- `AGENTFIELD_TELEMETRY_TIMEOUT` (default: `800ms`): Per-event send timeout. Failures are ignored.

### CORS (HTTP API)

These map to `api.cors.*` in config. When set via env, use comma-separated values.

- `AGENTFIELD_API_CORS_ALLOWED_ORIGINS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOWED_METHODS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOWED_HEADERS` (comma-separated): Include `X-Admin-Token` when browser clients access admin or `/debug/pprof/*` endpoints.
- `AGENTFIELD_API_CORS_EXPOSED_HEADERS` (comma-separated)
- `AGENTFIELD_API_CORS_ALLOW_CREDENTIALS` (`true`/`false`)

### Authorization (VC-Based Permissions)

When enabled, the control plane issues DID identities to agents and enforces tag-based access policies on agent-to-agent calls.

- `AGENTFIELD_AUTHORIZATION_ENABLED` (default: `false`): Enable VC-based authorization.
- `AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN` (recommended): Separate token for admin operations and `/debug/pprof/*`; clients send it in `X-Admin-Token`. Without it, those endpoints rely on the global API key, so do not expose pprof outside trusted networks unless authentication is configured.
- `AGENTFIELD_AUTHORIZATION_MASTER_SEED` (required when enabled): Master seed for deriving Ed25519 keypairs for agent DIDs. Keep this secret and consistent across restarts — changing it invalidates all existing DID signatures.
- `AGENTFIELD_AUTHORIZATION_TAG_APPROVAL_MODE` (default: `auto`): `auto` (tags approved immediately) or `admin` (tags require admin approval before the agent becomes ready).
- `AGENTFIELD_AUTHORIZATION_DEFAULT_DENY` (default: `false`): When `true`, the tag policy middleware returns HTTP 403 for any request where no access policy matches the `(caller_tags, target_tags, function)` tuple. Default is `false`, preserving the existing behavior of allowing unmatched requests. The unmatched tuple is logged at `DEBUG` in both modes for diagnosis. Equivalent YAML: `features.did.authorization.default_deny`.

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
- `AGENTFIELD_DISABLE_IP_DETECTION` (optional, default off): Set to `1`, `true` or `yes` (case-insensitive, surrounding whitespace ignored) to stop the Python SDK from probing the cloud metadata services (`169.254.169.254` for AWS/Azure, `metadata.google.internal` for GCP) and `https://api.ipify.org` for the node's public address. On Kubernetes those requests are typically denied by a `NetworkPolicy` and show up as deny-log noise. In detail:
  - **What it disables.** The probe is one step of callback-URL discovery (`_detect_container_ip()`, whose only caller is `_build_callback_candidates()`), and it only runs when the SDK believes it is inside a container — `/.dockerenv` exists, `/proc/1/cgroup` mentions docker/containerd/kubepods, any `KUBERNETES_*` variable is set, or `CONTAINER` / `DOCKER_CONTAINER` / `RAILWAY_ENVIRONMENT` is set. This variable gates that single call site, so with it on the SDK makes no metadata or `api.ipify.org` request on any code path.
  - **When you would set it.** Discovery already skips the probe on its own as soon as it has a callback URL to start from — the `callback_url=` constructor argument or `AGENT_CALLBACK_URL` — so most deployments need nothing. Note also that in the current SDK an agent started with `app.serve()` only enters callback discovery when it was constructed with `callback_url=...`; given `AGENT_CALLBACK_URL`, or neither, `serve()` derives `base_url` itself and never calls the discovery helpers. Set this variable when you want the no-egress guarantee to hold regardless of how the agent is constructed, or when your code calls `_build_callback_candidates()` / `_resolve_callback_url()` / `AgentFieldHandler.register_with_agentfield_server()` directly.
  - **What it does not disable.** It is not a network kill switch. The SDK still determines its local address with a UDP `socket.connect()` toward `8.8.8.8:80`, which only asks the kernel which interface would be used — no packets are sent and no DNS lookup happens — and the agent still registers and heartbeats against the control plane as usual.
  - **What it costs.** Only the public-IP entry disappears from the callback candidate list. The Railway internal hostname, the node's local-network address, the container hostname, `host.docker.internal` and the localhost fallbacks are all still offered.

Many Python examples also require model provider credentials (for example `OPENAI_API_KEY`), depending on the `AIConfig` you choose.

### MiniMax video generation

- `MINIMAX_API_KEY`: API key used by the Python SDK's MiniMax media provider.
- `MINIMAX_BASE_URL` (optional): API base URL. Defaults to `https://api.minimax.io/v1`; use `https://api.minimaxi.com/v1` for the China endpoint.

MiniMax video models are routed with the `minimax/` model prefix. The model suffix is sent unchanged to the video generation API.

### OpenRouter attribution

OpenRouter attribution is request metadata, not authentication. AgentField SDKs send these as `HTTP-Referer`, `X-OpenRouter-Title`, and `X-Title` for OpenRouter requests.

- `AGENTFIELD_OPENROUTER_SITE_URL` (default: `https://agentfield.ai`)
- `AGENTFIELD_OPENROUTER_APP_NAME` (default: `AgentField AI`)
- `OR_SITE_URL`, `OR_APP_NAME`: LiteLLM-compatible attribution env vars.
- `AGENTFIELD_OPENROUTER_ATTRIBUTION=false`: Disable OpenRouter attribution headers/env propagation.

Explicit SDK config or explicit request headers win over env defaults. `AGENTFIELD_API_KEY`, SDK `api_key` / `apiKey`, Go `WithAPIKey`, and the `X-API-Key` header are only for AgentField control-plane authentication and are not used for OpenRouter attribution.

### Infron

- `INFRON_API_KEY`: API key for the Infron gateway. When it is the only gateway key set, the Go SDK's `ai.DefaultConfig()` points at `https://llm.onerouter.pro/v1` (`onerouter.pro` is the domain Infron serves its gateway from). `OPENAI_API_KEY` and `OPENROUTER_API_KEY` both keep precedence over it, so adding this key never reroutes an existing deployment.

Infron is OpenAI-compatible and serves the standard `<provider>/<model>` ids, so a model moves across by prefix alone (`infron/moonshotai/kimi-k2.6`). The `infron/` prefix is a routing marker only and is stripped before the request is sent, since the gateway serves the bare id.

Attribution is sent as `HTTP-Referer` and `X-Title`:

- `AGENTFIELD_INFRON_SITE_URL` (default: `https://agentfield.ai`)
- `AGENTFIELD_INFRON_APP_NAME` (default: `AgentField AI`)
- `AGENTFIELD_INFRON_ATTRIBUTION=false`: Disable Infron attribution headers.

When the `AGENTFIELD_INFRON_*` vars are unset, these OpenRouter attribution values are used as fallbacks, so a deployment that already declares its identity keeps it after switching gateways: `AGENTFIELD_OPENROUTER_SITE_URL`, `OR_SITE_URL`, `AGENTFIELD_OPENROUTER_APP_NAME`, `OR_APP_NAME`. The opt-out travels with them: when `AGENTFIELD_OPENROUTER_ATTRIBUTION=false`, these values are not inherited and the Infron defaults apply instead. To control Infron attribution specifically, set the `AGENTFIELD_INFRON_*` vars explicitly or disable it with `AGENTFIELD_INFRON_ATTRIBUTION=false`.

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

### Logging

- `AGENTFIELD_LOG_LEVEL` (default: `info`): Minimum severity written to stderr — `debug`, `info`, `warn` or `error`. Equivalent YAML: `logging.level`. The `--verbose` flag on `af` overrides both. A value that is not one of those four is reported once at `warn` on startup (`unrecognized log level, falling back to info`) and the server runs at `info`. Successful HTTP requests are logged at `debug`; a `404` at `info` (a request for a route that does not exist is routine noise, not an operator signal); the other 4xx responses at `warn` and 5xx at `error`, so failures stay visible at the default level and alerting keyed on `warn` is not tripped by scanners. Every request is logged, including the ones rejected before they reach a route: a disallowed `Origin` is answered with `403` by the CORS middleware and still produces one `warn` line.
- `AGENTFIELD_LOG_REDACT_PAYLOADS` (default: `true`): When `true`, execution inputs/outputs and agent response bodies are kept out of log events and internal event-bus payloads; log lines carry the media type, byte length and a short keyed digest (an HMAC under a key minted at process start, so the digest correlates repeats within a run without committing to the plaintext) instead. Set to `false` only for local debugging. Equivalent YAML: `logging.redact_payloads`.

### Miscellaneous control-plane knobs

- `AGENTFIELD_MAX_CONCURRENT_PER_AGENT` (default: `0`): Maximum concurrent executions dispatched to one agent; `0` means unlimited.
- `AGENTFIELD_EXEC_ASYNC_WORKERS` (default: the greater of the available CPU count and `16`): Worker count for asynchronous execution and restart jobs, which are I/O-bound; non-positive values use the default.
- `AGENTFIELD_EXEC_ASYNC_QUEUE_CAPACITY` (default: `1024`): Maximum number of asynchronous executions waiting for a worker; non-positive values use the default. Requests arriving once the queue is saturated are rejected with `503`, a `Retry-After` header and a `retry_after` field, and no execution row is persisted for them.
- `AGENTFIELD_MAX_EXECUTE_BODY_BYTES` (default: `33554432`, 32 MiB): Maximum request body size, in bytes, for POST routes under `/api/v1/execute`. Oversize requests are rejected with `413` before any execution is persisted; other routes are not capped by this setting.
- `AGENTFIELD_MAX_REGISTER_BODY_BYTES` (default: `8388608`, 8 MiB): Maximum request body size, in bytes, for node registration POST routes (`/api/v1/nodes`, `/api/v1/nodes/register`, and `/api/v1/nodes/register-serverless`). Oversize requests are rejected with `413` before registration handling begins.
- `AGENTFIELD_SHUTDOWN_TIMEOUT` (default: `30s`): Grace period for draining the control plane HTTP server during shutdown.
- `AGENTFIELD_AGENT_RESTART_GRACE` (default: `15s`): How long an execution waits for an agent process to return during a coordinated restart; a negative duration disables the wait.
- `AGENTFIELD_AGENT_DRAIN_GRACE` (default: `60s`): How long instance-scoped non-terminal work may keep completing after a replacement agent instance registers, before it is marked `agent_restart_orphaned`. The deferred in-memory timer is lost on a control-plane restart; the stale-execution sweep configured by `AGENTFIELD_EXECUTION_STALE_TIMEOUT` is the backstop. Equivalent YAML: `agentfield.node_health.agent_drain_grace`.

For Kubernetes, set `terminationGracePeriodSeconds` above the SDK drain window so the departing pod can return accepted work. Keep the agent `version` stable across rolling updates: changing it creates a separate versioned registration, so its work is recovered only by the stale sweep rather than this re-registration drain timer.

Rate limiting is off by default and has no dedicated environment-variable overrides. Configure the YAML-only `agentfield.rate_limit` block with `enabled`, `execute_rps`, `execute_burst`, `discovery_rps`, `discovery_burst`, `bulk_status_rps`, `bulk_status_burst`, `global_rps`, and `global_burst`.

Execution cleanup also has YAML keys `agentfield.execution_cleanup.max_retries` (default `0`) and `retry_backoff` (default `30s`), with environment overrides `AGENTFIELD_EXECUTION_MAX_RETRIES` and `AGENTFIELD_EXECUTION_RETRY_BACKOFF`. Despite its name, `max_retries` only rewinds stale workflow rows to `pending`; nothing re-dispatches those rows, so do not rely on it for execution retries.

### Execution cleanup and retention (control plane)

- `AGENTFIELD_EXECUTION_CLEANUP_ENABLED` (default: `true`): Runs stale-execution maintenance, terminal-row retention and payload garbage collection. Cleanup is on unless `agentfield.execution_cleanup.enabled` is set explicitly to `false`.
- `AGENTFIELD_EXECUTION_CLEANUP_INTERVAL` (default: `5m`): Interval between cleanup passes. Zero or negative values fall back to the default rather than spinning the ticker.
- `AGENTFIELD_EXECUTION_STALE_TIMEOUT` (default: `30m`): Age after which an inactive running execution is marked timed out.
- `AGENTFIELD_EXECUTION_RETENTION_PERIOD` (default: `0s`): How long finished execution rows are kept. `0s` keeps them forever — deletion is opt-in; `72h` prunes finished rows older than three days.
- `AGENTFIELD_EXECUTION_CLEANUP_BATCH_SIZE` (default: `200`): Maximum finished execution rows removed in one database transaction.
- `AGENTFIELD_EXECUTION_PRESERVE_RECENT` (default: `1h`): Window of recent executions that retention never deletes.
- `AGENTFIELD_PAYLOAD_ORPHAN_GRACE` (default: `1h`): Minimum age before an unreferenced payload file may be swept, so in-flight writes are not removed.
- `AGENTFIELD_AGENT_CALL_TIMEOUT` (default: `90s`): Timeout for HTTP calls from the control plane to agent nodes. Set to `0s` or a negative duration such as `-1s` to disable the timeout through the dispatch layer. Equivalent YAML: `agentfield.execution_queue.agent_call_timeout`.

All durations use Go duration syntax (`30s`, `5m`, `72h`). The same settings are available in YAML under `agentfield.execution_cleanup`; environment variables take precedence. `AGENTFIELD_EXECUTION_MAX_RETRIES` and `AGENTFIELD_EXECUTION_RETRY_BACKOFF` are described under Miscellaneous control-plane knobs above. The effective values are logged once at startup.

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

- `AGENTFIELD_CONNECTOR_ENABLED` (default: `false`): Set to `true` to expose the `/connector/*` endpoints.
- `AGENTFIELD_CONNECTOR_TOKEN` (optional): Bearer token required for all `/connector/*` endpoints.

Capabilities are granted one variable at a time. Each accepts `true` (full
access), `readonly` (GET only — writes are rejected with HTTP 403) or `false`.
**A capability that is not set is disabled**, so grant only what the connector
needs. These map to `features.connector.capabilities.<name>` in the YAML config.

- `AGENTFIELD_CONNECTOR_CAP_POLICY_MANAGEMENT`
- `AGENTFIELD_CONNECTOR_CAP_TAG_MANAGEMENT`
- `AGENTFIELD_CONNECTOR_CAP_DID_MANAGEMENT`
- `AGENTFIELD_CONNECTOR_CAP_REASONER_MANAGEMENT`
- `AGENTFIELD_CONNECTOR_CAP_STATUS_READ`
- `AGENTFIELD_CONNECTOR_CAP_OBSERVABILITY_CONFIG`
- `AGENTFIELD_CONNECTOR_CAP_CONFIG_MANAGEMENT`

Example:
```
AGENTFIELD_CONNECTOR_ENABLED=true
AGENTFIELD_CONNECTOR_TOKEN=my-secret-token
AGENTFIELD_CONNECTOR_CAP_STATUS_READ=true
AGENTFIELD_CONNECTOR_CAP_REASONER_MANAGEMENT=readonly
AGENTFIELD_CONNECTOR_CAP_DID_MANAGEMENT=false
```

## Agent Nodes

### Structured logging (SDKs)

- `AGENTFIELD_LOGS_ENABLED` (default: `true`): Enables Python, Go, and TypeScript agent-node stdout/stderr capture and the `/agentfield/v1/logs` endpoint. This controls capture, not control-plane execution-log dispatch.
- `AGENTFIELD_LOG_TRUNCATE` (Python default: `200` characters): Truncates human-readable plain log messages and visible plain-log payloads. It does not truncate structured records.
- `AGENTFIELD_LOG_PAYLOADS` (Python default: `false`): Shows payloads in human-readable plain logs when `true`. Structured execution attributes are unaffected.
- `AGENTFIELD_LOG_MAX_LINE_BYTES` (default: `16384`, minimum: `256`): Maximum emitted process-log line size in bytes. Values below 256 bytes are raised to the minimum. The Python structured stdout mirror elides attributes, then the message or entire record as needed, so every emitted line—including the complete JSON envelope—is valid JSON and fits this cap.
- `AGENTFIELD_LOG_BUFFER_BYTES` (default: `4194304`): Approximate total byte capacity of the in-memory process-log capture ring; oldest entries are discarded when full.

Agent nodes run as separate processes/pods and register with the control plane. The most important Kubernetes-specific concept is:

- The **control plane must be able to reach the agent** at the URL the agent registers (its callback/public URL).
- In Kubernetes, this should usually be a `Service` DNS name (for example `http://my-agent.default.svc.cluster.local:8001`).

The same concept applies to **Docker**:

- If the control plane runs in a container and the agent runs on your host, set the agent’s callback/public URL to `host.docker.internal` (or the Docker host gateway on Linux).
- If both run in the same Docker network/Compose project, set the callback/public URL to the agent service name (for example `http://demo-go-agent:8001`).

### Graceful shutdown (Go & TypeScript SDK agents)

- `AGENTFIELD_SHUTDOWN_TIMEOUT` (default: `30s`): How long a Go or TypeScript **agent node** waits for its in-flight executions to drain during graceful shutdown, triggered by SIGTERM or SIGINT. The Go SDK also exposes `POST /shutdown`; this is not a TypeScript SDK route. The setting accepts bare seconds (`30`) or a duration string (`30s`, `5m`); an invalid value logs a warning and falls back to the default. In the Go SDK, `Config.ShutdownTimeout` takes precedence over this variable.

When the deadline expires the agent cancels whatever is still running and allows up to 5 additional seconds for those executions to settle and report terminal status. Total shutdown time is therefore the configured timeout, plus up to 5 seconds of post-cancel settlement, plus the time required to notify the control plane.

Note that this is the **agent-node** meaning of the variable. The control plane reads the same variable name for a different purpose — the grace period for draining its own HTTP server (see "Miscellaneous control-plane knobs" above). They are separate processes, so one exported value applies to each independently; give them different values by setting the variable per process rather than globally.

The Python SDK's equivalent setting is documented under "Python SDK agents" below.

In Kubernetes, set the agent pod's `terminationGracePeriodSeconds` at least 10 seconds higher than `AGENTFIELD_SHUTDOWN_TIMEOUT` to leave room for post-cancel settlement and the control-plane notification before the container is killed.

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
- `AGENTFIELD_LOG_STDOUT` (optional, default on): Controls whether the Python and Go SDKs mirror structured execution records to stdout as JSON. Set to `0`, `false`, `no` or `off` (case-insensitive, surrounding whitespace ignored) to suppress the mirror; forwarding to the control plane continues unchanged. Any other value — including `1`, `true`, `yes`, an unset variable and a set-but-empty one — keeps the mirror on, so a typo cannot silently drop log output.
- `AGENTFIELD_SHUTDOWN_TIMEOUT` (optional, default `30s`): Graceful-shutdown budget for both direct HTTP requests and control-plane-dispatched reasoners. Accepts bare seconds (`30`), seconds (`30s`), or minutes (`5m`). `app.serve(timeout_graceful_shutdown=N)` remains supported and sets both budgets unless this environment variable is explicitly set; when both are set, the `serve()` argument still controls uvicorn's direct-HTTP drain while this variable controls dispatched reasoners.
- `AGENTFIELD_DISABLE_IP_DETECTION` (optional, default off): Set to `1`, `true` or `yes` (case-insensitive, surrounding whitespace ignored) to stop the Python SDK from probing the cloud metadata services (`169.254.169.254` for AWS/Azure, `metadata.google.internal` for GCP) and `https://api.ipify.org` for the node's public address. On Kubernetes those requests are typically denied by a `NetworkPolicy` and show up as deny-log noise. In detail:
  - **What it disables.** The probe is one step of callback-URL discovery (`_detect_container_ip()`, whose only caller is `_build_callback_candidates()`), and it only runs when the SDK believes it is inside a container — `/.dockerenv` exists, `/proc/1/cgroup` mentions docker/containerd/kubepods, any `KUBERNETES_*` variable is set, or `CONTAINER` / `DOCKER_CONTAINER` / `RAILWAY_ENVIRONMENT` is set. This variable gates that single call site, so with it on the SDK makes no metadata or `api.ipify.org` request on any code path.
  - **When you would set it.** Discovery already skips the probe on its own as soon as it has a callback URL to start from — the `callback_url=` constructor argument or `AGENT_CALLBACK_URL` — so most deployments need nothing. Note also that in the current SDK an agent started with `app.serve()` only enters callback discovery when it was constructed with `callback_url=...`; given `AGENT_CALLBACK_URL`, or neither, `serve()` derives `base_url` itself and never calls the discovery helpers. Set this variable when you want the no-egress guarantee to hold regardless of how the agent is constructed, or when your code calls `_build_callback_candidates()` / `_resolve_callback_url()` / `AgentFieldHandler.register_with_agentfield_server()` directly.
  - **What it does not disable.** It is not a network kill switch. The SDK still determines its local address with a UDP `socket.connect()` toward `8.8.8.8:80`, which only asks the kernel which interface would be used — no packets are sent and no DNS lookup happens — and the agent still registers and heartbeats against the control plane as usual.
  - **What it costs.** Only the public-IP entry disappears from the callback candidate list. The Railway internal hostname, the node's local-network address, the container hostname, `host.docker.internal` and the localhost fallbacks are all still offered.

Many Python examples also require model provider credentials (for example `OPENAI_API_KEY`), depending on the `AIConfig` you choose.

### Graceful shutdown (Python SDK)

On SIGTERM or a graceful `POST /shutdown`, the node immediately stops heartbeats, notifies the control plane that it is stopping, and drains in-progress reasoners before closing the callback client. A reasoner still running when `AGENTFIELD_SHUTDOWN_TIMEOUT` expires is cancelled and receives a terminal `cancelled` status whose reason identifies shutdown, so the execution is not left running indefinitely.

`af stop` waits **at least** the node's shutdown budget, not exactly that duration: its total wait also includes the initial HTTP request and, when needed, the signal fallback after the budget expires.

For Kubernetes, set `terminationGracePeriodSeconds` to a value greater than the shutdown budget. This leaves time for the terminal callback and normal process teardown after the reasoner drain. The older `Agent.setup_signal_handlers()` API is retained for compatibility, but `app.serve()` owns the production uvicorn-aware signal lifecycle.

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

### Harness (SDKs)

- `AGENTFIELD_HARNESS_DEPTH`: Marks subprocesses running inside an AgentField
  harness session. The SDKs set it to `1` for a first-level child and increment
  an inherited numeric value for nested sessions. An explicit per-call `env`
  value wins over the derived depth.

### Tracing (control plane)

- `AGENTFIELD_TRACING_ENABLED`: Set to `true` or `1` to enable tracing. Setting any of the endpoint variables below also enables it.
- `AGENTFIELD_TRACING_EXPORTER`: `otlp-http` (default) or `otlp-grpc`. For any other value, the control plane logs a startup warning, continues running, and leaves tracing disabled.
- `AGENTFIELD_TRACING_ENDPOINT`: AgentField-native endpoint setting (the env equivalent of `features.tracing.endpoint` in YAML).
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Standard generic OTLP endpoint.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`: Standard trace-specific OTLP endpoint.
- `AGENTFIELD_TRACING_INSECURE`: Set to `true` or `1` to force plaintext transport for a bare `host:port` endpoint. An explicit `http://` URL is already plaintext; `https://` retains TLS.
- `OTEL_SERVICE_NAME`: Service name attached to exported spans (default `agentfield`).

Endpoint precedence, highest first: `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, then `OTEL_EXPORTER_OTLP_ENDPOINT`, then `AGENTFIELD_TRACING_ENDPOINT`. When none is set, the default is `localhost:4318` for `otlp-http` and `localhost:4317` for `otlp-grpc`.

Each endpoint accepts either a bare `host:port` or a full `http://` / `https://` URL. For an invalid endpoint or unsupported scheme, the control plane logs a startup warning, continues running, and leaves tracing disabled. For HTTP export a URL without a path sends the trace signal to `/v1/traces`.

For a local OpenTelemetry Collector, use `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`. For Langfuse through an OTel Collector, point this variable at the collector's OTLP/HTTP listener and configure the collector's authenticated Langfuse export pipeline; use `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces` when specifying the trace signal URL directly.

---
date: "2026-04-20"
researcher: "codebase-analyzer (auth-model)"
git_commit: "0f42e612"
branch: "copilot-cli-support"
repository: "agentfield"
topic: "AgentField authentication and external credential model"
tags: [research, codebase, auth, did-vc, credentials]
status: complete
last_updated: "2026-04-20"
last_updated_by: "codebase-analyzer"
---

## Analysis: AgentField Authentication & External Credential System

### Overview

AgentField uses a layered authentication architecture. The control plane enforces a single static **API key** on all HTTP routes and optionally an **Ed25519 DID-based signature** scheme for caller identity. Agents register with the control plane at startup by POSTing their metadata (node ID, reasoners, skills, callback URL) to `/api/v1/nodes/register` carrying that API key. The control plane then issues each agent a **W3C Verifiable Credential identity package** (a set of Ed25519 key pairs encoded as `did:key` or `did:web` DIDs, one per agent, reasoner, and skill) which the agent caches locally to sign subsequent requests. There is no dedicated secrets store; external API keys (e.g., OpenAI) are read from the **agent process's own environment variables** by LiteLLM and are never sent to or stored by the control plane.

---

### 1. Control-Plane Authentication Middleware

#### 1a. API Key Auth (`control-plane/internal/server/middleware/auth.go`)

The primary gate is `APIKeyAuth` (`auth.go:18`). It is applied as a global Gin middleware at `server.go:979`:

```go
s.Router.Use(middleware.APIKeyAuth(middleware.AuthConfig{
    APIKey:    s.config.API.Auth.APIKey,
    SkipPaths: s.config.API.Auth.SkipPaths,
}))
```

**Token extraction order** (`auth.go:71–87`):
1. `X-API-Key` header
2. `Authorization: Bearer <token>` header
3. `?api_key=` query parameter

**Bypass rules** (no token required — `auth.go:38–69`):
- `/api/v1/health`, `/health`, `/metrics` — health/metrics
- Paths starting with `/ui` or exactly `/` — UI static files
- `GET /api/v1/did/document/` and `/api/v1/did/resolve/` — public DID resolution (W3C spec)
- `GET /api/v1/agentic/kb/` — public knowledge base
- `POST /api/v1/connector/` — handled by `ConnectorTokenAuth` separately

Comparison is constant-time (`crypto/subtle.ConstantTimeCompare` at `auth.go:89`). Failed auth sets `auth_level=public` (`auth.go:91`) and returns `HTTP 401`.

The API key is sourced from `AGENTFIELD_API_KEY` env var (applied in `config.go:367`) or `agentfield.yaml` under `api.auth.api_key` (`config.go:311`). It is empty by default, which disables all auth (`auth.go:26–29`).

#### 1b. Admin Token Auth (`middleware/auth.go:115–134`)

A separate `AdminTokenAuth` middleware requires the `X-Admin-Token` header for admin routes. Mounted at `server.go:1686`:

```go
adminGroup.Use(middleware.AdminTokenAuth(s.config.Features.DID.Authorization.AdminToken))
```

Configured via `AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN` (`config.go:489`) or `features.did.authorization.admin_token` in YAML. The default in `agentfield.yaml:99` is `"admin-secret"`.

#### 1c. Connector Token Auth (`middleware/connector_auth.go`)

The connector subsystem uses its own token scheme (`connector_auth.go:13`). All `/api/v1/connector/` routes are skipped by the global API key middleware (`auth.go:66–69`) and instead gated by `ConnectorTokenAuth`, which checks the `X-Connector-Token` header (`connector_auth.go:23`). Additionally it injects audit headers `X-Command-ID` and `X-Command-Source` into the Gin context (`connector_auth.go:33–39`). Per-route capability enforcement is done by `ConnectorCapabilityCheck` middleware (`connector_capability.go:14`), which reads `config.ConnectorCapability.Enabled` and `ReadOnly` flags.

The connector token comes from `features.connector.token` in YAML (`config.go:191`) or `AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN` (this env var sets the *internal* token, not the connector token directly).

#### 1d. gRPC Admin Auth (`middleware/grpc_auth.go`)

An admin gRPC server runs on port `cfg.AgentField.Port + 100` (default 8180). The `APIKeyUnaryInterceptor` interceptor checks `x-api-key` metadata or `Authorization: Bearer` metadata (`grpc_auth.go:27–36`). Installed at `server.go:603–608`:

```go
grpc.UnaryInterceptor(middleware.APIKeyUnaryInterceptor(s.config.API.Auth.APIKey))
```

#### 1e. DID Auth Middleware (`middleware/did_auth.go`)

`DIDAuthMiddleware` is an **optional** layer mounted only when `features.did.authorization.did_auth_enabled=true` (`server.go:988`, default `false` per `agentfield.yaml:95`). When active, it runs after the API key middleware.

**Flow** (`did_auth.go:156–308`):
1. If `X-Caller-DID` header is absent → skip, set `did_auth_skipped=true` (`did_auth.go:179–183`)
2. If present, require `X-DID-Signature` and `X-DID-Timestamp` headers (`did_auth.go:195–202`)
3. Parse timestamp and reject if outside ±300s window (`did_auth.go:204–222`, configurable via `timestamp_window_seconds`)
4. Read and restore request body up to 1 MB (`did_auth.go:225–241`)
5. Build verification payload: `"{timestamp}:{nonce}:{sha256(body)}"` (or `"{timestamp}:{sha256(body)}"` if no nonce) (`did_auth.go:246–252`)
6. Base64-decode signature (`did_auth.go:255–261`)
7. Replay protection: SHA256 the signature bytes and check global `signatureCache` (`did_auth.go:265–273`); the cache has a TTL equal to `TimestampWindowSeconds`
8. Call `didService.VerifyDIDOwnership(ctx, callerDID, payload, sigBytes)` to verify against the DID document's Ed25519 public key (`did_auth.go:276–302`)
9. On success, store verified DID in Gin context under `"verified_caller_did"` (`did_auth.go:306`)

---

### 2. DID/VC Identity System

#### DID Service Initialization (`control-plane/internal/services/did_service.go`)

On server startup, when `features.did.enabled=true`, the following initialization chain runs (`server.go:188–263`):

1. `KeystoreService` is created from `features.did.keystore` config (`server.go:204`). It stores encrypted key files at `cfg.Path` (default `./data/keys`) using AES-GCM with a session-ephemeral random key (`keystore_service.go:25–49`).

2. A `DIDRegistry` is created with `storage.StorageProvider`, and if `keystore.encryption_passphrase` is set, an `EncryptionService` is wired in (`server.go:211–213`). The `EncryptionService` uses AES-256-GCM with PBKDF2 (600,000 rounds) key derivation (`encryption.go:37–38`). Encrypted blobs are prefixed with `"AFENC2"` and versioned as `"v2:<base64>"` (`encryption.go:19–22`, `115–127`).

3. `DIDService` is initialized with a server ID derived from `AGENTFIELD_HOME` (`server.go:234`). On first run, it generates a 32-byte random master seed (`did_service.go:57–59`), derives a root DID at BIP32 path `m/44'/0'` (`did_service.go:63`), and stores the `DIDRegistry` struct persistently (`did_service.go:69–81`).

#### Agent DID Registration (`did_service.go:150–205`)

When an agent registers at `POST /api/v1/did/register` (handler: `did_handlers.go:92`), `DIDService.RegisterAgent` is called. For a new agent:

- **Agent DID**: derived at path `m/44'/{serverHash}'/{agentIndex}'` (`did_service.go:235`)
- **Reasoner DIDs**: `m/44'/{serverHash}'/{agentIndex}'/0'/{reasonerIndex}'` (`did_service.go:258`)
- **Skill DIDs**: `m/44'/{serverHash}'/{agentIndex}'/1'/{skillIndex}'` (`did_service.go:304`)

Key generation uses HKDF over the master seed (`did_service.go` imports `golang.org/x/crypto/hkdf`), producing Ed25519 key pairs. Each DID is encoded as `did:key:z6Mk...` (multibase-encoded public key). The private keys are returned in the response's `IdentityPackage` as JWK objects with `"kty":"OKP"`, `"crv":"Ed25519"`, `"d":"<base64url>"`, `"x":"<base64url>"`.

If the agent already exists, a differential analysis compares known reasoner/skill IDs with the new request and only generates keys for new components (`did_service.go:182–201`).

#### DID/VC Endpoints (`did_handlers.go:532–548`)

Registered under `/api/v1/did/`:

| Endpoint | Handler | Notes |
|---|---|---|
| `POST /api/v1/did/register` | `RegisterAgent` | Issues identity package |
| `GET /api/v1/did/resolve/:did` | `ResolveDID` | Resolves `did:web` (DB) or `did:key` (memory) |
| `POST /api/v1/did/verify` | `VerifyVC` | Verifies a VC document |
| `POST /api/v1/did/verify-audit` | `VerifyAuditBundle` | Verifies exported provenance JSON |
| `GET /api/v1/did/workflow/:id/vc-chain` | `GetWorkflowVCChain` | Chains execution VCs |
| `POST /api/v1/did/workflow/:id/vc` | `CreateWorkflowVC` | Creates workflow-level VC |
| `GET /api/v1/did/status` | `GetDIDStatus` | Returns `"active"` |
| `GET /api/v1/did/export/vcs` | `ExportVCs` | Exports all VCs |
| `GET /api/v1/did/document/:did` | `GetDIDDocument` | Returns W3C DID document |
| `POST /api/v1/execution/vc` | `CreateExecutionVC` | Creates per-execution VC |

W3C `did:web` resolution paths are also mounted at `/.well-known/did.json` (server DID) and `/agents/:agentID/did.json` (per-agent DID) (`server.go:1012–1013`).

#### VC Generation

VCs are W3C Verifiable Credentials. The `VCService.GenerateExecutionVC` creates a credential that records `issuer_did`, `target_did`, `caller_did`, `input_hash`, `output_hash`, `status`, `duration_ms`, and a signature (`vc_service.go`). These are stored in the control-plane database. The `EncryptConfigurationValues` / `DecryptConfigurationValues` methods on `EncryptionService` (`encryption.go:163–211`) can encrypt specific fields in config maps, used to protect the master seed at rest.

---

### 3. Agent Registration Flow

#### Python SDK (`sdk/python/agentfield/`)

**Init** (`agent.py:464–709`):
1. The control-plane URL is resolved: explicit `agentfield_server` param → `AGENTFIELD_SERVER` env var → `AGENTFIELD_SERVER_URL` env var → `"http://localhost:8080"` (`agent.py:556–558`)
2. `AgentFieldClient` is instantiated with `base_url` and `api_key` (`agent.py:606–608`). The `api_key` is stored and later injected as `X-API-Key` on every request (`client.py:220–222`)
3. If `enable_did=True` (the default), `DIDManager(agentfield_server, node_id, api_key)` and `VCGenerator(agentfield_server, api_key)` are created (`agent.py:1149–1167`)

**Registration** (`agent_field_handler.py:41–184`):
1. `AgentFieldClient.register_agent()` is called with node ID, reasoner/skill schemas, `base_url`, and `tags` (`client.py:630–711`)
2. The POST goes to `/api/v1/nodes/register` with `X-API-Key` in headers (`client.py:690–695`)
3. On success, if `did_manager` exists, `_register_agent_with_did()` is called (`agent_field_handler.py:159`)
4. `_register_agent_with_did()` calls `did_manager.register_agent(reasoner_defs, skill_defs)` which POSTs to `/api/v1/did/register` (`did_manager.py:100`, `agent.py:1578`)
5. The returned `identity_package` is stored in `did_manager.identity_package`; the agent DID and private key JWK are extracted and passed to `client.set_did_credentials(did, private_key_jwk)` (`agent.py:1585–1592`), which wires them into `DIDAuthenticator` for request signing

**Heartbeat** (`agent_field_handler.py:226–247`):
- POST to `/api/v1/nodes/{node_id}/heartbeat` with `X-API-Key` header every 30 seconds

#### Go SDK (`sdk/go/agent/agent.go`, `sdk/go/client/client.go`)

**Init** (`agent.go:353–427`):
1. `Config.AgentFieldURL` and `Config.Token` are set by the caller (no automatic env var lookup in the Go SDK for the URL)
2. `client.New(cfg.AgentFieldURL, client.WithBearerToken(cfg.Token))` is called (`agent.go:415`)
3. If `cfg.DID != ""` and `cfg.PrivateKeyJWK != ""`, `client.WithDIDAuth(cfg.DID, cfg.PrivateKeyJWK)` is also passed (`agent.go:416–418`)

**HTTP client `do()` method** (`client.go:178–253`):
- Sets `Authorization: Bearer <token>` if `c.token != ""` (`client.go:212–214`)
- Sets `X-API-Key: <key>` if `c.apiKey != ""` (`client.go:215–217`)
- Calls `c.didAuthenticator.SignRequest(bodyBytes)` and adds the resulting headers if DID auth is configured (`client.go:220–225`)

**DID request signing** (`client/did_auth.go:60–93`):
1. Generate Unix timestamp (`did_auth.go:66`)
2. Generate 16-byte random nonce hex-encoded (`did_auth.go:70–74`)
3. SHA256 the body (`did_auth.go:77`)
4. Build payload: `"{timestamp}:{nonce}:{bodyHash}"` (`did_auth.go:80`)
5. Sign with Ed25519 (`did_auth.go:83`)
6. Set headers `X-Caller-DID`, `X-DID-Signature` (base64), `X-DID-Timestamp`, `X-DID-Nonce` (`did_auth.go:88–93`)

**Incoming request auth (RequireOriginAuth)** (`agent.go:949–982`):
- When `RequireOriginAuth=true`, the `originAuthMiddleware` checks that `Authorization: Bearer <InternalToken>` matches (`agent.go:973–974`). `/health` and `/discover` are exempt (`agent.go:968`).
- `InternalToken` falls back to `Token` if empty (`agent.go:949–951`)

**Registered endpoints** (`agent.go:940`):
- `POST /execute/{reasoner}` — main invocation path
- `POST /reasoners/{reasoner}` — alternate invocation path
- `GET /health` — health check (no auth)
- `GET /discover` — capability discovery (no auth)

---

### 4. Secrets / External Credentials Storage

AgentField has **no secrets store** for external service credentials. There is no per-agent secrets API, no way to upload a GitHub token or OpenAI key to the control plane, and no mechanism to forward env vars from the control plane to agents.

**Config-level encrypted storage** (`encryption.go`):

The `EncryptionService` (`encryption.go:26`) can be used to encrypt/decrypt arbitrary strings with AES-256-GCM + PBKDF2. It is wired exclusively to protect the DID registry's **master seed** at rest (`server.go:211–213`). The `EncryptConfigurationValues` (`encryption.go:163`) and `DecryptConfigurationValues` (`encryption.go:188`) methods exist to encrypt named fields in `map[string]interface{}` config payloads, but there is no handler that accepts external service credentials through this path.

**Config storage API** (`handlers/config_storage.go`):
Routes `GET/PUT/DELETE /api/v1/configs/:key` store arbitrary key/value YAML strings in the database (`config_storage.go:30–36`). These are generic config blobs (not encrypted). There is no special handling for secrets here.

**Per-agent memory** (`agent/memory.go`, `sdk/python/agentfield/memory.py`):
The memory system stores scoped key/value data per execution/session/workflow/agent scope. It is not used as a secrets channel.

---

### 5. External AI Credentials (OpenAI/Anthropic)

#### Python SDK `AIConfig` (`sdk/python/agentfield/types.py`)

`AIConfig` (`types.py:313`) delegates entirely to **LiteLLM** for API key management:

- `api_key` field (`types.py:458`): if set explicitly, it is passed as `params["api_key"]` to `litellm.acompletion()` via `get_litellm_params()` (`types.py:644`)
- `api_base` field (`types.py:461`): custom base URL (e.g., for a proxy)
- If `api_key` is not set, LiteLLM reads provider-specific env vars automatically (per docstring `types.py:318–320`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, etc. — these are **read directly from the agent process environment**
- `fal_api_key` field (`types.py:380`): for Fal.ai media generation; if unset, Fal reads `FAL_KEY` env var

`AIConfig.from_env()` (`types.py:692`) simply calls `cls(**overrides)` — it does not read any env var itself; it relies entirely on LiteLLM's own env var handling at call time.

The `AgentAI` class (`agent_ai.py:143`) uses the agent's `ai_config` lazily; when `ai()` is called, it ultimately calls `litellm.acompletion()` with the parameters from `get_litellm_params()`.

**OpenAI direct mode** (`agent_ai.py:1294`): when `mode="openai_direct"` is passed, the code reads `config.get("api_key")` from the call-specific config and passes it to `openai.OpenAI(api_key=api_key)` (`agent_ai.py:1329–1335`).

#### Go SDK `ai.Config` (`sdk/go/ai/`)

The Go AI client is configured via `ai.Config` passed to `agent.Config.AIConfig`. The API key is stored in the config struct and passed directly to the AI provider calls (no env var auto-read visible in the harness path; each provider CLI subprocess inherits the parent process environment via `os.Environ()` in `harness/cli.go:55`).

---

### 6. Harness / External Coding Agent Credentials

The harness system (`sdk/go/harness/`, `sdk/python/agentfield/harness/`) dispatches prompts to CLI-based coding agents (`claude-code`, `codex`, `opencode`, `gemini`) by spawning subprocesses.

**Environment variable pass-through** (`harness/cli.go:47–71`):
```go
// Merge environment: empty values unset the variable.
for _, entry := range os.Environ() { ... }  // inherits ALL parent env vars
for k, v := range env {
    if v != "" {
        mergedEnv = append(mergedEnv, k+"="+v)
    }
}
c.Env = mergedEnv
```

The subprocess inherits **all** env vars from the agent process. The `Options.Env` map (`harness/provider.go:43`) can override or unset specific vars (empty string value = unset). This means that if `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GITHUB_TOKEN` are set in the agent process environment, they are transparently forwarded to the CLI subprocess.

**Claude Code** (`harness/claudecode.go:70–78`): builds an env override with `env["CLAUDECODE"] = ""` to unset the variable that prevents nested Claude Code sessions. All other env vars (including API keys) are inherited.

**`Options.Env` field** (`provider.go:43`): callers can inject per-call env vars. For example, to pass a `GITHUB_TOKEN` to a coding agent, the caller would set `Options.Env["GITHUB_TOKEN"] = "<token>"`. There is no special handling for this; the token arrives in the subprocess's environment.

---

### 6. Search Results for "GITHUB_TOKEN", "github", "oauth", "bearer", "copilot"

**In source code (`.go`, `.py`, `.ts`, `.yaml`):**

- No occurrences of `GITHUB_TOKEN`, `github_token`, `copilot`, or `oauth` appear in any Go, Python, TypeScript, or YAML source file. The grep returned zero results (`bash` command output above).

**In CI/CD workflows (`.github/workflows/`):**
- `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` appears in `release.yml:248`, `coverage-report.yml:25`, `coverage-badge.yml:117`, `memory-metrics-report.yml:27` — these are standard GitHub Actions automation tokens, not part of the AgentField application itself.
- The AI detection label workflow (`ai-label.yml:74`) contains a regex pattern that checks PR descriptions for references to `github.com/features/copilot` — this is used to auto-label AI-assisted PRs, not a credential.

**Bearer in source code:**
- `"Authorization: Bearer "` pattern is used in:
  - `auth.go:79` — control plane extracts from incoming `Authorization: Bearer` header
  - `grpc_auth.go:32` — gRPC interceptor extracts from `authorization` metadata
  - `client.go:213` — Go SDK client sets `Authorization: Bearer <token>` on outbound requests
  - `execute.go:1238` — control plane sets `Authorization: Bearer <internalToken>` when forwarding to agents
  - `agent.go:973`, `1028` — Go agent validates incoming `Authorization: Bearer <internalToken>`
  - `client.py` (implicitly, via `_get_auth_headers` setting `X-API-Key` — the Python SDK uses `X-API-Key`, not `Bearer`, by default)

---

### Data Flow Summary

```
1. Agent Process Startup
   └─ Python: reads AGENTFIELD_SERVER / AGENTFIELD_SERVER_URL env var (agent.py:557-558)
   └─ Go: AgentFieldURL / Token set explicitly in Config struct (agent.go:181-202)

2. Control-Plane Registration
   └─ POST /api/v1/nodes/register
       ├─ Header: X-API-Key (Python) or Authorization: Bearer (Go)
       └─ Body: {id, base_url, reasoners[], skills[], tags[], ...}

3. DID Registration (after node registration succeeds)
   └─ POST /api/v1/did/register
       ├─ Header: X-API-Key
       └─ Body: {agent_node_id, reasoners[], skills[]}
   └─ Response: {identity_package: {agent_did: {did, private_key_jwk, public_key_jwk, ...},
                                     reasoner_dids: {...}, skill_dids: {...}}}
   └─ Agent caches private_key_jwk → wired into DIDAuthenticator for future requests

4. Agent-to-Agent Execution (via control plane)
   └─ Caller POSTs to /api/v1/execute/{target}
       ├─ Header: X-API-Key (control-plane auth)
       ├─ Header: X-Caller-DID, X-DID-Signature, X-DID-Timestamp, X-DID-Nonce (optional DID auth)
   └─ Control plane validates API key + optional DID signature
   └─ Control plane forwards to target agent:
       ├─ Header: Authorization: Bearer <InternalToken>
       ├─ Header: X-Caller-DID, X-Target-DID (from DID auth)
       └─ Body: original execution payload

5. External AI Calls (within agent process)
   └─ OPENAI_API_KEY / ANTHROPIC_API_KEY / FAL_KEY read from agent's own process env
   └─ Passed to litellm.acompletion() (Python) or subprocess env (Go harness)
   └─ Never sent to / stored by the control plane
```

---

### Configuration: All Auth-Related Env Vars

| Env Var | Effect | Set In |
|---|---|---|
| `AGENTFIELD_SERVER` | Agent's control-plane URL (Python SDK) | `agent.py:557` |
| `AGENTFIELD_SERVER_URL` | Fallback control-plane URL (Python SDK) | `agent.py:558` |
| `AGENTFIELD_API_KEY` | Control-plane API key (server-side) | `config.go:367` |
| `AGENTFIELD_API_AUTH_API_KEY` | Same, alternate path | `config.go:371` |
| `AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN` | Admin routes token | `config.go:489` |
| `AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN` | Token CP sends to agents | `config.go:493` |
| `AGENTFIELD_AUTHORIZATION_DID_AUTH_ENABLED` | Enable DID middleware | `config.go:483` |
| `AGENTFIELD_AUTHORIZATION_DOMAIN` | Domain for `did:web` DIDs | `config.go:486` |
| `AGENT_CALLBACK_URL` | Agent's public callback URL | `agent.py:293` |
| `OPENAI_API_KEY` | OpenAI key (LiteLLM auto-reads) | agent process env |
| `ANTHROPIC_API_KEY` | Anthropic key (LiteLLM auto-reads) | agent process env |
| `FAL_KEY` | Fal.ai key | agent process env |
